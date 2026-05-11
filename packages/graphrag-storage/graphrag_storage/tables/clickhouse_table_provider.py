# Copyright (c) 2026 Sougato
# Licensed under the MIT License

"""ClickHouse-based table provider implementation."""

from __future__ import annotations

import base64
import hashlib
import inspect
import json
import logging
import math
import os
import re
from datetime import date, datetime
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlparse

import pandas as pd

from graphrag_storage.tables.table import RowTransformer, Table
from graphrag_storage.tables.table_provider import TableProvider

if TYPE_CHECKING:
    from graphrag_storage.storage import Storage

try:
    from clickhouse_driver import Client
except ImportError:  # pragma: no cover - exercised at runtime when dependency missing
    Client = None  # type: ignore[assignment]

try:
    import clickhouse_connect
except ImportError:  # pragma: no cover - exercised at runtime when dependency missing
    clickhouse_connect = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

VALID_IDENTIFIER_REGEX = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]*$")
PARAM_PLACEHOLDER_REGEX = re.compile(r"%\(([^)]+)\)s")
INSERT_REGEX = re.compile(
    r"^\s*INSERT\s+INTO\s+(.+?)\s*\((.*?)\)\s*VALUES\s*$",
    re.IGNORECASE | re.DOTALL,
)


class _HTTPClientAdapter:
    """Adapter to mimic clickhouse-driver's execute API over clickhouse-connect."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def execute(self, query: str, params: Any | None = None) -> list[tuple[Any, ...]]:
        sql = query.strip()

        if params is None:
            return self._run_sql(sql)

        if isinstance(params, dict):
            rendered = _render_query_with_params(sql, params)
            return self._run_sql(rendered)

        if isinstance(params, (list, tuple)):
            table_name, columns = _parse_insert_statement(sql)
            rows: list[list[Any] | tuple[Any, ...]] = []
            for row in params:
                if isinstance(row, (list, tuple)):
                    rows.append(row)
                else:
                    msg = f"Unsupported row type for INSERT payload: {type(row)!r}"
                    raise TypeError(msg)

            self._client.insert(table=table_name, data=rows, column_names=columns)
            return []

        msg = f"Unsupported parameter type for execute(): {type(params)!r}"
        raise TypeError(msg)

    def _run_sql(self, sql: str) -> list[tuple[Any, ...]]:
        if _is_result_query(sql):
            return list(self._client.query(sql).result_rows)
        self._client.command(sql)
        return []


def _identity(row: dict[str, Any]) -> Any:
    return row


def _apply_transformer(transformer: RowTransformer, row: dict[str, Any]) -> Any:
    if inspect.isclass(transformer):
        return transformer(**row)
    return transformer(row)


def _validate_identifier(name: str, value: str) -> None:
    if not VALID_IDENTIFIER_REGEX.match(value):
        msg = f"Unsafe or invalid {name}: {value}"
        raise ValueError(msg)


def _normalize_namespace(raw: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", raw).strip("_").lower()
    if not safe:
        safe = "graphrag"
    if safe[0].isdigit():
        safe = f"ns_{safe}"
    safe = safe[-40:]
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return f"{safe}_{digest}"


def _parse_connection_uri(connection_uri: str) -> dict[str, str | int]:
    parsed = urlparse(connection_uri)
    params: dict[str, str | int] = {}

    if parsed.scheme:
        params["scheme"] = parsed.scheme

    if parsed.hostname:
        params["host"] = parsed.hostname
    if parsed.port:
        params["port"] = parsed.port
    if parsed.username:
        params["user"] = unquote(parsed.username)
    if parsed.password:
        params["password"] = unquote(parsed.password)

    database = parsed.path.lstrip("/")
    if database:
        params["database"] = database
    return params


def _resolve_protocol(
    protocol: str | None, scheme: str | None, port: int | None
) -> str:
    normalized = (
        (protocol or os.getenv("CLICKHOUSE_PROTOCOL") or "").strip().lower()
    )
    if not normalized:
        if (scheme or "").lower() in {"http", "https"}:
            normalized = "http"
        elif port in {8123, 8124}:
            normalized = "http"
        else:
            normalized = "native"

    if normalized in {"native", "tcp"}:
        return "native"
    if normalized in {"http", "https"}:
        return "http"

    msg = f"Unsupported ClickHouse protocol '{normalized}'. Expected 'native' or 'http'."
    raise ValueError(msg)


def _is_result_query(sql: str) -> bool:
    head = sql.lstrip().lower()
    return head.startswith(("select", "show", "describe", "explain", "with"))


def _parse_insert_statement(sql: str) -> tuple[str, list[str]]:
    match = INSERT_REGEX.match(sql)
    if not match:
        msg = f"Could not parse INSERT statement for HTTP ClickHouse client: {sql}"
        raise ValueError(msg)

    table_expr = match.group(1).strip().replace("`", "").replace('"', "")
    columns_expr = match.group(2)
    columns = [col.strip().replace("`", "").replace('"', "") for col in columns_expr.split(",")]
    return table_expr, columns


def _render_query_with_params(sql: str, params: dict[str, Any]) -> str:
    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in params:
            msg = f"Missing SQL parameter '{key}' for query template."
            raise KeyError(msg)
        return _to_sql_literal(params[key])

    return PARAM_PLACEHOLDER_REGEX.sub(_replace, sql)


def _to_sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "NULL"
        return repr(value)
    if isinstance(value, str):
        return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_to_sql_literal(item) for item in value) + "]"
    return _to_sql_literal(str(value))


def _jsonify(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_jsonify(v) for v in value]

    if hasattr(value, "item"):
        try:
            return _jsonify(value.item())
        except Exception:  # noqa: BLE001
            return str(value)

    return str(value)


class ClickHouseTableProvider(TableProvider):
    """Table provider that stores arbitrary row dicts in ClickHouse.

    Rows are stored in a generic schema:
      - _row_idx: monotonic insertion index for deterministic ordering
      - id: optional row id extracted from row["id"]
      - _row_json: full row payload as JSON
    """

    def __init__(
        self,
        storage: Storage | None = None,
        db_uri: str | None = None,
        url: str | None = None,
        host: str | None = None,
        port: int | None = None,
        protocol: str | None = None,
        database: str | None = None,
        database_name: str | None = None,
        user: str | None = None,
        password: str | None = None,
        secure: bool = False,
        verify: bool = True,
        settings: dict[str, Any] | None = None,
        namespace: str | None = None,
        use_storage_path_as_namespace: bool = True,
        **kwargs: Any,
    ) -> None:
        connection_uri = db_uri or url
        parsed_scheme: str | None = None
        if connection_uri:
            parsed = _parse_connection_uri(connection_uri)
            host = host or parsed.get("host")  # type: ignore[assignment]
            port = port or parsed.get("port")  # type: ignore[assignment]
            user = user or parsed.get("user")  # type: ignore[assignment]
            password = password or parsed.get("password")  # type: ignore[assignment]
            database = database or parsed.get("database")  # type: ignore[assignment]
            parsed_scheme = parsed.get("scheme")  # type: ignore[assignment]

        self._host = host or os.getenv("CLICKHOUSE_HOST", "127.0.0.1")
        self._port = int(port or os.getenv("CLICKHOUSE_PORT", "9000"))
        self._protocol = _resolve_protocol(protocol, parsed_scheme, self._port)
        self._database = (
            database
            or database_name
            or os.getenv("CLICKHOUSE_TABLE_DB")
            or os.getenv("CLICKHOUSE_DB")
            or os.getenv("CLICKHOUSE_DATABASE")
            or "default"
        )
        self._user = user or os.getenv("CLICKHOUSE_USER", "default")
        self._password = (
            password
            if password is not None
            else os.getenv("CLICKHOUSE_PASSWORD", "")
        )
        self._secure = secure
        self._verify = verify
        self._settings = settings or {}
        self._storage = storage

        if namespace:
            self._namespace = _normalize_namespace(namespace)
        elif use_storage_path_as_namespace:
            self._namespace = _normalize_namespace(self._derive_namespace_seed(storage))
        else:
            self._namespace = "graphrag"

        _validate_identifier("database", self._database)
        _validate_identifier("namespace", self._namespace)

        self._client = self._connect()
        self._client.execute(f"CREATE DATABASE IF NOT EXISTS `{self._database}`")

    def _derive_namespace_seed(self, storage: Storage | None) -> str:
        if storage is None:
            return "graphrag_default"

        if hasattr(storage, "get_path"):
            try:
                return str(storage.get_path(""))  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass

        for attr in ("_base_dir", "_container_name", "_database_name"):
            if hasattr(storage, attr):
                value = getattr(storage, attr)
                if value:
                    return str(value)

        return storage.__class__.__name__

    def _connect(self) -> Any:
        if self._protocol == "http":
            if clickhouse_connect is None:
                msg = (
                    "clickhouse-connect is required for ClickHouse HTTP protocol. "
                    "Install with: pip install clickhouse-connect"
                )
                raise ImportError(msg)
            http_client = clickhouse_connect.get_client(
                host=self._host,
                port=self._port,
                username=self._user,
                password=self._password,
                database="default",
                secure=self._secure,
                verify=self._verify,
                settings=self._settings,
            )
            return _HTTPClientAdapter(http_client)

        if Client is None:
            msg = "clickhouse-driver is required to use ClickHouseTableProvider."
            raise ImportError(msg)
        return Client(
            host=self._host,
            port=self._port,
            user=self._user,
            password=self._password,
            secure=self._secure,
            verify=self._verify,
            settings=self._settings,
        )

    def _encode_table_name(self, table_name: str) -> str:
        encoded = base64.urlsafe_b64encode(table_name.encode("utf-8")).decode("ascii")
        encoded = encoded.rstrip("=")
        physical = f"{self._namespace}__{encoded}"
        _validate_identifier("table_name", physical)
        return physical

    def _decode_table_name(self, physical_name: str) -> str | None:
        prefix = f"{self._namespace}__"
        if not physical_name.startswith(prefix):
            return None
        encoded = physical_name[len(prefix) :]
        padding = "=" * ((4 - len(encoded) % 4) % 4)
        try:
            return base64.urlsafe_b64decode(encoded + padding).decode("utf-8")
        except Exception:  # noqa: BLE001
            return None

    def _table_name(self, table_name: str) -> str:
        return self._encode_table_name(table_name)

    def _table_sql(self, table_name: str) -> str:
        return f"`{self._database}`.`{self._table_name(table_name)}`"

    async def read_dataframe(self, table_name: str) -> pd.DataFrame:
        if not await self.has(table_name):
            msg = f"Could not find {table_name} in ClickHouse!"
            raise ValueError(msg)

        rows = self._client.execute(
            f"""
            SELECT `_row_json`
            FROM {self._table_sql(table_name)}
            ORDER BY `_row_idx`
            """
        )
        if not rows:
            return pd.DataFrame()
        payload = [json.loads(row[0]) for row in rows]
        return pd.DataFrame(payload)

    async def write_dataframe(self, table_name: str, df: pd.DataFrame) -> None:
        self._drop_table(table_name)
        self._ensure_table(table_name)
        records = df.to_dict(orient="records")
        self._insert_rows(table_name, records, start_idx=0)

    async def has(self, table_name: str) -> bool:
        rows = self._client.execute(
            """
            SELECT count()
            FROM system.tables
            WHERE database = %(database)s AND name = %(table)s
            """,
            {
                "database": self._database,
                "table": self._table_name(table_name),
            },
        )
        return bool(rows and rows[0][0] > 0)

    def list(self) -> list[str]:
        rows = self._client.execute(f"SHOW TABLES FROM `{self._database}`")
        names: list[str] = []
        for row in rows:
            decoded = self._decode_table_name(row[0])
            if decoded:
                names.append(decoded)
        return sorted(names)

    def open(
        self,
        table_name: str,
        transformer: RowTransformer | None = None,
        truncate: bool = True,
    ) -> Table:
        from graphrag_storage.tables.clickhouse_table import ClickHouseTable

        return ClickHouseTable(
            provider=self,
            table_name=table_name,
            transformer=transformer or _identity,
            truncate=truncate,
        )

    def _drop_table(self, table_name: str) -> None:
        self._client.execute(f"DROP TABLE IF EXISTS {self._table_sql(table_name)}")

    def _ensure_table(self, table_name: str) -> None:
        self._client.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table_sql(table_name)}
            (
                `_row_idx` UInt64,
                `id` Nullable(String),
                `_row_json` String
            )
            ENGINE = MergeTree
            ORDER BY (`_row_idx`)
            """
        )

    def _next_row_idx(self, table_name: str) -> int:
        self._ensure_table(table_name)
        rows = self._client.execute(
            f"""
            SELECT COALESCE(max(`_row_idx`), -1) + 1
            FROM {self._table_sql(table_name)}
            """
        )
        return int(rows[0][0]) if rows else 0

    def _insert_rows(
        self, table_name: str, rows: list[dict[str, Any]], start_idx: int = 0
    ) -> None:
        if not rows:
            return

        self._ensure_table(table_name)
        payload = []
        for i, row in enumerate(rows):
            normalized = {str(k): _jsonify(v) for k, v in row.items()}
            row_id = normalized.get("id")
            payload.append(
                (
                    int(start_idx + i),
                    str(row_id) if row_id is not None else None,
                    json.dumps(normalized, ensure_ascii=False),
                )
            )

        self._client.execute(
            f"INSERT INTO {self._table_sql(table_name)} (`_row_idx`, `id`, `_row_json`) VALUES",
            payload,
        )

    def _read_rows(self, table_name: str) -> list[dict[str, Any]]:
        self._ensure_table(table_name)
        rows = self._client.execute(
            f"""
            SELECT `_row_json`
            FROM {self._table_sql(table_name)}
            ORDER BY `_row_idx`
            """
        )
        return [json.loads(row[0]) for row in rows]

    def _row_count(self, table_name: str) -> int:
        if not self._client:
            return 0
        rows = self._client.execute(
            f"""
            SELECT count()
            FROM {self._table_sql(table_name)}
            """
        )
        return int(rows[0][0]) if rows else 0

    def _has_row_id(self, table_name: str, row_id: str) -> bool:
        if not self._client:
            return False
        self._ensure_table(table_name)
        rows = self._client.execute(
            f"""
            SELECT count()
            FROM {self._table_sql(table_name)}
            WHERE `id` = %(row_id)s
            LIMIT 1
            """,
            {"row_id": row_id},
        )
        return bool(rows and rows[0][0] > 0)
