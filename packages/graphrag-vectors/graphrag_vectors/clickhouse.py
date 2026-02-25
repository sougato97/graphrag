# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A package containing the ClickHouse vector store implementation."""

from __future__ import annotations

import math
import os
import re
from typing import Any
from urllib.parse import unquote, urlparse

from graphrag_vectors.vector_store import (
    VectorStore,
    VectorStoreDocument,
    VectorStoreSearchResult,
)

try:
    from clickhouse_driver import Client
except ImportError:  # pragma: no cover - exercised at runtime when dependency missing
    Client = None  # type: ignore[assignment]

try:
    import clickhouse_connect
except ImportError:  # pragma: no cover - exercised at runtime when dependency missing
    clickhouse_connect = None  # type: ignore[assignment]


VALID_FIELD_IDENTIFIER_REGEX = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
VALID_OBJECT_IDENTIFIER_REGEX = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]*$")
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


def _validate_identifier(name: str, value: str, regex: re.Pattern[str]) -> None:
    if not regex.match(value):
        msg = f"Unsafe or invalid {name}: {value}"
        raise ValueError(msg)


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


class ClickHouseVectorStore(VectorStore):
    """ClickHouse vector storage implementation.

    This implementation uses native ClickHouse protocol (clickhouse-driver) and
    computes similarity with cosineDistance at query-time.
    """

    def __init__(
        self,
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
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

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

        self.host = host or os.getenv("CLICKHOUSE_HOST", "127.0.0.1")
        self.port = int(port or os.getenv("CLICKHOUSE_PORT", "9000"))
        self.protocol = _resolve_protocol(protocol, parsed_scheme, self.port)
        self.database = (
            database
            or database_name
            or os.getenv("CLICKHOUSE_DB")
            or os.getenv("CLICKHOUSE_DATABASE")
            or "default"
        )
        self.user = user or os.getenv("CLICKHOUSE_USER", "default")
        self.password = (
            password
            if password is not None
            else os.getenv("CLICKHOUSE_PASSWORD", "")
        )
        self.secure = secure
        self.verify = verify
        self.settings = settings or {}

        _validate_identifier("database", self.database, VALID_OBJECT_IDENTIFIER_REGEX)
        _validate_identifier("index_name", self.index_name, VALID_OBJECT_IDENTIFIER_REGEX)
        _validate_identifier("id_field", self.id_field, VALID_FIELD_IDENTIFIER_REGEX)
        _validate_identifier(
            "vector_field", self.vector_field, VALID_FIELD_IDENTIFIER_REGEX
        )

        self._client: Any | None = None

    @property
    def _table_name(self) -> str:
        return f"`{self.database}`.`{self.index_name}`"

    def _require_client(self) -> Any:
        if self._client is None:
            msg = "ClickHouse client is not initialized. Call connect() first."
            raise ValueError(msg)
        return self._client

    def connect(self) -> Any:
        """Connect to ClickHouse vector storage."""
        if self.protocol == "http":
            if clickhouse_connect is None:
                msg = (
                    "clickhouse-connect is required for ClickHouse HTTP protocol. "
                    "Install with: pip install clickhouse-connect"
                )
                raise ImportError(msg)
            http_client = clickhouse_connect.get_client(
                host=self.host,
                port=self.port,
                username=self.user,
                password=self.password,
                database="default",
                secure=self.secure,
                verify=self.verify,
                settings=self.settings,
            )
            self._client = _HTTPClientAdapter(http_client)
        else:
            if Client is None:
                msg = "clickhouse-driver is required to use ClickHouseVectorStore."
                raise ImportError(msg)

            self._client = Client(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                secure=self.secure,
                verify=self.verify,
                settings=self.settings,
            )

        self._client.execute(f"CREATE DATABASE IF NOT EXISTS `{self.database}`")
        return self._client

    def create_index(self) -> None:
        """Create index."""
        client = self._require_client()

        client.execute(f"DROP TABLE IF EXISTS {self._table_name}")
        client.execute(
            f"""
            CREATE TABLE {self._table_name}
            (
                `{self.id_field}` String,
                `{self.vector_field}` Array(Float32)
            )
            ENGINE = MergeTree
            ORDER BY (`{self.id_field}`)
            """
        )

    def load_documents(self, documents: list[VectorStoreDocument]) -> None:
        """Load documents into ClickHouse."""
        client = self._require_client()

        valid_vectors = [doc.vector for doc in documents if doc.vector is not None]
        if not valid_vectors:
            return

        expected_size = len(valid_vectors[0])
        self.vector_size = expected_size
        rows = [
            (
                str(doc.id),
                [float(v) for v in doc.vector],
            )
            for doc in documents
            if doc.vector is not None and len(doc.vector) == expected_size
        ]
        if not rows:
            return

        client.execute(
            f"INSERT INTO {self._table_name} (`{self.id_field}`, `{self.vector_field}`) VALUES",
            rows,
        )

    def similarity_search_by_vector(
        self, query_embedding: list[float], k: int = 10
    ) -> list[VectorStoreSearchResult]:
        """Perform a vector-based similarity search."""
        client = self._require_client()
        if not query_embedding:
            return []

        query_vector = [float(v) for v in query_embedding]
        self.vector_size = len(query_vector)

        rows = client.execute(
            f"""
            SELECT
                `{self.id_field}`,
                `{self.vector_field}`,
                cosineDistance(`{self.vector_field}`, %(query_vector)s) AS distance
            FROM {self._table_name}
            WHERE length(`{self.vector_field}`) = %(vector_size)s
            ORDER BY distance ASC
            LIMIT %(k)s
            """,
            {
                "query_vector": query_vector,
                "vector_size": self.vector_size,
                "k": int(k),
            },
        )

        return [
            VectorStoreSearchResult(
                document=VectorStoreDocument(
                    id=row[0],
                    vector=row[1],
                ),
                # cosineDistance in ClickHouse is [0, 2], convert to similarity in [-1, 1]
                score=1.0 - float(row[2]),
            )
            for row in rows
        ]

    def search_by_id(self, id: str) -> VectorStoreDocument:
        """Search for a document by id."""
        client = self._require_client()

        rows = client.execute(
            f"""
            SELECT `{self.id_field}`, `{self.vector_field}`
            FROM {self._table_name}
            WHERE `{self.id_field}` = %(id)s
            LIMIT 1
            """,
            {
                "id": str(id),
            },
        )
        if rows:
            return VectorStoreDocument(
                id=rows[0][0],
                vector=rows[0][1],
            )
        return VectorStoreDocument(id=id, vector=None)
