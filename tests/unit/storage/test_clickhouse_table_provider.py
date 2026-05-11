# Copyright (c) 2026 Sougato
# Licensed under the MIT License

import re
import unittest
from unittest.mock import patch

import pandas as pd
from graphrag_storage.tables.clickhouse_table_provider import ClickHouseTableProvider


class _FakeClickHouseClient:
    def __init__(self, **kwargs):
        self.tables: dict[str, list[tuple[int, str | None, str]]] = {}

    def _parse_table(self, sql: str) -> str:
        match = re.search(r"`([^`]+)`\.`([^`]+)`", sql)
        if not match:
            return ""
        return f"{match.group(1)}.{match.group(2)}"

    def execute(self, sql: str, params=None):
        compact = " ".join(sql.strip().split())
        upper = compact.upper()

        if upper.startswith("CREATE DATABASE IF NOT EXISTS"):
            return []

        if upper.startswith("DROP TABLE IF EXISTS"):
            table = self._parse_table(compact)
            self.tables.pop(table, None)
            return []

        if upper.startswith("CREATE TABLE IF NOT EXISTS"):
            table = self._parse_table(compact)
            self.tables.setdefault(table, [])
            return []

        if upper.startswith("INSERT INTO"):
            table = self._parse_table(compact)
            self.tables.setdefault(table, [])
            self.tables[table].extend(params or [])
            return []

        if "FROM SYSTEM.TABLES" in upper:
            database = params["database"]
            name = params["table"]
            key = f"{database}.{name}"
            return [(1 if key in self.tables else 0,)]

        if upper.startswith("SHOW TABLES FROM"):
            db_match = re.search(r"SHOW TABLES FROM `([^`]+)`", compact, re.IGNORECASE)
            if not db_match:
                return []
            db = db_match.group(1)
            return [(k.split(".", 1)[1],) for k in self.tables if k.startswith(f"{db}.")]

        if "SELECT COALESCE(MAX(`_ROW_IDX`), -1) + 1" in upper:
            table = self._parse_table(compact)
            rows = self.tables.get(table, [])
            if not rows:
                return [(0,)]
            return [(max(r[0] for r in rows) + 1,)]

        if "SELECT `_ROW_JSON`" in upper and "ORDER BY `_ROW_IDX`" in upper:
            table = self._parse_table(compact)
            rows = sorted(self.tables.get(table, []), key=lambda r: r[0])
            return [(r[2],) for r in rows]

        if "SELECT COUNT()" in upper and "WHERE `ID` =" in upper:
            table = self._parse_table(compact)
            rows = self.tables.get(table, [])
            row_id = params["row_id"]
            return [(sum(1 for r in rows if r[1] == row_id),)]

        if "SELECT COUNT()" in upper:
            table = self._parse_table(compact)
            return [(len(self.tables.get(table, [])),)]

        return []


class TestClickHouseTableProvider(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        patcher = patch(
            "graphrag_storage.tables.clickhouse_table_provider.Client",
            _FakeClickHouseClient,
        )
        self.addCleanup(patcher.stop)
        patcher.start()

        self.table_provider = ClickHouseTableProvider(
            host="127.0.0.1",
            port=9001,
            database="applicant_analytics",
            user="clickhouse",
            password="clickhouse",
            namespace="test_unit",
            use_storage_path_as_namespace=False,
        )

    async def test_write_and_read_dataframe(self):
        df = pd.DataFrame(
            {
                "id": ["1", "2"],
                "name": ["Alice", "Bob"],
                "tags": [["x", "y"], ["z"]],
                "meta": [{"a": 1}, {"b": 2}],
            }
        )
        await self.table_provider.write_dataframe("users", df)
        result = await self.table_provider.read_dataframe("users")
        assert result.to_dict(orient="records") == df.to_dict(orient="records")

    async def test_has_and_list(self):
        df = pd.DataFrame({"id": ["1"], "name": ["Alice"]})
        await self.table_provider.write_dataframe("users", df)

        assert await self.table_provider.has("users") is True
        assert await self.table_provider.has("missing") is False
        assert "users" in self.table_provider.list()

    async def test_open_streaming_append(self):
        async with self.table_provider.open("events", truncate=True) as table:
            await table.write({"id": "1", "value": "a"})
            await table.write({"id": "2", "value": "b"})

        async with self.table_provider.open("events", truncate=False) as table:
            await table.write({"id": "3", "value": "c"})

        df = await self.table_provider.read_dataframe("events")
        records = df.to_dict(orient="records")
        assert [r["id"] for r in records] == ["1", "2", "3"]

    async def test_uses_configured_database_for_created_tables(self):
        custom_provider = ClickHouseTableProvider(
            host="127.0.0.1",
            port=9001,
            database="graphrag_tables_custom",
            user="clickhouse",
            password="clickhouse",
            namespace="db_scope_test",
            use_storage_path_as_namespace=False,
        )

        await custom_provider.write_dataframe(
            "entities", pd.DataFrame({"id": ["1"], "name": ["Acme"]})
        )

        created_tables = getattr(custom_provider._client, "tables", {})
        assert created_tables
        assert all(
            name.startswith("graphrag_tables_custom.") for name in created_tables
        )
