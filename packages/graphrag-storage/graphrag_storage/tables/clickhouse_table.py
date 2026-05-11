# Copyright (c) 2026 Sougato
# Licensed under the MIT License

"""A ClickHouse-based implementation of the Table abstraction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from graphrag_storage.tables.table import RowTransformer, Table

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from graphrag_storage.tables.clickhouse_table_provider import ClickHouseTableProvider


class ClickHouseTable(Table):
    """Row-by-row table interface backed by ClickHouse."""

    def __init__(
        self,
        provider: "ClickHouseTableProvider",
        table_name: str,
        transformer: RowTransformer,
        truncate: bool = True,
    ):
        self._provider = provider
        self._table_name = table_name
        self._transformer = transformer
        self._truncate = truncate
        self._write_rows: list[dict[str, Any]] = []

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._aiter_impl()

    async def _aiter_impl(self) -> AsyncIterator[Any]:
        rows = self._provider._read_rows(self._table_name)
        for row in rows:
            yield self._transformer(row)

    async def length(self) -> int:
        return self._provider._row_count(self._table_name)

    async def has(self, row_id: str) -> bool:
        return self._provider._has_row_id(self._table_name, row_id)

    async def write(self, row: dict[str, Any]) -> None:
        self._write_rows.append(row)

    async def close(self) -> None:
        if not self._write_rows:
            return

        if self._truncate:
            self._provider._drop_table(self._table_name)
            self._provider._ensure_table(self._table_name)
            start_idx = 0
        else:
            start_idx = self._provider._next_row_idx(self._table_name)

        self._provider._insert_rows(
            self._table_name,
            self._write_rows,
            start_idx=start_idx,
        )
        self._write_rows = []
