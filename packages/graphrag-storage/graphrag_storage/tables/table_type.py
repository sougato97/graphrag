# Copyright (c) 2024 Microsoft Corporation.
# Modifications Copyright (c) 2026 Sougato
# Licensed under the MIT License


"""Builtin table storage implementation types."""

from enum import StrEnum


class TableType(StrEnum):
    """Enum for table storage types."""

    Parquet = "parquet"
    CSV = "csv"
    ClickHouse = "clickhouse"
