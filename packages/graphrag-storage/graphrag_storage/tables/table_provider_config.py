# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Storage configuration model."""

from pydantic import BaseModel, ConfigDict, Field

from graphrag_storage.tables.table_type import TableType


class TableProviderConfig(BaseModel):
    """The default configuration section for table providers."""

    model_config = ConfigDict(extra="allow")
    """Allow extra fields to support custom table provider implementations."""

    type: str = Field(
        description="The table type to use.",
        default=TableType.Parquet,
    )

    db_uri: str | None = Field(
        description="The database URI to use when type == clickhouse.",
        default=None,
    )

    url: str | None = Field(
        description="The database URL to use when type == clickhouse.",
        default=None,
    )

    host: str | None = Field(
        description="The database host to use when type == clickhouse.",
        default=None,
    )

    port: int | str | None = Field(
        description="The database port to use when type == clickhouse.",
        default=None,
    )

    protocol: str | None = Field(
        description="The ClickHouse protocol to use when type == clickhouse (native or http).",
        default=None,
    )

    database: str | None = Field(
        description="The database name to use when type == clickhouse.",
        default=None,
    )

    user: str | None = Field(
        description="The database user to use when type == clickhouse.",
        default=None,
    )

    password: str | None = Field(
        description="The database password to use when type == clickhouse.",
        default=None,
    )

    secure: bool = Field(
        description="Whether to use TLS when type == clickhouse.",
        default=False,
    )

    namespace: str | None = Field(
        description="Optional namespace prefix for clickhouse tables.",
        default=None,
    )

    use_storage_path_as_namespace: bool = Field(
        description="Whether clickhouse tables should be namespaced from the storage path.",
        default=True,
    )
