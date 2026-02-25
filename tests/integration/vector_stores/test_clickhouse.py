# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Integration-style tests for ClickHouse vector store implementation."""

import os
from uuid import uuid4
from unittest.mock import patch

import pytest
from graphrag_vectors import (
    IndexSchema,
    VectorStoreConfig,
    VectorStoreDocument,
    create_vector_store,
)
from graphrag_vectors.clickhouse import ClickHouseVectorStore

RUN_LIVE_CLICKHOUSE_TESTS = os.getenv("GRAPHRAG_TEST_CLICKHOUSE_LIVE", "0") == "1"


class TestClickHouseVectorStore:
    """Test class for ClickHouseVectorStore."""

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            VectorStoreDocument(
                id="doc1",
                vector=[0.1, 0.2, 0.3, 0.4, 0.5],
            ),
            VectorStoreDocument(
                id="doc2",
                vector=[0.2, 0.3, 0.4, 0.5, 0.6],
            ),
        ]

    @pytest.fixture
    def vector_store(self):
        """Create a ClickHouse vector store instance with mocked client."""
        with patch("graphrag_vectors.clickhouse.Client") as mock_client_class:
            vector_store = ClickHouseVectorStore(
                host="127.0.0.1",
                port=9001,
                database="applicant_analytics",
                user="clickhouse",
                password="clickhouse",
                index_name="test_vectors",
                vector_size=5,
            )

            vector_store.connect()
            mock_client = mock_client_class.return_value
            mock_client.execute.reset_mock()

            yield vector_store, mock_client

    def test_vector_store_operations(self, vector_store, sample_documents):
        """Test basic vector store operations with ClickHouse."""
        store, client = vector_store

        client.execute.side_effect = [
            [],  # drop table
            [],  # create table
            [],  # insert
            [
                ("doc1", [0.1, 0.2, 0.3, 0.4, 0.5], 0.1),
                ("doc2", [0.2, 0.3, 0.4, 0.5, 0.6], 0.2),
            ],  # similarity search
            [("doc1", [0.1, 0.2, 0.3, 0.4, 0.5])],  # search by id
        ]

        store.create_index()
        store.load_documents(sample_documents)

        vector_results = store.similarity_search_by_vector(
            [0.1, 0.2, 0.3, 0.4, 0.5], k=2
        )
        assert len(vector_results) == 2
        assert vector_results[0].document.id == "doc1"
        assert isinstance(vector_results[0].score, float)

        # Define a simple text embedder function for testing
        def mock_embedder(text: str) -> list[float]:
            return [0.1, 0.2, 0.3, 0.4, 0.5]

        text_results = store.similarity_search_by_text("test query", mock_embedder, k=2)
        assert len(text_results) == 2
        assert isinstance(text_results[0].score, float)

        doc = store.search_by_id("doc1")
        assert doc.id == "doc1"
        assert doc.vector is not None

    def test_vector_store_customization(self, sample_documents):
        """Test vector store customization with ClickHouse."""
        with patch("graphrag_vectors.clickhouse.Client") as mock_client_class:
            store = ClickHouseVectorStore(
                host="127.0.0.1",
                port=9001,
                database="applicant_analytics",
                user="clickhouse",
                password="clickhouse",
                index_name="text_embeddings",
                id_field="id_custom",
                vector_field="vector_custom",
                vector_size=5,
            )
            store.connect()
            client = mock_client_class.return_value
            client.execute.reset_mock()

            client.execute.side_effect = [
                [],  # drop table
                [],  # create table
                [],  # insert
                [("doc1", [0.1, 0.2, 0.3, 0.4, 0.5], 0.1)],  # similarity search
                [("doc1", [0.1, 0.2, 0.3, 0.4, 0.5])],  # search by id
            ]

            store.create_index()
            store.load_documents(sample_documents)

            vector_results = store.similarity_search_by_vector(
                [0.1, 0.2, 0.3, 0.4, 0.5], k=1
            )
            assert len(vector_results) == 1
            assert vector_results[0].document.id == "doc1"

            doc = store.search_by_id("doc1")
            assert doc.id == "doc1"
            assert doc.vector is not None

    def test_empty_embedding(self, vector_store):
        """Test similarity search by text with empty embedding."""
        store, client = vector_store

        def none_embedder(text: str):  # noqa: ANN201
            return None

        results = store.similarity_search_by_text("test query", none_embedder, k=1)
        assert not client.execute.called
        assert len(results) == 0


@pytest.mark.skipif(
    not RUN_LIVE_CLICKHOUSE_TESTS,
    reason="Set GRAPHRAG_TEST_CLICKHOUSE_LIVE=1 to run live ClickHouse tests.",
)
def test_clickhouse_live_roundtrip() -> None:
    """Round-trip test against a real ClickHouse container."""
    index_name = f"test_vectors_live_{uuid4().hex[:10]}"
    config = VectorStoreConfig(
        type="clickhouse",
        host=os.getenv("CLICKHOUSE_HOST", "127.0.0.1"),
        port=int(os.getenv("CLICKHOUSE_PORT", "9001")),
        database=os.getenv("CLICKHOUSE_DB", "applicant_analytics"),
        user=os.getenv("CLICKHOUSE_USER", "clickhouse"),
        password=os.getenv("CLICKHOUSE_PASSWORD", "clickhouse"),
    )
    schema = IndexSchema(
        index_name=index_name,
        id_field="id",
        vector_field="vector",
        vector_size=5,
    )

    store = create_vector_store(config, schema)
    store.connect()

    try:
        store.create_index()
        store.load_documents(
            [
                # doc1 should be the top hit for query vector below
                VectorStoreDocument(id="doc1", vector=[0.1, 0.2, 0.3, 0.4, 0.5]),
                VectorStoreDocument(id="doc2", vector=[0.2, 0.3, 0.4, 0.5, 0.6]),
            ]
        )

        doc = store.search_by_id("doc1")
        assert doc.id == "doc1"
        assert doc.vector is not None
        assert len(doc.vector) == 5

        vector_results = store.similarity_search_by_vector(
            [0.1, 0.2, 0.3, 0.4, 0.5], k=2
        )
        assert len(vector_results) >= 1
        assert vector_results[0].document.id == "doc1"

        text_results = store.similarity_search_by_text(
            "test query",
            lambda _text: [0.1, 0.2, 0.3, 0.4, 0.5],
            k=2,
        )
        assert len(text_results) >= 1
    finally:
        client = getattr(store, "_client", None)
        table_name = getattr(store, "_table_name", "")
        if client and table_name:
            client.execute(f"DROP TABLE IF EXISTS {table_name}")
