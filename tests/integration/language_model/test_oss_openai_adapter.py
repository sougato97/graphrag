# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Tests for the OSS adapter."""

from collections.abc import AsyncIterator, Iterator

import pytest

from graphrag_llm.adapters.oss_openai_adapter import OSSOpenAIAdapter
from graphrag_llm.types import LLMCompletionChunk


def _build_adapter() -> OSSOpenAIAdapter:
    return OSSOpenAIAdapter(
        api_base="http://localhost:8080/v1",
        default_model="qwen2.5-7b-instruct-awq",
        call_args={},
    )


def test_completion_stream_sync_returns_chunks() -> None:
    adapter = _build_adapter()
    adapter._post_json = lambda _u, _p: {"reply": "hello from adapter stream", "session_id": "sess-1"}  # type: ignore[method-assign]

    response = adapter.completion(messages="hi", stream=True)
    assert isinstance(response, Iterator)

    chunks = list(response)
    assert len(chunks) >= 2
    assert all(isinstance(chunk, LLMCompletionChunk) for chunk in chunks)

    content = "".join(chunk.choices[0].delta.content or "" for chunk in chunks)
    assert content == "hello from adapter stream"
    assert chunks[-1].choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_completion_stream_async_returns_async_chunks() -> None:
    adapter = _build_adapter()
    adapter._post_json = lambda _u, _p: {"reply": "hello async stream", "session_id": "sess-2"}  # type: ignore[method-assign]

    response = await adapter.completion_async(messages="hi", stream=True)
    assert isinstance(response, AsyncIterator)

    chunks = []
    async for chunk in response:
        chunks.append(chunk)

    assert len(chunks) >= 2
    assert all(isinstance(chunk, LLMCompletionChunk) for chunk in chunks)
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "hello async stream"
    assert chunks[-1].choices[0].finish_reason == "stop"
