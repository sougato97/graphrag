# Copyright (c) 2026 Sougato
# Licensed under the MIT License

"""Adapter for custom open-source model endpoints to GraphRAG/OpenAI-like response objects."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import Any
from urllib import error, parse, request

from graphrag_llm.types import (
    LLMChoiceChunk,
    LLMChoiceDelta,
    LLMChoice,
    LLMCompletionChunk,
    LLMCompletionMessage,
    LLMCompletionResponse,
    LLMCompletionUsage,
    LLMEmbedding,
    LLMEmbeddingResponse,
    LLMEmbeddingUsage,
)


def _normalize_oss_model_name(model: str | None) -> str | None:
    """Normalize common short aliases to served model names."""
    if model is None:
        return None
    raw_model = model.strip()
    if not raw_model:
        return None

    alias_map = {
        "qwen2.5-7b-instruct-awq": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "qwen3-embedding-4b": "Qwen/Qwen3-Embedding-4B",
        "intfloat-multilingual-e5-large-instruct": "intfloat/multilingual-e5-large-instruct",
    }
    normalized = raw_model.lower().replace("_", "-")
    normalized = re.sub(r"^qwen(\d+)-(\d+)(-.+)$", r"qwen\1.\2\3", normalized)
    return alias_map.get(normalized, raw_model)


def _strip_v1_path(api_base: str) -> str:
    """Allow api_base values like http://host:8080/v1 and map to custom root."""
    parsed = parse.urlsplit(api_base.strip())
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return parse.urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _flatten_message_content(content: Any) -> str:
    """Flatten text or OpenAI-style multimodal content blocks to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if isinstance(text, str) and text.strip():
                    parts.append(text)
        return "\n".join(p for p in parts if p)
    if isinstance(content, dict):
        text = content.get("text") or content.get("content")
        if isinstance(text, str):
            return text
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def _extract_system_and_user_message(messages: Any) -> tuple[str | None, str]:
    """Extract system text and latest user message for custom /chat payload."""
    if isinstance(messages, str):
        return None, messages

    if not isinstance(messages, list):
        return None, _flatten_message_content(messages)

    system_parts: list[str] = []
    user_parts: list[str] = []
    fallback_parts: list[str] = []

    for message in messages:
        if not isinstance(message, dict):
            fallback_parts.append(_flatten_message_content(message))
            continue

        role = str(message.get("role") or "").lower()
        content = _flatten_message_content(message.get("content"))
        if not content:
            continue
        if role == "system":
            system_parts.append(content)
        elif role == "user":
            user_parts.append(content)
        else:
            fallback_parts.append(content)

    system = "\n".join(system_parts).strip() or None
    if user_parts:
        return system, user_parts[-1]
    if fallback_parts:
        return system, fallback_parts[-1]
    return system, ""


def _response_format_instruction(response_format: Any) -> str:
    """Build a strict JSON instruction for structured response requests."""
    if response_format is None:
        return ""

    instruction = (
        "Return ONLY a valid JSON object. Do not include markdown, code fences, "
        "or explanatory prose."
    )

    if isinstance(response_format, type) and hasattr(response_format, "model_json_schema"):
        try:
            schema = response_format.model_json_schema()
            properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
            required = schema.get("required", []) if isinstance(schema, dict) else []
            keys = ", ".join(sorted(str(k) for k in properties.keys()))
            required_keys = ", ".join(str(k) for k in required)
            if keys:
                instruction += f" Include keys: {keys}."
            if required_keys:
                instruction += f" Required keys: {required_keys}."
        except Exception:  # noqa: BLE001
            pass

    return instruction


@dataclass
class OSSOpenAIAdapter:
    """Convert GraphRAG/OpenAI-like args to custom oss endpoint requests."""

    api_base: str
    default_model: str | None
    call_args: dict[str, Any]
    timeout_s: float = 120.0
    stream_chunk_chars: int = 96

    def __post_init__(self) -> None:
        if not self.api_base:
            msg = "api_base must be provided when model_provider='oss'."
            raise ValueError(msg)
        root = _strip_v1_path(self.api_base).rstrip("/")
        self._chat_url = f"{root}/chat"
        self._embeddings_url = f"{root}/embeddings"

        self.default_model = _normalize_oss_model_name(self.default_model)
        self._default_backend = self.call_args.get("backend")
        self._default_framework = self.call_args.get("framework")
        self._default_temperature = self.call_args.get("temperature")
        self._default_max_tokens = self.call_args.get("max_tokens")
        self._default_prefix = self.call_args.get("prefix")
        self._default_normalize = self.call_args.get("normalize")

    def completion(
        self, **kwargs: Any
    ) -> LLMCompletionResponse | Iterator[LLMCompletionChunk]:
        """Sync chat completion using custom /chat endpoint."""
        stream = bool(kwargs.pop("stream", False))

        messages = kwargs.pop("messages", "")
        system, user_message = _extract_system_and_user_message(messages)
        response_format = kwargs.pop("response_format", None)

        json_instruction = _response_format_instruction(response_format)
        if json_instruction:
            system = f"{system}\n\n{json_instruction}" if system else json_instruction

        max_tokens = kwargs.pop("max_tokens", None)
        if max_tokens is None:
            max_tokens = kwargs.pop("max_completion_tokens", None)
        if max_tokens is None:
            max_tokens = self._default_max_tokens
        if max_tokens is None:
            # Structured outputs (response_format) often need larger generations.
            max_tokens = 2048 if response_format is not None else 512

        model = _normalize_oss_model_name(kwargs.pop("model", None)) or self.default_model

        payload: dict[str, Any] = {
            "message": user_message,
            "session_id": None,
            "system": system,
            "temperature": kwargs.pop(
                "temperature",
                self._default_temperature if self._default_temperature is not None else 0.2,
            ),
            "max_tokens": int(max_tokens),
        }
        if model:
            payload["model"] = model
        if self._default_backend is not None:
            payload["backend"] = self._default_backend
        if self._default_framework is not None:
            payload["framework"] = self._default_framework

        data = self._post_json(self._chat_url, payload)
        reply = str(data.get("reply") or "")
        session_id = str(data.get("session_id") or "oss-session")
        created = int(time.time())
        resolved_model = model or "oss/chat"

        completion_response = LLMCompletionResponse(
            id=session_id,
            object="chat.completion",
            created=created,
            model=resolved_model,
            choices=[
                LLMChoice(
                    index=0,
                    message=LLMCompletionMessage(
                        role="assistant",
                        content=reply,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=LLMCompletionUsage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            ),
            formatted_response=None,
        )
        if not stream:
            return completion_response

        return self._stream_completion_chunks(
            reply=reply,
            session_id=session_id,
            model=resolved_model,
            created=created,
        )

    async def completion_async(
        self, **kwargs: Any
    ) -> LLMCompletionResponse | AsyncIterator[LLMCompletionChunk]:
        """Async chat completion using custom /chat endpoint."""
        response = await asyncio.to_thread(self.completion, **kwargs)
        if not isinstance(response, Iterator):
            return response
        return self._stream_completion_chunks_async(response)

    def embedding(self, **kwargs: Any) -> LLMEmbeddingResponse:
        """Sync embedding call using custom /embeddings endpoint."""
        texts = kwargs.pop("input")
        if not isinstance(texts, list):
            texts = [str(texts)]

        model = _normalize_oss_model_name(kwargs.pop("model", None)) or self.default_model

        payload: dict[str, Any] = {
            "texts": [str(t) for t in texts],
        }
        if model:
            payload["model"] = model

        normalize = kwargs.pop("normalize", self._default_normalize)
        prefix = kwargs.pop("prefix", self._default_prefix)
        backend = kwargs.pop("backend", self._default_backend)
        framework = kwargs.pop("framework", self._default_framework)

        if normalize is not None:
            payload["normalize"] = bool(normalize)
        if prefix is not None:
            payload["prefix"] = prefix
        if backend is not None:
            payload["backend"] = backend
        if framework is not None:
            payload["framework"] = framework

        data = self._post_json(self._embeddings_url, payload)
        vectors = data.get("vectors")
        if not isinstance(vectors, list):
            msg = f"Invalid oss embeddings response, expected 'vectors' list: {data}"
            raise RuntimeError(msg)

        embedding_objects = [
            LLMEmbedding(
                object="embedding",
                embedding=list(vector),
                index=index,
            )
            for index, vector in enumerate(vectors)
        ]

        resolved_model = str(data.get("model") or model or "oss/embedding")
        return LLMEmbeddingResponse(
            object="list",
            data=embedding_objects,
            model=resolved_model,
            usage=LLMEmbeddingUsage(prompt_tokens=0, total_tokens=0),
        )

    async def embedding_async(self, **kwargs: Any) -> LLMEmbeddingResponse:
        """Async embedding call using custom /embeddings endpoint."""
        return await asyncio.to_thread(self.embedding, **kwargs)

    def _stream_completion_chunks(
        self,
        *,
        reply: str,
        session_id: str,
        model: str,
        created: int,
    ) -> Iterator[LLMCompletionChunk]:
        content_chunks = list(self._split_for_streaming(reply))
        if not content_chunks:
            content_chunks = [""]

        for idx, text_chunk in enumerate(content_chunks):
            delta: dict[str, Any] = {"content": text_chunk}
            if idx == 0:
                delta["role"] = "assistant"
            yield LLMCompletionChunk(
                id=session_id,
                object="chat.completion.chunk",
                created=created,
                model=model,
                choices=[
                    LLMChoiceChunk(
                        index=0,
                        delta=LLMChoiceDelta(**delta),
                        finish_reason=None,
                    )
                ],
            )

        yield LLMCompletionChunk(
            id=session_id,
            object="chat.completion.chunk",
            created=created,
            model=model,
            choices=[
                LLMChoiceChunk(
                    index=0,
                    delta=LLMChoiceDelta(),
                    finish_reason="stop",
                )
            ],
        )

    async def _stream_completion_chunks_async(
        self, chunks: Iterator[LLMCompletionChunk]
    ) -> AsyncIterator[LLMCompletionChunk]:
        for chunk in chunks:
            yield chunk
            await asyncio.sleep(0)

    def _split_for_streaming(self, text: str) -> list[str]:
        if not text:
            return []
        chunk_chars = max(1, int(self.stream_chunk_chars))
        return [text[i : i + chunk_chars] for i in range(0, len(text), chunk_chars)]

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=url,
            method="POST",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with request.urlopen(req, timeout=self.timeout_s) as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OSS endpoint HTTP {exc.code} at {url}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"OSS endpoint unreachable at {url}: {exc}") from exc

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"OSS endpoint returned non-JSON response at {url}: {raw}") from exc

        if not isinstance(parsed, dict):
            msg = f"OSS endpoint returned invalid JSON payload at {url}: {parsed}"
            raise RuntimeError(msg)
        return parsed
