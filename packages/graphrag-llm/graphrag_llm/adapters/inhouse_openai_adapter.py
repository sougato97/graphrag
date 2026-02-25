# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Backward-compatible alias for OSS adapter."""

from graphrag_llm.adapters.oss_openai_adapter import OSSOpenAIAdapter


class InhouseOpenAIAdapter(OSSOpenAIAdapter):
    """Compatibility shim for legacy `model_provider: inhouse` references."""
