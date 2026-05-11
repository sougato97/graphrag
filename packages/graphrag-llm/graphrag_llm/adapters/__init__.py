# Copyright (c) 2026 Sougato
# Licensed under the MIT License

"""Adapters for custom model endpoint integrations."""

from graphrag_llm.adapters.oss_openai_adapter import OSSOpenAIAdapter

# Backward compatibility: keep old symbol name available.
InhouseOpenAIAdapter = OSSOpenAIAdapter

__all__ = ["OSSOpenAIAdapter", "InhouseOpenAIAdapter"]
