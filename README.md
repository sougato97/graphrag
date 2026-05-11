# GraphRAG with ClickHouse and OSS Model Support

👉 [Microsoft Research Blog Post](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/)<br/>
👉 [Read the docs](https://microsoft.github.io/graphrag)<br/>
👉 [GraphRAG Arxiv](https://arxiv.org/pdf/2404.16130)<br/>
👉 [Original GraphRAG Repository](https://github.com/microsoft/graphrag)

<div align="left">
  <a href="https://pypi.org/project/graphrag/">
    <img alt="PyPI - Version" src="https://img.shields.io/pypi/v/graphrag">
  </a>
  <a href="https://pypi.org/project/graphrag/">
    <img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/graphrag">
  </a>
  <a href="https://github.com/microsoft/graphrag/issues">
    <img alt="GitHub Issues" src="https://img.shields.io/github/issues/microsoft/graphrag">
  </a>
  <a href="https://github.com/microsoft/graphrag/discussions">
    <img alt="GitHub Discussions" src="https://img.shields.io/github/discussions/microsoft/graphrag">
  </a>
</div>

## Overview

This repository builds on Microsoft's GraphRAG and adapts it for self-hosted and open-source deployments.

The main goal of this fork is to make GraphRAG practical for environments where you want:

- ClickHouse as the backing store for GraphRAG tabular outputs
- ClickHouse as the vector store for embeddings
- Open-source or local LLMs and embedding models exposed through OpenAI-compatible APIs
- Streaming support for OSS-backed query flows without losing compatibility with existing GraphRAG behavior

The upstream GraphRAG methodology, documentation, and research links remain relevant. This fork extends the implementation for a more database-centric and self-hosted setup.

For model serving, this repo is designed to work well with [ModelServer](https://github.com/sougato97/ModelServer), a Docker-first inference stack for chat and embeddings with a unified API gateway and support for OpenAI-compatible self-hosted flows.

## What This Fork Adds

- `ClickHouse` table provider support so GraphRAG workflows can read and write tabular artifacts in ClickHouse
- `ClickHouse` vector store wiring for embedding persistence and similarity search
- `oss` model provider support for both completion and embedding adapters
- OpenAI-style adapter support for self-hosted endpoints
- Streaming adapter support for OSS providers to unblock `basic`, `local`, and `drift` query modes
- Backward-compatible handling for legacy `inhouse` provider references
- Updated CLI init templates and defaults so generated configs use `model_provider: oss`

## Why ClickHouse

`ClickHouse` is a strong fit here because GraphRAG workloads are heavily read-oriented once indexing is complete, especially for query-time retrieval, table access, and embedding lookups.

- it is one of the fastest read-heavy databases available for analytical workloads
- it lets you run a self-hosted setup without being locked into a cloud-specific vector or database service
- it works well for building locally first and scaling the deployment later as usage grows
- using the same database family for both workflow tables and embeddings keeps the storage story simpler

## Supported Deployment Pattern

This fork is designed for a setup where:

- GraphRAG orchestration runs locally or in your own environment
- completion models are served by an OpenAI-compatible endpoint
- embedding models are served by an OpenAI-compatible endpoint
- GraphRAG outputs and embedding vectors are stored in ClickHouse

That makes it suitable for local LLM stacks, self-hosted inference services, and OSS model serving layers that expose OpenAI-like APIs.

## Recommended Model Serving Stack

The recommended companion deployment for OSS and local models is [ModelServer](https://github.com/sougato97/ModelServer).

It provides:

- a unified FastAPI gateway for chat and embeddings
- `POST /chat`, `POST /chat/stream`, and `POST /embeddings` endpoints
- Docker-based deployment for CUDA and ROCm environments
- support for the model families already reflected in this fork's defaults, including `Qwen/Qwen2.5-7B-Instruct-AWQ`, `Qwen/Qwen3-Embedding-4B`, and `intfloat/multilingual-e5-large-instruct`

By default, the OSS init template in this repo points GraphRAG-style settings at a local endpoint shape such as `http://localhost:8080/v1`, which aligns with the self-hosted serving flow this fork is targeting.

## Quickstart

Initialize a project using the OSS-oriented defaults:

```bash
graphrag init --root ./ragtest --model-provider oss
```

The generated configuration path now supports:

- `model_provider: oss` for completion models
- `model_provider: oss` for embedding models
- `table_provider.type: clickhouse`
- `vector_store.type: clickhouse`

A typical configuration looks like:

```yaml
completion_models:
  default_completion_model:
    model_provider: oss
    model: ${CHAT_COMPLETIONS_MODEL}
    api_key: ${NA}
    api_base: ${CHAT_COMPLETIONS_MODEL_ENDPOINT}

embedding_models:
  default_embedding_model:
    model_provider: oss
    model: ${EMBEDDING_MODEL}
    api_key: ${NA}
    api_base: ${EMBEDDING_MODEL_ENDPOINT}

table_provider:
  type: clickhouse
  host: ${CLICKHOUSE_HOST}
  port: ${CLICKHOUSE_PORT}
  protocol: ${CLICKHOUSE_PROTOCOL}
  database: ${CLICKHOUSE_TABLE_DB}
  user: ${CLICKHOUSE_USER}
  password: ${CLICKHOUSE_PASSWORD}
  namespace: ${CLICKHOUSE_TABLE_NAMESPACE}
  use_storage_path_as_namespace: true

vector_store:
  type: clickhouse
  host: ${CLICKHOUSE_HOST}
  port: ${CLICKHOUSE_PORT}
  protocol: ${CLICKHOUSE_PROTOCOL}
  database: ${CLICKHOUSE_VECTOR_DB}
  user: ${CLICKHOUSE_USER}
  password: ${CLICKHOUSE_PASSWORD}
```

## Compatibility Notes

- Existing GraphRAG defaults for `openai` remain intact
- Legacy `inhouse` references are preserved through a compatibility shim and alias handling
- Upstream GraphRAG documentation is still the best reference for workflow semantics, prompt tuning, and indexing behavior
- This fork mainly changes provider and storage integration points, not the core GraphRAG methodology

## Repository Guidance

This repository extends the GraphRAG methodology for research and self-hosted experimentation. The codebase is based on Microsoft's GraphRAG project and should be understood alongside the upstream documentation.

⚠️ GraphRAG indexing can be expensive or time-consuming depending on your GPU capacity and model setup. Start with small datasets and validate your model endpoint and ClickHouse configuration before scaling up.

## Diving Deeper

- Upstream docs: [microsoft.github.io/graphrag](https://microsoft.github.io/graphrag)
- Contribution guide: [CONTRIBUTING.md](./CONTRIBUTING.md)
- Development guide: [DEVELOPING.md](./DEVELOPING.md)
- Versioning notes: [breaking-changes.md](./breaking-changes.md)
- Upstream discussions: [GitHub Discussions](https://github.com/microsoft/graphrag/discussions)

## Prompt Tuning

Using GraphRAG with your own data and local models may require prompt adjustments. The upstream [Prompt Tuning Guide](https://microsoft.github.io/graphrag/prompt_tuning/overview/) remains relevant for improving output quality.

## Responsible AI FAQ

See [RAI_TRANSPARENCY.md](./RAI_TRANSPARENCY.md)

- [What is GraphRAG?](./RAI_TRANSPARENCY.md#what-is-graphrag)
- [What can GraphRAG do?](./RAI_TRANSPARENCY.md#what-can-graphrag-do)
- [What are GraphRAG’s intended use(s)?](./RAI_TRANSPARENCY.md#what-are-graphrags-intended-uses)
- [How was GraphRAG evaluated? What metrics are used to measure performance?](./RAI_TRANSPARENCY.md#how-was-graphrag-evaluated-what-metrics-are-used-to-measure-performance)
- [What are the limitations of GraphRAG? How can users minimize the impact of GraphRAG’s limitations when using the system?](./RAI_TRANSPARENCY.md#what-are-the-limitations-of-graphrag-how-can-users-minimize-the-impact-of-graphrags-limitations-when-using-the-system)
- [What operational factors and settings allow for effective and responsible use of GraphRAG?](./RAI_TRANSPARENCY.md#what-operational-factors-and-settings-allow-for-effective-and-responsible-use-of-graphrag)

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

## Privacy

[Microsoft Privacy Statement](https://privacy.microsoft.com/en-us/privacystatement)
