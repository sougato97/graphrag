# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""CLI implementation of the initialization subcommand."""

import logging
from pathlib import Path

from graphrag.config.defaults import (
    DEFAULT_COMPLETION_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    graphrag_config_defaults,
)
from graphrag.config.init_content import (
    INIT_DOTENV,
    INIT_DOTENV_OSS,
    INIT_YAML,
    INIT_YAML_OSS,
)
from graphrag.prompts.index.community_report import (
    COMMUNITY_REPORT_PROMPT,
)
from graphrag.prompts.index.community_report_text_units import (
    COMMUNITY_REPORT_TEXT_PROMPT,
)
from graphrag.prompts.index.extract_claims import EXTRACT_CLAIMS_PROMPT
from graphrag.prompts.index.extract_graph import GRAPH_EXTRACTION_PROMPT
from graphrag.prompts.index.summarize_descriptions import SUMMARIZE_PROMPT
from graphrag.prompts.query.basic_search_system_prompt import BASIC_SEARCH_SYSTEM_PROMPT
from graphrag.prompts.query.drift_search_system_prompt import (
    DRIFT_LOCAL_SYSTEM_PROMPT,
    DRIFT_REDUCE_PROMPT,
)
from graphrag.prompts.query.global_search_knowledge_system_prompt import (
    GENERAL_KNOWLEDGE_INSTRUCTION,
)
from graphrag.prompts.query.global_search_map_system_prompt import MAP_SYSTEM_PROMPT
from graphrag.prompts.query.global_search_reduce_system_prompt import (
    REDUCE_SYSTEM_PROMPT,
)
from graphrag.prompts.query.local_search_system_prompt import LOCAL_SEARCH_SYSTEM_PROMPT
from graphrag.prompts.query.question_gen_system_prompt import QUESTION_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

GRAPHRAG_DEFAULTS_SELECTION = "graphrag-defaults"
OSS_SELECTION = "oss"
LEGACY_INHOUSE_SELECTION = "inhouse"
OPENAI_MODEL_PROVIDER = "openai"
OSS_MODEL_PROVIDER = "oss"
OSS_DEFAULT_COMPLETION_MODEL = "qwen2.5-7b-instruct-awq"
OSS_DEFAULT_EMBEDDING_MODEL = "Qwen3-Embedding-4B"


def initialize_project_at(
    path: Path,
    force: bool,
    model_provider: str,
    model: str | None,
    embedding_model: str | None,
) -> None:
    """
    Initialize the project at the given path.

    Parameters
    ----------
    path : Path
        The path at which to initialize the project.
    force : bool
        Whether to force initialization even if the project already exists.
    model_provider : str
        The model provider to use for generated model config values.
    model : str | None
        Optional completion model override.
    embedding_model : str | None
        Optional embedding model override.

    Raises
    ------
    ValueError
        If the project already exists and force is False.
    """
    logger.info("Initializing project at %s", path)
    root = Path(path).resolve()
    root.mkdir(parents=True, exist_ok=True)

    provider_selection = model_provider.strip().lower()
    # Backward-compatible alias: treat "openai" as "graphrag-defaults".
    if provider_selection == OPENAI_MODEL_PROVIDER:
        provider_selection = GRAPHRAG_DEFAULTS_SELECTION

    if provider_selection not in {
        GRAPHRAG_DEFAULTS_SELECTION,
        OSS_SELECTION,
        LEGACY_INHOUSE_SELECTION,
    }:
        msg = (
            f"Unsupported model provider '{model_provider}'. "
            f"Supported providers: {GRAPHRAG_DEFAULTS_SELECTION}, {OSS_SELECTION}."
        )
        raise ValueError(msg)

    if provider_selection in {OSS_SELECTION, LEGACY_INHOUSE_SELECTION}:
        resolved_model_provider = OSS_MODEL_PROVIDER
        resolved_model = model or OSS_DEFAULT_COMPLETION_MODEL
        resolved_embedding_model = embedding_model or OSS_DEFAULT_EMBEDDING_MODEL
        default_completion_api_key = "${NA}"
        default_embedding_api_key = "${NA}"
        init_yaml = INIT_YAML_OSS
        init_dotenv = INIT_DOTENV_OSS
    else:
        resolved_model_provider = OPENAI_MODEL_PROVIDER
        resolved_model = model or DEFAULT_COMPLETION_MODEL
        resolved_embedding_model = embedding_model or DEFAULT_EMBEDDING_MODEL
        default_completion_api_key = "${GRAPHRAG_API_KEY}"
        default_embedding_api_key = "${GRAPHRAG_API_KEY}"
        init_yaml = INIT_YAML
        init_dotenv = INIT_DOTENV

    settings_yaml = root / "settings.yaml"
    if settings_yaml.exists() and not force:
        msg = f"Project already initialized at {root}"
        raise ValueError(msg)

    input_path = (
        root / (graphrag_config_defaults.input_storage.base_dir or "input")
    ).resolve()
    input_path.mkdir(parents=True, exist_ok=True)
    formatted = (
        init_yaml.replace("<DEFAULT_MODEL_PROVIDER>", resolved_model_provider)
        .replace("<DEFAULT_COMPLETION_MODEL>", resolved_model)
        .replace("<DEFAULT_EMBEDDING_MODEL>", resolved_embedding_model)
        .replace("<DEFAULT_COMPLETION_API_KEY>", default_completion_api_key)
        .replace("<DEFAULT_EMBEDDING_API_KEY>", default_embedding_api_key)
    )
    settings_yaml.write_text(formatted, encoding="utf-8", errors="strict")

    dotenv = root / ".env"
    if not dotenv.exists() or force:
        dotenv.write_text(init_dotenv, encoding="utf-8", errors="strict")

    prompts_dir = root / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    prompts = {
        "extract_graph": GRAPH_EXTRACTION_PROMPT,
        "summarize_descriptions": SUMMARIZE_PROMPT,
        "extract_claims": EXTRACT_CLAIMS_PROMPT,
        "community_report_graph": COMMUNITY_REPORT_PROMPT,
        "community_report_text": COMMUNITY_REPORT_TEXT_PROMPT,
        "drift_search_system_prompt": DRIFT_LOCAL_SYSTEM_PROMPT,
        "drift_reduce_prompt": DRIFT_REDUCE_PROMPT,
        "global_search_map_system_prompt": MAP_SYSTEM_PROMPT,
        "global_search_reduce_system_prompt": REDUCE_SYSTEM_PROMPT,
        "global_search_knowledge_system_prompt": GENERAL_KNOWLEDGE_INSTRUCTION,
        "local_search_system_prompt": LOCAL_SEARCH_SYSTEM_PROMPT,
        "basic_search_system_prompt": BASIC_SEARCH_SYSTEM_PROMPT,
        "question_gen_system_prompt": QUESTION_SYSTEM_PROMPT,
    }

    for name, content in prompts.items():
        prompt_file = prompts_dir / f"{name}.txt"
        if not prompt_file.exists() or force:
            prompt_file.write_text(content, encoding="utf-8", errors="strict")
