# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Structure response as pydantic base model."""

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel, covariant=True)
JSON_FENCE_REGEX = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)


def structure_completion_response(response: str, model: type[T]) -> T:
    """Structure completion response as pydantic base model.

    Args
    ----
        response: str
            The completion response as a JSON string.
        model: type[T]
            The pydantic base model type to structure the response into.

    Returns
    -------
        The structured response as a pydantic base model.
    """
    parsed_dict = _parse_response_to_dict(response)
    return model(**parsed_dict)


def _parse_response_to_dict(response: str) -> dict[str, Any]:
    """Best-effort parse to handle fenced JSON or extra prose around JSON."""
    text = (response or "").strip()
    if not text:
        msg = "Empty completion response; expected JSON payload."
        raise ValueError(msg)

    direct = _try_load_json_dict(text)
    if direct is not None:
        return direct

    fenced_match = JSON_FENCE_REGEX.search(text)
    if fenced_match:
        fenced = fenced_match.group(1).strip()
        fenced_obj = _try_load_json_dict(fenced)
        if fenced_obj is not None:
            return fenced_obj

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        body = text[start : end + 1]
        body_obj = _try_load_json_dict(body)
        if body_obj is not None:
            return body_obj

    preview = text[:240].replace("\n", "\\n")
    msg = f"Could not parse completion into JSON object. Response preview: {preview}"
    raise ValueError(msg)


def _try_load_json_dict(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:  # noqa: BLE001
        return None
    return None
