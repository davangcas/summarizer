"""Cliente OpenAI-compatible (LM Studio) y llamadas parseadas con reintentos."""

from __future__ import annotations

import os
from typing import Any, TypeVar

from openai import APIConnectionError, APITimeoutError, BadRequestError, OpenAI
from openai.types.chat import ParsedChatCompletion
from pydantic import BaseModel

from summarizer.config import (
    LM_STUDIO_HOST,
    REQUEST_RETRIES,
    REQUEST_RETRY_BACKOFF_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
)
from summarizer.progress import progress_log
from summarizer.stop import check_stop_requested, sleep_with_stop

TModel = TypeVar("TModel", bound=BaseModel)

client = OpenAI(
    base_url=f"{LM_STUDIO_HOST}/v1",
    api_key=os.environ.get("LM_API_TOKEN", os.environ.get("LM_API_KEY", "lm-studio")),
)


def strip_json_fence(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def completion_parsed_or_validate(
    completion: ParsedChatCompletion[Any],
    model: type[TModel],
) -> TModel:
    msg = completion.choices[0].message
    if msg.parsed is not None:
        return msg.parsed
    content = (msg.content or "").strip()
    if content:
        return model.model_validate_json(strip_json_fence(content))
    raise ValueError("El modelo no devolvió contenido parseable")


def chat_parse_with_retry(**kwargs: Any) -> ParsedChatCompletion[Any]:
    last_error: Exception | None = None
    for attempt in range(1, REQUEST_RETRIES + 1):
        check_stop_requested()
        try:
            return client.chat.completions.parse(
                timeout=REQUEST_TIMEOUT_SECONDS,
                **kwargs,
            )
        except (APITimeoutError, APIConnectionError) as ex:
            last_error = ex
            if attempt >= REQUEST_RETRIES:
                break
            wait_seconds = REQUEST_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
            progress_log(
                "LM request timeout/conexión; "
                f"reintento {attempt}/{REQUEST_RETRIES - 1} en {wait_seconds:.1f}s..."
            )
            sleep_with_stop(wait_seconds)
    if last_error is not None:
        raise last_error
    raise RuntimeError("Fallo inesperado al ejecutar chat completion")


def is_context_overflow_error(ex: Exception) -> bool:
    if isinstance(ex, BadRequestError):
        text = str(ex).lower()
        return (
            "context length" in text
            or "context size has been exceeded" in text
            or "context size exceeded" in text
            or "context has been exceeded" in text
            or "n_keep" in text
            or "n_ctx" in text
            or "too many tokens" in text
        )
    return False
