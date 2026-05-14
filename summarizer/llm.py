"""Cliente OpenAI-compatible (LM Studio) y llamadas parseadas con reintentos."""

from __future__ import annotations

import json
import os
from typing import Any, TypeVar

from openai import (
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    OpenAI,
)
from openai.types.chat import ParsedChatCompletion
from pydantic import BaseModel, ValidationError

from summarizer.config import (
    LM_STUDIO_HOST,
    REQUEST_RETRIES,
    REQUEST_RETRY_BACKOFF_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
)
from summarizer.math_sanitize import sanitize_model
from summarizer.progress import progress_log
from summarizer.stop import check_stop_requested, sleep_with_stop

TModel = TypeVar("TModel", bound=BaseModel)

_STRUCTURED_VALIDATION_RETRIES = 2
_BADREQUEST_NONOVERFLOW_RETRIES = 1

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
    """Devuelve la instancia parseada o valida el JSON del content.

    Aplica saneado matemático (escapes JSON rotos en LaTeX) sobre el modelo
    resultante.
    """
    msg = completion.choices[0].message
    parsed: TModel | None = None
    if msg.parsed is not None:
        parsed = msg.parsed
    else:
        content = (msg.content or "").strip()
        if not content:
            raise ValueError("El modelo no devolvió contenido parseable")
        parsed = model.model_validate_json(strip_json_fence(content))
    sanitized = sanitize_model(parsed)
    return sanitized if isinstance(sanitized, model) else parsed


def chat_parse_with_retry(**kwargs: Any) -> ParsedChatCompletion[Any]:
    """Ejecuta ``chat.completions.parse`` con reintentos sobre errores transitorios.

    Reintenta ``APITimeoutError`` y ``APIConnectionError`` hasta
    ``REQUEST_RETRIES``. Para ``BadRequestError`` que NO sea de overflow se
    permite UN reintento (cubre fallos transitorios de validación / parseo
    del servidor). El overflow se propaga sin reintentos: el caller debe
    reaccionar bajando granularidad.
    """
    last_error: Exception | None = None
    badrequest_retries_left = _BADREQUEST_NONOVERFLOW_RETRIES
    attempt = 0
    while attempt < REQUEST_RETRIES:
        attempt += 1
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
        except BadRequestError as ex:
            if is_context_overflow_error(ex):
                raise
            if badrequest_retries_left <= 0:
                raise
            badrequest_retries_left -= 1
            progress_log(f"LM BadRequest no-overflow; reintentando una vez: {ex}")
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("Fallo inesperado al ejecutar chat completion")


def chat_structured_with_retry(
    *,
    response_format: type[TModel],
    messages: list[dict[str, Any]],
    model: str,
    **kwargs: Any,
) -> TModel:
    """Wrapper de alto nivel para llamadas con salida estructurada.

    1. Hace ``parse`` con reintentos transitorios y de BadRequest no-overflow.
    2. Si la respuesta no se valida contra el schema (``ValidationError`` /
       ``ValueError`` / ``JSONDecodeError``), reintenta hasta
       ``_STRUCTURED_VALIDATION_RETRIES`` veces inyectando un mensaje
       correctivo.
    3. Como último recurso, cae a un modo *free-form* (sin
       ``response_format``) inyectando el JSON schema en el prompt y
       validando manualmente la respuesta.
    """
    last_validation_error: Exception | None = None
    attempt_messages = list(messages)
    for attempt in range(1, _STRUCTURED_VALIDATION_RETRIES + 1):
        try:
            completion = chat_parse_with_retry(
                model=model,
                messages=attempt_messages,
                response_format=response_format,
                **kwargs,
            )
            return completion_parsed_or_validate(completion, response_format)
        except (ValidationError, ValueError, json.JSONDecodeError) as ex:
            last_validation_error = ex
            if attempt >= _STRUCTURED_VALIDATION_RETRIES:
                break
            progress_log(
                "LM respuesta no cumple el esquema "
                f"(intento {attempt}/{_STRUCTURED_VALIDATION_RETRIES}); "
                "reintentando con instrucción correctiva."
            )
            attempt_messages = list(messages) + [
                {
                    "role": "user",
                    "content": (
                        "Tu respuesta anterior no fue un JSON válido o no "
                        "cumple el esquema. Devuelve EXCLUSIVAMENTE el JSON "
                        "del esquema, sin Markdown envolvente, sin comentarios."
                    ),
                }
            ]
        except BadRequestError as ex:
            if is_context_overflow_error(ex):
                raise
            if _structured_outputs_unsupported(ex):
                progress_log(
                    "LM no soporta response_format estructurado; "
                    "intentando fallback free-form."
                )
                return _free_form_fallback(
                    response_format=response_format,
                    messages=messages,
                    model=model,
                    **kwargs,
                )
            raise

    progress_log(
        "Fallaron los reintentos de salida estructurada; intentando free-form."
    )
    try:
        return _free_form_fallback(
            response_format=response_format,
            messages=messages,
            model=model,
            **kwargs,
        )
    except Exception as ff_ex:
        if last_validation_error is not None:
            raise last_validation_error from ff_ex
        raise


def _structured_outputs_unsupported(ex: Exception) -> bool:
    """Heurística para detectar 'el modelo no soporta response_format'."""
    if not isinstance(ex, BadRequestError):
        return False
    text = str(ex).lower()
    indicators = (
        "response_format",
        "json_schema",
        "structured output",
        "not support",
        "unsupported parameter",
    )
    return any(indicator in text for indicator in indicators)


def _free_form_fallback(
    *,
    response_format: type[TModel],
    messages: list[dict[str, Any]],
    model: str,
    **kwargs: Any,
) -> TModel:
    """Llama sin ``response_format`` inyectando el schema en el prompt.

    Saca cualquier ``response_format`` previo de ``kwargs`` y agrega un
    mensaje user con el JSON schema. Devuelve una instancia validada y
    saneada.
    """
    kwargs.pop("response_format", None)
    schema_json = json.dumps(
        response_format.model_json_schema(),
        ensure_ascii=False,
        indent=2,
    )
    instructions = (
        "Devuelve EXCLUSIVAMENTE un JSON que cumpla este esquema. "
        "Sin Markdown, sin texto extra, sin comentarios.\n\n"
        f"Esquema:\n{schema_json}"
    )
    augmented = list(messages) + [{"role": "user", "content": instructions}]
    last_error: Exception | None = None
    for attempt in range(1, REQUEST_RETRIES + 1):
        check_stop_requested()
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=augmented,
                timeout=REQUEST_TIMEOUT_SECONDS,
                **kwargs,
            )
            content = (completion.choices[0].message.content or "").strip()
            if not content:
                raise ValueError("Fallback free-form: contenido vacío")
            parsed = response_format.model_validate_json(strip_json_fence(content))
            sanitized = sanitize_model(parsed)
            return sanitized if isinstance(sanitized, response_format) else parsed
        except (APITimeoutError, APIConnectionError) as ex:
            last_error = ex
            if attempt >= REQUEST_RETRIES:
                break
            wait_seconds = REQUEST_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
            sleep_with_stop(wait_seconds)
        except BadRequestError:
            raise
    if last_error is not None:
        raise last_error
    raise RuntimeError("Fallback free-form: fallo inesperado")


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
