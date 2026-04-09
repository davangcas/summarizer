"""Detección del modelo cargado en LM Studio."""

from __future__ import annotations

import os
from typing import Any

import httpx

from summarizer import state as app_state
from summarizer.config import (
    LM_STUDIO_HOST,
    PROMPT_CONTEXT_RATIO_TARGET,
)


def _effective_context_tokens(model_entry: dict[str, Any]) -> int:
    """Contexto real de instancias cargadas (fallback: max_context_length)."""
    best = 0
    for inst in model_entry.get("loaded_instances") or []:
        cfg = inst.get("config") or {}
        cl = int(cfg.get("context_length") or 0)
        best = max(best, cl)
    if best <= 0:
        best = int(model_entry.get("max_context_length") or 0)
    return best


def configure_lm_studio_model() -> None:
    """
    Consulta GET {LM_STUDIO_HOST}/api/v1/models y elige un LLM cargado con vision=True
    y el mayor contexto cargado. Ajusta MAX_CONTEXT_TOKENS y MAX_INPUT_TOKENS.

    Override: SUMMARIZER_COMPLETION_MODEL + SUMMARIZER_MAX_CONTEXT_TOKENS (sin llamar a la API).
    """
    manual_model = os.environ.get("SUMMARIZER_COMPLETION_MODEL", "").strip()
    manual_ctx = os.environ.get("SUMMARIZER_MAX_CONTEXT_TOKENS", "").strip()
    if manual_model and manual_ctx:
        app_state.completion_model = manual_model
        app_state.MAX_CONTEXT_TOKENS = max(1024, int(manual_ctx))
        app_state.MAX_INPUT_TOKENS = max(
            512, int(app_state.MAX_CONTEXT_TOKENS * PROMPT_CONTEXT_RATIO_TARGET)
        )
        print(
            f"Modelo (override entorno): {app_state.completion_model}, contexto {app_state.MAX_CONTEXT_TOKENS} tokens"
        )
        return

    url = f"{LM_STUDIO_HOST}/api/v1/models"
    headers: dict[str, str] = {}
    token = (
        os.environ.get("LM_API_TOKEN", "").strip()
        or os.environ.get("LM_API_KEY", "").strip()
    )
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        resp = httpx.get(url, headers=headers, timeout=60.0)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        raise SystemExit(
            f"No se pudo listar modelos en LM Studio ({url}). ¿Está el servidor activo? {e}"
        ) from e

    payload = resp.json()
    models = payload.get("models") or []
    best: tuple[str, int] | None = None

    for m in models:
        if m.get("type") != "llm":
            continue
        caps = m.get("capabilities") or {}
        if not caps.get("vision"):
            continue
        if not (m.get("loaded_instances") or []):
            continue
        key = str(m.get("key") or "").strip()
        if not key:
            continue
        ctx = _effective_context_tokens(m)
        if ctx <= 0:
            ctx = 8192
        if best is None or ctx > best[1]:
            best = (key, ctx)

    if best is None:
        raise SystemExit(
            "No hay ningún LLM cargado en LM Studio con capacidad de visión (capabilities.vision). "
            "Cargue uno en la app o defina SUMMARIZER_COMPLETION_MODEL y "
            "SUMMARIZER_MAX_CONTEXT_TOKENS."
        )

    app_state.completion_model, app_state.MAX_CONTEXT_TOKENS = best
    app_state.MAX_INPUT_TOKENS = max(
        512, int(app_state.MAX_CONTEXT_TOKENS * PROMPT_CONTEXT_RATIO_TARGET)
    )
    print(
        f"Modelo LM Studio (visión, mayor contexto cargado): {app_state.completion_model} — "
        f"contexto {app_state.MAX_CONTEXT_TOKENS} tokens (límite prompt objetivo "
        f"{int(PROMPT_CONTEXT_RATIO_TARGET * 100)}% ~{app_state.MAX_INPUT_TOKENS})"
    )
