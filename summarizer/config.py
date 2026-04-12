"""Constantes y lectura de variables de entorno."""

import os
import threading

LM_STUDIO_HOST = os.environ.get("LM_STUDIO_HOST", "http://localhost:1234").rstrip("/")

PROMPT_CONTEXT_RATIO_TARGET = float(
    os.environ.get("SUMMARIZER_PROMPT_CONTEXT_RATIO", "0.40")
)
PROMPT_CONTEXT_RATIO_START = float(
    os.environ.get("SUMMARIZER_PROMPT_CONTEXT_RATIO_START", "0.30")
)
PROMPT_CONTEXT_RATIO_MIN = float(
    os.environ.get("SUMMARIZER_PROMPT_CONTEXT_RATIO_MIN", "0.10")
)
PROMPT_CONTEXT_RATIO_STEP = float(
    os.environ.get("SUMMARIZER_PROMPT_CONTEXT_RATIO_STEP", "0.05")
)

TOKENIZER_FALLBACKS = (
    "google/gemma-3-12b-it",
    "google/gemma-3-270m",
    "mistralai/ministral-3-3b",
    "gpt2",
)


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def env_optional_timeout_seconds(name: str, default: float | None) -> float | None:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"none", "null", "off", "infinite", "inf", "0", "-1"}:
        return None
    try:
        value = float(raw)
        return None if value <= 0 else value
    except ValueError:
        return default


MAX_PARALLEL_PDFS = env_int("SUMMARIZER_MAX_PARALLEL_PDFS", 4)
MAX_PARALLEL_SUMMARIES = env_int("SUMMARIZER_MAX_PARALLEL_SUMMARIES", 4)
MAX_PARALLEL_CHUNKS = env_int("SUMMARIZER_MAX_PARALLEL_CHUNKS", 4)
MAX_PARALLEL_OCR_PAGES = env_int("SUMMARIZER_MAX_PARALLEL_OCR_PAGES", 4)
MAX_PARALLEL_WINDOW_SUMMARIES = env_int("SUMMARIZER_MAX_PARALLEL_WINDOW_SUMMARIES", 2)
SUMMARY_PAGE_OVERLAP = env_int("SUMMARIZER_SUMMARY_PAGE_OVERLAP", 1)
ASSEMBLE_DEDUP_BORDER = os.environ.get(
    "SUMMARIZER_ASSEMBLE_DEDUP_BORDER", ""
).strip().lower() in ("1", "true", "yes", "sí", "si", "on")


def env_flag(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in ("0", "false", "no", "n", "off"):
        return False
    if raw in ("1", "true", "yes", "sí", "si", "y", "on"):
        return True
    return default


# Tras ensamblar ventanas, una pasada LLM fusiona temas y limpia el índice (salvo límite de tokens).
POST_UNIFY_ENABLED = env_flag("SUMMARIZER_POST_UNIFY", default=True)
# Heurística de índice del libro en las primeras páginas (INDICE + líneas con puntos guía).
BOOK_OUTLINE_HEURISTIC_ENABLED = env_flag(
    "SUMMARIZER_BOOK_OUTLINE_HEURISTIC", default=True
)
# Si True, en ensamblado se omite un segundo tema con el mismo título normalizado (conserva el primero).
# Por defecto False: la consolidación post-unify fusiona duplicados sin perder notas entre ventanas.
ASSEMBLE_DEDUP_GLOBAL = env_flag("SUMMARIZER_ASSEMBLE_DEDUP_GLOBAL", default=False)

REQUEST_TIMEOUT_SECONDS = env_optional_timeout_seconds(
    "SUMMARIZER_REQUEST_TIMEOUT_SECONDS", None
)
REQUEST_RETRIES = env_int("SUMMARIZER_REQUEST_RETRIES", 4)
REQUEST_RETRY_BACKOFF_SECONDS = float(
    os.environ.get("SUMMARIZER_REQUEST_RETRY_BACKOFF_SECONDS", "2.0")
)

MAX_PARALLEL_ONEDRIVE_UPLOADS = max(
    1, env_int("SUMMARIZER_MAX_PARALLEL_ONEDRIVE_UPLOADS", 3)
)
ONEDRIVE_UPLOAD_SEM = threading.BoundedSemaphore(MAX_PARALLEL_ONEDRIVE_UPLOADS)
