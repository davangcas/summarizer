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
SUMMARY_MAX_PAGES_PER_WINDOW = env_int("SUMMARIZER_SUMMARY_MAX_PAGES_PER_WINDOW", 3)
SUMMARY_RETRY_MAX_PAGES_STEPDOWN = env_int(
    "SUMMARIZER_SUMMARY_RETRY_MAX_PAGES_STEPDOWN", 1
)
ASSEMBLE_DEDUP_BORDER = os.environ.get(
    "SUMMARIZER_ASSEMBLE_DEDUP_BORDER", ""
).strip().lower() in ("1", "true", "yes", "sí", "si", "on")
# Umbral Jaccard para considerar dos temas como duplicados semánticos
# (se aplica al máximo entre Jaccard de títulos y Jaccard de firmas completas).
SEMANTIC_DEDUP_THRESHOLD = float(
    os.environ.get("SUMMARIZER_SEMANTIC_DEDUP_THRESHOLD", "0.5")
)


def env_flag(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in ("0", "false", "no", "n", "off"):
        return False
    if raw in ("1", "true", "yes", "sí", "si", "y", "on"):
        return True
    return default


# Extracción híbrida: usa OCR sólo en páginas sin texto, no en el documento entero.
HYBRID_OCR_ENABLED = env_flag("SUMMARIZER_HYBRID_OCR", default=True)

# Pre-renderizado de fórmulas LaTeX a PNG (matplotlib.mathtext) antes de generar PDF.
MATH_RENDER_ENABLED = env_flag("SUMMARIZER_MATH_RENDER", default=True)

# DPI usado al renderizar fórmulas a PNG con matplotlib.
MATH_RENDER_DPI = env_int("SUMMARIZER_MATH_DPI", 200)

# Ventana de escaneo del índice (en caracteres) sobre el inicio del texto completado.
BOOK_OUTLINE_SCAN_CHARS = env_int("SUMMARIZER_BOOK_OUTLINE_SCAN_CHARS", 64000)

# Heurística de índice del libro en las primeras páginas (INDICE + líneas con puntos guía).
BOOK_OUTLINE_HEURISTIC_ENABLED = env_flag(
    "SUMMARIZER_BOOK_OUTLINE_HEURISTIC", default=True
)
# Si True, en ensamblado se omite un segundo tema con el mismo título normalizado (conserva el primero).
# Por defecto False: la consolidación post-unify fusiona duplicados sin perder notas entre ventanas.
ASSEMBLE_DEDUP_GLOBAL = env_flag("SUMMARIZER_ASSEMBLE_DEDUP_GLOBAL", default=False)
# Paso de unificación LLM tras ensamblar ventanas: fusiona duplicados semánticos y mejora coherencia.
SUMMARY_UNIFY_WINDOWS = env_flag("SUMMARIZER_SUMMARY_UNIFY_WINDOWS", default=True)
# Si True y la unificación está activa, usar troceo por lotes + reducción en niveles cuando el ensamblaje
# no cabe en un solo prompt (evita colapsar libros largos a un único JSON pequeño).
SUMMARY_UNIFY_HIERARCHICAL = env_flag(
    "SUMMARIZER_SUMMARY_UNIFY_HIERARCHICAL", default=True
)
# Tras la unificación jerárquica, ¿correr una pasada FINAL "single-pass" sobre todo el resultado?
# Esa pasada tiende a comprimir agresivamente (el LLM resume lo que ya estaba consolidado),
# por eso por defecto está apagada. Actívala si prefieres un texto más sintético a costa de
# perder detalle entre lotes. Equivale a SUMMARY_UNIFY_MODE=aggressive.
SUMMARY_FINAL_UNIFY_PASS = env_flag(
    "SUMMARIZER_SUMMARY_FINAL_UNIFY_PASS", default=False
)


_VALID_UNIFY_MODES = ("none", "lmless", "hierarchical", "aggressive")


def _resolve_unify_mode() -> str:
    """Modo de unificación post-ensamblado:

    - ``none``: devuelve el ensamblaje (Jaccard dedup ya aplicado) sin tocar.
      Cero llamadas LLM extra. Máximo detalle, conserva duplicados blandos.
    - ``lmless``: una segunda pasada Jaccard (umbral relajado) sobre el
      ensamblaje. Cero llamadas LLM extra. Más detalle que ``hierarchical``,
      menos duplicados blandos que ``none``.
    - ``hierarchical`` (default): unificación por lotes con LLM, sin pasada
      final. Es el comportamiento actual heredado.
    - ``aggressive``: hierarchical + pasada final single-pass LLM. Más
      síntesis a costa de detalle.

    Si la variable explícita no está seteada y el flag legado
    ``SUMMARIZER_SUMMARY_FINAL_UNIFY_PASS`` está activo, se mapea a
    ``aggressive`` (backward-compat).
    """
    raw = os.environ.get("SUMMARIZER_SUMMARY_UNIFY_MODE", "").strip().lower()
    if raw in _VALID_UNIFY_MODES:
        return raw
    if SUMMARY_FINAL_UNIFY_PASS:
        return "aggressive"
    return "hierarchical"


SUMMARY_UNIFY_MODE = _resolve_unify_mode()
# Escribe además `stem_full.md` / `stem_full.pdf` con el ensamblaje de ventanas antes de unificar.
SUMMARY_DUAL_OUTPUT = env_flag("SUMMARIZER_SUMMARY_DUAL_OUTPUT", default=False)
# No borrar `summary_partials/` tras éxito (checkpoints y `_combined_windows.md`).
SUMMARY_KEEP_PARTIALS = env_flag("SUMMARIZER_SUMMARY_KEEP_PARTIALS", default=False)


def cornell_depth_profile() -> str:
    """normal | high — controla instrucciones extra en prompts Cornell (ver prompts.py)."""
    raw = os.environ.get("SUMMARIZER_CORNELL_DEPTH", "normal").strip().lower()
    if raw in ("high", "deep", "alto", "max"):
        return "high"
    return "normal"


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
