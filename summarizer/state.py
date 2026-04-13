"""Estado mutable del pipeline (directorios, modelo, ratio adaptativo)."""

from __future__ import annotations

import pathlib
import threading

from summarizer.config import (
    PROMPT_CONTEXT_RATIO_MIN,
    PROMPT_CONTEXT_RATIO_START,
    PROMPT_CONTEXT_RATIO_STEP,
    PROMPT_CONTEXT_RATIO_TARGET,
)

files_directory: pathlib.Path | None = None
# Si no es None, solo se procesan estos PDF (rutas absolutas resueltas).
source_pdf_paths: frozenset[pathlib.Path] | None = None
# Carpeta para los PDF de resumen finales; None = usar ``paths.summary_pdfs`` del proyecto.
summary_pdfs_directory: pathlib.Path | None = None
use_vision_for_scanned_pdfs: bool = False


def source_pdf_is_in_scope(pdf_path: pathlib.Path) -> bool:
    """True si el PDF debe procesarse según ``source_pdf_paths`` (o todos si es None)."""
    if source_pdf_paths is None:
        return True
    return pdf_path.resolve() in source_pdf_paths


def completed_rel_matches_source_filter(rel: pathlib.Path) -> bool:
    """True si el .md bajo completed_texts corresponde a un PDF seleccionado."""
    assert files_directory is not None
    if source_pdf_paths is None:
        return True
    expected_pdf = (files_directory / rel).with_suffix(".pdf").resolve()
    return expected_pdf in source_pdf_paths


completion_model: str = ""
MAX_CONTEXT_TOKENS: int = 0
MAX_INPUT_TOKENS: int = 0

_prompt_ratio_lock = threading.Lock()
_adaptive_prompt_context_ratio: float = max(
    PROMPT_CONTEXT_RATIO_MIN,
    min(PROMPT_CONTEXT_RATIO_TARGET, PROMPT_CONTEXT_RATIO_START),
)


def get_adaptive_prompt_ratio() -> float:
    with _prompt_ratio_lock:
        return _adaptive_prompt_context_ratio


def record_prompt_ratio_success() -> float:
    global _adaptive_prompt_context_ratio
    with _prompt_ratio_lock:
        _adaptive_prompt_context_ratio = min(
            PROMPT_CONTEXT_RATIO_TARGET,
            _adaptive_prompt_context_ratio + max(0.01, PROMPT_CONTEXT_RATIO_STEP),
        )
        return _adaptive_prompt_context_ratio


def record_prompt_ratio_overflow() -> float:
    global _adaptive_prompt_context_ratio
    with _prompt_ratio_lock:
        _adaptive_prompt_context_ratio = max(
            PROMPT_CONTEXT_RATIO_MIN,
            _adaptive_prompt_context_ratio - max(0.01, PROMPT_CONTEXT_RATIO_STEP),
        )
        return _adaptive_prompt_context_ratio
