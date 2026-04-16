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
# Si no es None, solo se procesan estos archivos fuente (rutas absolutas resueltas).
source_file_paths: frozenset[pathlib.Path] | None = None
# Carpeta para los PDF de resumen finales; None = usar ``paths.summary_pdfs`` del proyecto.
summary_pdfs_directory: pathlib.Path | None = None
use_vision_for_scanned_pdfs: bool = False


def source_file_is_in_scope(source_path: pathlib.Path) -> bool:
    """True si el archivo fuente debe procesarse según ``source_file_paths``."""
    if source_file_paths is None:
        return True
    return source_path.resolve() in source_file_paths


_SOURCE_SUFFIXES = (
    ".pdf",
    ".docx",
    ".doc",
    ".pptx",
    ".xlsx",
    ".xls",
    ".html",
    ".htm",
    ".csv",
    ".json",
    ".xml",
    ".epub",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".tiff",
    ".tif",
    ".webp",
    ".mp3",
    ".wav",
    ".zip",
    ".msg",
    ".rst",
)


def completed_rel_matches_source_filter(rel: pathlib.Path) -> bool:
    """True si el .md bajo completed_texts corresponde a un archivo fuente seleccionado."""
    assert files_directory is not None
    if source_file_paths is None:
        return True
    source_base = (files_directory / rel).with_suffix("")
    for suffix in _SOURCE_SUFFIXES:
        if source_base.with_suffix(suffix).resolve() in source_file_paths:
            return True
    return False


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
