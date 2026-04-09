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
use_vision_for_scanned_pdfs: bool = False

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
