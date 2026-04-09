"""Checkpoints reanudables por ventana de resumen."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from summarizer import paths
from summarizer.fs import atomic_write_json
from summarizer.models import CornellSummaryStructured, WindowSummaryCheckpoint


def summary_partials_enabled() -> bool:
    raw = os.environ.get("SUMMARIZER_SUMMARY_PARTIALS", "").strip().lower()
    if raw in ("0", "false", "no", "n", "off"):
        return False
    return True


def summary_partials_dir_for_completed_rel(md_source_rel: Path) -> Path:
    """Carpeta dedicada al documento: summary_partials/<misma_jerarquía>/<stem>/"""
    stem = md_source_rel.stem
    parent = md_source_rel.parent
    return paths.summary_partials / parent / stem


def window_body_fingerprint(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def try_load_window_checkpoint(
    path: Path,
    *,
    start_p: int,
    end_p: int,
    body: str,
) -> CornellSummaryStructured | None:
    if not path.is_file():
        return None
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
        ck = WindowSummaryCheckpoint.model_validate_json(raw)
    except (OSError, ValueError):
        return None
    if ck.start_p != start_p or ck.end_p != end_p:
        return None
    if ck.body_sha256 != window_body_fingerprint(body):
        return None
    return ck.structured


def save_window_checkpoint(
    path: Path,
    *,
    start_p: int,
    end_p: int,
    body: str,
    structured: CornellSummaryStructured,
) -> None:
    ck = WindowSummaryCheckpoint(
        start_p=start_p,
        end_p=end_p,
        body_sha256=window_body_fingerprint(body),
        structured=structured,
    )
    atomic_write_json(path, ck)
