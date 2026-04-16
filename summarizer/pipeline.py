"""Orquestación del paso de resumen sobre `completed_texts`."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from summarizer import paths
from summarizer import state as app_state
from summarizer.config import MAX_PARALLEL_SUMMARIES
from summarizer.output import summarize_single_completed_md
from summarizer.progress import get_global_progress
from summarizer.stop import check_stop_requested


def collect_completed_md_for_summary() -> list[Path]:
    """Archivos .md en scope que entran al pipeline de resumen."""
    file_paths = sorted(paths.completed_texts.rglob("*.md"))
    file_paths = [
        p
        for p in file_paths
        if app_state.completed_rel_matches_source_filter(
            p.relative_to(paths.completed_texts)
        )
    ]
    return file_paths


def run_summarization_pipeline(*, file_paths: list[Path] | None = None) -> None:
    """Idempotent: skips when summary .md and PDF exist; rebuilds PDF solo si falta PDF."""
    file_paths = (
        file_paths if file_paths is not None else collect_completed_md_for_summary()
    )
    if not file_paths:
        return
    check_stop_requested()
    progress = get_global_progress()
    if progress is not None:
        progress.set_stage("Resumen académico")
    workers = min(MAX_PARALLEL_SUMMARIES, len(file_paths))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(summarize_single_completed_md, p) for p in file_paths]
        for fut in as_completed(futures):
            check_stop_requested()
            fut.result()
            if progress is not None:
                progress.advance(1)
