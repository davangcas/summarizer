"""Orquestación del paso de resumen sobre `completed_texts`."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from summarizer import paths
from summarizer.config import MAX_PARALLEL_SUMMARIES
from summarizer.output import summarize_single_completed_md
from summarizer.stop import check_stop_requested


def run_summarization_pipeline() -> None:
    """Idempotent: skips when summary .md and PDF exist; rebuilds PDF only if .md exists but PDF missing."""
    file_paths = sorted(paths.completed_texts.rglob("*.md"))
    if not file_paths:
        return
    check_stop_requested()
    workers = min(MAX_PARALLEL_SUMMARIES, len(file_paths))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(summarize_single_completed_md, p) for p in file_paths]
        for fut in as_completed(futures):
            check_stop_requested()
            fut.result()
