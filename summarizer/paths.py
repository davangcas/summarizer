"""Rutas de proyecto y carpetas de salida."""

import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
summarized_texts = PROJECT_ROOT / "summarized_texts"
completed_texts = PROJECT_ROOT / "completed_texts"
summary_pdfs = PROJECT_ROOT / "summary_pdfs"
summary_partials = PROJECT_ROOT / "summary_partials"

for _p in (
    summarized_texts,
    completed_texts,
    summary_pdfs,
    summary_partials,
):
    _p.mkdir(parents=True, exist_ok=True)
