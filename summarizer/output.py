"""Rutas y escritura de textos completados y resúmenes."""

from __future__ import annotations

from pathlib import Path

from markdown_pdf import MarkdownPdf, Section

from summarizer import paths
from summarizer import state as app_state
from summarizer.checkpoints import summary_partials_dir_for_completed_rel
from summarizer.cornell_summary import summarize_document_paged_windows
from summarizer.markdown_utils import pdf_has_selectable_text
from summarizer.stop import check_stop_requested
from summarizer.pdf_markdown import (
    ensure_markdown_h1_for_pdf,
    markdown_for_pymupdf_pdf,
    normalize_markdown_heading_hierarchy_for_pdf,
)


def summary_pdfs_output_dir() -> Path:
    """Directorio base para los PDF generados a partir de los resúmenes."""
    base = app_state.summary_pdfs_directory
    return base if base is not None else paths.summary_pdfs


def completed_md_path_for_pdf(src: Path) -> Path:
    assert app_state.files_directory is not None
    rel = src.relative_to(app_state.files_directory)
    return paths.completed_texts / rel.with_suffix(".md")


def completed_texts_ocr_pdf_path_for_pdf(src: Path) -> Path:
    assert app_state.files_directory is not None
    rel = src.relative_to(app_state.files_directory)
    return paths.completed_texts_ocr / rel.with_suffix(".pdf")


def maybe_write_completed_texts_ocr_clone_pdf(src: Path) -> None:
    """
    PDF con el mismo contenido que el .md en completed_texts, solo para PDFs
    escaneados procesados por visión (no se usa después en el pipeline).
    Idempotente: regenera si falta el PDF y el .md existe.
    """
    if pdf_has_selectable_text(src):
        return
    if not app_state.use_vision_for_scanned_pdfs:
        return
    out_md = completed_md_path_for_pdf(src)
    if not nonempty_utf8_file(out_md):
        return
    ocr_pdf = completed_texts_ocr_pdf_path_for_pdf(src)
    if nonempty_pdf_file(ocr_pdf):
        return
    render_markdown_to_pdf(
        ocr_pdf,
        out_md.read_text(encoding="utf-8", errors="replace"),
        fallback_h1=src.stem,
    )


def nonempty_utf8_file(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip() != ""
    except OSError:
        return False


def nonempty_pdf_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def write_completed_text(src: Path, file_text: str) -> None:
    out_path = completed_md_path_for_pdf(src)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(file_text, encoding="utf-8")


def write_summary_markdown(md_source_rel: Path, summary_md: str) -> None:
    out_md = paths.summarized_texts / md_source_rel
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(summary_md, encoding="utf-8")


def render_markdown_to_pdf(
    out_pdf: Path, markdown: str, *, fallback_h1: str = "Resumen"
) -> None:
    """Convierte Markdown a PDF (mismo pipeline que los resúmenes)."""
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf = MarkdownPdf()
    md_pdf = ensure_markdown_h1_for_pdf(
        markdown_for_pymupdf_pdf(markdown), fallback_h1=fallback_h1
    )
    md_pdf = normalize_markdown_heading_hierarchy_for_pdf(md_pdf)
    pdf.add_section(
        Section(
            text=md_pdf,
            toc=True,
        )
    )
    pdf.save(out_pdf)


def write_summary_pdf(md_source_rel: Path, summary_md: str) -> None:
    out_pdf = summary_pdfs_output_dir() / md_source_rel.with_suffix(".pdf")
    render_markdown_to_pdf(out_pdf, summary_md, fallback_h1=md_source_rel.stem)


def summarize_single_completed_md(md_path: Path) -> None:
    try:
        check_stop_requested()
        rel = md_path.relative_to(paths.completed_texts)
        out_summary_md = paths.summarized_texts / rel
        out_summary_pdf = summary_pdfs_output_dir() / rel.with_suffix(".pdf")

        if nonempty_utf8_file(out_summary_md) and nonempty_pdf_file(out_summary_pdf):
            print(f"Skip summarize (summary + PDF done): {rel}")
            return

        if nonempty_utf8_file(out_summary_md) and not nonempty_pdf_file(
            out_summary_pdf
        ):
            print(f"Resume PDF from summary: {rel}")
            write_summary_pdf(rel, out_summary_md.read_text(encoding="utf-8"))
            return

        print(f"Summarizing: {md_path}")
        full_text = md_path.read_text(encoding="utf-8")
        if not full_text.strip():
            return
        partials = summary_partials_dir_for_completed_rel(rel)
        summary_md = summarize_document_paged_windows(
            full_text, partials_dir=partials, h1_title=rel.stem
        )
        if summary_md.strip():
            write_summary_markdown(rel, summary_md)
            write_summary_pdf(rel, summary_md)
    except Exception as ex:
        print(f"Error summarizing {md_path}: {ex}")
