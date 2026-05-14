"""Rutas y escritura de textos completados y resúmenes."""

from __future__ import annotations

import shutil
from pathlib import Path

from markdown_pdf import MarkdownPdf, Section

from summarizer import paths
from summarizer import state as app_state
from summarizer.checkpoints import summary_partials_dir_for_completed_rel
from summarizer.config import (
    MATH_RENDER_ENABLED,
    SUMMARY_DUAL_OUTPUT,
    SUMMARY_KEEP_PARTIALS,
)
from summarizer.cornell_summary import summarize_document_paged_windows
from summarizer.markdown_utils import pdf_has_selectable_text
from summarizer.math_render import replace_math_with_images
from summarizer.math_sanitize import sanitize_math_text
from summarizer.progress import progress_log
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


def completed_md_path_for_source(src: Path) -> Path:
    assert app_state.files_directory is not None
    rel = src.relative_to(app_state.files_directory)
    return paths.completed_texts / rel.with_suffix(".md")


def completed_texts_ocr_pdf_path_for_pdf(src: Path) -> Path:
    assert app_state.files_directory is not None
    rel = src.relative_to(app_state.files_directory)
    return paths.completed_texts_ocr / rel.with_suffix(".pdf")


def write_completed_texts_ocr_clone_pdf(src: Path) -> None:
    """Escribe el PDF clon a partir del .md en completed_texts.

    Llamado por el flujo de extracción cuando se ha hecho OCR (parcial o
    total). Idempotente: no regenera si el PDF de destino ya existe.
    """
    if not app_state.use_vision_for_scanned_pdfs:
        return
    out_md = completed_md_path_for_source(src)
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


def maybe_write_completed_texts_ocr_clone_pdf(src: Path) -> None:
    """Variante usada en la fase de sincronización idempotente.

    Conserva la heurística histórica: si el PDF original no tiene texto
    seleccionable, se asume que fue OCR-eado y se escribe el clon.
    """
    if pdf_has_selectable_text(src):
        return
    write_completed_texts_ocr_clone_pdf(src)


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
    out_path = completed_md_path_for_source(src)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(file_text, encoding="utf-8")


def write_summary_markdown(md_source_rel: Path, summary_md: str) -> None:
    out_md = paths.summarized_texts / md_source_rel
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(summary_md, encoding="utf-8")


def math_assets_dir_for_pdf(out_pdf: Path) -> Path:
    """Carpeta dedicada a PNG de fórmulas para un PDF concreto."""
    return out_pdf.parent / f"{out_pdf.stem}_math"


def render_markdown_to_pdf(
    out_pdf: Path, markdown: str, *, fallback_h1: str = "Resumen"
) -> None:
    """Convierte Markdown a PDF.

    Si ``MATH_RENDER_ENABLED`` está activo, pre-renderiza los spans
    ``$...$`` y ``$$...$$`` a PNG via matplotlib.mathtext antes de
    entregar el Markdown al motor de PDF.

    ``markdown_for_pymupdf_pdf`` (que retira ``[label](#anchor)`` y
    ``{#id}`` de los encabezados) corre **antes** de la sustitución
    matemática. De lo contrario, un span ``$E=mc^2$`` dentro de un label
    de enlace del índice se reemplaza por ``![](path)`` y los corchetes
    interiores rompen el regex de stripping; el ``[label](#anchor)`` se
    cuela hacia ``fitz.Story`` y revienta con ``No destination with id=``.
    """
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    markdown = markdown_for_pymupdf_pdf(markdown)
    if MATH_RENDER_ENABLED:
        # Sanitize escape collisions ANTES de la sustitución math: cura
        # ``\\text`` (over-escape JSON) → ``\text`` y restaura comandos
        # huérfanos. Necesario también al re-renderizar PDF desde un .md
        # antiguo cuyo LLM no había pasado por el sanitizer (idempotente
        # sobre contenido ya limpio).
        markdown = sanitize_math_text(markdown)
        math_dir = math_assets_dir_for_pdf(out_pdf)
        markdown = replace_math_with_images(markdown, math_dir, base_dir=out_pdf.parent)
    pdf = MarkdownPdf()
    md_pdf = ensure_markdown_h1_for_pdf(markdown, fallback_h1=fallback_h1)
    md_pdf = normalize_markdown_heading_hierarchy_for_pdf(md_pdf)
    pdf.add_section(
        Section(
            text=md_pdf,
            toc=True,
            root=str(out_pdf.parent),
        )
    )
    pdf.save(out_pdf)


def write_summary_pdf(md_source_rel: Path, summary_md: str) -> None:
    out_pdf = summary_pdfs_output_dir() / md_source_rel.with_suffix(".pdf")
    render_markdown_to_pdf(out_pdf, summary_md, fallback_h1=md_source_rel.stem)


def _cleanup_success_artifacts(rel: Path, *, keep_partials: bool = False) -> None:
    completed_md = paths.completed_texts / rel
    summary_md = paths.summarized_texts / rel
    partials_dir = summary_partials_dir_for_completed_rel(rel)
    for candidate in (completed_md, summary_md):
        try:
            if candidate.exists():
                candidate.unlink()
        except OSError as ex:
            progress_log(f"Aviso limpieza (archivo): {candidate} -> {ex}")
    if keep_partials:
        return
    try:
        if partials_dir.exists():
            shutil.rmtree(partials_dir, ignore_errors=False)
    except OSError as ex:
        progress_log(f"Aviso limpieza (parciales): {partials_dir} -> {ex}")
    out_pdf = summary_pdfs_output_dir() / rel.with_suffix(".pdf")
    math_dir = math_assets_dir_for_pdf(out_pdf)
    try:
        if math_dir.exists():
            shutil.rmtree(math_dir, ignore_errors=True)
    except OSError as ex:
        progress_log(f"Aviso limpieza (math assets): {math_dir} -> {ex}")


def partial_summary_md_path(rel: Path) -> Path:
    """Ruta del Markdown parcial cuando el resumen falla mid-run."""
    return paths.summarized_texts / rel.with_name(f"{rel.stem}.partial{rel.suffix}")


def summarize_single_completed_md(md_path: Path) -> None:
    try:
        check_stop_requested()
        rel = md_path.relative_to(paths.completed_texts)
        out_summary_md = paths.summarized_texts / rel
        out_summary_pdf = summary_pdfs_output_dir() / rel.with_suffix(".pdf")
        partial_md = partial_summary_md_path(rel)

        if nonempty_utf8_file(out_summary_md) and nonempty_pdf_file(out_summary_pdf):
            progress_log(f"Skip summarize (summary + PDF done): {rel}")
            _drop_partial(partial_md)
            return

        if nonempty_utf8_file(out_summary_md) and not nonempty_pdf_file(
            out_summary_pdf
        ):
            progress_log(f"Resume PDF from summary: {rel}")
            write_summary_pdf(rel, out_summary_md.read_text(encoding="utf-8"))
            if nonempty_pdf_file(out_summary_pdf):
                _cleanup_success_artifacts(rel, keep_partials=SUMMARY_KEEP_PARTIALS)
            return

        if nonempty_utf8_file(partial_md):
            progress_log(
                f"Resumen parcial previo detectado: {partial_md.name}; "
                "reanudando con checkpoints existentes."
            )

        progress_log(f"Summarizing: {md_path}")
        full_text = md_path.read_text(encoding="utf-8")
        if not full_text.strip():
            return
        partials = summary_partials_dir_for_completed_rel(rel)
        summary_md, assembled_md = summarize_document_paged_windows(
            full_text,
            partials_dir=partials,
            h1_title=rel.stem,
            partial_md_path=partial_md,
        )
        if summary_md.strip():
            write_summary_markdown(rel, summary_md)
            write_summary_pdf(rel, summary_md)
            if SUMMARY_DUAL_OUTPUT and assembled_md.strip():
                asm = assembled_md.strip()
                fin = summary_md.strip()
                if asm != fin:
                    full_rel = rel.with_name(f"{rel.stem}_full{rel.suffix}")
                    write_summary_markdown(full_rel, assembled_md)
                    write_summary_pdf(full_rel, assembled_md)
                    progress_log(
                        f"Copia ensamblaje pre-unificación: {full_rel.name} "
                        f"(Markdown y PDF junto al resumen principal)."
                    )
            if nonempty_pdf_file(out_summary_pdf):
                _drop_partial(partial_md)
                _cleanup_success_artifacts(rel, keep_partials=SUMMARY_KEEP_PARTIALS)
    except Exception as ex:
        progress_log(f"Error summarizing {md_path}: {ex}")


def _drop_partial(partial_md: Path) -> None:
    try:
        if partial_md.exists():
            partial_md.unlink()
    except OSError as ex:
        progress_log(f"Aviso limpieza (parcial): {partial_md} -> {ex}")
