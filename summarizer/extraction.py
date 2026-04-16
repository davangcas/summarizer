"""Extracción de texto desde documentos fuente vía MarkItDown + pymupdf4llm."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pymupdf
import pymupdf4llm

from summarizer import state as app_state
from summarizer.config import MAX_PARALLEL_PDFS
from summarizer.markdown_utils import PAGE_HEADER_START_RE
from summarizer.output import (
    completed_md_path_for_source,
    maybe_write_completed_texts_ocr_clone_pdf,
    nonempty_utf8_file,
    write_completed_text,
)
from summarizer.progress import get_global_progress, progress_log
from summarizer.stop import check_stop_requested

_MARKITDOWN_EXTENSIONS = frozenset(
    {
        ".docx",
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
    }
)

SUPPORTED_EXTENSIONS = frozenset({".pdf"}) | _MARKITDOWN_EXTENSIONS

_markitdown_instance = None


def _get_markitdown():
    global _markitdown_instance
    if _markitdown_instance is None:
        from markitdown import MarkItDown

        if app_state.completion_model:
            from summarizer.llm import client as openai_client

            _markitdown_instance = MarkItDown(
                enable_plugins=True,
                llm_client=openai_client,
                llm_model=app_state.completion_model,
            )
        else:
            _markitdown_instance = MarkItDown(enable_plugins=False)
    return _markitdown_instance


def _convert_with_markitdown(src: Path) -> str:
    """Conversión unificada vía MarkItDown (DOCX, PPTX, XLSX, HTML, PDF escaneado, …)."""
    result = _get_markitdown().convert(str(src))
    return result.text_content


def extract_text_get_text_only(pdf_path: Path) -> str:
    """Extrae Markdown estructurado de un PDF con texto seleccionable (por página)."""
    try:
        page_chunks = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
        parts: list[str] = []
        for i, chunk in enumerate(page_chunks):
            text = chunk.get("text", "").strip()
            parts.append(f"## Página {i + 1}\n\n{text}")
        return "\n\n".join(parts)
    except Exception:
        parts: list[str] = []
        with pymupdf.open(pdf_path) as pdf:
            for i, page in enumerate(pdf):
                text = page.get_text().strip()
                parts.append(f"## Página {i + 1}\n\n{text}")
        return "\n\n".join(parts)


def _pdf_all_pages_have_text(pdf_path: Path) -> bool:
    """True si todas las páginas del PDF tienen texto seleccionable."""
    with pymupdf.open(pdf_path) as pdf:
        if len(pdf) == 0:
            return False
        return all(page.get_text().strip() for page in pdf)


def _completed_md_needs_page_markers(out_md: Path) -> bool:
    """True si hay texto completado pero sin ## Página (extracción antigua)."""
    if not nonempty_utf8_file(out_md):
        return False
    try:
        txt = out_md.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return not PAGE_HEADER_START_RE.search(txt)


def _extract_single_pdf_for_source(src_pdf: Path, src_for_output: Path) -> None:
    """Procesa un PDF y escribe salida usando la ruta del archivo fuente."""
    try:
        check_stop_requested()
        progress_log(str(src_for_output))
        if _pdf_all_pages_have_text(src_pdf):
            file_text = extract_text_get_text_only(src_pdf)
            write_completed_text(src_for_output, file_text)
        elif app_state.use_vision_for_scanned_pdfs:
            progress_log(f"OCR vía MarkItDown: {src_for_output}")
            file_text = _convert_with_markitdown(src_pdf)
            if file_text.strip():
                write_completed_text(src_for_output, file_text)
                if src_for_output.suffix.lower() == ".pdf":
                    maybe_write_completed_texts_ocr_clone_pdf(src_for_output)
            else:
                progress_log(f"Sin contenido extraíble vía OCR: {src_for_output}")
        else:
            file_text = extract_text_get_text_only(src_pdf)
            non_header = any(
                line.strip() and not line.startswith("## Página")
                for line in file_text.split("\n")
            )
            if non_header:
                write_completed_text(src_for_output, file_text)
            else:
                progress_log(
                    "Sin texto extraíble; visión desactivada — "
                    f"omitido: {src_for_output}"
                )
    except Exception as ex:
        progress_log(f"Error processing {src_for_output}: {ex}")


def _extract_single_source(src: Path) -> None:
    suffix = src.suffix.lower()
    if suffix == ".pdf":
        _extract_single_pdf_for_source(src, src)
        return
    if suffix in _MARKITDOWN_EXTENSIONS:
        try:
            text = _convert_with_markitdown(src)
            if text.strip():
                write_completed_text(src, text)
                return
            progress_log(f"Omitido {src}: sin contenido de texto extraíble.")
            return
        except Exception as ex:
            progress_log(f"Omitido {src}: error extrayendo con MarkItDown ({ex})")
            return
    if suffix == ".doc":
        progress_log(
            f"Omitido {src}: formato .doc no soportado. "
            "Convierta el archivo a .docx o .pdf."
        )
        return


def collect_pending_sources() -> list[Path]:
    """Devuelve documentos soportados pendientes de extracción."""
    assert app_state.files_directory is not None
    check_stop_requested()
    pending: list[Path] = []
    supported = SUPPORTED_EXTENSIONS | {".doc"}
    for root, _, files in os.walk(app_state.files_directory):
        for file in files:
            suffix = Path(file).suffix.lower()
            if suffix not in supported:
                continue
            src = Path(root) / file
            if not app_state.source_file_is_in_scope(src):
                continue
            out_md = completed_md_path_for_source(src)
            needs_markers = suffix == ".pdf" and _completed_md_needs_page_markers(
                out_md
            )
            if nonempty_utf8_file(out_md) and not needs_markers:
                progress_log(f"Skip extract (already in completed_texts): {src}")
                if suffix == ".pdf":
                    maybe_write_completed_texts_ocr_clone_pdf(src)
                continue
            if needs_markers:
                progress_log(f"Re-extracción (marcadores de página): {src}")
            pending.append(src)
    return pending


def run_document_extraction(*, pending_sources: list[Path] | None = None) -> None:
    """Idempotent para todos los formatos soportados."""
    pending = (
        pending_sources if pending_sources is not None else collect_pending_sources()
    )
    if not pending:
        return
    progress = get_global_progress()
    if progress is not None:
        progress.set_stage("Extracción")
    workers = min(MAX_PARALLEL_PDFS, len(pending))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_extract_single_source, p) for p in pending]
        for fut in as_completed(futures):
            check_stop_requested()
            fut.result()
            if progress is not None:
                progress.advance(1)


def run_pdf_extraction() -> None:
    """Compat: alias hacia el runner general de documentos."""
    run_document_extraction()
