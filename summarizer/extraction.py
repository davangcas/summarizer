"""Extracción de texto desde documentos fuente (PDF y Word)."""

from __future__ import annotations

import base64
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pymupdf

from summarizer import state as app_state
from summarizer.config import MAX_PARALLEL_OCR_PAGES, MAX_PARALLEL_PDFS
from summarizer.fs import atomic_write_text
from summarizer.llm import chat_parse_with_retry, completion_parsed_or_validate
from summarizer.markdown_utils import (
    PAGE_HEADER_START_RE,
    pdf_has_selectable_text,
    split_markdown_by_page_headers,
)
from summarizer.models import OCRPageOutput
from summarizer.output import (
    completed_md_path_for_source,
    maybe_write_completed_texts_ocr_clone_pdf,
    nonempty_utf8_file,
    write_completed_text,
)
from summarizer.prompts import OCR_PROMPT
from summarizer.stop import check_stop_requested
from summarizer.word_conversion import extract_docx_text


def extract_text_get_text_only(pdf_path: Path) -> str:
    parts: list[str] = []
    with pymupdf.open(pdf_path) as pdf:
        for i, page in enumerate(pdf):
            text = page.get_text().strip()
            parts.append(f"## Página {i + 1}\n\n{text}")
    return "\n\n".join(parts)


def _ocr_single_page(pdf_path: Path, page_index: int) -> str:
    """Abre el PDF en el hilo actual; una página por llamada (seguro en paralelo)."""
    mat = pymupdf.Matrix(2, 2)
    with pymupdf.open(pdf_path) as pdf:
        page = pdf[page_index]
        pix = page.get_pixmap(matrix=mat)
        png_bytes = pix.tobytes("png")
        b64 = base64.standard_b64encode(png_bytes).decode("ascii")
        data_url = f"data:image/png;base64,{b64}"
        completion = chat_parse_with_retry(
            model=app_state.completion_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": OCR_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                    ],
                }
            ],
            response_format=OCRPageOutput,
        )
        return completion_parsed_or_validate(completion, OCRPageOutput).markdown_text


def _ocr_page_body_or_empty(src: Path, page_index: int) -> str:
    try:
        return _ocr_single_page(src, page_index).strip()
    except Exception as ex:
        print(f"OCR error página {page_index + 1} de {src}: {ex}")
        return ""


def extract_vision_pdf_incremental(src: Path, out_path: Path) -> None:
    """OCR de páginas en paralelo; escribe el .md al avanzar el prefijo consecutivo (atómico, reanudable)."""
    with pymupdf.open(src) as pdf:
        n = len(pdf)
    if n == 0:
        atomic_write_text(out_path, "")
        return

    existing = ""
    if out_path.is_file():
        existing = out_path.read_text(encoding="utf-8", errors="replace")

    page_bodies: dict[int, str] = {}
    chunks_prefix: list[str] = []
    start_i = 0
    if existing.strip():
        if not PAGE_HEADER_START_RE.search(existing):
            print(f"OCR reinicio (sin marcadores de página previos): {src.name}")
            page_bodies.clear()
            chunks_prefix.clear()
            start_i = 0
        else:
            parsed = split_markdown_by_page_headers(existing)
            if len(parsed) >= n:
                print(f"OCR ya completo ({n} páginas): {src.name}")
                return
            for pnum, body in parsed:
                idx = int(pnum) - 1
                body = body.strip()
                page_bodies[idx] = body
                chunks_prefix.append(f"## Página {pnum}\n\n{body}")
            start_i = len(parsed)

    print(
        f"OCR vis {src.name}: páginas {start_i + 1}..{n} de {n} "
        f"(paralelo ≤{MAX_PARALLEL_OCR_PAGES})"
    )
    if start_i >= n:
        atomic_write_text(out_path, "\n\n".join(chunks_prefix))
        return
    check_stop_requested()

    written_len = start_i
    write_lock = threading.Lock()

    def try_flush_extended_locked() -> None:
        nonlocal written_len
        while written_len < n and written_len in page_bodies:
            b = page_bodies[written_len]
            chunks_prefix.append(f"## Página {written_len + 1}\n\n{b}")
            written_len += 1
        if chunks_prefix:
            atomic_write_text(out_path, "\n\n".join(chunks_prefix))

    pending = [i for i in range(start_i, n) if i not in page_bodies]
    workers = min(MAX_PARALLEL_OCR_PAGES, len(pending))
    workers = max(1, workers)

    def run_page(i: int) -> tuple[int, str]:
        return i, _ocr_page_body_or_empty(src, i)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_page, i): i for i in pending}
        for fut in as_completed(futures):
            check_stop_requested()
            i, text = fut.result()
            with write_lock:
                page_bodies[i] = text
                try_flush_extended_locked()

    with write_lock:
        try_flush_extended_locked()


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
        print(src_for_output)
        out_md = completed_md_path_for_source(src_for_output)
        if pdf_has_selectable_text(src_pdf):
            file_text = extract_text_get_text_only(src_pdf)
            write_completed_text(src_for_output, file_text)
        elif app_state.use_vision_for_scanned_pdfs:
            extract_vision_pdf_incremental(src_pdf, out_md)
            if src_for_output.suffix.lower() == ".pdf":
                maybe_write_completed_texts_ocr_clone_pdf(src_for_output)
        else:
            print(
                "Sin texto extraíble (get_text vacío); visión desactivada — "
                f"omitido: {src_for_output}"
            )
    except Exception as ex:
        print(f"Error processing {src_for_output}: {ex}")


def _extract_single_source(src: Path) -> None:
    suffix = src.suffix.lower()
    if suffix == ".pdf":
        _extract_single_pdf_for_source(src, src)
        return
    if suffix == ".docx":
        try:
            text = extract_docx_text(src)
            if text.strip():
                write_completed_text(src, text)
                return
            print(f"Omitido {src}: DOCX sin contenido de texto extraíble.")
            return
        except Exception as ex:
            print(f"Omitido {src}: error extrayendo DOCX ({ex})")
            return
    if suffix == ".doc":
        print(
            f"Omitido {src}: formato .doc no soportado en modo solo-Python. "
            "Convierta el archivo a .docx."
        )
        return


def run_document_extraction() -> None:
    """Idempotent para documentos soportados (.pdf/.docx/.doc)."""
    assert app_state.files_directory is not None
    check_stop_requested()
    pending: list[Path] = []
    supported = {".pdf", ".docx", ".doc"}
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
                print(f"Skip extract (already in completed_texts): {src}")
                if suffix == ".pdf":
                    maybe_write_completed_texts_ocr_clone_pdf(src)
                continue
            if needs_markers:
                print(f"Re-extracción (marcadores de página): {src}")
            pending.append(src)
    if not pending:
        return
    workers = min(MAX_PARALLEL_PDFS, len(pending))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_extract_single_source, p) for p in pending]
        for fut in as_completed(futures):
            check_stop_requested()
            fut.result()


def run_pdf_extraction() -> None:
    """Compat: alias hacia el runner general de documentos."""
    run_document_extraction()
