"""Extracción de texto desde documentos fuente vía MarkItDown + pymupdf4llm."""

from __future__ import annotations

import os
import re
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pymupdf
import pymupdf4llm

from summarizer import state as app_state
from summarizer.config import HYBRID_OCR_ENABLED, MAX_PARALLEL_PDFS
from summarizer.markdown_utils import PAGE_HEADER_START_RE
from summarizer.output import (
    completed_md_path_for_source,
    maybe_write_completed_texts_ocr_clone_pdf,
    nonempty_utf8_file,
    write_completed_text,
    write_completed_texts_ocr_clone_pdf,
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
_markitdown_lock = threading.Lock()


def _get_markitdown():
    """Inicializa el singleton de MarkItDown con doble-check thread-safe."""
    global _markitdown_instance
    if _markitdown_instance is not None:
        return _markitdown_instance
    with _markitdown_lock:
        if _markitdown_instance is not None:
            return _markitdown_instance
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


@dataclass(frozen=True)
class PageExtraction:
    """Resultado por página de la extracción directa.

    ``has_text`` indica si la página aportó texto seleccionable; las páginas
    con ``has_text == False`` son candidatas a OCR vía MarkItDown cuando la
    opción de visión está activa.
    """

    page_number: int
    text: str
    has_text: bool


_PLACEHOLDER_PATTERNS = (
    re.compile(
        r"\*\*==>\s*picture\s*\[[^\]]*\]\s*intentionally\s*omitted\s*<==\*\*",
        re.IGNORECASE,
    ),
    re.compile(
        r"\*\*==>\s*[^*]+\s*omitted\s*<==\*\*",
        re.IGNORECASE,
    ),
)


def _strip_extraction_noise(text: str) -> str:
    """Quita marcadores de imágenes omitidas que pymupdf4llm inyecta."""
    for pat in _PLACEHOLDER_PATTERNS:
        text = pat.sub("", text)
    return text


_MIN_PAGE_TEXT_CHARS = 8


def extract_pdf_pages(pdf_path: Path) -> list[PageExtraction]:
    """Extrae texto por página combinando ``pymupdf4llm`` y ``pymupdf``.

    Estrategia robusta:

    1. Se obtiene siempre ``page.get_text()`` con la API básica de PyMuPDF;
       es lenta de tipear pero **es la fuente de verdad** sobre si una
       página contiene texto seleccionable. Algunas combinaciones de
       fuentes y CMaps producen chunks vacíos en ``pymupdf4llm`` aunque la
       página tenga texto perfectamente extraíble.
    2. Se intenta ``pymupdf4llm.to_markdown(page_chunks=True)`` para tener
       Markdown enriquecido (encabezados, listas). Si funciona y devuelve
       más texto que el extracto plano, se prefiere su salida porque tiene
       mejor estructura. Si devuelve menos texto, se usa el plano.
    3. ``has_text`` se calcula sobre el mejor de ambos: una página queda
       marcada para OCR sólo cuando NINGUNA de las dos fuentes recuperó
       texto significativo.

    Esto evita disparar OCR en PDFs donde ``pymupdf4llm`` falla globalmente
    (un caso real observado en libros con fuentes con CMap parcial).
    """
    raw_pages: list[str] = []
    with pymupdf.open(pdf_path) as pdf:
        for page in pdf:
            raw_pages.append(page.get_text().strip())

    md_pages: list[str] = []
    try:
        page_chunks = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
        for chunk in page_chunks:
            raw_md = (chunk.get("text") or "").strip()
            md_pages.append(_strip_extraction_noise(raw_md).strip())
    except Exception:
        md_pages = []

    n = len(raw_pages)
    results: list[PageExtraction] = []
    for i in range(n):
        plain = raw_pages[i]
        md = md_pages[i] if i < len(md_pages) else ""
        text = _choose_richer_text(plain, md)
        results.append(
            PageExtraction(
                page_number=i + 1,
                text=text,
                has_text=len(text) >= _MIN_PAGE_TEXT_CHARS,
            )
        )
    return results


def _choose_richer_text(plain: str, md: str) -> str:
    """Elige entre el texto plano (``page.get_text``) y el Markdown estructurado.

    Si el Markdown aporta significativamente más contenido (al menos 70% de
    lo que aporta el plano), se prefiere por su mejor formato. Si el
    Markdown está vacío o es notablemente más corto, se conserva el plano:
    una página con texto seleccionable que ``pymupdf4llm`` no logró
    procesar todavía debe llegar al resumen.
    """
    if not plain and not md:
        return ""
    if not md:
        return plain
    if not plain:
        return md
    if len(md) >= max(_MIN_PAGE_TEXT_CHARS, int(0.7 * len(plain))):
        return md
    return plain


def extract_text_get_text_only(pdf_path: Path) -> str:
    """Compat: Markdown estructurado por página, asumiendo PDF con texto.

    Mantenido por retrocompatibilidad; el nuevo flujo híbrido usa
    :func:`extract_pdf_pages` directamente.
    """
    return _assemble_pages_markdown(extract_pdf_pages(pdf_path))


def _assemble_pages_markdown(pages: list[PageExtraction]) -> str:
    """Ensambla el Markdown final con encabezados ``## Página N``."""
    parts: list[str] = []
    for p in pages:
        body = p.text if p.text else "[contenido no extraíble en esta página]"
        parts.append(f"## Página {p.page_number}\n\n{body}")
    return "\n\n".join(parts)


def _ocr_single_page(src_pdf: Path, page_number_1based: int) -> str:
    """Extrae una página concreta a un PDF temporal y la envía a MarkItDown."""
    idx0 = page_number_1based - 1
    tmp_path: Path | None = None
    single = pymupdf.open()
    try:
        with pymupdf.open(src_pdf) as src:
            if idx0 < 0 or idx0 >= len(src):
                return ""
            single.insert_pdf(src, from_page=idx0, to_page=idx0)
        fd, name = tempfile.mkstemp(suffix=".pdf", prefix="summarizer_ocr_")
        os.close(fd)
        tmp_path = Path(name)
        single.save(str(tmp_path))
    finally:
        single.close()
    try:
        result = _get_markitdown().convert(str(tmp_path))
        return (result.text_content or "").strip()
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except OSError:
                pass


def ocr_pages_with_markitdown(src_pdf: Path, page_numbers: list[int]) -> dict[int, str]:
    """OCR-extrae las páginas indicadas (1-based) vía MarkItDown.

    Retorna ``{page_number: text}`` solo para páginas con contenido extraído.
    Errores por página se logean y se omiten sin bloquear el resto.
    """
    out: dict[int, str] = {}
    if not page_numbers:
        return out
    for pnum in page_numbers:
        try:
            check_stop_requested()
            text = _ocr_single_page(src_pdf, pnum)
            if text:
                out[pnum] = text
        except Exception as ex:
            progress_log(f"OCR fallido página {pnum} de {src_pdf}: {ex}")
    return out


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
    """Procesa un PDF y escribe salida usando la ruta del archivo fuente.

    Estrategia híbrida (por defecto): extrae texto seleccionable por página
    en una sola pasada y, si la visión está activa, OCR-ea solo las páginas
    que vinieron vacías. Cae al modo legacy "todo o nada" si
    ``SUMMARIZER_HYBRID_OCR`` está desactivado.
    """
    try:
        check_stop_requested()
        progress_log(str(src_for_output))
        pages = extract_pdf_pages(src_pdf)
        if not pages:
            progress_log(f"Sin páginas extraíbles: {src_for_output}")
            return

        missing = [p.page_number for p in pages if not p.has_text]
        all_have_text = not missing

        if all_have_text:
            file_text = _assemble_pages_markdown(pages)
            write_completed_text(src_for_output, file_text)
            return

        if not HYBRID_OCR_ENABLED:
            _legacy_extract_fallback(src_pdf, src_for_output, pages, missing)
            return

        if app_state.use_vision_for_scanned_pdfs:
            progress_log(
                f"OCR por página ({len(missing)} de {len(pages)}): {src_for_output}"
            )
            ocr_results = ocr_pages_with_markitdown(src_pdf, missing)
            patched = [
                PageExtraction(
                    page_number=p.page_number,
                    text=ocr_results.get(p.page_number, p.text),
                    has_text=p.has_text or p.page_number in ocr_results,
                )
                for p in pages
            ]
            non_header = any(pg.has_text for pg in patched)
            if non_header:
                file_text = _assemble_pages_markdown(patched)
                write_completed_text(src_for_output, file_text)
                if src_for_output.suffix.lower() == ".pdf" and ocr_results:
                    write_completed_texts_ocr_clone_pdf(src_for_output)
            else:
                progress_log(f"Sin contenido extraíble: {src_for_output}")
            return

        non_header = any(p.has_text for p in pages)
        if non_header:
            file_text = _assemble_pages_markdown(pages)
            write_completed_text(src_for_output, file_text)
        else:
            progress_log(
                f"Sin texto extraíble; visión desactivada — omitido: {src_for_output}"
            )
    except Exception as ex:
        progress_log(f"Error processing {src_for_output}: {ex}")


def _legacy_extract_fallback(
    src_pdf: Path,
    src_for_output: Path,
    pages: list[PageExtraction],
    missing: list[int],
) -> None:
    """Camino legacy 'todo o nada' (cuando SUMMARIZER_HYBRID_OCR=false)."""
    if app_state.use_vision_for_scanned_pdfs:
        progress_log(f"OCR vía MarkItDown (legacy todo-o-nada): {src_for_output}")
        try:
            file_text = _convert_with_markitdown(src_pdf)
        except Exception as ex:
            progress_log(f"OCR fallido: {src_for_output} ({ex})")
            return
        if file_text.strip():
            write_completed_text(src_for_output, file_text)
            if src_for_output.suffix.lower() == ".pdf":
                maybe_write_completed_texts_ocr_clone_pdf(src_for_output)
        else:
            progress_log(f"Sin contenido extraíble vía OCR: {src_for_output}")
        return
    non_header = any(p.has_text for p in pages)
    if non_header:
        write_completed_text(src_for_output, _assemble_pages_markdown(pages))
    else:
        progress_log(
            f"Sin texto extraíble; visión desactivada — omitido: {src_for_output}"
        )


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
