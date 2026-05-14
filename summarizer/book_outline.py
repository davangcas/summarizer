"""Detección opcional del índice del libro y lectura desde entorno."""

from __future__ import annotations

import os
import re

from summarizer.config import BOOK_OUTLINE_HEURISTIC_ENABLED, BOOK_OUTLINE_SCAN_CHARS

_INDICE_RE = re.compile(
    r"\b(INDICE|ÍNDICE|CONTENIDO|TABLE\s+OF\s+CONTENTS)\b",
    re.IGNORECASE,
)
# Línea tipo "TÍTULO ........................ 12" o puntos guía
_TOC_LINE_RE = re.compile(
    r"^(.+?)\s*\.{2,}\s*[\d\sivxlcdmIVXLCDM]+$",
    re.MULTILINE,
)


def outline_from_env() -> list[str] | None:
    """``SUMMARIZER_BOOK_CHAPTERS``: capítulos separados por ``;`` o saltos de línea."""
    raw = os.environ.get("SUMMARIZER_BOOK_CHAPTERS", "").strip()
    if not raw:
        return None
    parts: list[str] = []
    for chunk in raw.replace("\n", ";").split(";"):
        s = chunk.strip()
        if s:
            parts.append(s)
    return parts if parts else None


def _harvest_toc_from_chunk(chunk: str) -> list[str]:
    """Extrae títulos de un fragmento que contiene `INDICE`/`ÍNDICE`/`TOC`."""
    seen: set[str] = set()
    out: list[str] = []
    for m in _TOC_LINE_RE.finditer(chunk):
        title = m.group(1).strip()
        title = re.sub(r"\s+", " ", title)
        if len(title) < 3 or len(title) > 120:
            continue
        low = title.lower()
        if low in seen:
            continue
        if re.match(r"^##?\s*Página\s+\d+", title, re.I):
            continue
        seen.add(low)
        out.append(title)
        if len(out) >= 48:
            break
    return out


_PAGE_MARKER_RE = re.compile(r"(?m)^##\s+Página\s+(\d+)")


def _secondary_scan_window(full_text: str) -> str:
    """Ventana secundaria tras el prólogo, antes del primer tercio del libro.

    Heurística: desde el marcador `## Página 5` (o cercano) hasta el `## Página 40`
    o el final, lo que ocurra antes. Útil para libros con prólogo extenso donde
    el índice aparece después de las primeras 16k–64k caracteres iniciales.
    """
    matches = list(_PAGE_MARKER_RE.finditer(full_text))
    if not matches:
        return ""
    start_idx = None
    end_idx = None
    for m in matches:
        pnum = int(m.group(1))
        if start_idx is None and pnum >= 5:
            start_idx = m.start()
        if pnum >= 40:
            end_idx = m.end()
            break
    if start_idx is None:
        return ""
    if end_idx is None:
        end_idx = min(len(full_text), start_idx + 96000)
    return full_text[start_idx:end_idx]


def _heuristic_outline_from_text(
    full_text: str, *, max_scan_chars: int | None = None
) -> list[str] | None:
    if not BOOK_OUTLINE_HEURISTIC_ENABLED:
        return None
    scan_chars = (
        max_scan_chars if max_scan_chars is not None else BOOK_OUTLINE_SCAN_CHARS
    )
    head = full_text[:scan_chars]
    candidates: list[str] = []
    if _INDICE_RE.search(head):
        candidates = _harvest_toc_from_chunk(head)
    if len(candidates) < 2:
        secondary = _secondary_scan_window(full_text)
        if secondary and _INDICE_RE.search(secondary):
            candidates = _harvest_toc_from_chunk(secondary)
    return candidates if len(candidates) >= 2 else None


def chapter_outline_for_summary(full_text: str) -> list[str] | None:
    """Capítulos de referencia: variable de entorno o heurística sobre el texto."""
    env = outline_from_env()
    if env:
        return env
    return _heuristic_outline_from_text(full_text)
