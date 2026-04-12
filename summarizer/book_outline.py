"""Detección opcional del índice del libro y lectura desde entorno."""

from __future__ import annotations

import os
import re

from summarizer.config import BOOK_OUTLINE_HEURISTIC_ENABLED

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


def _heuristic_outline_from_text(
    full_text: str, *, max_scan_chars: int = 16000
) -> list[str] | None:
    if not BOOK_OUTLINE_HEURISTIC_ENABLED:
        return None
    head = full_text[:max_scan_chars]
    if not _INDICE_RE.search(head):
        return None
    seen: set[str] = set()
    out: list[str] = []
    for m in _TOC_LINE_RE.finditer(head):
        title = m.group(1).strip()
        title = re.sub(r"\s+", " ", title)
        if len(title) < 3 or len(title) > 120:
            continue
        low = title.lower()
        if low in seen:
            continue
        # Filtrar líneas que son solo número de página o encabezado de página OCR
        if re.match(r"^##?\s*Página\s+\d+", title, re.I):
            continue
        seen.add(low)
        out.append(title)
        if len(out) >= 48:
            break
    return out if len(out) >= 2 else None


def chapter_outline_for_summary(full_text: str) -> list[str] | None:
    """
    Lista de capítulos/títulos de referencia: primero variable de entorno,
    si no, heurística sobre el inicio del texto completado.
    """
    env = outline_from_env()
    if env:
        return env
    return _heuristic_outline_from_text(full_text)
