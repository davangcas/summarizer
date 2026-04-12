"""Normalización de Markdown para generación de PDF (PyMuPDF / markdown_pdf)."""

from __future__ import annotations

import re
from collections.abc import Callable

_HEADING_LINE_RE = re.compile(r"^(#{1,6})(\s+.*)$")
_HTML_HEADING_START = re.compile(r"^<h([1-6])\b", re.IGNORECASE)
# Enlaces solo a fragmento: Story exige destinos con id; markdown_it no siempre los genera.
_FRAGMENT_LINK_RE = re.compile(r"\[([^\]]+)\]\(#[^)]*\)")
_LINE_PANDOC_HEADING_ID = re.compile(r"^(#{1,6}\s+.+?)\s*\{#[^}]+\}\s*$")


def _normalize_newlines(markdown: str) -> str:
    return re.sub(r"\r\n|\r", "\n", markdown)


def _apply_outside_fences(
    markdown: str,
    line_fn: Callable[[str], str],
) -> str:
    out: list[str] = []
    in_fence = False
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue
        out.append(line_fn(line))
    return "\n".join(out)


def _strip_fragment_links_line(line: str) -> str:
    return _FRAGMENT_LINK_RE.sub(r"\1", line)


def _strip_pandoc_heading_id_line(line: str) -> str:
    m = _LINE_PANDOC_HEADING_ID.match(line.rstrip())
    if m:
        return m.group(1)
    return line


def markdown_for_pymupdf_pdf(markdown: str) -> str:
    """
    Ajusta el Markdown para el pipeline markdown_it → PyMuPDF Story.

    Elimina enlaces ``[texto](#ancla)`` y atributos ``{#id}`` en encabezados: Story
    falla con «No destination with id=…» si el HTML no define esos destinos.
    El contenido visible (índice y títulos) se conserva.
    """
    text = _normalize_newlines(markdown)
    text = _apply_outside_fences(text, _strip_fragment_links_line)
    text = _apply_outside_fences(text, _strip_pandoc_heading_id_line)
    return text


def ensure_markdown_h1_for_pdf(markdown: str) -> str:
    """PyMuPDF falla con set_toc si el primer heading no es # (ver 'hierarchy level of item 0 must be 1')."""
    text = markdown.strip()
    if not text:
        return "# Resumen\n"
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#") and not s.startswith("##"):
            return markdown
        break
    return f"# Resumen\n\n{text}"


def normalize_markdown_heading_hierarchy_for_pdf(markdown: str) -> str:
    """
    PyMuPDF set_toc exige que los niveles del índice no salten (p. ej. h1 → h3).

    Tras ``## Índice``, los temas pueden ser Markdown o HTML ``<h3 id=…>``; las
    líneas HTML actualizan el nivel esperado para los ``####`` siguientes.
    """
    last_level = 0
    out: list[str] = []
    in_fence = False
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue
        m_html = _HTML_HEADING_START.match(stripped)
        if m_html:
            last_level = int(m_html.group(1))
            out.append(line)
            continue
        m = _HEADING_LINE_RE.match(line)
        if not m:
            out.append(line)
            continue
        raw_level = len(m.group(1))
        rest = m.group(2)
        if last_level == 0:
            adj = 1
        elif raw_level > last_level + 1:
            adj = last_level + 1
        else:
            adj = raw_level
        last_level = adj
        out.append("#" * adj + rest)
    return "\n".join(out)
