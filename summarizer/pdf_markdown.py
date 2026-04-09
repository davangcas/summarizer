"""Normalización de Markdown para generación de PDF (PyMuPDF / markdown_pdf)."""

import re

_HEADING_LINE_RE = re.compile(r"^(#{1,6})(\s+.*)$")


def markdown_for_pymupdf_pdf(markdown: str) -> str:
    """
    Ajusta el Markdown para el pipeline markdown_it → PyMuPDF Story.

    En modo commonmark, los atributos tipo Pandoc ``### Título {#slug}`` no se traducen a
    ``id=`` en el HTML; el índice manual con ``[texto](#slug)`` sí genera enlaces internos.
    Story falla luego con: No destination with id=...
    """
    text = markdown
    text = re.sub(
        r"(?ms)^## Índice\s*\n.*?\n---\s*\n+",
        "",
        text,
        count=1,
    )
    text = re.sub(
        r"^(#{1,6}\s+.+?)\s*\{#([^}]+)\}\s*$",
        r"\1",
        text,
        flags=re.MULTILINE,
    )
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
    PyMuPDF set_toc exige que los niveles del índice no salten (p. ej. h1 → h3),
    lo que provoca «bad hierarchy level in row …» al guardar el PDF.

    Tras quitar el bloque «## Índice» en markdown_for_pymupdf_pdf, los temas Cornell
    (### …) quedan colgando directamente de # Resumen; aquí se reajustan los «#».
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
