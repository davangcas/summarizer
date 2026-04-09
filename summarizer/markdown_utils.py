"""Utilidades Markdown y comprobaciones sobre PDF."""

import re
import unicodedata
from pathlib import Path

import pymupdf

PAGE_HEADER_START_RE = re.compile(r"^## Página (\d+)(?:[^\n]*)?$", re.MULTILINE)


def split_markdown_by_page_headers(text: str) -> list[tuple[int, str]]:
    """Devuelve (número_página, cuerpo) ordenado. Sin marcadores: todo el texto como página 1."""
    text = text.strip()
    if not text:
        return []
    if not PAGE_HEADER_START_RE.search(text):
        return [(1, text)]
    pieces = re.split(r"(?m)^(?=## Página \d+)", text)
    out: list[tuple[int, str]] = []
    header_re = re.compile(r"^## Página (\d+)(?:[^\n]*)?$", re.MULTILINE)
    for raw in pieces:
        raw = raw.strip()
        if not raw:
            continue
        first = raw.split("\n", 1)[0].strip()
        m = header_re.match(first)
        if not m:
            continue
        pn = int(m.group(1))
        body = header_re.sub("", raw, count=1).strip()
        out.append((pn, body))
    return sorted(out, key=lambda x: x[0])


def pdf_has_selectable_text(pdf_path: Path) -> bool:
    with pymupdf.open(pdf_path) as pdf:
        return any(page.get_text().strip() for page in pdf)


def last_page_from_completed_md(content: str) -> int | None:
    nums = [int(m) for m in PAGE_HEADER_START_RE.findall(content)]
    return max(nums) if nums else None


def slugify_anchor(label: str, *, fallback: str) -> str:
    s = unicodedata.normalize("NFKD", label)
    s = "".join(c if c.isalnum() or c in " -_" else "" for c in s)
    s = "-".join(s.lower().split())
    return s if s else fallback
