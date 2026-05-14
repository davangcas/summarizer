"""Tests de la extracción híbrida: una sola apertura del PDF, OCR por página."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import pymupdf


class _SyntheticPdfMixin:
    """Helpers para crear PDFs sintéticos con páginas mixtas (texto / sólo imagen)."""

    def _build_text_page(self, doc: pymupdf.Document, body: str) -> None:
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 72), body, fontsize=14)

    def _build_image_only_page(self, doc: pymupdf.Document) -> None:
        page = doc.new_page(width=595, height=842)
        rect = pymupdf.Rect(72, 72, 522, 200)
        page.draw_rect(rect, fill=(0.85, 0.85, 0.85), color=(0, 0, 0))

    def _make_mixed_pdf(self, path: Path) -> None:
        doc = pymupdf.open()
        try:
            self._build_text_page(doc, "Pagina uno con texto seleccionable.")
            self._build_image_only_page(doc)
            self._build_text_page(doc, "Pagina tres con mas texto seleccionable.")
            doc.save(str(path))
        finally:
            doc.close()


class ExtractPdfPagesTests(_SyntheticPdfMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="extract_test_"))
        self.pdf = self.tmp / "mixed.pdf"
        self._make_mixed_pdf(self.pdf)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_pages_with_text_flagged_correctly(self) -> None:
        from summarizer.extraction import extract_pdf_pages

        pages = extract_pdf_pages(self.pdf)
        self.assertEqual(len(pages), 3)
        self.assertTrue(pages[0].has_text)
        self.assertFalse(pages[1].has_text)
        self.assertTrue(pages[2].has_text)
        self.assertIn("Pagina uno", pages[0].text)
        self.assertIn("Pagina tres", pages[2].text)

    def test_assemble_pages_markdown_marks_empty(self) -> None:
        from summarizer.extraction import (
            _assemble_pages_markdown,
            extract_pdf_pages,
        )

        md = _assemble_pages_markdown(extract_pdf_pages(self.pdf))
        self.assertIn("## Página 1", md)
        self.assertIn("## Página 2", md)
        self.assertIn("## Página 3", md)
        self.assertIn("[contenido no extraíble en esta página]", md)


class ChooseRicherTextTests(unittest.TestCase):
    """Cubre el fallback cuando pymupdf4llm devuelve menos texto que el plano."""

    def test_uses_plain_when_md_is_empty(self) -> None:
        from summarizer.extraction import _choose_richer_text

        plain = "Texto plano de la página con suficiente contenido."
        self.assertEqual(_choose_richer_text(plain, ""), plain)

    def test_prefers_md_when_richer(self) -> None:
        from summarizer.extraction import _choose_richer_text

        plain = "Encabezado breve"
        md = "# Encabezado breve\n\nContenido enriquecido con muchísimo más texto."
        self.assertEqual(_choose_richer_text(plain, md), md)

    def test_falls_back_to_plain_when_md_truncated(self) -> None:
        from summarizer.extraction import _choose_richer_text

        plain = "Texto largo y completo de la página con muchos caracteres reales."
        md = "# \n\n"
        self.assertEqual(_choose_richer_text(plain, md), plain)

    def test_both_empty_returns_empty(self) -> None:
        from summarizer.extraction import _choose_richer_text

        self.assertEqual(_choose_richer_text("", ""), "")


if __name__ == "__main__":
    unittest.main()
