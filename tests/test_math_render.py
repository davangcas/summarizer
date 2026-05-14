"""Tests del pre-render de fórmulas LaTeX a PNG y reemplazo en Markdown."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from summarizer.math_render import (
    render_math_to_png,
    replace_math_with_images,
)


class RenderMathToPngTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="math_render_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_basic_fraction_renders(self) -> None:
        path = render_math_to_png(r"\frac{a}{b}", inline=True, out_dir=self.tmp)
        self.assertIsNotNone(path)
        self.assertTrue(path.exists())
        self.assertGreater(path.stat().st_size, 0)

    def test_idempotent_cache(self) -> None:
        path1 = render_math_to_png("x^2", inline=True, out_dir=self.tmp)
        path2 = render_math_to_png("x^2", inline=True, out_dir=self.tmp)
        self.assertEqual(path1, path2)

    def test_invalid_latex_returns_none(self) -> None:
        path = render_math_to_png(r"\unknowncmd{thing}", inline=True, out_dir=self.tmp)
        self.assertIsNone(path)


class ReplaceMathWithImagesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="math_replace_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_inline_math_replaced(self) -> None:
        md = r"Velocidad: $v = \frac{d}{t}$ por segundo."
        out = replace_math_with_images(md, self.tmp)
        self.assertIn("![](", out)
        self.assertNotIn(r"$v = \frac{d}{t}$", out)

    def test_currency_left_alone(self) -> None:
        md = "Cuesta $5 y luego $10."
        out = replace_math_with_images(md, self.tmp)
        self.assertEqual(out, md)

    def test_block_math_replaced(self) -> None:
        md = r"Antes" + "\n\n$$\\frac{a}{b}$$" + "\n\nDespués"
        out = replace_math_with_images(md, self.tmp)
        self.assertIn("![](", out)
        self.assertNotIn("$$", out)

    def test_fenced_code_block_preserved(self) -> None:
        md = "```\n$x^2$ no se debe procesar\n```\nfuera $y^2$"
        out = replace_math_with_images(md, self.tmp)
        self.assertIn("$x^2$ no se debe procesar", out)
        self.assertIn("![](", out)

    def test_unsupported_block_falls_back_to_code_fence(self) -> None:
        md = (
            "Texto antes\n\n"
            "$$\\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix}$$\n\n"
            "Texto después"
        )
        out = replace_math_with_images(md, self.tmp)
        self.assertIn("```latex", out)
        self.assertIn("\\begin{pmatrix}", out)
        self.assertNotIn("![](", out)

    def test_unsupported_inline_with_newlines_promoted(self) -> None:
        md = "Antes $\\begin{matrix} a \\\\ b \\end{matrix}$ después"
        out = replace_math_with_images(md, self.tmp)
        self.assertIn("\\begin{matrix}", out)

    def test_absolute_path_is_url_encoded(self) -> None:
        space_dir = self.tmp / "dir with space"
        space_dir.mkdir()
        md = r"Velocidad: $v = \frac{d}{t}$ por segundo."
        out = replace_math_with_images(md, space_dir)
        self.assertIn("![](", out)
        self.assertIn("dir%20with%20space", out)
        self.assertNotIn("dir with space/", out)

    def test_base_dir_emits_relative_encoded_url(self) -> None:
        pdf_dir = self.tmp / "out folder"
        math_dir = pdf_dir / "stem_math"
        pdf_dir.mkdir()
        math_dir.mkdir()
        md = r"Velocidad: $v = \frac{d}{t}$ por segundo."
        out = replace_math_with_images(md, math_dir, base_dir=pdf_dir)
        self.assertIn("![](stem_math/math_", out)
        self.assertNotIn("out%20folder", out)
        self.assertNotIn(str(pdf_dir.as_posix()), out)


class RenderMarkdownToPdfImageEmbeddingTests(unittest.TestCase):
    """Verifica que las fórmulas se embeben como imágenes en el PDF final.

    Reproduce el caso original (ruta con espacios) que generaba ``![](...)``
    como texto literal en el PDF. El fix combina URL-encoding y
    ``Section.root`` para que ``fitz.Story`` resuelva la imagen.
    """

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="math_pdf_e2e_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_image_embedded_with_space_in_output_dir(self) -> None:
        import pymupdf

        from summarizer.output import render_markdown_to_pdf

        out_dir = self.tmp / "Universo - Libros"
        out_dir.mkdir()
        out_pdf = out_dir / "doc test.pdf"
        markdown = "# Título\n\nFórmula simple: $E = mc^2$ ¡saludos!\n"
        render_markdown_to_pdf(out_pdf, markdown, fallback_h1="Test")
        self.assertTrue(out_pdf.exists())
        with pymupdf.open(out_pdf) as doc:
            n_imgs = sum(len(p.get_images()) for p in doc)
            page_text = doc[0].get_text()
        self.assertGreaterEqual(n_imgs, 1, "Se esperaba al menos una imagen embebida")
        self.assertNotIn("![](", page_text)

    def test_block_math_inside_list_item_does_not_break_toc(self) -> None:
        """Regresión: ``$$...$$`` en medio de un list item.

        El LLM a veces emite ``  - $$x$$  - resto``. Antes del fix,
        ``_to_image_md_block`` reemplazaba con ``\\n\\n![](path)\\n\\n``,
        inyectando blank lines dentro del list item; markdown_it
        interpretaba el texto previo como **setext heading h2**, lo cual
        producía un salto de nivel inválido (h2 → h4) y revienta
        ``fitz.Story.set_toc`` con ``bad hierarchy level in row N``.
        """
        import pymupdf

        from summarizer.output import render_markdown_to_pdf

        out_pdf = self.tmp / "doc_list_block_math.pdf"
        markdown = (
            "# Test\n\n"
            "### Tema con bloque math en list item\n\n"
            "#### Notas\n"
            "- **Galileo:** transformación entre marcos a velocidad $u$.\n"
            "  - $$X = X' + u t'$$  - Demuestra invariancia.\n"
            "- Otro punto importante.\n\n"
            "#### Resumen del tema\n"
            "Texto del resumen.\n"
        )
        render_markdown_to_pdf(out_pdf, markdown, fallback_h1="Test")
        self.assertTrue(out_pdf.exists())
        with pymupdf.open(out_pdf) as doc:
            n_imgs = sum(len(p.get_images()) for p in doc)
        self.assertGreaterEqual(n_imgs, 1)

    def test_block_math_via_render_emits_inline_image(self) -> None:
        """``_to_image_md_block`` ya no inyecta ``\\n\\n`` alrededor.

        Verifica que la sustitución produce solo ``![](path)``, lo cual
        permite preservar la estructura del Markdown circundante (list
        items, párrafos con math al medio, etc.).
        """
        from summarizer.math_render import replace_math_with_images

        md = "  - antes $$\\frac{a}{b}$$ despues"
        out = replace_math_with_images(md, self.tmp, base_dir=self.tmp.parent)
        self.assertIn("![](", out)
        self.assertNotIn("\n\n![](", out)

    def test_math_inside_index_link_label_does_not_break_pdf(self) -> None:
        """Regresión: span ``$E=mc^2$`` dentro de ``[label](#anchor)`` del índice.

        Antes del fix, ``replace_math_with_images`` inyectaba un ``![](path)``
        en el label, creando un ``]`` interno que el regex de
        ``_strip_fragment_links_line`` no manejaba. El link se colaba al
        renderer y ``fitz.Story`` reventaba con
        ``No destination with id=...``. Tras reordenar el pipeline (strip
        primero, luego math), el PDF se genera sin error y la imagen
        queda embebida.
        """
        import pymupdf

        from summarizer.output import render_markdown_to_pdf

        out_pdf = self.tmp / "doc.pdf"
        markdown = (
            "# Resumen\n\n"
            "## Índice\n\n"
            "- [El poder nuclear: De $E=mc^2$ a la reacción](#el-poder-nuclear)\n"
            "- [Otro tema sin matemática](#otro-tema)\n\n"
            "---\n\n"
            "### El poder nuclear: De $E=mc^2$ a la reacción {#el-poder-nuclear}\n\n"
            "Contenido del tema.\n\n"
            "### Otro tema sin matemática {#otro-tema}\n\n"
            "Contenido del segundo tema.\n"
        )
        render_markdown_to_pdf(out_pdf, markdown, fallback_h1="Test")
        self.assertTrue(out_pdf.exists())
        with pymupdf.open(out_pdf) as doc:
            n_imgs = sum(len(p.get_images()) for p in doc)
            full_text = "\n".join(p.get_text() for p in doc)
        self.assertGreaterEqual(n_imgs, 1)
        self.assertNotIn("](#", full_text)
        self.assertNotIn("![](", full_text)
        self.assertNotIn("{#el-poder-nuclear}", full_text)


if __name__ == "__main__":
    unittest.main()
