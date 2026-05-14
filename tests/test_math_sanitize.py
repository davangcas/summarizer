"""Tests del saneador de fórmulas LaTeX dañadas por escapes JSON."""

from __future__ import annotations

import unittest

from summarizer.math_sanitize import sanitize_math_text


class SanitizeMathTextTests(unittest.TestCase):
    def test_no_changes_when_clean(self) -> None:
        text = "Texto plano sin fórmulas ni caracteres raros."
        self.assertEqual(sanitize_math_text(text), text)

    def test_no_changes_with_currency(self) -> None:
        text = "Cuesta $5 y luego $10 más."
        self.assertEqual(sanitize_math_text(text), text)

    def test_tab_restores_text_command(self) -> None:
        broken = "Tierra ($10 \text{m/s}^2$)."
        out = sanitize_math_text(broken)
        self.assertIn("\\text{m/s}^2", out)
        self.assertNotIn("\t", out)

    def test_form_feed_restores_frac(self) -> None:
        broken = "Velocidad: $\frac{a}{b}$."
        out = sanitize_math_text(broken)
        self.assertIn("\\frac{a}{b}", out)
        self.assertNotIn("\x0c", out)

    def test_collapses_newlines_inside_inline_math(self) -> None:
        # El matcher de math inline NO debe cruzar saltos de línea: si el
        # LLM rompió un span con un newline, lo dejamos sin recombinar para
        # no comerse estructura del Markdown circundante. Sí restauramos
        # cualquier comando latex over-escapado dentro del prefijo válido.
        broken = "($10 \text{\nm/s}^2$)"
        out = sanitize_math_text(broken)
        self.assertIn("\\text", out)

    def test_orphan_text_command_recovered(self) -> None:
        sanitized = sanitize_math_text("$ ext{km}$")
        self.assertIn("\\text{km}", sanitized)

    def test_double_backslash_collapsed_for_known_command(self) -> None:
        # Real-world overscape: el LLM produjo `\\\\text` en JSON (4 backslashes)
        # → tras parsear queda `\\text` (2 reales). matplotlib lo interpreta
        # como salto de línea + literal "text", rompiendo el span.
        broken = "$\\approx 10^{-35}\\\\text{m}$"
        out = sanitize_math_text(broken)
        self.assertIn("\\text{m}", out)
        self.assertNotIn("\\\\text", out)

    def test_double_backslash_collapsed_in_block_math(self) -> None:
        broken = "$$10^{16}\\\\text{TeV}$$"
        out = sanitize_math_text(broken)
        self.assertIn("\\text{TeV}", out)
        self.assertNotIn("\\\\text", out)

    def test_double_backslash_preserved_for_unknown_command(self) -> None:
        # Caso conservador: `\\\\foo` no está en la whitelist; se deja igual
        # para no romper usos legítimos de `\\` (salto de línea en matrices).
        body = "$$a \\\\ b \\\\customcmd{x}$$"
        out = sanitize_math_text(body)
        self.assertIn("\\\\ b", out)
        self.assertIn("\\\\customcmd", out)

    def test_block_math_round_trip(self) -> None:
        original = "$$\\frac{a}{b}$$"
        self.assertEqual(sanitize_math_text(original), original)

    def test_preserves_text_outside_math(self) -> None:
        text = "Comprar al $10 por unidad."
        self.assertEqual(sanitize_math_text(text), text)


if __name__ == "__main__":
    unittest.main()
