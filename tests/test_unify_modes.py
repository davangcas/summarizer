"""Tests del modo de unificación post-ensamblado (SUMMARIZER_SUMMARY_UNIFY_MODE).

Cada modo cambia el comportamiento de ``_try_unify_assembled``:

- ``none``: el ensamblaje pasa sin tocar y sin llamadas LLM.
- ``lmless``: segunda pasada Jaccard, sin llamadas LLM.
- ``hierarchical`` (default heredado): unificación por lotes con LLM.
- ``aggressive``: hierarchical + pasada final single-pass LLM.
"""

from __future__ import annotations

import importlib
import os
import unittest
from unittest import mock


def _reload_modules() -> tuple[object, object]:
    """Recarga config y cornell_summary para tomar env vars frescas."""
    import summarizer.config as cfg

    importlib.reload(cfg)
    import summarizer.cornell_summary as cs

    importlib.reload(cs)
    return cfg, cs


class UnifyModeResolutionTests(unittest.TestCase):
    """``SUMMARY_UNIFY_MODE`` se resuelve correctamente desde el entorno."""

    def tearDown(self) -> None:
        os.environ.pop("SUMMARIZER_SUMMARY_UNIFY_MODE", None)
        os.environ.pop("SUMMARIZER_SUMMARY_FINAL_UNIFY_PASS", None)
        _reload_modules()

    def test_default_is_hierarchical(self) -> None:
        os.environ.pop("SUMMARIZER_SUMMARY_UNIFY_MODE", None)
        os.environ.pop("SUMMARIZER_SUMMARY_FINAL_UNIFY_PASS", None)
        cfg, _ = _reload_modules()
        self.assertEqual(cfg.SUMMARY_UNIFY_MODE, "hierarchical")

    def test_explicit_mode_wins(self) -> None:
        os.environ["SUMMARIZER_SUMMARY_UNIFY_MODE"] = "lmless"
        cfg, _ = _reload_modules()
        self.assertEqual(cfg.SUMMARY_UNIFY_MODE, "lmless")

    def test_legacy_final_pass_flag_maps_to_aggressive(self) -> None:
        os.environ.pop("SUMMARIZER_SUMMARY_UNIFY_MODE", None)
        os.environ["SUMMARIZER_SUMMARY_FINAL_UNIFY_PASS"] = "true"
        cfg, _ = _reload_modules()
        self.assertEqual(cfg.SUMMARY_UNIFY_MODE, "aggressive")

    def test_invalid_mode_falls_back_to_default(self) -> None:
        os.environ["SUMMARIZER_SUMMARY_UNIFY_MODE"] = "weird"
        cfg, _ = _reload_modules()
        self.assertEqual(cfg.SUMMARY_UNIFY_MODE, "hierarchical")


def _assembled_with_two_topics(*, similar: bool) -> str:
    """Construye un Markdown ensamblado con 2 bloques `### ...`.

    ``similar=True`` produce dos títulos casi idénticos para que la
    similitud Jaccard supere el umbral; ``similar=False`` produce temas
    distintos.
    """
    if similar:
        return (
            "# Resumen\n\n## Índice\n\n"
            "- [Gravedad superficial estelar](#t1)\n"
            "- [Superficial gravedad estelar](#t2)\n\n"
            "---\n\n"
            "### Gravedad superficial estelar {#t1}\n\n"
            "#### Pistas (claves)\n- fuerza\n- intensa\n\n"
            "#### Notas\nDefinición de g superficial.\nValor 2e12 m/s².\n\n"
            "#### Resumen del tema\nGravedad muy alta.\n\n"
            "### Superficial gravedad estelar {#t2}\n\n"
            "#### Pistas (claves)\n- g\n- estelar\n\n"
            "#### Notas\nObjeto cae a 1800 km/s.\nFórmula compleja.\n\n"
            "#### Resumen del tema\nAlta gravedad estelar.\n"
        )
    return (
        "# Resumen\n\n## Índice\n\n"
        "- [Galaxias en cúmulos](#t1)\n"
        "- [Composición química solar](#t2)\n\n"
        "---\n\n"
        "### Galaxias en cúmulos {#t1}\n\n"
        "#### Pistas (claves)\n- cúmulo\n- supercúmulo\n\n"
        "#### Notas\nLas galaxias forman estructuras grandes.\n\n"
        "#### Resumen del tema\nEstructura del universo.\n\n"
        "### Composición química solar {#t2}\n\n"
        "#### Pistas (claves)\n- hidrógeno\n- helio\n\n"
        "#### Notas\nEl Sol tiene H y He principalmente.\n\n"
        "#### Resumen del tema\nQuímica solar.\n"
    )


class UnifyRoutingNoLLMCallsTests(unittest.TestCase):
    """``none`` y ``lmless`` nunca tocan el LLM."""

    def tearDown(self) -> None:
        os.environ.pop("SUMMARIZER_SUMMARY_UNIFY_MODE", None)
        _reload_modules()

    def test_mode_none_returns_input_unchanged(self) -> None:
        os.environ["SUMMARIZER_SUMMARY_UNIFY_MODE"] = "none"
        _, cs = _reload_modules()
        md = _assembled_with_two_topics(similar=True)
        with mock.patch.object(cs, "_chat_cornell_structured") as patched:
            out = cs._try_unify_assembled(md, h1_title="Resumen")
        patched.assert_not_called()
        self.assertEqual(out, md)

    def test_mode_lmless_does_not_call_llm(self) -> None:
        os.environ["SUMMARIZER_SUMMARY_UNIFY_MODE"] = "lmless"
        _, cs = _reload_modules()
        md = _assembled_with_two_topics(similar=True)
        with mock.patch.object(cs, "_chat_cornell_structured") as patched:
            cs._try_unify_assembled(md, h1_title="Resumen")
        patched.assert_not_called()

    def test_mode_lmless_merges_similar_topics(self) -> None:
        os.environ["SUMMARIZER_SUMMARY_UNIFY_MODE"] = "lmless"
        _, cs = _reload_modules()
        md = _assembled_with_two_topics(similar=True)
        out = cs._try_unify_assembled(md, h1_title="Resumen")
        heading_count = sum(1 for line in out.splitlines() if line.startswith("### "))
        self.assertEqual(
            heading_count,
            1,
            f"Esperaba 1 tema tras LM-less merge, obtuve {heading_count}:\n{out}",
        )
        self.assertIn("Valor 2e12 m/s²", out)
        self.assertIn("Objeto cae a 1800 km/s.", out)

    def test_mode_lmless_keeps_distinct_topics(self) -> None:
        os.environ["SUMMARIZER_SUMMARY_UNIFY_MODE"] = "lmless"
        _, cs = _reload_modules()
        md = _assembled_with_two_topics(similar=False)
        out = cs._try_unify_assembled(md, h1_title="Resumen")
        heading_count = sum(1 for line in out.splitlines() if line.startswith("### "))
        self.assertEqual(heading_count, 2)


class ParseTopicSectionTests(unittest.TestCase):
    """El parser de bloques `### ...` reconstruye CornellTopicBlock fielmente."""

    def setUp(self) -> None:
        _, self.cs = _reload_modules()

    def test_parses_full_block(self) -> None:
        section = (
            "### Gravedad estelar {#grav}\n\n"
            "#### Pistas (claves)\n- pista1\n- pista2\n\n"
            "#### Notas\nLínea uno.\nLínea dos.\n\n"
            "#### Resumen del tema\nResumen breve."
        )
        block = self.cs._parse_topic_section(section)
        self.assertIsNotNone(block)
        self.assertEqual(block.title, "Gravedad estelar")
        self.assertEqual(block.cues, ["pista1", "pista2"])
        self.assertIn("Línea uno.", block.notes)
        self.assertIn("Línea dos.", block.notes)
        self.assertEqual(block.topic_summary, "Resumen breve.")

    def test_returns_none_for_invalid(self) -> None:
        self.assertIsNone(self.cs._parse_topic_section(""))
        self.assertIsNone(self.cs._parse_topic_section("texto sin heading"))

    def test_handles_dash_only_cues(self) -> None:
        section = (
            "### T {#x}\n\n"
            "#### Pistas (claves)\n-\n\n"
            "#### Notas\nN\n\n"
            "#### Resumen del tema\nR"
        )
        block = self.cs._parse_topic_section(section)
        self.assertIsNotNone(block)
        self.assertEqual(block.cues, [])


if __name__ == "__main__":
    unittest.main()
