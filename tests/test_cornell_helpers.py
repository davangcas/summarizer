"""Tests sin LLM: troceo de Markdown ensamblado y flags de configuración."""

from __future__ import annotations

import importlib
import os
import unittest


class TopicSectionSplitTests(unittest.TestCase):
    def test_topic_sections_standard_shape(self) -> None:
        from summarizer.cornell_summary import _topic_sections_from_assembled_markdown

        md = (
            "# Mi libro\n\n## Índice\n\n- [A](#a)\n\n---\n\n"
            "### A {#a}\n\n#### Pistas\n- p\n\n#### Notas\nx\n\n"
            "### B {#b}\n\n#### Pistas\n- q\n\n#### Notas\ny"
        )
        parts = _topic_sections_from_assembled_markdown(md)
        self.assertEqual(len(parts), 2)
        self.assertTrue(parts[0].startswith("### A"))
        self.assertTrue(parts[1].startswith("### B"))

    def test_topic_sections_single_block(self) -> None:
        from summarizer.cornell_summary import _topic_sections_from_assembled_markdown

        md = "# T\n\n## Índice\n\n- [x](#x)\n\n---\n\n### X {#x}\nbody"
        parts = _topic_sections_from_assembled_markdown(md)
        self.assertEqual(len(parts), 1)


class UnifyWindowsEnvTests(unittest.TestCase):
    def test_unify_windows_false_via_env(self) -> None:
        os.environ["SUMMARIZER_SUMMARY_UNIFY_WINDOWS"] = "false"
        import summarizer.config as cfg

        importlib.reload(cfg)
        self.assertFalse(cfg.SUMMARY_UNIFY_WINDOWS)
        os.environ.pop("SUMMARIZER_SUMMARY_UNIFY_WINDOWS", None)
        importlib.reload(cfg)


if __name__ == "__main__":
    unittest.main()
