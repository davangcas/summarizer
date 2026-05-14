"""Tests del dedup semántico y fusión de notas en assemble_cornell_windows_markdown."""

from __future__ import annotations

import unittest


class TopicSimilarityTests(unittest.TestCase):
    def test_paraphrased_titles_meet_threshold(self) -> None:
        from summarizer.cornell_summary import _topic_similarity

        sim = _topic_similarity(
            "La gravedad superficial de la estrella",
            "",
            "Gravedad superficial estelar",
            "",
        )
        self.assertGreaterEqual(sim, 0.5)

    def test_distinct_topics_below_threshold(self) -> None:
        from summarizer.cornell_summary import _topic_similarity

        sim = _topic_similarity(
            "Composición química del Sol",
            "El sol tiene hidrógeno y helio.",
            "Distribución de galaxias",
            "Cúmulos y supercúmulos.",
        )
        self.assertLess(sim, 0.5)

    def test_stopwords_filtered_from_title_tokens(self) -> None:
        from summarizer.cornell_summary import _title_tokens

        toks = _title_tokens("La distribución de las galaxias")
        self.assertIn("distribucion", toks)
        self.assertIn("galaxias", toks)
        self.assertNotIn("la", toks)
        self.assertNotIn("de", toks)

    def test_identical_titles_score_one(self) -> None:
        from summarizer.cornell_summary import _topic_similarity

        sim = _topic_similarity("Gravedad estelar", "", "Gravedad estelar", "")
        self.assertEqual(sim, 1.0)


class MergeNotesTests(unittest.TestCase):
    def test_dedup_identical_lines(self) -> None:
        from summarizer.cornell_summary import _merge_notes

        merged = _merge_notes("- punto A\n- punto B", "- punto B\n- punto C")
        self.assertEqual(merged.count("- punto B"), 1)
        self.assertIn("- punto A", merged)
        self.assertIn("- punto C", merged)


class MergeTopicBlocksTests(unittest.TestCase):
    def test_cues_deduplicated_and_capped(self) -> None:
        from summarizer.cornell_summary import _merge_topic_blocks
        from summarizer.models import CornellTopicBlock

        a = CornellTopicBlock(
            title="Tema",
            cues=["pista1", "pista2"],
            notes="nota A",
            topic_summary="resumen",
        )
        b = CornellTopicBlock(
            title="Tema (variante)",
            cues=["pista2", "pista3"],
            notes="nota B",
            topic_summary="",
        )
        merged = _merge_topic_blocks(a, b)
        self.assertEqual(merged.title, "Tema")
        self.assertEqual(merged.cues.count("pista2"), 1)
        self.assertIn("nota A", merged.notes)
        self.assertIn("nota B", merged.notes)


class AssembleSemanticDedupTests(unittest.TestCase):
    def test_paraphrased_titles_merged(self) -> None:
        import os

        os.environ["SUMMARIZER_ASSEMBLE_DEDUP_GLOBAL"] = "true"
        import importlib

        import summarizer.config as cfg

        importlib.reload(cfg)
        import summarizer.cornell_summary as cs

        importlib.reload(cs)

        from summarizer.models import (
            CornellSummaryStructured,
            CornellTopicBlock,
        )

        t1 = CornellTopicBlock(
            title="Gravedad superficial de la estrella",
            cues=["g", "fuerza"],
            notes="Definicion A.\nValor 2e12 m/s².",
            topic_summary="",
        )
        t2 = CornellTopicBlock(
            title="Gravedad estelar superficial",
            cues=["g superficie", "intensa"],
            notes="Definicion A.\nObjeto cae a 1800 km/s.",
            topic_summary="",
        )
        ordered = [
            (1, 1, CornellSummaryStructured(topics=[t1])),
            (2, 2, CornellSummaryStructured(topics=[t2])),
        ]
        md = cs.assemble_cornell_windows_markdown(ordered, h1_title="Test")
        topic_headings = [line for line in md.splitlines() if line.startswith("### ")]
        self.assertEqual(len(topic_headings), 1, f"Esperaba 1 tema, obtuve:\n{md}")
        self.assertIn("Objeto cae a 1800 km/s.", md)
        self.assertIn("Valor 2e12 m/s²", md)

        os.environ.pop("SUMMARIZER_ASSEMBLE_DEDUP_GLOBAL", None)
        importlib.reload(cfg)
        importlib.reload(cs)


if __name__ == "__main__":
    unittest.main()
