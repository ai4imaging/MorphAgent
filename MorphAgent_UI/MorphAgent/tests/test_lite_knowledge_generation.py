from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from knowledge.deep_research import extract_deep_research
from knowledge.precomputed_lite import (
    DEEP_RESEARCH_SYSTEM_PROMPT,
    LITERATURE_SUMMARY_SYSTEM_PROMPT,
    load_or_generate_summary,
)
from knowledge.rag import extract_rag_knowledge


class LiteKnowledgeGenerationTests(unittest.TestCase):
    def _llm(self, content: str = "generated summary") -> Mock:
        llm = Mock()
        llm.invoke.return_value = SimpleNamespace(content=content)
        return llm

    def test_precomputed_summary_wins_without_model_call(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            prepared = root / "precomputed"
            prepared.mkdir()
            (prepared / "deep_research_summary.txt").write_text(
                "prepared deep research",
                encoding="utf-8",
            )
            with patch("config.make_chat_llm") as make_llm:
                summary = load_or_generate_summary(
                    root,
                    "deep_research",
                    "How do mitochondria change?",
                    "Confocal images",
                )
            self.assertEqual(summary, "prepared deep research")
            make_llm.assert_not_called()

    def test_missing_summaries_are_generated_and_cached_per_question(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            deep_llm = self._llm("deep result")
            rag_llm = self._llm("literature result")
            with patch("config.make_chat_llm", side_effect=[deep_llm, rag_llm]):
                deep = extract_deep_research(
                    root,
                    user_query="Quantify mitochondrial fragmentation",
                    dataset_description="Confocal cell images",
                )
                rag = extract_rag_knowledge(
                    root,
                    user_query="Quantify mitochondrial fragmentation",
                    dataset_description="Confocal cell images",
                )

            self.assertEqual(deep, "deep result")
            self.assertEqual(rag, "literature result")
            cache = root / ".knowledge_precomputed"
            self.assertEqual(
                (cache / "deep_research_summary.txt").read_text(encoding="utf-8").strip(),
                "deep result",
            )
            self.assertEqual(
                (cache / "rag_knowledge_summary.txt").read_text(encoding="utf-8").strip(),
                "literature result",
            )
            deep_messages = deep_llm.invoke.call_args.args[0]
            rag_messages = rag_llm.invoke.call_args.args[0]
            self.assertEqual(deep_messages[0].content, DEEP_RESEARCH_SYSTEM_PROMPT)
            self.assertEqual(rag_messages[0].content, LITERATURE_SUMMARY_SYSTEM_PROMPT)
            self.assertIn("Quantify mitochondrial fragmentation", deep_messages[1].content)

            with patch("config.make_chat_llm") as make_llm:
                reused = extract_deep_research(
                    root,
                    user_query="Quantify mitochondrial fragmentation",
                    dataset_description="Confocal cell images",
                )
            self.assertEqual(reused, "deep result")
            make_llm.assert_not_called()

    def test_changed_question_regenerates_cached_summary(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            first_llm = self._llm("first")
            second_llm = self._llm("second")
            with patch("config.make_chat_llm", side_effect=[first_llm, second_llm]):
                first = load_or_generate_summary(root, "rag", "Question one")
                second = load_or_generate_summary(root, "rag", "Question two")
            self.assertEqual(first, "first")
            self.assertEqual(second, "second")

    def test_model_failure_is_non_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            llm = Mock()
            llm.invoke.side_effect = RuntimeError("temporary API error")
            with patch("config.make_chat_llm", return_value=llm):
                summary = extract_rag_knowledge(
                    root,
                    user_query="Profile cell morphology",
                )
            self.assertIsNone(summary)

    def test_disabled_sources_do_not_generate(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            with patch("config.make_chat_llm") as make_llm:
                self.assertIsNone(
                    extract_deep_research(
                        root,
                        enable_deep_research=False,
                        user_query="Question",
                    )
                )
                self.assertIsNone(
                    extract_rag_knowledge(
                        root,
                        enable_rag=False,
                        user_query="Question",
                    )
                )
            make_llm.assert_not_called()


if __name__ == "__main__":
    unittest.main()
