"""Tests for the reviewer-facing Ask MorphAgent knowledge service."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from morphagent_ui.reviewer_chat import (
    ReviewerChatClient,
    ReviewerKnowledgeBase,
    build_chat_messages,
    build_system_prompt,
)


class ReviewerKnowledgeBaseTests(unittest.TestCase):
    def _write_bundle(self, chunks: list[dict[str, str]]) -> Path:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "knowledge.json"
        path.write_text(
            json.dumps({"version": 1, "chunks": chunks}),
            encoding="utf-8",
        )
        return path

    def test_bundle_preserves_human_readable_source_labels(self) -> None:
        path = self._write_bundle(
            [
                {
                    "source": "Manuscript",
                    "title": "Validation",
                    "kind": "paper",
                    "text": "MorphAgent validates candidate features across rounds.",
                }
            ]
        )

        knowledge = ReviewerKnowledgeBase.from_path(path)

        self.assertEqual(len(knowledge.chunks), 1)
        self.assertEqual(knowledge.chunks[0].source, "Manuscript")
        self.assertEqual(knowledge.chunks[0].title, "Validation")

    def test_malformed_bundle_is_rejected_instead_of_silently_ignored(self) -> None:
        path = self._write_bundle(
            [
                {
                    "source": "Manuscript",
                    "title": "Missing text",
                    "kind": "paper",
                    "text": "",
                }
            ]
        )

        with self.assertRaisesRegex(ValueError, "text"):
            ReviewerKnowledgeBase.from_path(path)

    def test_search_ranks_matching_code_and_manuscript_chunks(self) -> None:
        path = self._write_bundle(
            [
                {
                    "source": "Manuscript",
                    "title": "Feature validation",
                    "kind": "paper",
                    "text": "Deterministic validation checks variation and redundancy.",
                },
                {
                    "source": "Code: tools/segmentation.py",
                    "title": "segment_all_samples",
                    "kind": "code",
                    "text": "Segmentation masks are reused when they already exist.",
                },
                {
                    "source": "Supplementary",
                    "title": "Unrelated benchmark",
                    "kind": "supplement",
                    "text": "A general ablation study is described here.",
                },
            ]
        )
        knowledge = ReviewerKnowledgeBase.from_path(path)

        selected = knowledge.search(
            "How does the code reuse existing segmentation masks?",
            top_k=2,
            max_chars=5000,
        )

        self.assertEqual(selected[0].source, "Code: tools/segmentation.py")
        self.assertTrue(any(chunk.source == "Manuscript" for chunk in selected) or len(selected) == 2)
        self.assertLessEqual(sum(len(chunk.text) for chunk in selected), 5000)


class ReviewerPromptTests(unittest.TestCase):
    def test_system_prompt_is_positive_evidence_bound_and_honest_about_limits(self) -> None:
        prompt = build_system_prompt()

        self.assertIn("foreground", prompt.lower())
        self.assertIn("contributions", prompt.lower())
        self.assertIn("never fabricate", prompt.lower())
        self.assertIn("limitations", prompt.lower())
        self.assertIn("dismissive", prompt.lower())

    def test_messages_include_source_labels_and_only_recent_history(self) -> None:
        path = self._write_bundle(
            [
                {
                    "source": "Manuscript",
                    "title": "Method",
                    "kind": "paper",
                    "text": "MorphAgent plans biologically grounded features.",
                }
            ]
        )
        chunk = ReviewerKnowledgeBase.from_path(path).chunks[0]
        history = []
        for index in range(6):
            history.extend(
                [
                    {"role": "user", "content": f"old-question-{index}"},
                    {"role": "assistant", "content": f"old-answer-{index}"},
                ]
            )

        messages = build_chat_messages(
            "What is the main method?",
            [chunk],
            history,
            history_turns=2,
        )

        combined = "\n".join(message["content"] for message in messages)
        self.assertIn("[Manuscript — Method]", combined)
        self.assertNotIn("old-question-0", combined)
        self.assertIn("old-question-5", combined)
        self.assertEqual(messages[-1]["content"], "What is the main method?")

    def _write_bundle(self, chunks: list[dict[str, str]]) -> Path:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "knowledge.json"
        path.write_text(json.dumps({"version": 1, "chunks": chunks}), encoding="utf-8")
        return path


class ReviewerChatClientTests(unittest.TestCase):
    def test_client_retries_missing_v1_once_and_returns_text(self) -> None:
        calls: list[dict[str, object]] = []

        class NotFoundError(RuntimeError):
            status_code = 404

        class Completions:
            def __init__(self, base_url: str) -> None:
                self.base_url = base_url

            def create(self, **kwargs):
                calls.append({"base_url": self.base_url, **kwargs})
                if not self.base_url.endswith("/v1"):
                    raise NotFoundError("404 not found")
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="Grounded answer."))]
                )

        def factory(**kwargs):
            return SimpleNamespace(
                chat=SimpleNamespace(completions=Completions(kwargs["base_url"]))
            )

        # Use a real KnowledgeChunk produced by a bundle for search/clipping behavior.
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp) / "knowledge.json"
            bundle.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "chunks": [
                            {
                                "source": "Manuscript",
                                "title": "Overview",
                                "kind": "paper",
                                "text": "MorphAgent discovers grounded features.",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            knowledge = ReviewerKnowledgeBase.from_path(bundle)

        client = ReviewerChatClient(
            base_url="https://provider.example",
            api_key="secret",
            model="review-model",
            client_factory=factory,
        )

        answer = client.ask("What does MorphAgent discover?", knowledge, [])

        self.assertEqual(answer, "Grounded answer.")
        self.assertEqual([call["base_url"] for call in calls], [
            "https://provider.example",
            "https://provider.example/v1",
        ])
        self.assertTrue(all(call["model"] == "review-model" for call in calls))


if __name__ == "__main__":
    unittest.main()
