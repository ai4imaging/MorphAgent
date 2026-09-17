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


def test_maintained_manuscript_folder_builds_paper_without_cover_letter(tmp_path, monkeypatch):
    from scripts import build_reviewer_knowledge as builder
    folder = tmp_path / 'manuscript'
    folder.mkdir()
    for name in ('manucript.pdf', 'SI.pdf', 'feature_list_1.pdf', 'cover_letter.pdf'):
        (folder / name).write_bytes(name.encode())
    monkeypatch.setattr(builder, 'extract_pdf_bytes', lambda data: data.decode() + ' evidence')
    chunks = builder.collect_manuscript_chunks(folder)
    sources = {chunk['source'] for chunk in chunks}
    assert {'Manuscript', 'Supplementary', 'Feature catalogue: feature_list_1'} <= sources
    assert not any('cover_letter' in chunk['text'] for chunk in chunks)


def test_code_bundle_skips_virtual_environments_and_hidden_outputs(tmp_path):
    from scripts.build_reviewer_knowledge import collect_code_chunks
    (tmp_path / 'main.py').write_text('def analyze(): pass')
    for directory in ('.venv-ui-check', '.web_workspace'):
        hidden = tmp_path / directory / 'pkg.py'
        hidden.parent.mkdir()
        hidden.write_text('SECRET_SHOULD_NOT_BE_INDEXED')
    chunks = collect_code_chunks(tmp_path)
    assert any(chunk['source'] == 'Code: main.py' for chunk in chunks)
    assert not any('SECRET_SHOULD_NOT_BE_INDEXED' in chunk['text'] for chunk in chunks)


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

    def test_chinese_question_retrieves_relevant_code(self) -> None:
        path = self._write_bundle([
            {'source':'Manuscript','title':'Overview','kind':'paper','text':'Cell profiling overview.'},
            {'source':'Code: tools/segmentation.py','title':'Segmentation','kind':'code',
             'text':'Segmentation masks are reused for cell feature computation.'},
        ])
        found = ReviewerKnowledgeBase.from_path(path).search('代码如何实现细胞分割？',top_k=1)
        self.assertEqual(found[0].source,'Code: tools/segmentation.py')

    def test_chinese_paper_question_prioritizes_manuscript(self) -> None:
        path = self._write_bundle([
            {'source':'Code: service.py','title':'Planning','kind':'code',
             'text':'Manuscript contribution contribution contribution.'},
            {'source':'Manuscript','title':'Overview','kind':'paper',
             'text':'The manuscript describes biologically grounded cell profiles.'},
        ])
        found = ReviewerKnowledgeBase.from_path(path).search('这篇论文的主要贡献是什么？',top_k=1)
        self.assertEqual(found[0].source,'Manuscript')


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
