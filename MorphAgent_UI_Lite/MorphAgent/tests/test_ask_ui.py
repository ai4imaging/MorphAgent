"""Qt tests for the Ask MorphAgent setup dialog and chat page."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication, QDialog, QLineEdit

from morphagent_ui.environment import read_model_environment
from morphagent_ui.demo_api import load_free_demo_credentials
from morphagent_ui.main import MorphAgentWidget
from morphagent_ui.reviewer_chat import WELCOME_MESSAGE
from morphagent_ui.theme import STYLESHEET
from morphagent_ui.widgets.ask import AskApiDialog, AskMorphAgentPage, ChatWorker


class AskMorphAgentUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self._previous_env = {
            name: os.environ.get(name)
            for name in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL")
        }
        for name in self._previous_env:
            os.environ.pop(name, None)

    def tearDown(self) -> None:
        for name, value in self._previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def _bundle(self, root: Path) -> Path:
        path = root / "knowledge.json"
        path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "chunks": [
                        {
                            "source": "Manuscript",
                            "title": "Overview",
                            "kind": "paper",
                            "text": "MorphAgent discovers biologically grounded features.",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_api_dialog_reuses_saved_secret_without_displaying_it(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / ".env").write_text(
                'LLM_BASE_URL="https://provider.example/v1"\n'
                'LLM_API_KEY="saved-secret"\n'
                'LLM_MODEL="review-model"\n',
                encoding="utf-8",
            )

            dialog = AskApiDialog(root)

            self.assertEqual(dialog.base_url_edit.text(), "https://provider.example/v1")
            self.assertEqual(dialog.model_edit.text(), "review-model")
            self.assertEqual(dialog.api_key_edit.echoMode(), QLineEdit.Password)
            self.assertEqual(dialog.api_key_edit.text(), "")
            self.assertIn("already saved", dialog.api_key_edit.placeholderText().lower())

            dialog._save_and_accept()
            self.assertEqual(dialog.result(), QDialog.Accepted)
            values = read_model_environment(root)
            self.assertEqual(values["LLM_API_KEY"], "saved-secret")

    def test_api_dialog_default_api_is_one_click_and_enters_chat(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            dialog = AskApiDialog(root)

            button = getattr(dialog, "free_api_button", None)
            self.assertIsNotNone(button)
            self.assertEqual(button.text(), "Use default API and start chatting")
            self.assertTrue(button.property("primary"))
            self.assertFalse(dialog.continue_button.property("primary"))
            self.assertIn("choose the default", dialog.status_label.text().lower())
            button.click()

            self.assertEqual(dialog.result(), QDialog.Accepted)
            expected = load_free_demo_credentials()
            values = read_model_environment(root)
            self.assertEqual(values["LLM_BASE_URL"], expected["base_url"])
            self.assertEqual(values["LLM_API_KEY"], expected["api_key"])
            self.assertEqual(values["LLM_MODEL"], expected["model"])
            self.assertNotEqual(dialog.api_key_edit.text(), expected["api_key"])

    def test_api_dialog_keeps_open_when_default_api_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            dialog = AskApiDialog(Path(raw))
            with patch(
                "morphagent_ui.widgets.ask.load_free_demo_credentials",
                side_effect=RuntimeError("Default service unavailable"),
            ):
                dialog.free_api_button.click()

            self.assertNotEqual(dialog.result(), QDialog.Accepted)
            self.assertIn("unavailable", dialog.status_label.text().lower())

    def test_theme_has_distinct_restrained_chat_messages_and_composer(self) -> None:
        self.assertIn('QFrame[chatMessage="true"]', STYLESHEET)
        self.assertIn('QFrame[chatRole="user"]', STYLESHEET)
        self.assertIn('QFrame[chatRole="assistant"]', STYLESHEET)
        self.assertIn('QFrame[chatComposer="true"]', STYLESHEET)

    def test_reviewer_knowledge_is_declared_as_package_data(self) -> None:
        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        self.assertIn(
            '"reviewer_knowledge/*.json"',
            pyproject.read_text(encoding="utf-8"),
        )

    def test_api_dialog_blocks_missing_required_values(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            dialog = AskApiDialog(Path(raw))

            dialog._save_and_accept()

            self.assertNotEqual(dialog.result(), QDialog.Accepted)
            self.assertIn("required", dialog.status_label.text().lower())

    def test_chat_page_starts_with_welcome_and_recovers_from_error(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            page = AskMorphAgentPage(root, self._bundle(root))

            self.assertEqual(page.welcome_label.text(), WELCOME_MESSAGE)
            self.assertIn("Manuscript", page.context_label.text())
            self.assertTrue(page.send_button.isEnabled())

            page.question_edit.setPlainText("How is the method validated?")
            page._set_busy(True)
            self.assertFalse(page.send_button.isEnabled())
            self.assertIn("thinking", page.status_label.text().lower())

            page._handle_error("Provider unavailable")
            self.assertTrue(page.send_button.isEnabled())
            self.assertIn("provider unavailable", page.status_label.text().lower())

            page._set_busy(True)
            page._handle_error("Timed out while contacting the model")
            self.assertIn("timed out", page.status_label.text().lower())

            page._set_busy(True)
            page._handle_answer("The manuscript reports an evidence-grounded result.")
            self.assertIn("grounded", page.status_label.text().lower())
            page.close()

    def test_assistant_markdown_is_rendered_as_safe_rich_text(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            page = AskMorphAgentPage(root, self._bundle(root))
            markdown = (
                "## Validation\n\n**Strong evidence** supports the result.\n\n"
                "- Manuscript evidence\n- Code evidence\n\n"
                "<script>alert('unsafe')</script>"
            )

            label = page._add_message("assistant", markdown)

            self.assertEqual(label.textFormat(), Qt.RichText)
            self.assertEqual(label.property("sourceMarkdown"), markdown)
            rendered = label.text()
            self.assertIn("<h2", rendered)
            self.assertIn("<ul", rendered)
            self.assertNotIn("**Strong evidence**", rendered)
            self.assertIn("&lt;script&gt;", rendered)
            page.close()

    def test_chat_worker_emits_answer_without_writing_history(self) -> None:
        class FakeClient:
            def ask(self, question, knowledge, history):
                self.received = (question, tuple(history), len(knowledge.chunks))
                return "Evidence-grounded answer."

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            page = AskMorphAgentPage(root, self._bundle(root))
            client = FakeClient()
            worker = ChatWorker(client, page.knowledge, "Question?", [])
            answers: list[str] = []
            worker.answer_ready.connect(answers.append)

            worker.run()

            self.assertEqual(answers, ["Evidence-grounded answer."])
            self.assertEqual(client.received, ("Question?", (), 1))
            page.close()

    def test_home_ask_action_opens_hidden_chat_page_after_api_setup(self) -> None:
        widget = MorphAgentWidget()
        home = widget.home_page

        self.assertEqual(home.ask_button.text(), "Ask MorphAgent")
        self.assertTrue(home.ask_button.property("homeAsk"))
        self.assertGreater(home.action_layout.indexOf(home.ask_button), home.action_layout.indexOf(home.previous_run_button))
        self.assertEqual(widget.navigation.count(), 5)
        self.assertEqual(widget.pages.count(), 6)

        with patch("morphagent_ui.main.AskApiDialog") as dialog_class:
            dialog_class.return_value.exec.return_value = QDialog.Accepted
            home.ask_button.click()

        self.assertEqual(widget.pages.currentWidget(), widget.ask_page)
        self.assertEqual(widget.navigation.currentRow(), -1)

        widget.ask_page.back_requested.emit()
        self.assertEqual(widget.pages.currentWidget(), widget.home_page)
        self.assertEqual(widget.navigation.currentRow(), 0)
        widget.close()

    def test_cancelled_ask_api_setup_stays_on_home(self) -> None:
        widget = MorphAgentWidget()
        with patch("morphagent_ui.main.AskApiDialog") as dialog_class:
            dialog_class.return_value.exec.return_value = QDialog.Rejected
            widget.home_page.ask_button.click()

        self.assertEqual(widget.pages.currentWidget(), widget.home_page)
        self.assertEqual(widget.navigation.currentRow(), 0)
        widget.close()


if __name__ == "__main__":
    unittest.main()
