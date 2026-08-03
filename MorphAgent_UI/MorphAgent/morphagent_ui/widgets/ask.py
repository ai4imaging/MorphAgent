"""Reviewer-facing paper and code chat for Ask MorphAgent."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from qtpy.QtCore import QEvent, QObject, Qt, QThread, Signal
from qtpy.QtGui import QCursor
from qtpy.QtWidgets import (
    QDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..demo_api import load_free_demo_credentials
from ..environment import read_model_environment, save_model_environment
from ..reviewer_chat import ReviewerChatClient, ReviewerKnowledgeBase, WELCOME_MESSAGE
from .common import Card, set_dynamic_property


DEFAULT_KNOWLEDGE_PATH = Path(__file__).resolve().parents[1] / "reviewer_knowledge" / "knowledge.json"


class AskApiDialog(QDialog):
    """Minimal LLM connection dialog shared with the Configure environment model."""

    def __init__(self, repository_root: str | Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.repository_root = Path(repository_root).expanduser().resolve()
        self.credentials: dict[str, str] = {}
        self.setWindowTitle("Ask MorphAgent · Model API")
        self.setModal(True)
        self.setMinimumWidth(660)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 22, 24, 22)
        outer.setSpacing(14)

        eyebrow = QLabel("MODEL API")
        eyebrow.setProperty("role", "eyebrow")
        title = QLabel("Connect Ask MorphAgent")
        title.setProperty("role", "title")
        body = QLabel(
            "Use the same OpenAI-compatible LLM connection as Configure. "
            "The paper excerpts needed for each question are sent to this provider."
        )
        body.setProperty("role", "muted")
        body.setWordWrap(True)
        outer.addWidget(eyebrow)
        outer.addWidget(title)
        outer.addWidget(body)

        default_card = Card()
        default_layout = QVBoxLayout(default_card)
        default_layout.setContentsMargins(18, 16, 18, 16)
        default_layout.setSpacing(9)
        default_label = QLabel("RECOMMENDED · NO SETUP REQUIRED")
        default_label.setProperty("role", "eyebrow")
        default_copy = QLabel(
            "Use MorphAgent’s token-limited default connection for reviewer questions."
        )
        default_copy.setWordWrap(True)
        self.free_api_button = QPushButton("Use default API and start chatting")
        self.free_api_button.setProperty("primary", True)
        self.free_api_button.setMinimumHeight(38)
        self.free_api_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.free_api_button.setAccessibleName("Use the default MorphAgent API and open chat")
        self.free_api_button.clicked.connect(self._use_default_api)
        default_layout.addWidget(default_label)
        default_layout.addWidget(default_copy)
        default_layout.addWidget(self.free_api_button)
        outer.addWidget(default_card)

        own_api_label = QLabel("OR USE YOUR OWN API")
        own_api_label.setProperty("role", "eyebrow")
        outer.addWidget(own_api_label)

        form = QFormLayout()
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(10)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.base_url_edit = QLineEdit()
        self.base_url_edit.setPlaceholderText("https://provider.example/v1")
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText("Model name")
        form.addRow("Base URL", self.base_url_edit)
        form.addRow("API key", self.api_key_edit)
        form.addRow("Model", self.model_edit)
        outer.addLayout(form)

        privacy = QLabel(
            "The API key is saved only in the repository-local .env file and is never "
            "added to prompts, manifests, chat messages, or logs."
        )
        privacy.setProperty("role", "muted")
        privacy.setWordWrap(True)
        outer.addWidget(privacy)

        self.status_label = QLabel("")
        self.status_label.setProperty("role", "muted")
        self.status_label.setWordWrap(True)
        outer.addWidget(self.status_label)

        actions = QHBoxLayout()
        actions.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        self.continue_button = QPushButton("Continue to chat")
        self.continue_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.continue_button.clicked.connect(self._save_and_accept)
        actions.addWidget(cancel)
        actions.addWidget(self.continue_button)
        outer.addLayout(actions)

        self._load()

    def _use_default_api(self) -> None:
        """Persist the existing restricted demo connection and enter chat in one click."""

        try:
            credentials = load_free_demo_credentials()
            values = {
                "LLM_BASE_URL": credentials["base_url"],
                "LLM_API_KEY": credentials["api_key"],
                "LLM_MODEL": credentials["model"],
            }
            save_model_environment(self.repository_root, values)
        except (OSError, RuntimeError) as exc:
            self.status_label.setText(f"Default API unavailable: {exc}")
            set_dynamic_property(self.status_label, "role", "warning")
            return
        self.credentials = values
        self.api_key_edit.clear()
        self.accept()

    def _load(self) -> None:
        current = read_model_environment(self.repository_root)
        self.base_url_edit.setText(current.get("LLM_BASE_URL", "").strip())
        self.model_edit.setText(current.get("LLM_MODEL", "").strip())
        if current.get("LLM_API_KEY", "").strip():
            self.api_key_edit.setPlaceholderText("API key already saved · leave blank to keep it")
            self.status_label.setText("Saved credentials detected · review and continue")
            set_dynamic_property(self.status_label, "role", "success")
        else:
            self.api_key_edit.setPlaceholderText("API key")
            self.status_label.setText("Choose the default API above, or enter your own connection")
            set_dynamic_property(self.status_label, "role", "muted")

    def _save_and_accept(self) -> None:
        current = read_model_environment(self.repository_root)
        base_url = self.base_url_edit.text().strip() or current.get("LLM_BASE_URL", "").strip()
        api_key = self.api_key_edit.text().strip() or current.get("LLM_API_KEY", "").strip()
        model = self.model_edit.text().strip() or current.get("LLM_MODEL", "").strip()
        if not base_url or not api_key or not model:
            self.status_label.setText("Base URL, API key, and model are required.")
            set_dynamic_property(self.status_label, "role", "warning")
            return
        values = {"LLM_BASE_URL": base_url, "LLM_API_KEY": api_key, "LLM_MODEL": model}
        save_model_environment(self.repository_root, values)
        self.credentials = values
        self.api_key_edit.clear()
        self.accept()


class ChatWorker(QThread):
    """Execute one reviewer request away from the Qt event loop."""

    answer_ready = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        client: ReviewerChatClient,
        knowledge: ReviewerKnowledgeBase,
        question: str,
        history: Sequence[Mapping[str, str]],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.client = client
        self.knowledge = knowledge
        self.question = question
        self.history = [dict(item) for item in history]

    def run(self) -> None:
        try:
            answer = self.client.ask(self.question, self.knowledge, self.history)
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.answer_ready.emit(answer)


class AskMorphAgentPage(QWidget):
    """A restrained ChatGPT-like reviewer conversation surface."""

    back_requested = Signal()
    api_setup_requested = Signal()

    def __init__(
        self,
        repository_root: str | Path,
        knowledge_path: str | Path | None = None,
        parent: QWidget | None = None,
        *,
        client_factory=None,
    ) -> None:
        super().__init__(parent)
        self.repository_root = Path(repository_root).expanduser().resolve()
        self.knowledge_path = Path(knowledge_path or DEFAULT_KNOWLEDGE_PATH)
        self.knowledge = ReviewerKnowledgeBase.from_path(self.knowledge_path)
        self.client_factory = client_factory
        self.history: list[dict[str, str]] = []
        self.worker: ChatWorker | None = None
        self._busy = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(30, 24, 30, 24)
        outer.setSpacing(14)

        header = QHBoxLayout()
        self.back_button = QPushButton("← Back to Home")
        self.back_button.setProperty("homeSecondary", True)
        self.back_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.back_button.clicked.connect(self.back_requested)
        header.addWidget(self.back_button, 0, Qt.AlignTop)
        heading = QVBoxLayout()
        eyebrow = QLabel("PAPER COMPANION")
        eyebrow.setProperty("role", "eyebrow")
        title = QLabel("Ask MorphAgent")
        title.setProperty("role", "display")
        subtitle = QLabel(
            "Discuss the manuscript, supplementary evidence, figures, methods, and implementation."
        )
        subtitle.setProperty("role", "subtitle")
        subtitle.setWordWrap(True)
        heading.addWidget(eyebrow)
        heading.addWidget(title)
        heading.addWidget(subtitle)
        header.addLayout(heading, 1)
        outer.addLayout(header)

        self.context_label = QLabel("Grounded in Manuscript · Supplementary · MorphAgent source code")
        self.context_label.setProperty("role", "scaleSummary")
        outer.addWidget(self.context_label)

        self.message_scroll = QScrollArea()
        self.message_scroll.setWidgetResizable(True)
        self.message_scroll.setFrameShape(QScrollArea.NoFrame)
        self.message_scroll.setProperty("chatTranscript", True)
        self.message_container = QWidget()
        self.message_layout = QVBoxLayout(self.message_container)
        self.message_layout.setContentsMargins(20, 18, 20, 18)
        self.message_layout.setSpacing(12)
        self.welcome_label = self._add_message("assistant", WELCOME_MESSAGE)
        self.message_layout.addStretch(1)
        self.message_scroll.setWidget(self.message_container)
        outer.addWidget(self.message_scroll, 1)

        composer = Card()
        composer.setProperty("chatComposer", True)
        composer_layout = QHBoxLayout(composer)
        composer_layout.setContentsMargins(14, 12, 14, 12)
        composer_layout.setSpacing(10)
        self.question_edit = QTextEdit()
        self.question_edit.setPlaceholderText("Ask about the paper, a figure, validation, or the code…")
        self.question_edit.setMinimumHeight(58)
        self.question_edit.setMaximumHeight(116)
        self.question_edit.setAcceptRichText(False)
        self.question_edit.installEventFilter(self)
        self.send_button = QPushButton("Send")
        self.send_button.setProperty("primary", True)
        self.send_button.setMinimumWidth(92)
        self.send_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.send_button.clicked.connect(self._send)
        composer_layout.addWidget(self.question_edit, 1)
        composer_layout.addWidget(self.send_button, 0, Qt.AlignBottom)
        outer.addWidget(composer)

        footer = QHBoxLayout()
        self.status_label = QLabel("Ready · conversations stay in memory for this session")
        self.status_label.setProperty("role", "muted")
        self.status_label.setWordWrap(True)
        footer.addWidget(self.status_label, 1)
        self.reconnect_button = QPushButton("API settings")
        self.reconnect_button.clicked.connect(self.api_setup_requested)
        footer.addWidget(self.reconnect_button, 0)
        outer.addLayout(footer)

    def _add_message(self, role: str, text: str) -> QLabel:
        frame = QFrame()
        frame.setProperty("chatMessage", True)
        frame.setProperty("chatRole", role)
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        row = QHBoxLayout(frame)
        row.setContentsMargins(16, 12, 16, 12)
        label = QLabel(text)
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.LinksAccessibleByMouse)
        label.setOpenExternalLinks(False)
        label.setProperty("role", "chatText")
        if role == "user":
            row.addStretch(1)
            row.addWidget(label, 4)
        else:
            row.addWidget(label, 5)
            row.addStretch(1)
        insert_at = max(0, self.message_layout.count() - 1)
        self.message_layout.insertWidget(insert_at, frame)
        return label

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self.question_edit and event.type() == QEvent.KeyPress:
            if event.key() in (Qt.Key_Return, Qt.Key_Enter) and not (event.modifiers() & Qt.ShiftModifier):
                self._send()
                return True
        return super().eventFilter(watched, event)

    def _scroll_to_bottom(self) -> None:
        bar = self.message_scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _set_busy(self, busy: bool) -> None:
        self._busy = bool(busy)
        self.send_button.setEnabled(not busy)
        self.question_edit.setEnabled(not busy)
        self.reconnect_button.setEnabled(not busy)
        if busy:
            self.status_label.setText("MorphAgent is thinking with manuscript and code evidence…")
            set_dynamic_property(self.status_label, "role", "success")
        elif not self.status_label.text().lower().startswith("provider"):
            self.status_label.setText("Ready · conversations stay in memory for this session")
            set_dynamic_property(self.status_label, "role", "muted")

    def _send(self) -> None:
        if self._busy:
            return
        question = self.question_edit.toPlainText().strip()
        if not question:
            self.status_label.setText("Enter a question before sending.")
            set_dynamic_property(self.status_label, "role", "warning")
            return
        settings = read_model_environment(self.repository_root)
        base_url = settings.get("LLM_BASE_URL", "").strip()
        api_key = settings.get("LLM_API_KEY", "").strip()
        model = settings.get("LLM_MODEL", "").strip()
        if not (base_url and api_key and model):
            self.status_label.setText("Complete Model API settings before asking a question.")
            set_dynamic_property(self.status_label, "role", "warning")
            self.api_setup_requested.emit()
            return

        previous = list(self.history)
        self._add_message("user", question)
        self.history.append({"role": "user", "content": question})
        self.question_edit.clear()
        self._set_busy(True)
        kwargs = {"base_url": base_url, "api_key": api_key, "model": model}
        if self.client_factory is not None:
            kwargs["client_factory"] = self.client_factory
        client = ReviewerChatClient(**kwargs)
        self.worker = ChatWorker(client, self.knowledge, question, previous, self)
        self.worker.answer_ready.connect(self._handle_answer)
        self.worker.failed.connect(self._handle_error)
        self.worker.finished.connect(self._worker_finished)
        self.worker.start()
        self._scroll_to_bottom()

    def _handle_answer(self, answer: str) -> None:
        self._add_message("assistant", answer)
        self.history.append({"role": "assistant", "content": answer})
        self._set_busy(False)
        self.status_label.setText("Answer grounded in the most relevant bundled sources")
        set_dynamic_property(self.status_label, "role", "success")
        self._scroll_to_bottom()

    def _handle_error(self, message: str) -> None:
        clean = message.strip() or "The model provider returned an unknown error."
        self._set_busy(False)
        self.status_label.setText(clean)
        set_dynamic_property(self.status_label, "role", "error")

    def _worker_finished(self) -> None:
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None
