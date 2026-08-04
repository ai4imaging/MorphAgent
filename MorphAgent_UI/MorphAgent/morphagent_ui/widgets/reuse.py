"""Reuse historical MorphAgent code features on a new dataset."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QCursor, QTextCursor
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..controller import ReuseConfig, ReuseController
from ..models import diagnose_dataset_selection, scan_dataset
from .common import Card, PathPicker, set_dynamic_property

try:
    from tools.code_reuse import diagnose_reuse_inputs, summarize_source_results
except Exception:  # pragma: no cover - UI may start before tools are importable
    diagnose_reuse_inputs = None  # type: ignore[assignment]
    summarize_source_results = None  # type: ignore[assignment]


def default_results_browse_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "demo" / "data" / "results"


class ReusePage(QWidget):
    """Hidden work page: pick history results + new dataset, then execute code reuse."""

    back_requested = Signal()
    review_requested = Signal(str)

    def __init__(
        self,
        repository_root: str | Path,
        controller: ReuseController | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.repository_root = Path(repository_root).expanduser().resolve()
        self.controller = controller or ReuseController(self)
        self.results_dir = ""
        self._active = False
        self._started_at: datetime | None = None

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
        eyebrow = QLabel("CODE REUSE")
        eyebrow.setProperty("role", "eyebrow")
        title = QLabel("Reuse history features")
        title.setProperty("role", "display")
        subtitle = QLabel(
            "Apply completed MorphAgent code features to a new dataset. "
            "Rounds run in order from historical merged code — Code only · No LLM/VLM calls."
        )
        subtitle.setProperty("role", "subtitle")
        subtitle.setWordWrap(True)
        heading.addWidget(eyebrow)
        heading.addWidget(title)
        heading.addWidget(subtitle)
        header.addLayout(heading, 1)
        outer.addLayout(header)

        note = QLabel("Code only · No LLM/VLM calls · VLM features in the history run are skipped")
        note.setProperty("role", "muted")
        note.setWordWrap(True)
        outer.addWidget(note)

        inputs = Card()
        inputs_layout = QVBoxLayout(inputs)
        inputs_layout.setContentsMargins(18, 16, 18, 16)
        inputs_layout.setSpacing(10)

        source_label = QLabel("History results")
        source_label.setStyleSheet("font-weight: 700;")
        self.source_picker = PathPicker("Choose a completed MorphAgent results folder")
        self.source_summary = QLabel("Select a completed run that contains round_*/merged feature code.")
        self.source_summary.setProperty("role", "muted")
        self.source_summary.setWordWrap(True)
        inputs_layout.addWidget(source_label)
        inputs_layout.addWidget(self.source_picker)
        inputs_layout.addWidget(self.source_summary)

        data_label = QLabel("New dataset")
        data_label.setStyleSheet("font-weight: 700;")
        self.dataset_picker = PathPicker("Choose a dataset folder (project root or dataset/)")
        self.dataset_summary = QLabel("Expected layout: dataset/<sample>/*.tif")
        self.dataset_summary.setProperty("role", "muted")
        self.dataset_summary.setWordWrap(True)
        inputs_layout.addWidget(data_label)
        inputs_layout.addWidget(self.dataset_picker)
        inputs_layout.addWidget(self.dataset_summary)
        outer.addWidget(inputs)

        status_card = Card()
        status_layout = QVBoxLayout(status_card)
        status_layout.setContentsMargins(16, 14, 16, 14)
        status_layout.setSpacing(8)
        status_top = QHBoxLayout()
        self.status_label = QLabel("Ready")
        self.status_label.setProperty("role", "title")
        self.elapsed_label = QLabel("")
        self.elapsed_label.setProperty("role", "muted")
        status_top.addWidget(self.status_label)
        status_top.addStretch(1)
        status_top.addWidget(self.elapsed_label)
        status_layout.addLayout(status_top)
        self.status_detail = QLabel("Choose history results and a new dataset, then start reuse.")
        self.status_detail.setProperty("role", "muted")
        self.status_detail.setWordWrap(True)
        status_layout.addWidget(self.status_detail)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        status_layout.addWidget(self.progress)
        self.blocker_label = QLabel("")
        self.blocker_label.setWordWrap(True)
        self.blocker_label.hide()
        status_layout.addWidget(self.blocker_label)
        outer.addWidget(status_card)

        actions = QHBoxLayout()
        actions.addStretch(1)
        self.start_button = QPushButton("Start reuse")
        self.start_button.setProperty("primary", True)
        self.start_button.setProperty("largePrimary", True)
        self.start_button.setMinimumSize(180, 48)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.review_button = QPushButton("View results")
        self.review_button.setEnabled(False)
        actions.addWidget(self.cancel_button)
        actions.addWidget(self.review_button)
        actions.addWidget(self.start_button)
        outer.addLayout(actions)

        log_card = Card()
        log_layout = QVBoxLayout(log_card)
        log_layout.setContentsMargins(16, 14, 16, 14)
        log_title = QLabel("Reuse log")
        log_title.setStyleSheet("font-weight: 700;")
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(180)
        log_layout.addWidget(log_title)
        log_layout.addWidget(self.log_view)
        outer.addWidget(log_card, 1)

        self.source_picker.path_changed.connect(self._refresh_source_summary)
        self.dataset_picker.path_changed.connect(self._refresh_dataset_summary)
        self.start_button.clicked.connect(self._start_reuse)
        self.cancel_button.clicked.connect(self.controller.cancel)
        self.review_button.clicked.connect(self._emit_review)
        self.controller.log_line.connect(self.append_log)
        self.controller.progress_changed.connect(self.progress.setValue)
        self.controller.state_changed.connect(self._state_changed)
        self.controller.run_finished.connect(self._run_finished)

        default_results = default_results_browse_dir()
        if default_results.is_dir():
            # Prefer browsing near packaged demo results without auto-selecting.
            self.source_picker.edit.setPlaceholderText(str(default_results))

    def prepare(self) -> None:
        """Reset interactive state when opening from Home."""

        if not self._active:
            self.status_label.setText("Ready")
            self.status_detail.setText("Choose history results and a new dataset, then start reuse.")
            self.progress.setValue(0)
            self.review_button.setEnabled(bool(self.results_dir))
            self.cancel_button.setEnabled(False)
            self.start_button.setEnabled(True)
            self.elapsed_label.setText("")
        self._refresh_source_summary(self.source_picker.text())
        self._refresh_dataset_summary(self.dataset_picker.text())

    def append_log(self, line: str) -> None:
        self.log_view.appendPlainText(line)
        self.log_view.moveCursor(QTextCursor.End)

    def _refresh_source_summary(self, value: str = "") -> None:
        path = (value or self.source_picker.text()).strip()
        if not path:
            self.source_summary.setText("Select a completed run that contains round_*/merged feature code.")
            set_dynamic_property(self.source_summary, "role", "muted")
            self._update_blockers()
            return
        if summarize_source_results is None:
            self.source_summary.setText(path)
            set_dynamic_property(self.source_summary, "role", "muted")
            self._update_blockers()
            return
        try:
            summary = summarize_source_results(path)
        except Exception as exc:  # noqa: BLE001
            self.source_summary.setText(f"Could not scan history results: {exc}")
            set_dynamic_property(self.source_summary, "role", "error")
            self._update_blockers()
            return
        rounds = summary.get("reusable_rounds", 0)
        features = summary.get("code_feature_count", 0)
        skipped = summary.get("skipped_rounds", 0)
        text = f"{rounds} reusable code round(s) · {features} code feature(s)"
        if skipped:
            text += f" · {skipped} round(s) skipped"
        if rounds:
            set_dynamic_property(self.source_summary, "role", "success")
        else:
            set_dynamic_property(self.source_summary, "role", "warning")
        self.source_summary.setText(text)
        self._update_blockers()

    def _refresh_dataset_summary(self, value: str = "") -> None:
        path = (value or self.dataset_picker.text()).strip()
        if not path:
            self.dataset_summary.setText("Expected layout: dataset/<sample>/*.tif")
            set_dynamic_property(self.dataset_summary, "role", "muted")
            self._update_blockers()
            return
        problem = diagnose_dataset_selection(path)
        if problem:
            self.dataset_summary.setText(problem.splitlines()[0])
            set_dynamic_property(self.dataset_summary, "role", "warning")
            self._update_blockers()
            return
        summary = scan_dataset(path)
        parts = [f"{summary.sample_count} samples", f"{summary.primary_image_count} primary images"]
        if summary.mask_count:
            parts.append(f"{summary.mask_count} masks")
        else:
            parts.append("no masks detected (features may return NaN)")
        self.dataset_summary.setText("Ready · " + " · ".join(parts))
        role = "warning" if summary.mask_count == 0 or summary.empty_samples else "success"
        set_dynamic_property(self.dataset_summary, "role", role)
        self._update_blockers()

    def _update_blockers(self) -> None:
        if diagnose_reuse_inputs is None:
            blockers = []
        else:
            blockers = diagnose_reuse_inputs(self.source_picker.text(), self.dataset_picker.text())
        if blockers and not self._active:
            self.blocker_label.setText(" · ".join(blockers[:2]))
            set_dynamic_property(self.blocker_label, "role", "warning")
            self.blocker_label.show()
        else:
            self.blocker_label.hide()

    def _start_reuse(self) -> None:
        if self.controller.running:
            return
        source = self.source_picker.text()
        data_root = self.dataset_picker.text()
        if diagnose_reuse_inputs is not None:
            blockers = diagnose_reuse_inputs(source, data_root)
        else:
            blockers = []
            if not source:
                blockers.append("Choose a completed MorphAgent results folder.")
            if not data_root:
                blockers.append("Choose a dataset folder.")
        if blockers:
            QMessageBox.warning(
                self,
                "Reuse setup needs attention",
                "\n".join(f"• {item}" for item in blockers),
            )
            self._update_blockers()
            return

        config = ReuseConfig(
            repository_root=str(self.repository_root),
            source_results=source,
            data_root=data_root,
            python_executable=sys.executable,
        )
        self.log_view.clear()
        self.progress.setValue(0)
        self.results_dir = ""
        self.review_button.setEnabled(False)
        self._started_at = datetime.now()
        try:
            results_dir = self.controller.start(config)
        except (RuntimeError, OSError) as exc:
            QMessageBox.critical(self, "Could not start reuse", str(exc))
            return
        self.results_dir = results_dir
        self._active = True
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.status_label.setText("Reusing…")
        self.status_detail.setText(f"Writing to {results_dir}")
        self.append_log(f"[UI] Started reuse → {results_dir}")

    def _state_changed(self, state: str, detail: str) -> None:
        self.status_detail.setText(detail)
        if state == "running":
            self.status_label.setText("Reusing…")
        elif state == "cancelling":
            self.status_label.setText("Cancelling…")
        elif state == "complete":
            self.status_label.setText("Complete")
        elif state == "cancelled":
            self.status_label.setText("Cancelled")
        elif state == "failed":
            self.status_label.setText("Failed")

    def _run_finished(self, success: bool, path: str) -> None:
        self._active = False
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        if path and path != "cancelled":
            # Prefer results dir on success; failure path may be the log file.
            candidate = Path(path)
            if candidate.is_dir():
                self.results_dir = str(candidate)
            elif candidate.parent.is_dir() and (candidate.parent / "features.csv").is_file():
                self.results_dir = str(candidate.parent)
        if self._started_at is not None:
            elapsed = datetime.now() - self._started_at
            total = int(elapsed.total_seconds())
            hours, rem = divmod(total, 3600)
            minutes, seconds = divmod(rem, 60)
            self.elapsed_label.setText(f"Elapsed {hours:02d}:{minutes:02d}:{seconds:02d}")
        self.review_button.setEnabled(bool(success and self.results_dir))
        if success:
            self.progress.setValue(100)
            self.status_label.setText("Complete")
            self.status_detail.setText(f"Reuse complete · {self.results_dir}")
        self._update_blockers()

    def _emit_review(self) -> None:
        if self.results_dir:
            self.review_requested.emit(self.results_dir)
