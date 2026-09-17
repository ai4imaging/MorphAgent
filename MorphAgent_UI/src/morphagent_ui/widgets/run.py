"""Live, observable execution screen."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from qtpy.QtCore import QTimer, QUrl, Signal
from qtpy.QtGui import QDesktopServices, QColor
from qtpy.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..controller import DynamicEta, STAGES, RunController, estimate_run_seconds
from ..models import DatasetSummary, RunConfig
from ..theme import COLORS
from .common import Card, PageHeader, set_dynamic_property


class StageCard(QFrame):
    def __init__(self, number: int, title: str, detail: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("stageState", "pending")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 9, 10, 9)
        layout.setSpacing(3)
        self.number_label = QLabel(f"{number:02d}")
        self.number_label.setProperty("role", "eyebrow")
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-weight: 700;")
        detail_label = QLabel(detail)
        detail_label.setProperty("role", "muted")
        detail_label.setWordWrap(True)
        layout.addWidget(self.number_label)
        layout.addWidget(self.title_label)
        layout.addWidget(detail_label)

    def set_state(self, state: str) -> None:
        set_dynamic_property(self, "stageState", state)


def format_estimated_duration(seconds: int) -> str:
    """Format an approximate duration for compact UI display."""

    total = max(0, int(seconds))
    if total < 60:
        return "<1 min"
    minutes = max(1, int(round(total / 60)))
    if minutes < 60:
        return f"~{minutes} min"
    hours, remainder = divmod(minutes, 60)
    return f"~{hours} h" if remainder == 0 else f"~{hours} h {remainder} min"


class RunPage(QWidget):
    cancel_requested = Signal()
    edit_requested = Signal()
    review_requested = Signal()

    def __init__(self, controller: RunController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.results_dir = ""
        self.started_at: datetime | None = None
        self.eta: DynamicEta | None = None
        self.run_active = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(28, 24, 28, 24)
        outer.setSpacing(14)
        outer.addWidget(PageHeader(
            "Live run",
            "Live run",
            "The scientific pipeline runs in a background process. Logs and completed artifacts remain available if you cancel.",
        ))

        status_card = Card()
        status_layout = QVBoxLayout(status_card)
        status_layout.setContentsMargins(16, 14, 16, 14)
        status_top = QHBoxLayout()
        self.status_label = QLabel("Not started")
        self.status_label.setProperty("role", "title")
        self.elapsed_label = QLabel("Elapsed 00:00:00")
        self.elapsed_label.setProperty("role", "muted")
        self.eta_label = QLabel("Remaining —")
        self.eta_label.setProperty("role", "muted")
        status_top.addWidget(self.status_label)
        status_top.addStretch(1)
        status_top.addWidget(self.elapsed_label)
        status_top.addWidget(self.eta_label)
        self.status_detail = QLabel("Configure a run to begin.")
        self.status_detail.setProperty("role", "subtitle")
        self.status_detail.setWordWrap(True)
        self.workload_label = QLabel("Estimate appears when a run starts.")
        self.workload_label.setProperty("role", "muted")
        self.workload_label.setWordWrap(True)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("%p%")
        status_layout.addLayout(status_top)
        status_layout.addWidget(self.status_detail)
        status_layout.addWidget(self.workload_label)
        status_layout.addWidget(self.progress)
        outer.addWidget(status_card)

        stages_row = QHBoxLayout()
        stages_row.setSpacing(8)
        self.stage_cards: list[StageCard] = []
        for index, stage in enumerate(STAGES):
            card = StageCard(index + 1, stage.title, stage.description)
            self.stage_cards.append(card)
            stages_row.addWidget(card, 1)
        outer.addLayout(stages_row)

        work_row = QHBoxLayout()
        work_row.setSpacing(12)
        log_card = Card()
        log_layout = QVBoxLayout(log_card)
        log_layout.setContentsMargins(14, 12, 14, 14)
        log_title = QLabel("Pipeline log")
        log_title.setProperty("role", "title")
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.document().setMaximumBlockCount(5000)
        self.log_view.setStyleSheet('font-family: "Fira Code", "SFMono-Regular", monospace; font-size: 11px;')
        log_layout.addWidget(log_title)
        log_layout.addWidget(self.log_view, 1)
        work_row.addWidget(log_card, 4)

        artifact_card = Card()
        artifact_layout = QVBoxLayout(artifact_card)
        artifact_layout.setContentsMargins(14, 12, 14, 14)
        artifact_title = QLabel("Audit artifacts")
        artifact_title.setProperty("role", "title")
        artifact_help = QLabel("Updated from files actually written by the run.")
        artifact_help.setProperty("role", "muted")
        artifact_help.setWordWrap(True)
        self.artifact_list = QListWidget()
        self.artifact_list.setObjectName("Artifacts")
        artifact_layout.addWidget(artifact_title)
        artifact_layout.addWidget(artifact_help)
        artifact_layout.addWidget(self.artifact_list, 1)
        work_row.addWidget(artifact_card, 2)
        outer.addLayout(work_row, 1)

        actions = QHBoxLayout()
        self.edit_button = QPushButton("Edit setup")
        self.cancel_button = QPushButton("Cancel safely")
        self.cancel_button.setProperty("danger", True)
        self.open_button = QPushButton("Open output folder")
        self.open_button.setEnabled(False)
        self.review_button = QPushButton("Review features")
        self.review_button.setProperty("primary", True)
        self.review_button.setEnabled(False)
        actions.addWidget(self.edit_button)
        actions.addWidget(self.cancel_button)
        actions.addStretch(1)
        actions.addWidget(self.open_button)
        actions.addWidget(self.review_button)
        outer.addLayout(actions)

        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self._update_elapsed)
        self._connect_controller()
        self.cancel_button.clicked.connect(self.cancel_requested)
        self.edit_button.clicked.connect(self.edit_requested)
        self.review_button.clicked.connect(self.review_requested)
        self.open_button.clicked.connect(self._open_output)

    def _connect_controller(self) -> None:
        self.controller.log_line.connect(self.append_log)
        self.controller.stage_changed.connect(self.set_stage)
        self.controller.progress_changed.connect(self.set_progress)
        self.controller.artifacts_changed.connect(self.set_artifacts)
        self.controller.state_changed.connect(self.set_state)
        self.controller.run_finished.connect(self._run_finished)

    def begin(
        self,
        results_dir: str,
        config: RunConfig,
        dataset: DatasetSummary | None = None,
    ) -> None:
        self.results_dir = results_dir
        self.started_at = datetime.now()
        initial_seconds = estimate_run_seconds(config, dataset)
        self.eta = DynamicEta(initial_seconds, max(1, int(config.num_rounds)))
        self.run_active = True
        self.timer.start()
        self.log_view.clear()
        self.artifact_list.clear()
        self.progress.setValue(5)
        image_count = 1 if dataset is None else max(
            1,
            dataset.sample_count,
            dataset.primary_image_count,
            dataset.vlm_source_count,
        )
        self.workload_label.setText(
            f"Initial estimate {format_estimated_duration(initial_seconds)} · "
            f"{config.num_rounds} round{'s' if config.num_rounds != 1 else ''} × "
            f"{config.features_per_iteration} features × {image_count} images · "
            "adjusts as stages and rounds complete"
        )
        self._refresh_time_labels(0)
        self.cancel_button.setEnabled(True)
        self.edit_button.setEnabled(False)
        self.open_button.setEnabled(True)
        self.review_button.setEnabled(False)
        for card in self.stage_cards:
            card.set_state("pending")
        self.stage_cards[0].set_state("active")

    def append_log(self, line: str) -> None:
        self.log_view.appendPlainText(line)
        bar = self.log_view.verticalScrollBar()
        bar.setValue(bar.maximum())

    def set_stage(self, index: int, _key: str, title: str) -> None:
        for card_index, card in enumerate(self.stage_cards):
            state = "done" if card_index < index else "active" if card_index == index else "pending"
            card.set_state(state)
        self.status_label.setText(title)

    def set_progress(self, value: int) -> None:
        self.progress.setValue(value)
        if self.eta is not None:
            self.eta.update_progress(value)
            self._update_elapsed()

    def set_state(self, state: str, detail: str) -> None:
        labels = {
            "running": "Run in progress",
            "cancelling": "Cancelling",
            "complete": "Run complete",
            "cancelled": "Run cancelled",
            "failed": "Run needs attention",
        }
        self.status_label.setText(labels.get(state, state.replace("_", " ").title()))
        self.status_detail.setText(detail)
        role = "success" if state == "complete" else "error" if state == "failed" else "warning" if state in {"cancelled", "cancelling"} else "title"
        self.status_label.setProperty("role", role)
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def set_artifacts(self, snapshot: dict) -> None:
        self.results_dir = str(snapshot.get("results_dir", self.results_dir))
        self.artifact_list.clear()
        labels = {
            "features.csv": "Complete feature matrix",
            "retained_features.csv": "Validated feature subset",
            "feature_registry.json": "Feature validation registry",
            "segmentation_summary.json": "Segmentation audit",
        }
        for filename, exists in snapshot.get("files", {}).items():
            prefix = "READY" if exists else "WAIT"
            item = QListWidgetItem(f"{prefix}  {labels.get(filename, filename)}")
            item.setForeground(QColor(COLORS["success"] if exists else COLORS["muted"]))
            self.artifact_list.addItem(item)
        rounds = snapshot.get("completed_rounds", [])
        if self.eta is not None:
            self.eta.update_completed_rounds(len(rounds))
            self._update_elapsed()
        if rounds:
            item = QListWidgetItem(f"READY  Completed rounds: {', '.join(map(str, rounds))}")
            item.setForeground(QColor(COLORS["success"]))
            self.artifact_list.addItem(item)

    def _run_finished(self, success: bool, path: str) -> None:
        self.timer.stop()
        self.run_active = False
        self.cancel_button.setEnabled(False)
        self.edit_button.setEnabled(True)
        self.open_button.setEnabled(True)
        self.review_button.setEnabled(Path(self.results_dir, "features.csv").is_file() or Path(self.results_dir, "feature_registry.json").is_file())
        if success:
            self.progress.setValue(100)
            if self.eta is not None:
                self.eta.update_progress(100)
            self.eta_label.setText("Remaining 0 min")
        else:
            self.eta_label.setText("Estimate stopped")
            for card in self.stage_cards:
                card.set_state("done")
        if path.endswith(".log"):
            self.append_log(f"Detailed log: {path}")

    def _update_elapsed(self) -> None:
        if self.started_at is None:
            return
        elapsed = datetime.now() - self.started_at
        total_seconds = int(elapsed.total_seconds())
        self._refresh_time_labels(total_seconds)

    def _refresh_time_labels(self, elapsed_seconds: int) -> None:
        self.elapsed_label.setText(
            f"Elapsed {timedelta(seconds=max(0, int(elapsed_seconds)))}"
        )
        if self.run_active and self.eta is not None:
            remaining = self.eta.remaining_seconds(elapsed_seconds)
            self.eta_label.setText(
                f"Remaining {format_estimated_duration(remaining)}"
            )

    def _open_output(self) -> None:
        if self.results_dir:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.results_dir))

    def show_demo_state(self) -> None:
        """Populate an explicit visual-only state for screenshot regression."""
        self.results_dir = "demo_results"
        self.set_state("running", "Quantifying 20 candidate feature cards across 48 microscopy samples.")
        self.progress.setValue(62)
        self.run_active = True
        self.eta = DynamicEta(2_400, 2, progress_percent=62)
        self.set_stage(3, "quantify", "Quantify")
        self.elapsed_label.setText("Elapsed 00:18:42")
        self.eta_label.setText("Remaining ~21 min")
        self.workload_label.setText(
            "Initial estimate ~40 min · 2 rounds × 20 features × 48 images · "
            "adjusts as stages and rounds complete"
        )
        self.log_view.setPlainText(
            "Step 1: Read the dataset index\n"
            "  Found 48 samples · 144 primary image files\n"
            "Step 2.4: Data segmentation (running)\n"
            "  Reused masks for 48/48 samples\n"
            "Step 3: Feature planning (using sample HSC_001)\n"
            "  Generated 20 structured feature cards\n"
            "Step 4: Batch feature extraction\n"
            "  CODE  mitochondrial_network_fragmentation · complete\n"
            "  VLM   perinuclear_puncta_enrichment · scoring 31/48"
        )
        self.set_artifacts({
            "results_dir": "demo_results",
            "files": {
                "features.csv": False,
                "retained_features.csv": False,
                "feature_registry.json": False,
                "segmentation_summary.json": True,
            },
            "completed_rounds": [],
        })
