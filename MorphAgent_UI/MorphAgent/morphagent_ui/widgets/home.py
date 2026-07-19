"""Home/onboarding screen."""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .common import Card


class HomePage(QWidget):
    new_run_requested = Signal()
    previous_run_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(30, 28, 30, 28)
        outer.setSpacing(0)

        hero = Card()
        hero.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        hero_layout = QHBoxLayout(hero)
        hero_layout.setContentsMargins(34, 32, 20, 32)
        hero_layout.setSpacing(30)
        copy = QVBoxLayout()
        copy.setSpacing(14)
        eyebrow = QLabel("BIOLOGICALLY GROUNDED PROFILING")
        eyebrow.setProperty("role", "eyebrow")
        title = QLabel("From microscopy to biologically grounded features")
        title.setProperty("role", "homeDisplay")
        title.setWordWrap(True)
        body = QLabel(
            "Turn microscopy images into validated, biologically grounded features in one guided run."
        )
        body.setProperty("role", "subtitle")
        body.setWordWrap(True)
        self.new_button = QPushButton("Start a discovery run")
        self.new_button.setProperty("primary", True)
        self.new_button.setProperty("homePrimary", True)
        self.new_button.setMinimumWidth(230)
        self.new_button.setAccessibleName("Start a discovery run")
        self.previous_run_button = QPushButton("Load a previous run")
        self.previous_run_button.setProperty("homeSecondary", True)
        self.previous_run_button.setMinimumWidth(230)
        self.previous_run_button.setAccessibleName("Load results from a previous MorphAgent run")
        copy.addStretch(2)
        copy.addWidget(eyebrow)
        copy.addWidget(title)
        copy.addWidget(body)
        copy.addSpacing(16)
        copy.addWidget(self.new_button, 0, Qt.AlignLeft)
        copy.addWidget(self.previous_run_button, 0, Qt.AlignLeft)
        copy.addStretch(3)
        hero_layout.addLayout(copy, 5)

        self.hero_image = QLabel()
        self.hero_image.setMinimumSize(400, 340)
        self.hero_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.hero_image.setAlignment(Qt.AlignCenter)
        self._hero_source = QPixmap(str(Path(__file__).resolve().parents[1] / "resources" / "morphagent_hero.png"))
        hero_layout.addWidget(self.hero_image, 7)
        outer.addWidget(hero, 1)

        self.new_button.clicked.connect(self.new_run_requested)
        self.previous_run_button.clicked.connect(self._choose_previous_run)

    def _choose_previous_run(self) -> None:
        default_results = Path(__file__).resolve().parents[2] / "demo" / "data" / "results"
        start = default_results if default_results.is_dir() else Path.home()
        selected = QFileDialog.getExistingDirectory(
            self,
            "Choose a completed MorphAgent run",
            str(start),
        )
        if selected:
            self.previous_run_requested.emit(selected)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        if self._hero_source.isNull():
            return
        target = self.hero_image.size()
        pixmap = self._hero_source.scaled(target, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        self.hero_image.setPixmap(pixmap)
