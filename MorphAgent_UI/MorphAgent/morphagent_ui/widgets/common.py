"""Small reusable Qt components."""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class PageHeader(QWidget):
    def __init__(self, eyebrow: str, title: str, subtitle: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 8)
        layout.setSpacing(4)
        eyebrow_label = QLabel(eyebrow.upper())
        eyebrow_label.setProperty("role", "eyebrow")
        title_label = QLabel(title)
        title_label.setProperty("role", "display")
        subtitle_label = QLabel(subtitle)
        subtitle_label.setProperty("role", "subtitle")
        subtitle_label.setWordWrap(True)
        layout.addWidget(eyebrow_label)
        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)


class Card(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("card", True)


class MetricCard(Card):
    def __init__(self, label: str, value: str = "—", accent: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(2)
        self.value_label = QLabel(value)
        self.value_label.setProperty("role", "title")
        if accent:
            self.value_label.setStyleSheet(f"color: {accent};")
        label_widget = QLabel(label)
        label_widget.setProperty("role", "muted")
        layout.addWidget(self.value_label)
        layout.addWidget(label_widget)

    def set_value(self, value: str) -> None:
        self.value_label.setText(value)


class PathPicker(QWidget):
    path_changed = Signal(str)

    def __init__(self, placeholder: str, mode: str = "directory", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.mode = mode
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(placeholder)
        self.button = QPushButton("Browse…")
        self.button.setAccessibleName(f"Browse for {placeholder}")
        layout.addWidget(self.edit, 1)
        layout.addWidget(self.button)
        self.button.clicked.connect(self._browse)
        self.edit.textChanged.connect(self.path_changed)

    def text(self) -> str:
        return self.edit.text().strip()

    def setText(self, value: str) -> None:  # noqa: N802 - mirrors Qt
        self.edit.setText(value)

    def _browse(self) -> None:
        start = self.text() or str(Path.home())
        if self.mode == "directory":
            selected = QFileDialog.getExistingDirectory(self, "Choose folder", start)
        elif self.mode == "csv":
            selected, _ = QFileDialog.getOpenFileName(self, "Choose metadata", start, "CSV files (*.csv);;All files (*)")
        else:
            selected, _ = QFileDialog.getOpenFileName(self, "Choose file", start, "Text and JSON (*.txt *.md *.json);;All files (*)")
        if selected:
            self.setText(selected)


def set_dynamic_property(widget: QWidget, name: str, value: str) -> None:
    widget.setProperty(name, value)
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()
