"""Feature library and evidence inspection screens."""

from __future__ import annotations

import csv
import html
import io
import json
import re
from pathlib import Path

from qtpy.QtCore import Qt, QUrl, Signal
from qtpy.QtGui import QColor, QDesktopServices, QPixmap
from qtpy.QtWidgets import (
    QButtonGroup,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..models import FeatureCard, list_result_artifacts, load_feature_cards
from ..theme import COLORS
from .common import Card, MetricCard, PageHeader, PathPicker


EVIDENCE_GROUPS = (
    ("measurements", "Measurements"),
    ("validation", "Validation"),
    ("production", "How it was produced"),
    ("images", "Images & segmentation"),
)
TEXT_PREVIEW_SUFFIXES = {".json", ".csv", ".txt", ".md", ".log", ".py", ".yaml", ".yml", ".toml"}
IMAGE_PREVIEW_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MAX_TEXT_PREVIEW_BYTES = 1_000_000


def _route_color(method: str) -> str:
    if method.lower() == "code":
        return COLORS["aqua"]
    if method.lower() == "vlm":
        return COLORS["violet"]
    return COLORS["muted"]


class FeatureDetail(Card):
    def __init__(self, parent: QWidget | None = None, compact: bool = False) -> None:
        super().__init__(parent)
        self.compact = compact
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 14, 15, 15)
        layout.setSpacing(8)
        self.route = QLabel("SELECT A FEATURE")
        self.route.setProperty("role", "eyebrow")
        self.name = QLabel("Feature card")
        self.name.setProperty("role", "title")
        self.name.setWordWrap(True)
        self.description = QLabel("Planned biological interpretation and validation state appear here.")
        self.description.setProperty("role", "subtitle")
        self.description.setWordWrap(True)
        self.fields = QTextBrowser()
        self.fields.setOpenExternalLinks(False)
        self.fields.setFrameShape(QFrame.NoFrame)
        self.fields.setStyleSheet("background: transparent;")
        layout.addWidget(self.route)
        layout.addWidget(self.name)
        layout.addWidget(self.description)
        if not compact:
            layout.addWidget(self.fields, 1)
        else:
            self.fields.hide()

    def set_card(self, card: FeatureCard | None) -> None:
        if card is None:
            self.route.setText("NO MATCH")
            self.route.setStyleSheet(f"color: {COLORS['muted']}; font-weight: 700;")
            self.name.setText("No matching feature")
            self.description.setText("Choose another route or state to see matching feature cards.")
            self.fields.clear()
            return
        self.route.setText(f"{card.method.upper()}  ·  {card.status.upper()}")
        self.route.setStyleSheet(f"color: {_route_color(card.method)}; font-weight: 700;")
        self.name.setText(card.name)
        self.description.setText(card.description or "No biological interpretation was recorded in the available artifacts.")
        score = "—" if card.validation_score is None else f"{card.validation_score:.3f}"
        rows = (
            ("Category", card.category),
            ("Round", str(card.round_number or "—")),
            ("Validation score", score),
            ("Visual signature", card.expected_visual_signature or "—"),
            ("Channels", card.required_channels or "—"),
            ("Masks / context", card.required_masks or "—"),
            ("Operators", card.candidate_operators or "—"),
            ("Summary", card.summary_statistics or "—"),
            ("Route rationale", card.method_rationale or "—"),
            ("Decision reasons", ", ".join(card.reason_codes) or "—"),
        )
        detail_html = "".join(
            f'<p><span style="color:{COLORS["muted"]}; font-size:11px; font-weight:700;">{html.escape(label.upper())}</span><br>{html.escape(value)}</p>'
            for label, value in rows
        )
        if not self.compact:
            self.fields.setHtml(detail_html)


class FeaturesPage(QWidget):
    feature_selected = Signal(object)

    def __init__(self, viewer=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.results_dir = ""
        self.cards: list[FeatureCard] = []
        self.filtered_cards: list[FeatureCard] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(28, 24, 28, 24)
        outer.setSpacing(12)
        outer.addWidget(PageHeader(
            "Feature library",
            "Interpretable profiles, not anonymous dimensions",
            "Inspect each feature's route, biological meaning, validation state, and structured design fields.",
        ))
        source_row = QHBoxLayout()
        self.results_picker = PathPicker("Run results folder")
        self.reload_button = QPushButton("Reload artifacts")
        source_row.addWidget(self.results_picker, 1)
        source_row.addWidget(self.reload_button)
        outer.addLayout(source_row)

        metric_row = QHBoxLayout()
        self.total_metric = MetricCard("feature cards", "0", COLORS["text"])
        self.code_metric = MetricCard("code route", "0", COLORS["aqua"])
        self.vlm_metric = MetricCard("VLM route", "0", COLORS["violet"])
        self.live_metric = MetricCard("retained / live", "0", COLORS["success"])
        for metric in (self.total_metric, self.code_metric, self.vlm_metric, self.live_metric):
            metric_row.addWidget(metric)
        outer.addLayout(metric_row)

        filters = QVBoxLayout()
        filters.setSpacing(8)
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search feature name, category, or biological interpretation")
        filters.addWidget(self.search_edit)

        choice_row = QHBoxLayout()
        choice_row.setSpacing(8)
        route_label = QLabel("Route")
        route_label.setProperty("role", "filterLabel")
        choice_row.addWidget(route_label)
        self.route_group, self.route_buttons, route_choices = self._make_filter_choices({
            "all": "All",
            "code": "Code",
            "vlm": "VLM",
        })
        choice_row.addLayout(route_choices)
        choice_row.addSpacing(18)
        state_label = QLabel("State")
        state_label.setProperty("role", "filterLabel")
        choice_row.addWidget(state_label)
        self.status_group, self.status_buttons, self.status_choices = self._make_filter_choices({"all": "All"})
        choice_row.addLayout(self.status_choices)
        choice_row.addStretch(1)
        filters.addLayout(choice_row)
        outer.addLayout(filters)

        self.splitter = QSplitter()
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Feature", "Route", "Category", "State", "Score"])
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for column in range(1, 5):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        self.detail = FeatureDetail()
        self.detail.setMinimumWidth(430)
        self.splitter.addWidget(self.table)
        self.splitter.addWidget(self.detail)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([600, 600])
        outer.addWidget(self.splitter, 1)

        self.reload_button.clicked.connect(lambda: self.load_results(self.results_picker.text()))
        self.results_picker.edit.editingFinished.connect(lambda: self.load_results(self.results_picker.text()))
        self.search_edit.textChanged.connect(self._filter)
        self.table.itemSelectionChanged.connect(self._selection_changed)

    def _make_filter_choices(
        self,
        choices: dict[str, str],
    ) -> tuple[QButtonGroup, dict[str, QRadioButton], QHBoxLayout]:
        group = QButtonGroup(self)
        group.setExclusive(True)
        buttons: dict[str, QRadioButton] = {}
        row = QHBoxLayout()
        row.setSpacing(6)
        for value, label in choices.items():
            button = QRadioButton(label)
            button.setProperty("filterChoice", True)
            button.setProperty("filterValue", value)
            button.setMinimumHeight(36)
            button.setCursor(Qt.PointingHandCursor)
            button.setAccessibleName(f"Filter by {label}")
            button.setToolTip(f"Show {label.lower()} features")
            group.addButton(button)
            buttons[value] = button
            row.addWidget(button)
        buttons["all"].setChecked(True)
        for button in buttons.values():
            button.toggled.connect(lambda checked: self._filter() if checked else None)
        return group, buttons, row

    @staticmethod
    def _selected_filter(buttons: dict[str, QRadioButton]) -> str:
        return next((value for value, button in buttons.items() if button.isChecked()), "all")

    def _replace_status_choices(self, choices: dict[str, str], selected: str) -> None:
        while self.status_choices.count():
            item = self.status_choices.takeAt(0)
            button = item.widget()
            if button is not None:
                self.status_group.removeButton(button)
                button.hide()
                button.deleteLater()
        self.status_buttons = {}
        for value, label in choices.items():
            button = QRadioButton(label)
            button.setProperty("filterChoice", True)
            button.setProperty("filterValue", value)
            button.setMinimumHeight(36)
            button.setCursor(Qt.PointingHandCursor)
            button.setAccessibleName(f"Filter by {label}")
            button.setToolTip(f"Show features with state: {label.lower()}")
            button.toggled.connect(lambda checked: self._filter() if checked else None)
            self.status_group.addButton(button)
            self.status_buttons[value] = button
            self.status_choices.addWidget(button)
        self.status_buttons.get(selected, self.status_buttons["all"]).setChecked(True)

    def load_results(self, results_dir: str) -> None:
        self.results_dir = results_dir
        self.results_picker.setText(results_dir)
        self.cards = load_feature_cards(results_dir) if results_dir else []
        self._update_filters()
        self._update_metrics()
        self._filter()

    def _update_filters(self) -> None:
        selected = self._selected_filter(self.status_buttons)
        # Only two feature states are exposed in the UI.
        present = {card.status for card in self.cards if card.status in {"retained", "dropped"}}
        choices = {"all": "All"}
        for state in ("retained", "dropped"):
            if state in present:
                choices[state] = state.title()
        self._replace_status_choices(choices, selected)

    def _update_metrics(self) -> None:
        self.total_metric.set_value(str(len(self.cards)))
        self.code_metric.set_value(str(sum(card.method.lower() == "code" for card in self.cards)))
        self.vlm_metric.set_value(str(sum(card.method.lower() == "vlm" for card in self.cards)))
        self.live_metric.set_value(str(sum(card.status == "retained" for card in self.cards)))

    def _filter(self, *_args) -> None:
        query = self.search_edit.text().strip().lower()
        route = self._selected_filter(self.route_buttons)
        status = self._selected_filter(self.status_buttons)
        self.filtered_cards = []
        for card in self.cards:
            haystack = " ".join((card.name, card.category, card.description)).lower()
            if query and query not in haystack:
                continue
            if route != "all" and card.method.lower() != route:
                continue
            if status != "all" and card.status != status:
                continue
            self.filtered_cards.append(card)
        self.table.setRowCount(len(self.filtered_cards))
        for row, card in enumerate(self.filtered_cards):
            name_item = QTableWidgetItem(card.name)
            name_item.setData(Qt.UserRole, row)
            route_item = QTableWidgetItem(card.method.upper())
            route_item.setForeground(QColor(_route_color(card.method)))
            score = "—" if card.validation_score is None else f"{card.validation_score:.3f}"
            values = (name_item, route_item, QTableWidgetItem(card.category), QTableWidgetItem(card.status.replace("_", " ")), QTableWidgetItem(score))
            for column, item in enumerate(values):
                self.table.setItem(row, column, item)
        if self.filtered_cards:
            self.table.selectRow(0)
            # QTableWidget may keep row 0 selected across reloads and therefore
            # emit no selection signal; refresh the detail pane explicitly.
            self._selection_changed()
        else:
            self.detail.set_card(None)

    def _selection_changed(self) -> None:
        row = self.table.currentRow()
        if 0 <= row < len(self.filtered_cards):
            card = self.filtered_cards[row]
            self.detail.set_card(card)
            self.feature_selected.emit(card)

    def show_demo_state(self) -> None:
        self.cards = [
            FeatureCard("f1", "mitochondrial_network_fragmentation", "code", "spatial", "Measures discontinuity and branch loss in the mitochondrial network.", "retained", 1, 0.91, ("high_unsupervised_signal",), candidate_operators="skeletonize, connected components", summary_statistics="fragment count / network length"),
            FeatureCard("f2", "perinuclear_puncta_enrichment", "vlm", "distribution", "Scores whether punctate structures concentrate around the nucleus.", "retained", 1, 0.86, ("metadata_alignment",), expected_visual_signature="Bright puncta enriched in a perinuclear ring", required_channels="mitochondria + nucleus"),
            FeatureCard("f3", "reticular_bundle_coherence", "vlm", "architecture", "Semantic evidence for coherent reticular bundles across the cell.", "dropped", 1, None, (), expected_visual_signature="Long connected bundles with aligned orientation"),
            FeatureCard("f4", "cell_edge_intensity_gradient", "code", "intensity", "Radial intensity change from cell center to boundary.", "retained", 1, 0.79, (), candidate_operators="distance transform, radial binning"),
            FeatureCard("f5", "organelle_clusteredness_variogram", "code", "texture", "Captures the spatial scale of organelle clustering.", "retained", 1, 0.88, (), candidate_operators="semivariogram"),
        ]
        self._update_filters()
        self._update_metrics()
        self._filter()


class EvidencePage(QWidget):
    def __init__(self, viewer=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self.results_dir = ""
        self.cards: list[FeatureCard] = []
        self.artifacts: list[Path] = []
        self.current_card: FeatureCard | None = None
        self._preview_pixmap = QPixmap()
        self.feature_columns = 3
        self.feature_buttons: dict[str, QPushButton] = {}
        self._feature_cards_by_button: dict[QPushButton, FeatureCard] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(28, 24, 28, 24)
        outer.setSpacing(12)
        outer.addWidget(PageHeader(
            "Evidence",
            "Trace each feature to its evidence",
            "Choose a feature on the left; inspect its measurements, validation, provenance, and visual context on the right.",
        ))
        source_row = QHBoxLayout()
        self.results_picker = PathPicker("Run results folder")
        self.reload_button = QPushButton("Reload evidence")
        source_row.addWidget(self.results_picker, 1)
        source_row.addWidget(self.reload_button)
        outer.addLayout(source_row)

        self.layout_splitter = QSplitter(Qt.Horizontal)

        feature_column = QWidget()
        feature_column_layout = QVBoxLayout(feature_column)
        feature_column_layout.setContentsMargins(0, 0, 0, 0)
        feature_column_layout.setSpacing(12)
        feature_card = Card()
        feature_layout = QVBoxLayout(feature_card)
        feature_layout.setContentsMargins(15, 13, 15, 15)
        feature_layout.setSpacing(8)
        feature_title = QLabel("Choose a feature")
        feature_title.setProperty("role", "evidenceSectionTitle")
        self.feature_search = QLineEdit()
        self.feature_search.setPlaceholderText("Search feature name or category")
        self.feature_group = QButtonGroup(self)
        self.feature_group.setExclusive(True)
        self.feature_scroll = QScrollArea()
        self.feature_scroll.setObjectName("EvidenceFeatureScroll")
        self.feature_scroll.setFrameShape(QFrame.NoFrame)
        self.feature_scroll.setWidgetResizable(True)
        self.feature_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.feature_scroll.setMinimumHeight(300)
        self.feature_grid_widget = QWidget()
        self.feature_grid_widget.setObjectName("EvidenceFeatureGrid")
        self.feature_grid_widget.setMinimumWidth(0)
        self.feature_grid = QGridLayout(self.feature_grid_widget)
        self.feature_grid.setContentsMargins(0, 2, 4, 2)
        self.feature_grid.setHorizontalSpacing(8)
        self.feature_grid.setVerticalSpacing(8)
        self.feature_grid.setAlignment(Qt.AlignTop)
        for column in range(self.feature_columns):
            self.feature_grid.setColumnStretch(column, 1)
        self.feature_scroll.setWidget(self.feature_grid_widget)
        feature_layout.addWidget(feature_title)
        feature_layout.addWidget(self.feature_search)
        feature_layout.addWidget(self.feature_scroll, 1)
        feature_column_layout.addWidget(feature_card)

        self.feature_summary_card = Card()
        self.feature_summary_card.setMinimumHeight(290)
        feature_summary_layout = QVBoxLayout(self.feature_summary_card)
        feature_summary_layout.setContentsMargins(17, 15, 17, 17)
        feature_summary_layout.setSpacing(8)
        self.selected_feature_name = QLabel("Select a feature")
        self.selected_feature_name.setProperty("role", "title")
        self.selected_feature_name.setWordWrap(True)
        self.selected_feature_description = QLabel("Choose a feature above to see its biological description.")
        self.selected_feature_description.setProperty("role", "subtitle")
        self.selected_feature_description.setWordWrap(True)
        feature_summary_layout.addWidget(self.selected_feature_name)
        feature_summary_layout.addWidget(self.selected_feature_description)
        feature_summary_layout.addStretch(1)
        feature_column_layout.addWidget(self.feature_summary_card)
        feature_column_layout.addStretch(1)

        artifact_card = Card()
        artifact_layout = QVBoxLayout(artifact_card)
        artifact_layout.setContentsMargins(15, 13, 15, 15)
        artifact_layout.setSpacing(8)

        self.image_context_note = QLabel(
            "Images and segmentation previews are included as run-level context. "
            "Unless a file is explicitly named for a feature, do not interpret it as a feature-specific heatmap or overlay."
        )
        self.image_context_note.setProperty("role", "evidenceNote")
        self.image_context_note.setWordWrap(True)
        self.image_context_note.hide()
        self.evidence_title = QLabel("Evidence for selected feature")
        self.evidence_title.setProperty("role", "evidenceSectionTitle")
        self.evidence_title.setWordWrap(True)
        self.evidence_summary = QLabel("Choose a feature to see its curated evidence.")
        self.evidence_summary.setProperty("role", "muted")
        self.evidence_summary.setWordWrap(True)
        self.artifact_tree = QTreeWidget()
        self.artifact_tree.setObjectName("EvidenceTree")
        self.artifact_tree.setHeaderLabels(["Evidence", "Type"])
        self.artifact_tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.artifact_tree.setCursor(Qt.PointingHandCursor)
        self.artifact_tree.setMinimumHeight(180)

        preview_shell = QFrame()
        preview_shell.setProperty("evidencePreview", True)
        preview_layout = QVBoxLayout(preview_shell)
        preview_layout.setContentsMargins(12, 10, 12, 10)
        preview_layout.setSpacing(5)
        self.preview_title = QLabel("Preview")
        self.preview_title.setProperty("role", "evidencePreviewTitle")
        self.preview_meta = QLabel("Select a file above.")
        self.preview_meta.setProperty("role", "muted")
        self.preview_meta.setWordWrap(True)
        self.preview_stack = QStackedWidget()
        self.preview_stack.setMinimumHeight(210)
        self.message_preview = QLabel("Select a file to preview it here.")
        self.message_preview.setAlignment(Qt.AlignCenter)
        self.message_preview.setWordWrap(True)
        self.message_preview.setProperty("role", "muted")
        self.image_preview = QLabel()
        self.image_preview.setAlignment(Qt.AlignCenter)
        self.text_preview = QPlainTextEdit()
        self.text_preview.setReadOnly(True)
        self.text_preview.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.text_preview.setProperty("evidenceText", True)
        self.preview_stack.addWidget(self.message_preview)
        self.preview_stack.addWidget(self.image_preview)
        self.preview_stack.addWidget(self.text_preview)
        # Compatibility alias for code that previously addressed the single preview label.
        self.preview = self.message_preview
        preview_layout.addWidget(self.preview_title)
        preview_layout.addWidget(self.preview_meta)
        preview_layout.addWidget(self.preview_stack, 1)

        actions = QHBoxLayout()
        self.open_artifact_button = QPushButton("Open externally")
        self.open_artifact_button.setEnabled(False)
        self.add_to_viewer_button = QPushButton("Add image to napari")
        self.add_to_viewer_button.setVisible(viewer is not None)
        self.add_to_viewer_button.setEnabled(False)
        actions.addWidget(self.open_artifact_button)
        actions.addWidget(self.add_to_viewer_button)
        actions.addStretch(1)
        artifact_layout.addWidget(self.evidence_title)
        artifact_layout.addWidget(self.evidence_summary)
        artifact_layout.addWidget(self.image_context_note)
        artifact_layout.addWidget(self.artifact_tree, 1)
        artifact_layout.addWidget(preview_shell, 2)
        artifact_layout.addLayout(actions)
        self.layout_splitter.addWidget(feature_column)
        self.layout_splitter.addWidget(artifact_card)
        self.layout_splitter.setStretchFactor(0, 1)
        self.layout_splitter.setStretchFactor(1, 1)
        self.layout_splitter.setSizes([600, 600])
        outer.addWidget(self.layout_splitter, 1)

        self.feature_search.textChanged.connect(self._filter_feature_buttons)
        self.artifact_tree.itemSelectionChanged.connect(self._artifact_selected)
        self.artifact_tree.itemDoubleClicked.connect(lambda _item, _column: self._open_artifact())
        self.open_artifact_button.clicked.connect(self._open_artifact)
        self.add_to_viewer_button.clicked.connect(self._add_to_viewer)
        self.reload_button.clicked.connect(lambda: self.set_results(self.results_picker.text()))
        self.results_picker.edit.editingFinished.connect(lambda: self.set_results(self.results_picker.text()))

    def set_results(self, results_dir: str, cards: list[FeatureCard] | None = None) -> None:
        self.results_dir = results_dir
        self.results_picker.setText(results_dir)
        self.cards = list(cards) if cards is not None else load_feature_cards(results_dir)
        self.artifacts = list_result_artifacts(results_dir)
        self._populate_feature_buttons()
        if self.cards:
            self.select_feature(self.cards[0])
        else:
            self.select_feature(None)

    def load_results(self, results_dir: str) -> None:
        self.set_results(results_dir)

    @staticmethod
    def _feature_button_text(name: str, line_length: int = 20) -> str:
        """Wrap underscore-delimited names without hiding any part of the feature name."""
        parts = [part for part in name.split("_") if part]
        if not parts:
            return name or "Unnamed feature"
        lines: list[str] = []
        current = parts[0]
        for part in parts[1:]:
            candidate = f"{current}_{part}"
            if len(candidate) > line_length:
                lines.append(current)
                current = part
            else:
                current = candidate
        lines.append(current)
        return "_\n".join(lines)

    def _populate_feature_buttons(self) -> None:
        for button in tuple(self._feature_cards_by_button):
            self.feature_group.removeButton(button)
            self.feature_grid.removeWidget(button)
            button.hide()
            button.deleteLater()
        self.feature_buttons.clear()
        self._feature_cards_by_button.clear()

        for index, card in enumerate(self.cards):
            button = QPushButton(self._feature_button_text(card.name))
            button.setCheckable(True)
            button.setCursor(Qt.PointingHandCursor)
            button.setMinimumHeight(62)
            button.setMinimumWidth(0)
            button.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            button.setProperty("featureChoice", True)
            button.setProperty("featureName", card.name)
            button.setToolTip(card.name)
            button.clicked.connect(lambda _checked=False, selected=card: self.select_feature(selected))
            self.feature_group.addButton(button)
            key = card.feature_id or card.name
            if key in self.feature_buttons:
                key = f"{key}-{index}"
            self.feature_buttons[key] = button
            self._feature_cards_by_button[button] = card

        self._filter_feature_buttons(self.feature_search.text())

    def _filter_feature_buttons(self, query: str = "") -> None:
        needle = query.strip().lower()
        visible: list[tuple[QPushButton, FeatureCard]] = []
        for button, card in self._feature_cards_by_button.items():
            self.feature_grid.removeWidget(button)
            haystack = " ".join((card.name, card.category, card.description, card.method, card.status)).lower()
            matches = not needle or needle in haystack
            button.setVisible(matches)
            if matches:
                visible.append((button, card))

        for index, (button, _card) in enumerate(visible):
            self.feature_grid.addWidget(button, index // self.feature_columns, index % self.feature_columns)

        current_button = next(
            (
                button
                for button, card in visible
                if self.current_card is not None
                and (card.feature_id == self.current_card.feature_id or card.name == self.current_card.name)
            ),
            None,
        )
        if visible and current_button is None:
            self.select_feature(visible[0][1])

    def select_feature(self, card: FeatureCard | None) -> None:
        self.current_card = card
        if card is not None:
            self.selected_feature_name.setText(card.name)
            self.selected_feature_description.setText(
                card.description or "No biological description was recorded for this feature."
            )
            for button, listed in self._feature_cards_by_button.items():
                if listed.feature_id == card.feature_id or listed.name == card.name:
                    button.setChecked(True)
                    break
        else:
            self.selected_feature_name.setText("Select a feature")
            self.selected_feature_description.setText("Choose a feature above to see its biological description.")
        self._populate_artifacts()

    @staticmethod
    def _artifact_group(path: Path) -> str:
        name = path.name.lower()
        suffix = path.suffix.lower()
        if name in {"features.csv", "retained_features.csv"}:
            return "measurements"
        if suffix in IMAGE_PREVIEW_SUFFIXES or "segmentation" in name:
            return "images"
        if name == "feature_registry.json" or name == "round_results.json" or name.startswith("validation_"):
            return "validation"
        return "production"

    def _feature_artifacts(self) -> list[Path]:
        candidates = list(self.artifacts)
        card = self.current_card
        direct_paths: set[Path] = set()
        if card is not None:
            for raw_path in card.source_paths.values():
                path = Path(raw_path)
                if path.is_file():
                    direct_paths.add(path.resolve())
                    if path not in candidates:
                        candidates.append(path)
        selected: list[Path] = []
        seen: set[Path] = set()
        root = Path(self.results_dir)
        for path in candidates:
            if not path.is_file():
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            name = path.name.lower()
            suffix = path.suffix.lower()
            if suffix == ".py" or suffix == ".log" or name.endswith("_knowledge_summary.txt"):
                continue

            is_image = suffix in IMAGE_PREVIEW_SUFFIXES
            # Run-level shared previews must not appear as per-feature evidence.
            lowered_parts = {part.lower() for part in path.parts}
            if is_image and "first_sample_visualization" in lowered_parts:
                continue
            is_global_context = name in {"segmentation_summary.json", "ui_run_manifest.json"}
            is_feature_source = False
            if card is not None:
                try:
                    relative = path.relative_to(root)
                except ValueError:
                    relative = Path(path.name)
                round_part = next((part for part in relative.parts if part.startswith("round_")), "")
                same_round = not round_part or not card.round_number or round_part == f"round_{card.round_number}"
                candidate_name = (
                    name in {"features.csv", "retained_features.csv", "feature_registry.json", "feature_plan.json", "round_results.json"}
                    or name.startswith("validation_")
                )
                is_feature_source = same_round and candidate_name and self._artifact_mentions_feature(path, card)
                if resolved in direct_paths and suffix in {".json", ".csv"}:
                    is_feature_source = True

            if not (is_image or is_global_context or is_feature_source):
                continue
            seen.add(resolved)
            selected.append(path)
        return selected

    @staticmethod
    def _artifact_mentions_feature(path: Path, card: FeatureCard) -> bool:
        name = card.name
        suffix = path.suffix.lower()
        try:
            if suffix == ".csv":
                with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                    reader = csv.reader(handle)
                    header = next(reader, [])
                    if name in header:
                        return True
                    return any(any(name in cell for cell in row) for row in reader)
            if suffix in TEXT_PREVIEW_SUFFIXES and path.stat().st_size <= 5_000_000:
                return name in path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
        return False

    @staticmethod
    def _artifact_type(path: Path) -> str:
        labels = {
            ".py": "PYTHON",
            ".json": "JSON",
            ".csv": "CSV",
            ".log": "LOG",
            ".txt": "TEXT",
            ".png": "IMAGE",
            ".jpg": "IMAGE",
            ".jpeg": "IMAGE",
            ".bmp": "IMAGE",
            ".tif": "IMAGE",
            ".tiff": "IMAGE",
            ".pdf": "PDF",
        }
        return labels.get(path.suffix.lower(), path.suffix.lstrip(".").upper() or "FILE")

    def _populate_artifacts(self) -> None:
        self.artifact_tree.clear()
        card = self.current_card
        grouped: dict[str, list[Path]] = {key: [] for key, _label in EVIDENCE_GROUPS}
        for path in self._feature_artifacts():
            grouped[self._artifact_group(path)].append(path)
        self.image_context_note.setVisible(bool(grouped["images"]))

        if card is None:
            self.evidence_title.setText("Evidence for selected feature")
        else:
            self.evidence_title.setText(f"Evidence for {card.name}")
        evidence_count = sum(len(paths) for paths in grouped.values())
        self.evidence_summary.setText(
            f"{evidence_count} curated sources · measurements, validation, provenance, and visual context"
            if card is not None
            else f"{evidence_count} run-level context files · choose a feature on the left"
        )
        first_file_item: QTreeWidgetItem | None = None
        root = Path(self.results_dir)
        for key, label in EVIDENCE_GROUPS:
            paths = grouped[key]
            if not paths:
                continue
            group_item = QTreeWidgetItem([label, f"{len(paths)} file{'s' if len(paths) != 1 else ''}"])
            group_item.setFlags(group_item.flags() & ~Qt.ItemIsSelectable)
            group_font = group_item.font(0)
            group_font.setBold(True)
            group_item.setFont(0, group_font)
            group_item.setForeground(0, QColor(COLORS["muted"]))
            self.artifact_tree.addTopLevelItem(group_item)
            for path in paths:
                try:
                    relative = path.relative_to(root)
                except ValueError:
                    relative = Path(path.name)
                item = QTreeWidgetItem([str(relative), self._artifact_type(path)])
                item.setData(0, Qt.UserRole, str(path))
                item.setToolTip(0, "Click to preview · double-click to open externally")
                group_item.addChild(item)
                if first_file_item is None:
                    first_file_item = item
            group_item.setExpanded(True)

        if first_file_item is not None:
            self.artifact_tree.setCurrentItem(first_file_item)
        else:
            self._reset_preview("No result artifacts were found in this run folder.")

    def _selected_artifact(self) -> Path | None:
        items = self.artifact_tree.selectedItems()
        if not items:
            return None
        raw_path = items[0].data(0, Qt.UserRole)
        if not isinstance(raw_path, str) or not raw_path:
            return None
        path = Path(raw_path)
        return path if path.is_file() else None

    @staticmethod
    def _format_size(size: int) -> str:
        if size < 1024:
            return f"{size} B"
        if size < 1024 * 1024:
            return f"{size / 1024:.1f} KB"
        return f"{size / (1024 * 1024):.1f} MB"

    def _reset_preview(self, message: str) -> None:
        self._preview_pixmap = QPixmap()
        self.image_preview.clear()
        self.text_preview.clear()
        self.message_preview.setText(message)
        self.preview_title.setText("Preview")
        self.preview_meta.setText("Select a file above.")
        self.preview_stack.setCurrentWidget(self.message_preview)
        self.open_artifact_button.setEnabled(False)
        self.add_to_viewer_button.setEnabled(False)

    def _show_preview_message(self, message: str) -> None:
        self._preview_pixmap = QPixmap()
        self.image_preview.clear()
        self.message_preview.setText(message)
        self.preview_stack.setCurrentWidget(self.message_preview)

    def _scale_preview_image(self) -> None:
        if self._preview_pixmap.isNull():
            return
        target = self.image_preview.size()
        if target.width() > 0 and target.height() > 0:
            self.image_preview.setPixmap(
                self._preview_pixmap.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

    @staticmethod
    def _filtered_csv_preview(path: Path, card: FeatureCard) -> str:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            rows = list(reader)
        if path.name in {"features.csv", "retained_features.csv"} and card.name in fieldnames:
            id_field = next((field for field in ("sample_id", "image_id", "id") if field in fieldnames), fieldnames[0] if fieldnames else "")
            selected_fields = [field for field in (id_field, card.name) if field]
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=selected_fields, lineterminator="\n")
            writer.writeheader()
            for row in rows[:500]:
                writer.writerow({field: row.get(field, "") for field in selected_fields})
            if len(rows) > 500:
                output.write(f"# Preview truncated: showing 500 of {len(rows)} rows.\n")
            return output.getvalue().rstrip()

        matching = [
            row for row in rows
            if any(card.name in str(value) or card.feature_id in str(value) for value in row.values())
        ]
        if not matching:
            return "No record for the selected feature was found in this source file."
        return json.dumps(matching[0] if len(matching) == 1 else matching, ensure_ascii=False, indent=2)

    @staticmethod
    def _prune_json_for_feature(value, card: FeatureCard):
        identifiers = {card.name, card.feature_id}
        if isinstance(value, dict):
            identity_fields = ("name", "feature_name", "actual_column_name", "canonical_name", "feature_id")
            if any(str(value.get(field, "")) in identifiers for field in identity_fields):
                return value
            pruned = {}
            for key, item in value.items():
                if any(identifier and identifier in str(key) for identifier in identifiers):
                    pruned[key] = item
                    continue
                child = EvidencePage._prune_json_for_feature(item, card)
                if child not in (None, {}, []):
                    pruned[key] = child
            return pruned or None
        if isinstance(value, list):
            pruned_items = [EvidencePage._prune_json_for_feature(item, card) for item in value]
            return [item for item in pruned_items if item not in (None, {}, [])] or None
        if isinstance(value, str) and any(identifier and identifier in value for identifier in identifiers):
            return value
        return None

    @staticmethod
    def _code_preview(text: str, card: FeatureCard) -> str:
        marker = re.compile(rf"(?m)^\s*#\s*Feature\s+\d+\s*:\s*{re.escape(card.name)}\s*$")
        match = marker.search(text)
        if match:
            next_marker = re.compile(r"(?m)^\s*#\s*Feature\s+\d+\s*:").search(text, match.end())
            end = next_marker.start() if next_marker else len(text)
            return text[match.start():end].strip()
        lines = [line for line in text.splitlines() if card.name in line or card.feature_id in line]
        return "\n".join(lines) or "No implementation snippet for the selected feature was found in this source file."

    def _feature_specific_text(self, path: Path, text: str) -> str:
        card = self.current_card
        if card is None:
            return text
        if path.name.lower() in {"segmentation_summary.json", "ui_run_manifest.json"}:
            return text
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return self._filtered_csv_preview(path, card)
        if suffix == ".json":
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                return text
            pruned = self._prune_json_for_feature(payload, card)
            if pruned in (None, {}, []):
                return "No record for the selected feature was found in this source file."
            return json.dumps(pruned, ensure_ascii=False, indent=2)
        if suffix == ".py":
            return self._code_preview(text, card)
        matching_lines = [line for line in text.splitlines() if card.name in line or card.feature_id in line]
        return "\n".join(matching_lines) or "No record for the selected feature was found in this source file."

    def _artifact_selected(self) -> None:
        path = self._selected_artifact()
        if path is None:
            self._reset_preview("Select a file to preview it here.")
            return
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        root = Path(self.results_dir)
        try:
            relative = path.relative_to(root)
        except ValueError:
            relative = Path(path.name)
        self.preview_title.setText(path.name)
        self.preview_meta.setText(f"{self._artifact_type(path)} · {self._format_size(size)} · {relative}")
        self.open_artifact_button.setEnabled(True)
        self.add_to_viewer_button.setEnabled(self.viewer is not None and path.suffix.lower() in IMAGE_PREVIEW_SUFFIXES)

        suffix = path.suffix.lower()
        if suffix in IMAGE_PREVIEW_SUFFIXES:
            pixmap = QPixmap(str(path))
            if not pixmap.isNull():
                self._preview_pixmap = pixmap
                self.preview_stack.setCurrentWidget(self.image_preview)
                self._scale_preview_image()
                return
            self._show_preview_message("This image format cannot be previewed by the current Qt installation. Open it externally to inspect it.")
            return

        if suffix in TEXT_PREVIEW_SUFFIXES:
            if size > MAX_TEXT_PREVIEW_BYTES:
                self._show_preview_message("This text file is too large for an in-app preview. Open it externally to inspect the full file.")
                return
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
                text = self._feature_specific_text(path, text)
            except OSError as exc:
                self._show_preview_message(f"Could not read this file: {exc}")
                return
            self._preview_pixmap = QPixmap()
            self.text_preview.setPlainText(text)
            self.text_preview.verticalScrollBar().setValue(0)
            self.text_preview.horizontalScrollBar().setValue(0)
            self.preview_stack.setCurrentWidget(self.text_preview)
            return

        self._show_preview_message("Preview is not available for this file type. Open it externally to inspect it.")

    def _open_artifact(self) -> None:
        path = self._selected_artifact()
        if path:
            opened = QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))
            if not opened:
                self._show_preview_message("The system could not open this file. Its path is shown above so it can be opened manually.")

    def _add_to_viewer(self) -> None:
        path = self._selected_artifact()
        if path and self.viewer is not None:
            try:
                self.viewer.open(str(path))
            except Exception as exc:
                self._show_preview_message(f"Could not add this artifact to napari: {exc}")

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._scale_preview_image()
