"""Low-friction run configuration and preflight validation."""

from __future__ import annotations

import os
from pathlib import Path

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor, QCursor
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QRadioButton,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..models import (
    DatasetSummary,
    RunConfig,
    Severity,
    ValidationIssue,
    diagnose_dataset_selection,
    scan_dataset,
)
from ..environment import read_model_environment, save_model_environment
from ..theme import COLORS
from .common import Card, PageHeader, PathPicker


METHOD_LABELS = {
    "both": "Code + VLM · demo default",
    "code": "Code only",
    "vlm": "VLM only",
}

FREE_API_URL = "https://platform.xiaomimimo.com/"

DATASET_LAYOUT_HELP = (
    "Required input path layout:\n"
    "  <folder you select>/\n"
    "    dataset/\n"
    "      sample_1/\n"
    "        image.tif   (or any .tif / .tiff / .png)\n"
    "      sample_2/\n"
    "        image.tif\n"
    "Select the parent folder that contains `dataset/` (one subfolder per sample)."
)


class ConfigurePage(QWidget):
    run_requested = Signal(object, object)
    configuration_changed = Signal()

    def __init__(self, config: RunConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.config = config
        self.dataset_summary: DatasetSummary | None = None
        self._loading = False

        shell = QVBoxLayout(self)
        shell.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        content = QWidget()
        self.content_layout = QVBoxLayout(content)
        self.content_layout.setContentsMargins(30, 24, 30, 30)
        self.content_layout.setSpacing(18)
        scroll.setWidget(content)
        shell.addWidget(scroll)

        self.content_layout.addWidget(
            PageHeader(
                "Setup",
                "Set up your run",
                "Choose the data, ask a biological question, and select how MorphAgent should measure it.",
            )
        )

        self._build_data_section()
        self._build_question_section()
        self._build_model_api_section()
        self._build_analysis_section()
        self._build_ready_section()
        self.content_layout.addStretch(1)

        # The exact CLI remains available to tests/manifests without adding a
        # dense technical panel to the first-run path.
        self.command_preview = QPlainTextEdit(self)
        self.command_preview.setReadOnly(True)
        self.command_preview.hide()

        self._connect_signals()
        self.load_from_config()
        self.load_api_settings()
        self.refresh_preflight(scan=False)

    @staticmethod
    def _mark_step(card: Card) -> None:
        card.setProperty("stepCard", True)

    @staticmethod
    def _section_title(text: str) -> QLabel:
        label = QLabel(text)
        label.setProperty("role", "title")
        return label

    def _build_data_section(self) -> None:
        card = Card()
        self._mark_step(card)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(22, 18, 22, 20)
        layout.setSpacing(12)
        layout.addWidget(self._section_title("1 · Data"))

        path_title = QLabel("Input data path")
        path_title.setProperty("role", "fieldLabel")
        layout.addWidget(path_title)

        layout_help = QLabel(DATASET_LAYOUT_HELP)
        layout_help.setProperty("role", "muted")
        layout_help.setWordWrap(True)
        layout_help.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(layout_help)

        help_text = QLabel(
            "Load the demo dataset, or browse to your own folder that matches the layout above."
        )
        help_text.setProperty("role", "muted")
        help_text.setWordWrap(True)
        layout.addWidget(help_text)

        demo_row = QHBoxLayout()
        demo_row.setSpacing(10)
        self.demo_guide = QLabel("👉")
        self.demo_guide.setToolTip("Start here for a ready-to-run dataset")
        self.demo_guide.setStyleSheet("font-size: 22px;")
        self.load_demo_button = QPushButton("Load demo dataset")
        self.load_demo_button.setProperty("choiceAction", True)
        self.load_demo_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.load_demo_button.setToolTip(
            "Load the demo samples, biological question, existing masks, and prepared knowledge sources."
        )
        demo_row.addWidget(self.demo_guide, 0, Qt.AlignVCenter)
        demo_row.addWidget(self.load_demo_button, 1)
        layout.addLayout(demo_row)

        own_data = QLabel("OR CHOOSE YOUR DATASET")
        own_data.setProperty("role", "eyebrowMuted")
        layout.addWidget(own_data)
        self.dataset_picker = PathPicker("Project or dataset folder")
        layout.addWidget(self.dataset_picker)

        self.dataset_note = QLabel("No dataset selected yet.")
        self.dataset_note.setProperty("role", "muted")
        self.dataset_note.setWordWrap(True)
        layout.addWidget(self.dataset_note)
        self.content_layout.addWidget(card)

    def _build_question_section(self) -> None:
        card = Card()
        self._mark_step(card)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(22, 18, 22, 20)
        layout.setSpacing(11)
        layout.addWidget(self._section_title("2 · Biological question"))
        tip = QLabel("What do you want to quantify or compare in these images?")
        tip.setProperty("role", "muted")
        layout.addWidget(tip)
        self.query_edit = QTextEdit()
        self.query_edit.setPlaceholderText("Example: Quantify Tau aggregation and neuronal structure in these images.")
        self.query_edit.setMinimumHeight(84)
        self.query_edit.setMaximumHeight(100)
        self.query_edit.setAccessibleName("Biological question")
        layout.addWidget(self.query_edit)
        self.content_layout.addWidget(card)

    def _make_choice_row(self, choices: dict[str, str]) -> tuple[QButtonGroup, dict[str, QRadioButton], QHBoxLayout]:
        group = QButtonGroup(self)
        group.setExclusive(True)
        buttons: dict[str, QRadioButton] = {}
        row = QHBoxLayout()
        row.setSpacing(8)
        for value, label in choices.items():
            button = QRadioButton(label)
            button.setProperty("choiceTile", True)
            button.setProperty("choiceValue", value)
            button.setMinimumHeight(42)
            group.addButton(button)
            buttons[value] = button
            row.addWidget(button, 1)
        return group, buttons, row

    def _build_analysis_section(self) -> None:
        card = Card()
        self._mark_step(card)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(22, 18, 22, 22)
        layout.setSpacing(13)
        layout.addWidget(self._section_title("4 · Analysis"))

        self.scale_summary = QLabel("Demo scale · 1 round × 5 candidates · target 5")
        self.scale_summary.setProperty("role", "scaleSummary")
        layout.addWidget(self.scale_summary)

        route_label = QLabel("Analysis route · choose one")
        route_label.setProperty("role", "fieldLabel")
        layout.addWidget(route_label)
        self.method_group, self.method_buttons, method_row = self._make_choice_row(METHOD_LABELS)
        layout.addLayout(method_row)

        # Mask preparation is internal: reuse existing masks when present,
        # otherwise Allen runs automatically. Compatibility aliases for tests.
        self.mask_group = QButtonGroup(self)
        self.mask_buttons: dict[str, QRadioButton] = {}

        knowledge_label = QLabel("Knowledge sources · multiple choice")
        knowledge_label.setProperty("role", "fieldLabel")
        layout.addWidget(knowledge_label)
        knowledge = QHBoxLayout()
        knowledge.setSpacing(14)
        self.expert_check = QCheckBox("Expert notes")
        self.deep_check = QCheckBox("Deep research")
        self.rag_check = QCheckBox("Literature / RAG")
        for checkbox in (self.expert_check, self.deep_check, self.rag_check):
            checkbox.setChecked(True)
            checkbox.setProperty("choiceTile", True)
            checkbox.setMinimumHeight(46)
            knowledge.addWidget(checkbox, 1)
        layout.addLayout(knowledge)

        config_row = QHBoxLayout()
        self.advanced_toggle = QPushButton("Config")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setChecked(False)
        self.advanced_toggle.setProperty("choiceAction", True)
        self.advanced_toggle.setCursor(QCursor(Qt.PointingHandCursor))
        self.advanced_toggle.setToolTip("Show advanced analysis parameters")
        config_row.addWidget(self.advanced_toggle, 0, Qt.AlignLeft)
        config_row.addStretch(1)
        layout.addLayout(config_row)

        self.advanced_panel = QWidget()
        advanced = QFormLayout(self.advanced_panel)
        advanced.setContentsMargins(0, 4, 0, 0)
        advanced.setHorizontalSpacing(16)
        advanced.setVerticalSpacing(8)
        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.0, 2.0)
        self.temperature_spin.setSingleStep(0.1)
        self.temperature_spin.setDecimals(2)
        self.rounds_spin = QSpinBox()
        self.rounds_spin.setRange(1, 20)
        self.candidates_spin = QSpinBox()
        self.candidates_spin.setRange(1, 50)
        self.target_spin = QSpinBox()
        self.target_spin.setRange(1, 500)
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 32)
        self.vlm_concurrency_spin = QSpinBox()
        self.vlm_concurrency_spin.setRange(1, 32)
        advanced.addRow("Temperature", self.temperature_spin)
        advanced.addRow("Rounds", self.rounds_spin)
        advanced.addRow("Candidates per round", self.candidates_spin)
        advanced.addRow("Target feature count", self.target_spin)
        advanced.addRow("Code workers", self.workers_spin)
        advanced.addRow("VLM concurrency", self.vlm_concurrency_spin)
        self.advanced_panel.hide()
        layout.addWidget(self.advanced_panel)

        self.content_layout.addWidget(card)

    def _build_model_api_section(self) -> None:
        card = Card()
        self._mark_step(card)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(22, 18, 22, 22)
        layout.setSpacing(12)
        layout.addWidget(self._section_title("3 · Model API"))

        help_text = QLabel(
            "Enter your OpenAI-compatible endpoint. Credentials are applied automatically when you click Run — "
            "no separate Save step. Keys stay in the local .env and are never written to manifests or logs."
        )
        help_text.setProperty("role", "muted")
        help_text.setWordWrap(True)
        layout.addWidget(help_text)

        free_api = QLabel(
            f'<a href="{FREE_API_URL}">Free OpenAI-compatible APIs available here</a>'
        )
        free_api.setOpenExternalLinks(True)
        free_api.setTextInteractionFlags(Qt.TextBrowserInteraction)
        free_api.setProperty("role", "muted")
        layout.addWidget(free_api)

        self.llm_form = QFormLayout()
        self.llm_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.llm_form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.llm_form.setHorizontalSpacing(16)
        self.llm_form.setVerticalSpacing(9)
        self.llm_base_url_edit = QLineEdit()
        self.llm_base_url_edit.setPlaceholderText("Base URL · e.g. https://api.openai.com/v1")
        self.llm_api_key_edit = QLineEdit()
        self.llm_api_key_edit.setEchoMode(QLineEdit.Password)
        self.llm_api_key_edit.setPlaceholderText("API key")
        self.llm_model_edit = QLineEdit()
        self.llm_model_edit.setPlaceholderText("Model name")
        self.llm_form.addRow("Base URL", self.llm_base_url_edit)
        self.llm_form.addRow("API key", self.llm_api_key_edit)
        self.llm_form.addRow("Model", self.llm_model_edit)
        layout.addLayout(self.llm_form)

        self.vlm_connection_fields = QWidget()
        self.vlm_form = QFormLayout(self.vlm_connection_fields)
        self.vlm_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.vlm_form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.vlm_form.setContentsMargins(0, 2, 0, 0)
        self.vlm_form.setHorizontalSpacing(16)
        self.vlm_form.setVerticalSpacing(9)
        self.vlm_base_url_edit = QLineEdit()
        self.vlm_base_url_edit.setPlaceholderText("VLM Base URL (optional if same as above)")
        self.vlm_api_key_edit = QLineEdit()
        self.vlm_api_key_edit.setEchoMode(QLineEdit.Password)
        self.vlm_api_key_edit.setPlaceholderText("VLM API key")
        self.vlm_model_edit = QLineEdit()
        self.vlm_model_edit.setPlaceholderText("Multimodal model name")
        self.vlm_form.addRow("VLM Base URL", self.vlm_base_url_edit)
        self.vlm_form.addRow("VLM API key", self.vlm_api_key_edit)
        self.vlm_form.addRow("VLM model", self.vlm_model_edit)
        layout.addWidget(self.vlm_connection_fields)

        self.reuse_llm_for_vlm = QCheckBox("Use the same connection for image scoring")
        self.reuse_llm_for_vlm.setChecked(False)
        self.reuse_llm_for_vlm.setProperty("choiceTile", True)
        self.reuse_llm_for_vlm.setMinimumHeight(46)
        layout.addWidget(self.reuse_llm_for_vlm)

        self.api_status_label = QLabel("Fill the fields above · applied automatically on Run")
        self.api_status_label.setProperty("role", "muted")
        self.api_status_label.setWordWrap(True)
        # Kept for tests that previously clicked Save; no longer shown.
        self.save_api_button = QPushButton("Save API configuration")
        self.save_api_button.hide()
        layout.addWidget(self.api_status_label)
        self.content_layout.addWidget(card)

    def _build_ready_section(self) -> None:
        self.readiness_list = QListWidget(self)
        self.readiness_list.setObjectName("Readiness")
        self.readiness_list.hide()
        self.blocker_label = QLabel(self)
        self.blocker_label.setWordWrap(True)
        self.blocker_label.hide()

        action_bar = QWidget()
        action = QHBoxLayout(action_bar)
        action.setContentsMargins(0, 2, 0, 10)
        self.run_button = QPushButton("Run MorphAgent")
        self.run_button.setProperty("primary", True)
        self.run_button.setProperty("largePrimary", True)
        self.run_button.setProperty("runCta", True)
        self.run_button.setMinimumSize(260, 54)
        self.run_button.setAccessibleName("Run MorphAgent")
        action.addStretch(1)
        action.addWidget(self.run_button)
        self.content_layout.addWidget(action_bar)

    def _connect_signals(self) -> None:
        self.load_demo_button.clicked.connect(self._load_reference_demo)
        self.dataset_picker.path_changed.connect(self._dataset_path_changed)
        self.query_edit.textChanged.connect(self._fields_changed)
        for value, button in self.method_buttons.items():
            button.toggled.connect(lambda checked, selected=value: self._method_changed(selected) if checked else None)
        for widget in (
            self.expert_check,
            self.deep_check,
            self.rag_check,
        ):
            widget.toggled.connect(self._fields_changed)
        self.advanced_toggle.toggled.connect(self._toggle_advanced)
        for spin in (
            self.temperature_spin,
            self.rounds_spin,
            self.candidates_spin,
            self.target_spin,
            self.workers_spin,
            self.vlm_concurrency_spin,
        ):
            spin.valueChanged.connect(self._fields_changed)
        self.run_button.clicked.connect(self._request_run)
        self.reuse_llm_for_vlm.toggled.connect(self._toggle_vlm_fields)
        for edit in (
            self.llm_base_url_edit,
            self.llm_api_key_edit,
            self.llm_model_edit,
            self.vlm_base_url_edit,
            self.vlm_api_key_edit,
            self.vlm_model_edit,
        ):
            edit.textChanged.connect(self._fields_changed)
        self.reuse_llm_for_vlm.toggled.connect(self._fields_changed)
        self.save_api_button.clicked.connect(self._persist_api_settings)

    def _toggle_advanced(self, checked: bool) -> None:
        self.advanced_panel.setVisible(checked)
        self.advanced_toggle.setText("Hide config" if checked else "Config")

    def _toggle_vlm_fields(self, reuse: bool) -> None:
        self.vlm_connection_fields.setVisible(not reuse)

    def load_api_settings(self) -> None:
        values = read_model_environment(self.config.repository_root)
        for name, value in values.items():
            if value:
                os.environ[name] = value

        # Always open with blank fields — never prefill Base URL / Model from .env.
        llm_key = values.get("LLM_API_KEY", "").strip()
        vlm_key = values.get("VLM_API_KEY", "").strip()
        llm_ready = bool(
            llm_key
            and values.get("LLM_BASE_URL", "").strip()
            and values.get("LLM_MODEL", "").strip()
        )

        self.llm_base_url_edit.clear()
        self.llm_model_edit.clear()
        self.llm_api_key_edit.clear()
        self.llm_api_key_edit.setPlaceholderText(
            "API key already on file · leave blank to keep it" if llm_key else "API key"
        )

        self.reuse_llm_for_vlm.setChecked(False)
        self.vlm_base_url_edit.clear()
        self.vlm_model_edit.clear()
        self.vlm_api_key_edit.clear()
        self.vlm_api_key_edit.setPlaceholderText(
            "VLM API key already on file · leave blank to keep it" if vlm_key else "VLM API key"
        )
        self._toggle_vlm_fields(False)
        if llm_ready:
            self.api_status_label.setText(
                "Credentials on file · leave fields blank to reuse them on Run, or type new values"
            )
            self.api_status_label.setProperty("role", "success")
        else:
            self.api_status_label.setText("Fill the fields above · applied automatically on Run")
            self.api_status_label.setProperty("role", "muted")
        self.api_status_label.style().unpolish(self.api_status_label)
        self.api_status_label.style().polish(self.api_status_label)

    def _persist_api_settings(self) -> bool:
        """Write form credentials to .env / environ. Called automatically on Run."""

        current = read_model_environment(self.config.repository_root)
        llm_base = self.llm_base_url_edit.text().strip() or current.get("LLM_BASE_URL", "").strip()
        llm_model = self.llm_model_edit.text().strip() or current.get("LLM_MODEL", "").strip()
        llm_key = self.llm_api_key_edit.text().strip() or current.get("LLM_API_KEY", "").strip()
        if not llm_base or not llm_model or not llm_key:
            self.api_status_label.setText("Base URL, model, and API key are required before running.")
            self.api_status_label.setProperty("role", "warning")
            self.api_status_label.style().unpolish(self.api_status_label)
            self.api_status_label.style().polish(self.api_status_label)
            return False

        values = {
            "LLM_BASE_URL": llm_base,
            "LLM_API_KEY": llm_key,
            "LLM_MODEL": llm_model,
        }
        reuse = self.reuse_llm_for_vlm.isChecked()
        vlm_base = self.vlm_base_url_edit.text().strip() or current.get("VLM_BASE_URL", "").strip()
        vlm_model = self.vlm_model_edit.text().strip() or current.get("VLM_MODEL", "").strip()
        vlm_key = self.vlm_api_key_edit.text().strip() or current.get("VLM_API_KEY", "").strip()
        # Comfort: empty VLM fields reuse the LLM connection automatically.
        if reuse or not (vlm_base and vlm_model and vlm_key):
            values.update({
                "VLM_BASE_URL": llm_base,
                "VLM_API_KEY": llm_key,
                "VLM_MODEL": llm_model,
            })
        else:
            values.update({
                "VLM_BASE_URL": vlm_base,
                "VLM_API_KEY": vlm_key,
                "VLM_MODEL": vlm_model,
            })

        save_model_environment(self.config.repository_root, values)
        self.llm_api_key_edit.clear()
        self.vlm_api_key_edit.clear()
        self.llm_api_key_edit.setPlaceholderText("API key already on file · leave blank to keep it")
        self.vlm_api_key_edit.setPlaceholderText("VLM API key already on file · leave blank to keep it")
        self.api_status_label.setText("Credentials applied · ready to run")
        self.api_status_label.setProperty("role", "success")
        self.api_status_label.style().unpolish(self.api_status_label)
        self.api_status_label.style().polish(self.api_status_label)
        self.refresh_preflight(scan=False)
        self.configuration_changed.emit()
        return True

    def _save_api_settings(self) -> None:
        """Backward-compatible alias used by older tests."""
        self._persist_api_settings()

    def _load_reference_demo(self) -> None:
        try:
            self.config.apply_reference_demo()
        except (OSError, ValueError, ImportError) as exc:
            QMessageBox.critical(self, "Demo dataset unavailable", str(exc))
            return
        self.load_from_config()
        self.dataset_summary = scan_dataset(self.config.data_root)
        self.refresh_preflight(scan=False)
        self.configuration_changed.emit()

    def load_from_config(self) -> None:
        self._loading = True
        self.dataset_picker.setText(self.config.data_root)
        self.query_edit.setPlainText(self.config.query)
        self.method_buttons.get(self.config.method, self.method_buttons["both"]).setChecked(True)
        self.config.enable_segmentation = True
        self.config.segmentation_skip_if_present = True
        self.expert_check.setChecked(self.config.enable_expert_knowledge)
        self.deep_check.setChecked(self.config.enable_deep_research)
        self.rag_check.setChecked(self.config.enable_rag)
        self.temperature_spin.setValue(float(self.config.temperature))
        self.config.reproduce = float(self.config.temperature) <= 0.0
        self.rounds_spin.setValue(int(self.config.num_rounds))
        self.candidates_spin.setValue(int(self.config.features_per_iteration))
        self.target_spin.setValue(int(self.config.target_feature_count))
        self.workers_spin.setValue(int(self.config.code_parallel_workers))
        self.vlm_concurrency_spin.setValue(int(self.config.vlm_online_concurrency))
        self._refresh_scale_summary()
        self._loading = False

    def _refresh_scale_summary(self) -> None:
        source = "Demo" if self.config.dataset_source == "demo" else "Custom"
        self.scale_summary.setText(
            f"{source} scale · {self.config.num_rounds} round"
            f"{'s' if self.config.num_rounds != 1 else ''} × "
            f"{self.config.features_per_iteration} candidates · target {self.config.target_feature_count}"
        )

    def _dataset_path_changed(self, value: str = "") -> None:
        if self._loading:
            return
        path = Path(value).expanduser() if value.strip() else None
        problem = diagnose_dataset_selection(path)
        if problem:
            self.dataset_summary = None
            self.config.description_path = ""
            self.config.metadata_path = ""
            self.config.dataset_source = "custom"
            self.dataset_note.setText("Path not usable — see the dialog for the required layout.")
            self.dataset_note.setProperty("role", "warning")
            self.dataset_note.style().unpolish(self.dataset_note)
            self.dataset_note.style().polish(self.dataset_note)
            QMessageBox.warning(self, "Dataset path not usable", problem)
            self._fields_changed()
            return

        if path is not None and path.is_dir():
            self.dataset_summary = scan_dataset(path)
            self._detect_dataset_context(path)
            demo_root = Path(self.config.repository_root).expanduser().resolve() / "demo" / "data"
            try:
                self.config.dataset_source = "demo" if path.resolve() == demo_root else "custom"
            except OSError:
                self.config.dataset_source = "custom"
        else:
            self.dataset_summary = None
            self.config.description_path = ""
            self.config.metadata_path = ""
            self.config.dataset_source = "custom"
        self._fields_changed()

    def _detect_dataset_context(self, root: Path) -> None:
        """Use conventional optional files without adding more form fields."""

        description_candidates = (
            root / "dataset" / "dataset_index.txt",
            root / "dataset_index.txt",
        )
        metadata_candidates = (
            root / "metadata.csv",
            root / "dataset" / "metadata.csv",
        )
        self.config.description_path = next(
            (str(candidate.resolve()) for candidate in description_candidates if candidate.is_file()),
            "",
        )
        self.config.metadata_path = next(
            (str(candidate.resolve()) for candidate in metadata_candidates if candidate.is_file()),
            "",
        )

    def _method_changed(self, _method: str) -> None:
        if not self._loading:
            self._fields_changed()

    def _selected_method(self) -> str:
        for value, button in self.method_buttons.items():
            if button.isChecked():
                return value
        return "both"

    def _sync_config(self) -> None:
        current = read_model_environment(self.config.repository_root)
        self.config.data_root = self.dataset_picker.text()
        self.config.query = self.query_edit.toPlainText().strip()
        self.config.method = self._selected_method()
        self.config.enable_segmentation = True
        self.config.segmentation_skip_if_present = True
        self.config.enable_expert_knowledge = self.expert_check.isChecked()
        self.config.enable_deep_research = self.deep_check.isChecked()
        self.config.enable_rag = self.rag_check.isChecked()
        self.config.temperature = float(self.temperature_spin.value())
        # Temperature 0 ⇒ reproducible Code + VLM (seed, deterministic decoding, cache).
        self.config.reproduce = self.config.temperature <= 0.0
        self.config.num_rounds = int(self.rounds_spin.value())
        self.config.features_per_iteration = int(self.candidates_spin.value())
        self.config.target_feature_count = int(self.target_spin.value())
        workers = int(self.workers_spin.value())
        vlm_conc = int(self.vlm_concurrency_spin.value())
        if self.config.reproduce:
            workers = 1
            vlm_conc = 1
        self.config.code_parallel_workers = workers
        self.config.vlm_online_concurrency = vlm_conc
        self.config.llm_base_url = self.llm_base_url_edit.text().strip()
        self.config.llm_api_key = self.llm_api_key_edit.text().strip() or current.get("LLM_API_KEY", "").strip()
        self.config.llm_model = self.llm_model_edit.text().strip() or current.get("LLM_MODEL", "").strip()
        self.config.reuse_llm_for_vlm = self.reuse_llm_for_vlm.isChecked()
        if self.config.reuse_llm_for_vlm:
            self.config.vlm_base_url = self.config.llm_base_url
            self.config.vlm_api_key = self.config.llm_api_key
            self.config.vlm_online_model = self.config.llm_model
        else:
            self.config.vlm_base_url = self.vlm_base_url_edit.text().strip()
            self.config.vlm_api_key = (
                self.vlm_api_key_edit.text().strip() or current.get("VLM_API_KEY", "").strip()
            )
            self.config.vlm_online_model = (
                self.vlm_model_edit.text().strip() or current.get("VLM_MODEL", "").strip()
            )
            # Empty VLM → treat as same connection for preflight comfort.
            if not (self.config.vlm_base_url and self.config.vlm_online_model and self.config.vlm_api_key):
                self.config.vlm_base_url = self.config.llm_base_url or current.get("LLM_BASE_URL", "").strip()
                self.config.vlm_api_key = self.config.llm_api_key
                self.config.vlm_online_model = self.config.llm_model
        if not self.config.llm_base_url:
            self.config.llm_base_url = current.get("LLM_BASE_URL", "").strip()
        self._refresh_scale_summary()

    def _fields_changed(self, *_args) -> None:
        if self._loading:
            return
        self._sync_config()
        self.refresh_preflight(scan=False)
        self.configuration_changed.emit()

    def refresh_preflight(self, scan: bool = False) -> list[ValidationIssue]:
        self._sync_config()
        if scan and self.config.data_root:
            self.dataset_summary = scan_dataset(self.config.data_root)
        self._update_dataset_summary()
        issues = self.config.validate(self.dataset_summary)
        self._render_issues(issues)
        if self.config.data_root and self.config.query:
            self.command_preview.setPlainText(self.config.command_preview())
        else:
            self.command_preview.clear()
        blockers = [issue for issue in issues if issue.severity is Severity.BLOCKER]
        usable_dataset = (
            self.dataset_summary is not None
            and self.dataset_summary.sample_count > 0
            and len(self.dataset_summary.empty_samples) < self.dataset_summary.sample_count
        )
        self.run_button.setEnabled(not blockers and usable_dataset)
        if blockers:
            self.blocker_label.setText(f"Complete {len(blockers)} required item{'s' if len(blockers) != 1 else ''}")
            self.blocker_label.setProperty("role", "error")
            details = "\n".join(f"• {issue.message}" for issue in blockers)
            self.run_button.setToolTip(f"Complete the required setup before running:\n{details}")
            self.run_button.setAccessibleDescription(details)
        elif not usable_dataset:
            self.blocker_label.setText("Choose a usable dataset folder")
            self.blocker_label.setProperty("role", "warning")
            self.run_button.setToolTip("Choose a folder that contains dataset/<sample>/*.tif")
            self.run_button.setAccessibleDescription("Choose a usable dataset folder")
        else:
            self.blocker_label.setText("Everything required is ready")
            self.blocker_label.setProperty("role", "success")
            self.run_button.setToolTip("Start the configured MorphAgent workflow")
            self.run_button.setAccessibleDescription("All required inputs are ready")
        self.blocker_label.style().unpolish(self.blocker_label)
        self.blocker_label.style().polish(self.blocker_label)
        return issues

    def _update_dataset_summary(self) -> None:
        summary = self.dataset_summary
        if summary is None:
            self.dataset_note.setText("No dataset selected yet.")
            self.dataset_note.setProperty("role", "muted")
        else:
            parts = [f"{summary.sample_count} samples", f"{summary.primary_image_count} primary images"]
            if summary.mask_count:
                parts.append(f"{summary.mask_count} masks")
            if summary.empty_samples:
                parts.append(f"{len(summary.empty_samples)} samples need attention")
                role = "warning"
            elif summary.sample_count < 5:
                parts.append("recommend ≥5 samples")
                role = "warning"
            else:
                role = "success"
            self.dataset_note.setText("Ready · " + " · ".join(parts))
            self.dataset_note.setProperty("role", role)
        self.dataset_note.style().unpolish(self.dataset_note)
        self.dataset_note.style().polish(self.dataset_note)

    def _render_issues(self, issues: list[ValidationIssue]) -> None:
        self.readiness_list.clear()
        blockers = [issue for issue in issues if issue.severity is Severity.BLOCKER]
        if not blockers:
            item = QListWidgetItem("READY  Required inputs are complete")
            item.setForeground(QColor(COLORS["success"]))
            self.readiness_list.addItem(item)
            return
        for issue in blockers:
            text = issue.message
            if issue.recovery:
                text += f"  {issue.recovery}"
            item = QListWidgetItem(text)
            item.setForeground(QColor(COLORS["error"]))
            item.setToolTip(issue.code)
            self.readiness_list.addItem(item)

    def _request_run(self) -> None:
        if not self._persist_api_settings():
            QMessageBox.warning(
                self,
                "Model API incomplete",
                "Fill Base URL, API key, and Model. Credentials are applied automatically when you click Run.",
            )
            self.refresh_preflight(scan=False)
            return
        problem = diagnose_dataset_selection(self.dataset_picker.text())
        if problem:
            QMessageBox.warning(self, "Dataset path not usable", problem)
            return
        issues = self.refresh_preflight(scan=True)
        if any(issue.severity is Severity.BLOCKER for issue in issues):
            return
        self.run_requested.emit(self.config, self.dataset_summary)
