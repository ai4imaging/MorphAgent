"""Low-friction run configuration and preflight validation."""

from __future__ import annotations

import os
from pathlib import Path

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
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
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..models import DatasetSummary, RunConfig, Severity, ValidationIssue, scan_dataset
from ..environment import read_model_environment, save_model_environment
from ..theme import COLORS
from .common import Card, PageHeader, PathPicker


METHOD_LABELS = {
    "both": "Code + VLM · demo default",
    "code": "Code only",
    "vlm": "VLM only",
}

MASK_PREPARATION_LABELS = {
    "reuse": "Reuse existing masks",
    "recreate": "Regenerate Cellpose masks",
}


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

        help_text = QLabel("Start with the bundled Tau dataset, or choose a folder containing one subfolder per sample.")
        help_text.setProperty("role", "muted")
        help_text.setWordWrap(True)
        layout.addWidget(help_text)

        self.load_demo_button = QPushButton("Use bundled Tau demo")
        self.load_demo_button.setProperty("choiceAction", True)
        self.load_demo_button.setToolTip("Load the five teacher-demo samples, question, masks, and knowledge sources.")
        layout.addWidget(self.load_demo_button, 0)

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

        scale_summary = QLabel("Teacher demo scale · 2 rounds × 5 candidates · target 10")
        scale_summary.setProperty("role", "scaleSummary")
        layout.addWidget(scale_summary)

        route_label = QLabel("Analysis route · choose one")
        route_label.setProperty("role", "fieldLabel")
        layout.addWidget(route_label)
        self.method_group, self.method_buttons, method_row = self._make_choice_row(METHOD_LABELS)
        layout.addLayout(method_row)

        preparation_label = QLabel("Mask preparation · choose one")
        preparation_label.setProperty("role", "fieldLabel")
        layout.addWidget(preparation_label)
        self.mask_group, self.mask_buttons, mask_row = self._make_choice_row(MASK_PREPARATION_LABELS)
        self.mask_buttons["reuse"].setToolTip(
            "Keep masks already present and run Cellpose only for samples without masks."
        )
        self.mask_buttons["recreate"].setToolTip(
            "Run Cellpose for every sample and overwrite cyto.tif, nuclei.tif, and cytoplasm.tif."
        )
        layout.addLayout(mask_row)

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
        self.content_layout.addWidget(card)

    def _build_model_api_section(self) -> None:
        card = Card()
        self._mark_step(card)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(22, 18, 22, 22)
        layout.setSpacing(12)
        layout.addWidget(self._section_title("3 · Model API"))

        help_text = QLabel(
            "Saved only in the repository .env file. The key is masked and never added to commands, manifests, or logs."
        )
        help_text.setProperty("role", "muted")
        help_text.setWordWrap(True)
        layout.addWidget(help_text)

        self.llm_form = QFormLayout()
        self.llm_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.llm_form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.llm_form.setHorizontalSpacing(16)
        self.llm_form.setVerticalSpacing(9)
        self.llm_base_url_edit = QLineEdit()
        self.llm_base_url_edit.setPlaceholderText("https://api.example.com/v1")
        self.llm_api_key_edit = QLineEdit()
        self.llm_api_key_edit.setEchoMode(QLineEdit.Password)
        self.llm_model_edit = QLineEdit()
        self.llm_model_edit.setPlaceholderText("Model name")
        self.llm_form.addRow("Base URL", self.llm_base_url_edit)
        self.llm_form.addRow("API key", self.llm_api_key_edit)
        self.llm_form.addRow("Model", self.llm_model_edit)
        layout.addLayout(self.llm_form)

        self.reuse_llm_for_vlm = QCheckBox("Use the same connection for image scoring")
        self.reuse_llm_for_vlm.setChecked(True)
        self.reuse_llm_for_vlm.setProperty("choiceTile", True)
        self.reuse_llm_for_vlm.setMinimumHeight(46)
        layout.addWidget(self.reuse_llm_for_vlm)

        self.vlm_connection_fields = QWidget()
        self.vlm_form = QFormLayout(self.vlm_connection_fields)
        self.vlm_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.vlm_form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.vlm_form.setContentsMargins(0, 2, 0, 0)
        self.vlm_form.setHorizontalSpacing(16)
        self.vlm_form.setVerticalSpacing(9)
        self.vlm_base_url_edit = QLineEdit()
        self.vlm_base_url_edit.setPlaceholderText("Multimodal API base URL")
        self.vlm_api_key_edit = QLineEdit()
        self.vlm_api_key_edit.setEchoMode(QLineEdit.Password)
        self.vlm_model_edit = QLineEdit()
        self.vlm_model_edit.setPlaceholderText("Multimodal model name")
        self.vlm_form.addRow("VLM Base URL", self.vlm_base_url_edit)
        self.vlm_form.addRow("VLM API key", self.vlm_api_key_edit)
        self.vlm_form.addRow("VLM model", self.vlm_model_edit)
        layout.addWidget(self.vlm_connection_fields)

        save_row = QHBoxLayout()
        self.api_status_label = QLabel("Model API configuration has not been checked yet.")
        self.api_status_label.setProperty("role", "muted")
        self.api_status_label.setWordWrap(True)
        self.save_api_button = QPushButton("Save API configuration")
        self.save_api_button.setProperty("choiceAction", True)
        save_row.addWidget(self.api_status_label, 1)
        save_row.addWidget(self.save_api_button)
        layout.addLayout(save_row)
        self.content_layout.addWidget(card)

    def _build_ready_section(self) -> None:
        # Validation remains active, but the first-run surface ends with one
        # unmistakable action instead of another large explanatory panel.
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
        for button in self.mask_buttons.values():
            button.toggled.connect(self._fields_changed)
        for widget in (
            self.expert_check,
            self.deep_check,
            self.rag_check,
        ):
            widget.toggled.connect(self._fields_changed)
        self.run_button.clicked.connect(self._request_run)
        self.reuse_llm_for_vlm.toggled.connect(self._toggle_vlm_fields)
        self.save_api_button.clicked.connect(self._save_api_settings)

    def _toggle_vlm_fields(self, reuse: bool) -> None:
        self.vlm_connection_fields.setVisible(not reuse)

    def load_api_settings(self) -> None:
        values = read_model_environment(self.config.repository_root)
        for name, value in values.items():
            if value:
                os.environ[name] = value

        llm_base = values.get("LLM_BASE_URL", "") or "https://api.openai.com/v1"
        llm_model = values.get("LLM_MODEL", "") or "gpt-4o"
        vlm_base = values.get("VLM_BASE_URL", "")
        vlm_model = values.get("VLM_MODEL", "")
        llm_key = values.get("LLM_API_KEY", "").strip()
        vlm_key = values.get("VLM_API_KEY", "").strip()
        llm_key_ready = bool(llm_key)
        vlm_key_ready = bool(vlm_key)

        self.llm_base_url_edit.setText(llm_base)
        self.llm_model_edit.setText(llm_model)
        self.llm_api_key_edit.clear()
        self.llm_api_key_edit.setPlaceholderText(
            "API key already saved · leave blank to keep it" if llm_key_ready else "Enter API key"
        )

        reuse = (
            not vlm_base
            or (
                vlm_base == llm_base
                and (not vlm_model or vlm_model == llm_model)
                and (not vlm_key or vlm_key == llm_key)
            )
        )
        self.reuse_llm_for_vlm.setChecked(reuse)
        self.vlm_base_url_edit.setText(vlm_base or llm_base)
        self.vlm_model_edit.setText(vlm_model or llm_model)
        self.vlm_api_key_edit.clear()
        self.vlm_api_key_edit.setPlaceholderText(
            "VLM API key already saved · leave blank to keep it" if vlm_key_ready else "Enter VLM API key"
        )
        self._toggle_vlm_fields(reuse)
        self._set_api_status(llm_key_ready, vlm_key_ready or (reuse and llm_key_ready))

    def _set_api_status(self, llm_ready: bool, vlm_ready: bool, saved: bool = False) -> None:
        if llm_ready and vlm_ready:
            text = "Saved to .env · planning and image scoring are ready" if saved else "Loaded from .env · planning and image scoring are ready"
            role = "success"
        elif llm_ready:
            text = "Planning API is ready · image-scoring API still needs a key"
            role = "warning"
        else:
            text = "Enter an API key, then save this configuration"
            role = "warning"
        self.api_status_label.setText(text)
        self.api_status_label.setProperty("role", role)
        self.api_status_label.style().unpolish(self.api_status_label)
        self.api_status_label.style().polish(self.api_status_label)

    def _save_api_settings(self) -> None:
        current = read_model_environment(self.config.repository_root)
        llm_base = self.llm_base_url_edit.text().strip()
        llm_model = self.llm_model_edit.text().strip()
        llm_key = self.llm_api_key_edit.text().strip() or current.get("LLM_API_KEY", "").strip()
        if not llm_base or not llm_model or not llm_key:
            self.api_status_label.setText("Base URL, model, and API key are required before saving.")
            self.api_status_label.setProperty("role", "warning")
            return

        values = {
            "LLM_BASE_URL": llm_base,
            "LLM_API_KEY": llm_key,
            "LLM_MODEL": llm_model,
        }
        if self.reuse_llm_for_vlm.isChecked():
            values.update({
                "VLM_BASE_URL": llm_base,
                "VLM_API_KEY": llm_key,
                "VLM_MODEL": llm_model,
            })
            vlm_ready = True
        else:
            vlm_base = self.vlm_base_url_edit.text().strip()
            vlm_model = self.vlm_model_edit.text().strip()
            vlm_key = self.vlm_api_key_edit.text().strip() or current.get("VLM_API_KEY", "").strip()
            if not vlm_base or not vlm_model or not vlm_key:
                self.api_status_label.setText("Complete the separate VLM Base URL, model, and API key before saving.")
                self.api_status_label.setProperty("role", "warning")
                return
            values.update({
                "VLM_BASE_URL": vlm_base,
                "VLM_API_KEY": vlm_key,
                "VLM_MODEL": vlm_model,
            })
            vlm_ready = True

        save_model_environment(self.config.repository_root, values)
        self.llm_api_key_edit.clear()
        self.llm_api_key_edit.setPlaceholderText("API key already saved · leave blank to keep it")
        self.vlm_api_key_edit.clear()
        self.vlm_api_key_edit.setPlaceholderText("VLM API key already saved · leave blank to keep it")
        self._set_api_status(True, vlm_ready, saved=True)
        self.refresh_preflight(scan=False)
        self.configuration_changed.emit()

    def _load_reference_demo(self) -> None:
        try:
            self.config.apply_reference_demo()
        except (OSError, ValueError, ImportError) as exc:
            QMessageBox.critical(self, "Reference demo unavailable", str(exc))
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
        # Reproducibility is a fixed UI policy, not another decision the user
        # has to make on the streamlined first-run path.
        self.config.reproduce = True
        self.config.enable_segmentation = True
        mask_mode = "reuse" if self.config.segmentation_skip_if_present else "recreate"
        self.mask_buttons[mask_mode].setChecked(True)
        self.expert_check.setChecked(self.config.enable_expert_knowledge)
        self.deep_check.setChecked(self.config.enable_deep_research)
        self.rag_check.setChecked(self.config.enable_rag)
        self._loading = False

    def _dataset_path_changed(self, value: str = "") -> None:
        if self._loading:
            return
        path = Path(value).expanduser() if value.strip() else None
        if path is not None and path.is_dir():
            self.dataset_summary = scan_dataset(path)
            self._detect_dataset_context(path)
        else:
            self.dataset_summary = None
            self.config.description_path = ""
            self.config.metadata_path = ""
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
        self.config.data_root = self.dataset_picker.text()
        self.config.query = self.query_edit.toPlainText().strip()
        self.config.method = self._selected_method()
        self.config.reproduce = True
        self.config.enable_segmentation = True
        self.config.segmentation_skip_if_present = self.mask_buttons["reuse"].isChecked()
        self.config.enable_expert_knowledge = self.expert_check.isChecked()
        self.config.enable_deep_research = self.deep_check.isChecked()
        self.config.enable_rag = self.rag_check.isChecked()

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
        self.run_button.setEnabled(not blockers and self.dataset_summary is not None)
        if blockers:
            self.blocker_label.setText(f"Complete {len(blockers)} required item{'s' if len(blockers) != 1 else ''}")
            self.blocker_label.setProperty("role", "error")
            details = "\n".join(f"• {issue.message}" for issue in blockers)
            self.run_button.setToolTip(f"Complete the required setup before running:\n{details}")
            self.run_button.setAccessibleDescription(details)
        elif self.dataset_summary is None:
            self.blocker_label.setText("Choose a dataset to continue")
            self.blocker_label.setProperty("role", "warning")
            self.run_button.setToolTip("Choose a dataset to continue")
            self.run_button.setAccessibleDescription("Choose a dataset to continue")
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
        issues = self.refresh_preflight(scan=True)
        if any(issue.severity is Severity.BLOCKER for issue in issues):
            return
        self.run_requested.emit(self.config, self.dataset_summary)
