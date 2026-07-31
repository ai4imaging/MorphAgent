from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QPixmap
from dotenv import dotenv_values
from qtpy.QtWidgets import QApplication, QCheckBox, QComboBox, QFormLayout, QFrame, QLabel, QLineEdit, QPlainTextEdit, QRadioButton

from launch_ui import build_parser, create_standalone_window
from morphagent_ui.demo_api import FREE_DEMO_CANDIDATES, FREE_DEMO_ROUNDS, FREE_DEMO_TARGET
from morphagent_ui.main import MorphAgentWidget
from morphagent_ui.models import FeatureCard
from morphagent_ui.theme import STYLESHEET, apply_theme
from morphagent_ui.widgets.common import Card


class WidgetSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])
        apply_theme(cls.app)

    def test_five_destinations_with_split_features_and_evidence(self) -> None:
        widget = MorphAgentWidget()
        self.assertEqual(widget.navigation.count(), 5)
        self.assertEqual(widget.pages.count(), 5)
        destinations = [widget.navigation.item(index).text() for index in range(widget.navigation.count())]
        self.assertTrue(any("Features" in destination for destination in destinations))
        self.assertTrue(any("Evidence" in destination for destination in destinations))
        self.assertFalse(any("Settings" in destination for destination in destinations))
        self.assertFalse(hasattr(widget, "settings_page"))
        self.assertIn("#22D3EE", widget.styleSheet())
        widget.show_demo("features")
        self.app.processEvents()
        self.assertEqual(len(widget.features_page.cards), 5)
        self.assertEqual(widget.pages.currentIndex(), 3)
        widget.show_demo("evidence")
        self.app.processEvents()
        self.assertEqual(widget.pages.currentIndex(), 4)
        widget.close()

    def test_focused_launcher_is_default_and_has_no_napari_viewer(self) -> None:
        args = build_parser().parse_args([])
        self.assertFalse(args.with_napari)
        _application, window, widget = create_standalone_window("home")
        self.app.processEvents()
        self.assertIs(window.centralWidget(), widget)
        self.assertIsNone(widget.viewer)
        self.assertEqual(widget.evidence_page.viewer, None)
        self.assertTrue(widget.evidence_page.add_to_viewer_button.isHidden())
        self.assertGreaterEqual(window.minimumWidth(), 1100)
        window.close()

    def test_napari_mode_requires_explicit_flag(self) -> None:
        args = build_parser().parse_args(["--with-napari"])
        self.assertTrue(args.with_napari)

    def test_home_has_one_clear_primary_entry(self) -> None:
        widget = MorphAgentWidget()
        home = widget.home_page
        labels = {label.text() for label in home.findChildren(QLabel)}

        # Hero card + optional bundled demo-sample panel.
        self.assertGreaterEqual(len(home.findChildren(Card)), 1)
        self.assertEqual(home.new_button.text(), "Start a discovery run")
        self.assertTrue(home.new_button.property("homePrimary"))
        self.assertEqual(home.previous_run_button.text(), "Load a previous run")
        self.assertTrue(home.previous_run_button.property("homeSecondary"))
        self.assertEqual(home.demo_sample_button.text(), "Load demo standard output")
        self.assertFalse(hasattr(home, "resume_button"))
        self.assertIn("From microscopy to biologically grounded features", labels)
        self.assertFalse(any("interpretable features" in label.lower() for label in labels))
        self.assertNotIn("Configure", labels)
        self.assertNotIn("Run", labels)
        self.assertNotIn("Review", labels)
        widget.close()

    def test_home_demo_sample_preloads_and_clears_after_own_results(self) -> None:
        from morphagent_ui.widgets.home import bundled_demo_results_dir

        sample = bundled_demo_results_dir()
        if not sample.is_dir():
            self.skipTest("bundled completed_demo_run is not present")

        widget = MorphAgentWidget()
        self.app.processEvents()
        # isVisible() is False until the window is shown; isHidden() tracks our flag.
        self.assertFalse(widget.home_page.sample_panel.isHidden())
        self.assertTrue(widget._sample_results_active)
        self.assertTrue(widget.features_page.cards)

        with tempfile.TemporaryDirectory() as raw:
            own = Path(raw)
            (own / "features.csv").write_text(
                "sample_id,own_feature\nWT_1,0.1\n",
                encoding="utf-8",
            )
            widget.home_page.previous_run_requested.emit(str(own))
            self.app.processEvents()
            self.assertTrue(widget.home_page.sample_panel.isHidden())
            self.assertFalse(widget._sample_results_active)
            self.assertEqual([card.name for card in widget.features_page.cards], ["own_feature"])
        widget.close()

    def test_home_loads_previous_results_without_starting_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "features.csv").write_text(
                "sample_id,tau_ratio\nWT_1,0.8\n",
                encoding="utf-8",
            )
            widget = MorphAgentWidget()
            widget.home_page.previous_run_requested.emit(str(root))
            self.app.processEvents()

            self.assertFalse(widget.controller.running)
            self.assertEqual(widget.pages.currentIndex(), 3)
            self.assertEqual(widget.features_page.results_dir, str(root))
            self.assertEqual(widget.evidence_page.results_dir, str(root))
            self.assertEqual([card.name for card in widget.features_page.cards], ["tau_ratio"])
            widget.close()

    def test_configure_uses_grounded_choices_and_distinct_sections(self) -> None:
        widget = MorphAgentWidget()
        page = widget.configure_page
        labels = {label.text() for label in page.findChildren(QLabel)}

        self.assertEqual(page.findChildren(QComboBox), [])
        self.assertFalse(hasattr(page, "preset_buttons"))
        self.assertEqual(set(page.method_buttons), {"both", "code", "vlm"})
        for section in ("1 · Data", "2 · Biological question", "3 · Model API", "4 · Analysis"):
            self.assertIn(section, labels)
        self.assertNotIn("4 · Ready to run", labels)
        self.assertIn(
            f"Custom scale · {widget.config.num_rounds} round"
            f"{'s' if widget.config.num_rounds != 1 else ''} × "
            f"{widget.config.features_per_iteration} candidates · target {widget.config.target_feature_count}",
            labels,
        )
        self.assertIn("Analysis route · choose one", labels)
        self.assertNotIn("Mask preparation · choose one", labels)
        self.assertNotIn("Preparation · multiple choice", labels)
        self.assertNotIn("Reproducible results", {checkbox.text() for checkbox in page.findChildren(QCheckBox)})
        self.assertIn("Knowledge sources · multiple choice", labels)
        step_cards = [card for card in page.findChildren(Card) if card.property("stepCard")]
        self.assertEqual(len(step_cards), 4)
        self.assertTrue(all(not card.property("stepTone") for card in step_cards))
        step_headers = [frame for frame in page.findChildren(QFrame) if frame.property("stepHeader")]
        self.assertEqual(step_headers, [])

        self.assertEqual(page.mask_buttons, {})
        for button in page.method_buttons.values():
            self.assertIsInstance(button, QRadioButton)
            self.assertTrue(button.property("choiceTile"))
            self.assertGreaterEqual(button.minimumHeight(), 42)

        multi_choices = (
            page.expert_check,
            page.deep_check,
            page.rag_check,
        )
        for checkbox in multi_choices:
            self.assertIsInstance(checkbox, QCheckBox)
            self.assertTrue(checkbox.property("choiceTile"))
            self.assertGreaterEqual(checkbox.minimumHeight(), 46)
            self.assertFalse(checkbox.isChecked())
        self.assertFalse(page.validation_check.isChecked())
        self.assertFalse(widget.config.enable_expert_knowledge)
        self.assertFalse(widget.config.enable_deep_research)
        self.assertFalse(widget.config.enable_rag)
        self.assertFalse(widget.config.enable_feature_analysis)
        self.assertFalse(hasattr(page, "segment_check"))
        self.assertFalse(hasattr(page, "reproduce_check"))
        self.assertTrue(widget.config.enable_segmentation)
        self.assertTrue(widget.config.segmentation_skip_if_present)
        self.assertEqual(widget.config.temperature, 0.0)
        self.assertTrue(widget.config.reproduce)
        self.assertFalse(page.reuse_llm_for_vlm.isChecked())
        self.assertTrue(hasattr(page, "advanced_toggle"))
        # Blank API fields = own-API path -> Run config is available and open by default.
        # Use isHidden(): isVisible() is False until the top-level window is shown.
        self.assertFalse(page.config_section.isHidden())
        self.assertFalse(page.advanced_toggle.isHidden())
        self.assertIn("run config", page.advanced_toggle.text().lower())
        self.assertFalse(page.advanced_panel.isHidden())
        self.assertFalse(page.save_config_button.isHidden())
        self.assertTrue(page.save_api_button.isHidden())
        self.assertIn("Input data path", labels)

        widget.config.reproduce = False
        widget.config.temperature = 0.0
        page.load_from_config()
        self.assertTrue(widget.config.reproduce)
        self.assertIn("--reproduce", widget.config.build_command())

        self.assertTrue(page.load_demo_button.property("choiceAction"))
        self.assertEqual(page.load_demo_button.text(), "Load demo dataset")
        self.assertEqual(page.demo_guide.text(), ">>")
        self.assertEqual(page.dataset_picker.button.text(), "Browse…")
        self.assertTrue(page.run_button.property("runCta"))
        self.assertFalse(hasattr(page, "scan_button"))
        widget.close()

    def test_mask_preparation_is_internal_skip_if_present(self) -> None:
        widget = MorphAgentWidget()
        page = widget.configure_page

        page._sync_config()
        self.app.processEvents()
        self.assertTrue(widget.config.enable_segmentation)
        self.assertTrue(widget.config.segmentation_skip_if_present)
        self.assertIn("--segmentation-skip-if-present", widget.config.build_command())
        self.assertNotIn("--segmentation-run-even-if-present", widget.config.build_command())
        widget.close()

    def test_free_restricted_api_button_fills_and_locks_scale(self) -> None:
        widget = MorphAgentWidget()
        page = widget.configure_page
        page.rounds_spin.setValue(3)
        page.candidates_spin.setValue(10)
        page.target_spin.setValue(20)

        page.free_api_button.click()
        self.app.processEvents()

        self.assertEqual(page.llm_base_url_edit.text(), "https://api.gpugeek.com/v1")
        self.assertEqual(page.llm_model_edit.text(), "gpt-5.5")
        self.assertTrue(page.llm_model_edit.isReadOnly())
        self.assertTrue(page.llm_api_key_edit.text())
        self.assertTrue(page.reuse_llm_for_vlm.isChecked())
        self.assertEqual(page.rounds_spin.value(), FREE_DEMO_ROUNDS)
        self.assertEqual(page.candidates_spin.value(), FREE_DEMO_CANDIDATES)
        self.assertEqual(page.target_spin.value(), FREE_DEMO_TARGET)
        self.assertFalse(page.rounds_spin.isEnabled())
        self.assertFalse(page.candidates_spin.isEnabled())
        self.assertFalse(page.target_spin.isEnabled())
        self.assertTrue(page.config_section.isHidden())
        self.assertIn("token", page.free_api_note.text().lower())
        # Model must stay fixed even if the user tries to edit it.
        page.llm_model_edit.setText("should-not-stick")
        page._fields_changed()
        self.assertEqual(page.llm_model_edit.text(), "gpt-5.5")
        self.assertEqual(widget.config.llm_model, "gpt-5.5")

        page.llm_base_url_edit.setText("https://api.openai.com/v1")
        self.app.processEvents()
        self.assertFalse(page.config_section.isHidden())
        self.assertFalse(page.advanced_panel.isHidden())
        self.assertTrue(page.rounds_spin.isEnabled())
        self.assertTrue(page.candidates_spin.isEnabled())
        self.assertTrue(page.target_spin.isEnabled())
        widget.close()

    def test_gpugeek_host_with_own_key_unlocks_run_config(self) -> None:
        widget = MorphAgentWidget()
        page = widget.configure_page
        page.free_api_button.click()
        self.app.processEvents()
        self.assertTrue(page.config_section.isHidden())

        # Keep the free host, replace only the API key -> unrestricted scale.
        page.llm_api_key_edit.setText("sk-user-own-gpugeek-key")
        self.app.processEvents()
        self.assertFalse(page.config_section.isHidden())
        self.assertTrue(page.rounds_spin.isEnabled())
        page.rounds_spin.setValue(1)
        page.candidates_spin.setValue(10)
        page.target_spin.setValue(10)
        page._fields_changed()
        self.assertEqual(widget.config.features_per_iteration, 10)
        self.assertEqual(widget.config.target_feature_count, 10)
        self.assertEqual(widget.config.num_rounds, 1)
        command = widget.config.build_command()
        self.assertIn("--features-per-iteration", command)
        self.assertEqual(command[command.index("--features-per-iteration") + 1], "10")
        widget.close()

    def test_own_api_can_save_run_config_to_env(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repository = Path(raw)
            (repository / ".env").write_text("KEEP_ME=untouched\n", encoding="utf-8")
            widget = MorphAgentWidget()
            page = widget.configure_page
            widget.config.repository_root = str(repository)
            page.llm_base_url_edit.setText("https://api.openai.com/v1")
            page.llm_api_key_edit.setText("sk-test")
            page.llm_model_edit.setText("gpt-4o")
            page.rounds_spin.setValue(2)
            page.candidates_spin.setValue(8)
            page.target_spin.setValue(16)
            page.temperature_spin.setValue(0.2)
            page.save_config_button.click()
            self.app.processEvents()

            env_text = (repository / ".env").read_text(encoding="utf-8")
            self.assertIn("NUM_ROUNDS", env_text)
            self.assertIn("2", env_text)
            self.assertIn("FEATURES_PER_ITERATION", env_text)
            self.assertIn("TARGET_FEATURE_COUNT", env_text)
            self.assertIn("KEEP_ME", env_text)
            self.assertEqual(widget.config.num_rounds, 2)
            self.assertEqual(widget.config.features_per_iteration, 8)
            self.assertEqual(widget.config.target_feature_count, 16)
            widget.close()

    def test_configure_saves_masked_model_api_to_repository_env(self) -> None:
        names = (
            "LLM_BASE_URL",
            "LLM_API_KEY",
            "LLM_MODEL",
            "VLM_BASE_URL",
            "VLM_API_KEY",
            "VLM_MODEL",
        )
        previous = {name: os.environ.get(name) for name in names}
        try:
            with tempfile.TemporaryDirectory() as raw:
                repository = Path(raw)
                env_path = repository / ".env"
                env_path.write_text(
                    'LLM_BASE_URL="https://old.example/v1"\n'
                    'LLM_API_KEY="existing-secret"\n'
                    'LLM_MODEL="old-model"\n'
                    'VLM_BASE_URL="${LLM_BASE_URL}"\n'
                    'VLM_API_KEY="${LLM_API_KEY}"\n'
                    'VLM_MODEL="${LLM_MODEL}"\n'
                    'KEEP_ME="untouched"\n',
                    encoding="utf-8",
                )

                widget = MorphAgentWidget()
                widget.config.repository_root = str(repository)
                page = widget.configure_page
                self.assertTrue(hasattr(page, "load_api_settings"))
                page.load_api_settings()

                # Form stays empty on open; .env values are reused only when fields stay blank on Run.
                self.assertEqual(page.llm_base_url_edit.text(), "")
                self.assertEqual(page.llm_model_edit.text(), "")
                self.assertEqual(page.llm_api_key_edit.echoMode(), QLineEdit.Password)
                self.assertEqual(page.llm_api_key_edit.text(), "")
                self.assertIn("already on file", page.llm_api_key_edit.placeholderText().lower())
                # Default is unchecked; VLM fields stay visible even when values match LLM.
                self.assertFalse(page.reuse_llm_for_vlm.isChecked())
                self.assertFalse(page.vlm_connection_fields.isHidden())
                self.assertNotIn("existing-secret", page.api_status_label.text())
                self.assertFalse(page.save_api_button.isVisible())

                page.reuse_llm_for_vlm.setChecked(True)
                self.app.processEvents()
                self.assertTrue(page.vlm_connection_fields.isHidden())

                page.llm_base_url_edit.setText("https://new.example/v1")
                page.llm_model_edit.setText("new-model")
                page.llm_api_key_edit.setText("replacement-secret")
                self.assertTrue(page._persist_api_settings())
                self.app.processEvents()

                saved = dotenv_values(env_path)
                self.assertEqual(saved["LLM_BASE_URL"], "https://new.example/v1")
                self.assertEqual(saved["LLM_API_KEY"], "replacement-secret")
                self.assertEqual(saved["LLM_MODEL"], "new-model")
                self.assertEqual(saved["VLM_BASE_URL"], "https://new.example/v1")
                self.assertEqual(saved["VLM_API_KEY"], "replacement-secret")
                self.assertEqual(saved["VLM_MODEL"], "new-model")
                self.assertEqual(saved["KEEP_ME"], "untouched")
                self.assertEqual(page.llm_api_key_edit.text(), "")
                self.assertEqual(os.environ["LLM_API_KEY"], "replacement-secret")
                self.assertNotIn("replacement-secret", page.api_status_label.text())
                widget.close()
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def test_model_api_fields_are_forced_to_fill_the_available_width(self) -> None:
        widget = MorphAgentWidget()
        page = widget.configure_page

        self.assertEqual(page.llm_form.fieldGrowthPolicy(), QFormLayout.AllNonFixedFieldsGrow)
        self.assertEqual(page.vlm_form.fieldGrowthPolicy(), QFormLayout.AllNonFixedFieldsGrow)
        self.assertEqual(page.llm_form.formAlignment(), Qt.AlignLeft | Qt.AlignTop)
        self.assertEqual(page.vlm_form.formAlignment(), Qt.AlignLeft | Qt.AlignTop)
        widget.close()

    def test_configure_selected_choices_use_original_aqua_highlight(self) -> None:
        self.assertIn(
            'QCheckBox[choiceTile="true"]:checked {\n    background: #103448;',
            STYLESHEET,
        )
        self.assertIn(
            'QRadioButton[choiceTile="true"]:checked {\n    background: #102C3E;',
            STYLESHEET,
        )
        self.assertIn(
            'QCheckBox[choiceTile="true"]::indicator:checked { background: #0891B2; border: 1px solid #22D3EE;',
            STYLESHEET,
        )

    def test_feature_filters_are_visible_click_choices(self) -> None:
        widget = MorphAgentWidget()
        page = widget.features_page
        page.cards = [
            FeatureCard("code-retained", "code_retained", "code", status="retained"),
            FeatureCard("vlm-retained", "vlm_retained", "vlm", status="retained"),
            FeatureCard("vlm-dropped", "vlm_dropped", "vlm", status="dropped"),
            FeatureCard("code-dropped", "code_dropped", "code", status="dropped"),
        ]
        page._update_filters()
        page._filter()
        self.app.processEvents()

        self.assertEqual(page.findChildren(QComboBox), [])
        self.assertEqual(set(page.route_buttons), {"all", "code", "vlm"})
        self.assertEqual(set(page.status_buttons), {"all", "retained", "dropped"})
        nonhidden_filter_buttons = [
            button for button in page.findChildren(QRadioButton)
            if button.property("filterChoice") and not button.isHidden()
        ]
        self.assertEqual(
            len(nonhidden_filter_buttons),
            len(page.route_buttons) + len(page.status_buttons),
        )
        for button in (*page.route_buttons.values(), *page.status_buttons.values()):
            self.assertIsInstance(button, QRadioButton)
            self.assertTrue(button.property("filterChoice"))
            self.assertEqual(button.cursor().shape(), Qt.PointingHandCursor)

        page.route_buttons["vlm"].click()
        self.app.processEvents()
        self.assertEqual({card.method for card in page.filtered_cards}, {"vlm"})
        self.assertEqual(page.table.rowCount(), 2)

        page.status_buttons["dropped"].click()
        self.app.processEvents()
        self.assertEqual([card.name for card in page.filtered_cards], ["vlm_dropped"])
        self.assertEqual(page.table.rowCount(), 1)

        page.route_buttons["code"].click()
        self.app.processEvents()
        self.assertEqual(page.table.rowCount(), 1)
        self.assertEqual([card.name for card in page.filtered_cards], ["code_dropped"])
        widget.close()

    def test_feature_table_and_detail_use_equal_splitter_widths(self) -> None:
        _application, window, widget = create_standalone_window("features")
        window.resize(1512, 982)
        window.show()
        self.app.processEvents()

        page = widget.features_page
        left, right = page.splitter.sizes()
        self.assertLessEqual(abs(left - right), 8)
        window.close()

    def test_evidence_feature_selector_uses_three_columns_and_only_a_compact_description(self) -> None:
        cards = [
            FeatureCard(
                f"feature-{index}",
                name,
                "vlm" if index % 2 else "code",
                "morphology",
                f"Description for {name}",
                "retained",
                1,
                0.8,
                (),
            )
            for index, name in enumerate(
                (
                    "tau_ratio",
                    "neurite_bead_density",
                    "somatic_nft_density",
                    "subcellular_tau_distribution_ratio",
                )
            )
        ]
        with tempfile.TemporaryDirectory() as raw:
            widget = MorphAgentWidget()
            evidence = widget.evidence_page
            evidence.set_results(raw, cards)
            self.app.processEvents()

            self.assertEqual(evidence.feature_columns, 3)
            self.assertEqual(len(evidence.feature_buttons), 4)
            self.assertFalse(hasattr(evidence, "feature_detail"))
            self.assertFalse(hasattr(evidence, "feature_facts"))
            self.assertFalse(hasattr(evidence, "semantic_note"))
            self.assertEqual(evidence.selected_feature_name.text(), "tau_ratio")
            self.assertEqual(evidence.selected_feature_description.text(), "Description for tau_ratio")

            buttons = list(evidence.feature_buttons.values())
            self.assertTrue(all(button.isCheckable() for button in buttons))
            self.assertTrue(all(button.property("featureChoice") for button in buttons))
            self.assertTrue(all("·" not in button.text() for button in buttons))
            self.assertEqual([button.text().replace("\n", "") for button in buttons], [card.name for card in cards])
            positions = [evidence.feature_grid.getItemPosition(index)[:2] for index in range(4)]
            self.assertEqual(positions, [(0, 0), (0, 1), (0, 2), (1, 0)])

            buttons[-1].click()
            self.app.processEvents()
            self.assertTrue(buttons[-1].isChecked())
            self.assertEqual(evidence.current_card.name, "subcellular_tau_distribution_ratio")
            self.assertIn("subcellular_tau_distribution_ratio", evidence.evidence_title.text())
            self.assertEqual(evidence.selected_feature_name.text(), "subcellular_tau_distribution_ratio")
            self.assertEqual(
                evidence.selected_feature_description.text(),
                "Description for subcellular_tau_distribution_ratio",
            )

            evidence.set_results(raw, cards[:2])
            self.assertTrue(all(button.isHidden() for button in buttons))
            evidence.set_results(raw, [])
            self.assertEqual(evidence.selected_feature_name.text(), "Select a feature")
            self.assertIn("Choose a feature", evidence.selected_feature_description.text())
            widget.close()

    def test_evidence_selector_is_equal_width_and_never_scrolls_horizontally(self) -> None:
        names = (
            "axon_to_soma_tau_intensity_ratio",
            "distal_to_proximal_axon_tau_intensity_ratio",
            "tau_neurite_bead_density",
            "vlm_somatic_nft_density",
            "vlm_tau_aggregation_pattern_proportion",
            "extracellular_ghost_tangle_count",
            "soma_tau_aggregate_compactness",
            "axon_tau_gradient_uniformity",
            "vlm_distal_dendrite_tau_inclusion_density",
            "vlm_somatic_tau_texture_variance",
        )
        long_description = (
            "This feature measures the uniformity of the Tau intensity gradient along axons, capturing deviations "
            "from healthy axonal distribution across proximal and distal regions. It compares slope and correlation "
            "values across the axonal length to identify disrupted localization patterns."
        )
        cards = [
            FeatureCard(
                f"feature-{index}",
                name,
                "code",
                "morphology",
                long_description if name == "axon_tau_gradient_uniformity" else name,
                "retained",
                1,
                0.8,
                (),
            )
            for index, name in enumerate(names)
        ]
        with tempfile.TemporaryDirectory() as raw:
            _application, window, widget = create_standalone_window("evidence")
            window.resize(1512, 982)
            evidence = widget.evidence_page
            evidence.set_results(raw, cards)
            widget.navigate(4)
            window.show()
            self.app.processEvents()

            left, right = evidence.layout_splitter.sizes()
            self.assertLessEqual(abs(left - right), 8)
            self.assertEqual(evidence.feature_scroll.horizontalScrollBarPolicy(), Qt.ScrollBarAlwaysOff)
            self.assertEqual(evidence.feature_scroll.horizontalScrollBar().maximum(), 0)
            gradient_button = next(
                button
                for button in evidence.feature_buttons.values()
                if button.property("featureName") == "axon_tau_gradient_uniformity"
            )
            gradient_button.click()
            self.app.processEvents()
            self.assertGreaterEqual(
                evidence.selected_feature_description.height(),
                evidence.selected_feature_description.heightForWidth(
                    evidence.selected_feature_description.width()
                ),
            )
            window.close()

    def test_evidence_uses_two_columns_and_curates_selected_feature_sources(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            round_one = root / "round_1"
            visual = root / "first_sample_visualization"
            round_one.mkdir()
            visual.mkdir()
            (root / "features.csv").write_text(
                "sample_id,tau_ratio,other_feature\nWT_1,0.8,9.1\nWT_2,0.6,8.2\n",
                encoding="utf-8",
            )
            (root / "retained_features.csv").write_text(
                "sample_id,tau_ratio,other_feature\nWT_1,0.8,9.1\nWT_2,0.6,8.2\n",
                encoding="utf-8",
            )
            (root / "feature_registry.json").write_text(json.dumps({
                "entries": [
                    {
                        "feature_id": "tau-ratio",
                        "name": "tau_ratio",
                        "actual_column_name": "tau_ratio",
                        "method": "code",
                        "category": "distribution",
                        "description": "Tau ratio only.",
                        "latest_round": 1,
                        "current_status": "retained",
                        "decision_history": [{"validation_score": 0.8, "reason_codes": ["passed_hard_filters"]}],
                    },
                    {
                        "feature_id": "other-id",
                        "name": "other_feature",
                        "actual_column_name": "other_feature",
                        "method": "code",
                        "category": "morphology",
                        "description": "Other feature only.",
                        "latest_round": 1,
                        "current_status": "retained",
                        "decision_history": [{"validation_score": 0.7, "reason_codes": ["passed_hard_filters"]}],
                    },
                ]
            }), encoding="utf-8")
            (root / "segmentation_summary.json").write_text('{"success": 5}', encoding="utf-8")
            (round_one / "feature_plan.json").write_text(json.dumps({
                "features": [
                    {"name": "tau_ratio", "method": "code", "description": "Tau plan"},
                    {"name": "other_feature", "method": "code", "description": "Other plan"},
                ]
            }), encoding="utf-8")
            (round_one / "validation_decisions.csv").write_text(
                "feature_name,status,validation_score\ntau_ratio,retained,0.8\nother_feature,retained,0.7\n",
                encoding="utf-8",
            )
            (round_one / "merged_feature_code.py").write_text(
                "# Feature 1: tau_ratio\nresults['tau_ratio'] = 0.8\n\n"
                "# Feature 2: other_feature\nresults['other_feature'] = 9.1\n",
                encoding="utf-8",
            )
            extra = round_one / "audit" / "extra_result.json"
            extra.parent.mkdir()
            extra.write_text('{"complete": true}', encoding="utf-8")
            manifest = root / "ui_run_manifest.json"
            manifest.write_text('{"method": "both", "num_rounds": 2}', encoding="utf-8")
            console_log = root / "ui_console.log"
            console_log.write_text("large runtime log", encoding="utf-8")
            image_path = visual / "summary.png"
            image = QPixmap(12, 8)
            image.fill(QColor("#22D3EE"))
            self.assertTrue(image.save(str(image_path)))

            widget = MorphAgentWidget()
            page = widget.features_page
            evidence = widget.evidence_page
            self.assertFalse(hasattr(page, "evidence_page"))
            self.assertIsNot(evidence, page)
            page.load_results(str(root))
            evidence.set_results(str(root), page.cards)
            self.app.processEvents()

            self.assertEqual(evidence.layout_splitter.count(), 2)
            self.assertEqual(len(evidence.feature_buttons), 2)
            self.assertEqual(evidence._selected_artifact(), root / "features.csv")
            self.assertIs(evidence.preview_stack.currentWidget(), evidence.text_preview)
            self.assertIn("sample_id,tau_ratio", evidence.text_preview.toPlainText())

            tau_row = next(row for row in range(page.table.rowCount()) if page.table.item(row, 0).text() == "tau_ratio")
            page.table.selectRow(tau_row)
            self.app.processEvents()
            self.assertEqual(page.detail.name.text(), "tau_ratio")
            self.assertEqual(evidence.current_card.name, "tau_ratio")
            self.assertIn("tau_ratio", evidence.evidence_title.text())

            listed_paths = {
                Path(group.child(child).data(0, Qt.UserRole))
                for index in range(evidence.artifact_tree.topLevelItemCount())
                for group in [evidence.artifact_tree.topLevelItem(index)]
                for child in range(group.childCount())
            }
            self.assertNotIn(image_path, listed_paths)
            self.assertIn(root / "segmentation_summary.json", listed_paths)
            self.assertIn(manifest, listed_paths)
            self.assertIn(root / "features.csv", listed_paths)
            self.assertIn(root / "feature_registry.json", listed_paths)
            self.assertIn(round_one / "feature_plan.json", listed_paths)
            self.assertIn(round_one / "validation_decisions.csv", listed_paths)
            self.assertNotIn(round_one / "merged_feature_code.py", listed_paths)
            self.assertNotIn(console_log, listed_paths)
            self.assertNotIn(extra, listed_paths)
            self.assertFalse(any(path.suffix.lower() == ".py" for path in listed_paths))

            features_item = next(
                group.child(child)
                for index in range(evidence.artifact_tree.topLevelItemCount())
                for group in [evidence.artifact_tree.topLevelItem(index)]
                for child in range(group.childCount())
                if Path(group.child(child).data(0, Qt.UserRole)).name == "features.csv"
            )
            evidence.artifact_tree.setCurrentItem(features_item)
            self.app.processEvents()
            tau_preview = evidence.text_preview.toPlainText()
            self.assertIn("sample_id,tau_ratio", tau_preview)
            self.assertIn("WT_1,0.8", tau_preview)
            self.assertNotIn("other_feature", tau_preview)

            other_button = next(
                button
                for button in evidence.feature_buttons.values()
                if button.property("featureName") == "other_feature"
            )
            other_button.click()
            self.app.processEvents()
            self.assertEqual(evidence.current_card.name, "other_feature")
            self.assertTrue(other_button.isChecked())
            self.assertEqual(evidence._selected_artifact(), root / "features.csv")
            self.assertIs(evidence.preview_stack.currentWidget(), evidence.text_preview)
            # Shared first_sample_visualization images stay out of per-feature evidence.
            listed_after = {
                Path(group.child(child).data(0, Qt.UserRole))
                for index in range(evidence.artifact_tree.topLevelItemCount())
                for group in [evidence.artifact_tree.topLevelItem(index)]
                for child in range(group.childCount())
            }
            self.assertNotIn(image_path, listed_after)
            widget.close()

    def test_ready_configuration_enables_launch(self) -> None:
        names = ("LLM_API_KEY", "VLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "VLM_BASE_URL", "VLM_MODEL")
        previous = {name: os.environ.get(name) for name in names}
        os.environ.update({
            "LLM_API_KEY": "ui-test-key",
            "VLM_API_KEY": "ui-test-vlm",
            "LLM_BASE_URL": "https://example.com/v1",
            "LLM_MODEL": "gpt-4o",
            "VLM_BASE_URL": "https://example.com/v1",
            "VLM_MODEL": "gpt-4o",
        })
        try:
            with tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                sample = root / "sample_1"
                sample.mkdir()
                (sample / "image.png").touch()
                widget = MorphAgentWidget()
                page = widget.configure_page
                page.llm_base_url_edit.setText("https://example.com/v1")
                page.llm_model_edit.setText("gpt-4o")
                page.llm_api_key_edit.setText("ui-test-key")
                page.reuse_llm_for_vlm.setChecked(True)
                page.dataset_picker.setText(str(root))
                page.query_edit.setPlainText("Profile interpretable nuclear morphology")
                page.refresh_preflight(scan=True)
                self.app.processEvents()
                self.assertTrue(page.run_button.isEnabled())
                self.assertIn("--data-root", page.command_preview.toPlainText())
                widget.close()
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def test_reference_demo_button_loads_teacher_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            demo = repo / "demo"
            dataset = demo / "data" / "dataset"
            sample = dataset / "WT_1"
            rag = demo / "data" / "RAG"
            precomputed = demo / "precomputed"
            sample.mkdir(parents=True)
            rag.mkdir(parents=True)
            precomputed.mkdir(parents=True)
            (sample / "image.tif").touch()
            (dataset / "dataset_index.txt").write_text("Tau demo", encoding="utf-8")
            (rag / "paper.pdf").write_bytes(b"reference")
            (precomputed / "rag_knowledge_summary.txt").write_text("cached knowledge", encoding="utf-8")

            widget = MorphAgentWidget()
            widget.config.repository_root = str(repo)
            page = widget.configure_page
            page.rounds_spin.setValue(2)
            page.candidates_spin.setValue(10)
            page.target_spin.setValue(10)
            page._fields_changed()

            self.assertTrue(hasattr(page, "load_demo_button"))
            page.load_demo_button.click()
            self.app.processEvents()

            self.assertEqual(page.dataset_summary.sample_count, 1)
            self.assertEqual(page.dataset_picker.text(), str((demo / "data").resolve()))
            self.assertIn("Tau protein aggregation", page.query_edit.toPlainText())
            self.assertEqual(widget.config.features_per_iteration, 10)
            self.assertEqual(widget.config.target_feature_count, 10)
            self.assertEqual(widget.config.num_rounds, 2)
            self.assertEqual(widget.config.dataset_source, "demo")
            command = widget.config.build_command()
            self.assertNotIn("--auto-deep-research", command)
            self.assertNotIn("--auto-literature-retrieval", command)
            widget.close()

    def test_custom_dataset_auto_detects_context_and_clears_demo_paths(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            sample = root / "dataset" / "sample_1"
            sample.mkdir(parents=True)
            (sample / "image.tif").touch()
            description = root / "dataset" / "dataset_index.txt"
            description.write_text("Custom dataset", encoding="utf-8")

            widget = MorphAgentWidget()
            widget.config.description_path = "/stale/demo/dataset_index.txt"
            widget.config.metadata_path = "/stale/demo/metadata.csv"
            page = widget.configure_page
            page.dataset_picker.setText(str(root))
            self.app.processEvents()

            self.assertEqual(widget.config.description_path, str(description.resolve()))
            self.assertEqual(widget.config.metadata_path, "")
            widget.close()


if __name__ == "__main__":
    unittest.main()
