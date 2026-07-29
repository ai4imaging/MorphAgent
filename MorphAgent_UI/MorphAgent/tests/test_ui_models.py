from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from morphagent_ui.models import (
    RunConfig,
    RunPreset,
    Severity,
    find_completed_rounds,
    list_result_artifacts,
    load_feature_cards,
    scan_dataset,
)


class DatasetScanTests(unittest.TestCase):
    def test_nested_dataset_and_route_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            sample_a = root / "dataset" / "sample_a"
            sample_b = root / "dataset" / "sample_b"
            (sample_a / "slices").mkdir(parents=True)
            (sample_a / "segmentation").mkdir()
            sample_b.mkdir(parents=True)
            (sample_a / "raw.tif").touch()
            (sample_a / "slices" / "view_1.png").touch()
            (sample_a / "slices" / "view_2.png").touch()
            (sample_a / "segmentation" / "cell_mask.tif").touch()
            (sample_b / "raw.tiff").touch()

            summary = scan_dataset(root)

            self.assertTrue(summary.used_dataset_child)
            self.assertEqual(summary.sample_count, 2)
            self.assertEqual(summary.primary_image_count, 2)
            self.assertEqual(summary.vlm_source_count, 3)
            self.assertEqual(summary.vlm_native_count, 2)
            self.assertEqual(summary.mask_count, 1)
            self.assertEqual(summary.samples[0].vlm_source, "folder:slices")

    def test_empty_sample_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "empty_sample").mkdir()
            summary = scan_dataset(root)
            self.assertEqual(summary.empty_samples, ("empty_sample",))


class RunConfigTests(unittest.TestCase):
    def test_ui_run_config_defaults_to_reproducible_mode(self) -> None:
        config = RunConfig(query="Profile cells")

        self.assertEqual(config.temperature, 0.0)
        self.assertTrue(config.reproduce)
        self.assertIn("--reproduce", config.build_command())
        self.assertIn("--reproduce-seed", config.build_command())
        env = config.pipeline_environment()
        self.assertEqual(env["CODE_TEMPERATURE"], "0")
        self.assertEqual(env["VLM_TEMPERATURE"], "0")

    def test_diagnose_dataset_selection_for_custom_images_only(self) -> None:
        from morphagent_ui.models import diagnose_dataset_selection

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            dataset = root / "dataset"
            sample = dataset / "WT_1"
            sample.mkdir(parents=True)
            (sample / "image.tif").touch()
            (dataset / "dataset_index.txt").write_text("custom images", encoding="utf-8")

            self.assertIsNone(diagnose_dataset_selection(root))
            self.assertIsNotNone(diagnose_dataset_selection(sample))
        self.assertIsNotNone(diagnose_dataset_selection(Path("/does/not/exist")))

    def _ready_dataset(self, root: Path):
        sample = root / "sample_1"
        sample.mkdir(parents=True)
        (sample / "image.png").touch()
        return scan_dataset(root)

    def test_only_repository_grounded_preset_is_exposed(self) -> None:
        self.assertEqual(list(RunPreset), [RunPreset.PILOT])

    def test_pilot_preset_matches_demo_scale(self) -> None:
        config = RunConfig()

        config.apply_preset(RunPreset.PILOT)

        self.assertEqual(config.method, "both")
        self.assertEqual(config.features_per_iteration, 5)
        self.assertEqual(config.target_feature_count, 5)
        self.assertEqual(config.num_rounds, 1)

    def test_low_frequency_run_defaults_can_come_from_environment(self) -> None:
        values = {
            "FEATURES_PER_ITERATION": "7",
            "TARGET_FEATURE_COUNT": "21",
            "NUM_ROUNDS": "3",
            "CODE_VLM_RATIO": "0.6",
            "KNOWLEDGE_DEPENDENCY": "0.4",
            "CODE_PARALLEL_WORKERS": "2",
            "VLM_ONLINE_CONCURRENCY": "6",
        }
        previous = {name: os.environ.get(name) for name in values}
        try:
            os.environ.update(values)
            config = RunConfig()
            self.assertEqual(config.features_per_iteration, 7)
            self.assertEqual(config.target_feature_count, 21)
            self.assertEqual(config.num_rounds, 3)
            self.assertEqual(config.code_vlm_ratio, 0.6)
            self.assertEqual(config.knowledge_dependency, 0.4)
            self.assertEqual(config.code_parallel_workers, 2)
            self.assertEqual(config.vlm_online_concurrency, 6)

            config.apply_preset(RunPreset.PILOT)
            self.assertEqual(config.features_per_iteration, 5)
            self.assertEqual(config.target_feature_count, 5)
            self.assertEqual(config.num_rounds, 1)
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def test_reference_demo_config_seeds_rag_cache(self) -> None:
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
            metadata_csv = demo / "data" / "metadata.csv"
            metadata_csv.write_text(
                "sample_id,group,genotype\nWT_1,WT,wild_type\n",
                encoding="utf-8",
            )

            config = RunConfig(repository_root=str(repo))
            self.assertTrue(hasattr(config, "apply_reference_demo"))
            cache_path = config.apply_reference_demo()

            self.assertEqual(config.data_root, str((demo / "data").resolve()))
            self.assertEqual(config.description_path, str((dataset / "dataset_index.txt").resolve()))
            self.assertEqual(config.metadata_path, str(metadata_csv.resolve()))
            self.assertTrue(config.enable_feature_analysis)
            self.assertEqual(config.results_dir, "")
            self.assertIn("Tau protein aggregation", config.query)
            self.assertEqual(config.method, "both")
            self.assertEqual(config.features_per_iteration, 5)
            self.assertEqual(config.target_feature_count, 5)
            self.assertEqual(config.num_rounds, 1)
            self.assertEqual(config.dataset_source, "demo")
            self.assertTrue(config.enable_expert_knowledge)
            self.assertTrue(config.enable_deep_research)
            self.assertTrue(config.enable_rag)
            self.assertTrue(config.enable_segmentation)
            self.assertTrue(config.segmentation_skip_if_present)
            self.assertTrue(cache_path.is_file())
            self.assertIn("cached knowledge", cache_path.read_text(encoding="utf-8"))
            command = config.build_command()
            self.assertIn("--metadata-path", command)
            self.assertNotIn("--disable-feature-analysis", command)
            self.assertNotIn("--auto-deep-research", command)
            self.assertNotIn("--auto-literature-retrieval", command)
            env = config.pipeline_environment()
            self.assertEqual(env["CODE_MAX_RETRIES"], "3")
            self.assertEqual(env["SEGMENTATION_BACKEND"], "allen")
            self.assertEqual(env["SEGMENTATION_CONDA_ENV"], "morphagent_allen")

    def test_custom_dataset_passes_auto_knowledge_flags(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            config = RunConfig(
                data_root=str(root),
                query="Profile cells",
                python_executable=sys.executable,
                dataset_source="custom",
                enable_deep_research=True,
                enable_rag=True,
            )
            command = config.build_command()
            self.assertIn("--auto-deep-research", command)
            self.assertIn("--auto-literature-retrieval", command)
            self.assertIn("--pubmed-max-results", command)
            self.assertIn("10", command)

    def test_sample_count_warning_for_small_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            summary = self._ready_dataset(root)
            config = RunConfig(
                data_root=str(root),
                query="Profile cells",
                python_executable=sys.executable,
                enable_segmentation=False,
            )
            issues = config.validate(summary, {
                "LLM_API_KEY": "x",
                "LLM_BASE_URL": "https://example.com/v1",
                "LLM_MODEL": "gpt-4o",
            })
            self.assertIn("sample_count_low", {issue.code for issue in issues})

    def test_preflight_blocks_missing_key_and_accepts_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            summary = self._ready_dataset(root)
            config = RunConfig(data_root=str(root), query="Profile nuclear texture", python_executable=sys.executable, enable_segmentation=False)
            missing = config.validate(summary, {})
            self.assertIn("llm_key_missing", {issue.code for issue in missing})

            ready = config.validate(summary, {
                "LLM_API_KEY": "test-only",
                "LLM_BASE_URL": "https://example.com/v1",
                "LLM_MODEL": "gpt-4o",
                "VLM_API_KEY": "test-only",
                "VLM_BASE_URL": "https://example.com/v1",
                "VLM_MODEL": "gpt-4o",
            })
            self.assertFalse(any(issue.severity is Severity.BLOCKER for issue in ready))

    def test_command_is_explicit_and_manifest_has_no_secret(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            config = RunConfig(
                data_root=str(root),
                query="Profile mitochondrial networks",
                python_executable=sys.executable,
                method="code",
                enable_deep_research=False,
                enable_rag=False,
                enable_segmentation=False,
                reproduce=True,
            )
            command = config.build_command()
            self.assertEqual(
                command[:4],
                [sys.executable, "-u", str(Path(config.repository_root) / "main.py"), config.query],
            )
            self.assertIn("--disable-deep-research", command)
            self.assertIn("--disable-rag", command)
            self.assertIn("--disable-segmentation", command)
            self.assertIn("--reproduce", command)
            self.assertNotIn("--auto-deep-research", command)
            self.assertNotIn("--auto-literature-retrieval", command)
            manifest = config.manifest()
            serialized = json.dumps(manifest)
            self.assertNotIn("API_KEY", serialized.replace("API keys", ""))
            self.assertNotIn("test-only", serialized)

    def test_resume_requires_completed_round_marker(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            sample = root / "data" / "sample"
            sample.mkdir(parents=True)
            (sample / "image.png").touch()
            results = root / "results"
            results.mkdir()
            config = RunConfig(data_root=str(root / "data"), query="Profile cells", results_dir=str(results), resume=True, python_executable=sys.executable)
            issues = config.validate(scan_dataset(root / "data"), {"LLM_API_KEY": "x"})
            self.assertIn("resume_marker_missing", {issue.code for issue in issues})
            marker = results / "round_1" / "round_results.json"
            marker.parent.mkdir()
            marker.write_text("{}", encoding="utf-8")
            self.assertEqual(find_completed_rounds(results), [1])

    def test_nonempty_config_source_fallback_is_detected_without_exposing_it(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repository = root / "repo"
            dataset = root / "dataset"
            repository.mkdir()
            (repository / "config.py").write_text(
                "import os\n"
                "DEFAULT_LLM_API_KEY = os.getenv('LLM_API_KEY', 'configured-in-source')\n"
                "DEFAULT_VLM_API_KEY = os.getenv('VLM_API_KEY', DEFAULT_LLM_API_KEY)\n",
                encoding="utf-8",
            )
            summary = self._ready_dataset(dataset)
            config = RunConfig(
                data_root=str(dataset),
                query="Profile cells",
                repository_root=str(repository),
                python_executable=sys.executable,
                enable_segmentation=False,
                llm_base_url="https://example.com/v1",
                llm_model="gpt-4o",
            )
            issues = config.validate(summary, {})
            self.assertNotIn("llm_key_missing", {issue.code for issue in issues})
            self.assertNotIn("vlm_key_missing", {issue.code for issue in issues})
            self.assertNotIn("configured-in-source", json.dumps(config.manifest()))


class FeatureArtifactTests(unittest.TestCase):
    def test_artifact_discovery_includes_nested_images_code_validation_and_logs(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            visual = root / "first_sample_visualization"
            round_dir = root / "round_1"
            merged = round_dir / "merged_features"
            visual.mkdir()
            merged.mkdir(parents=True)
            expected = {
                visual / "all_channels_summary.png",
                visual / "channel_01.tif",
                root / "features.csv",
                root / "ui_console.log",
                round_dir / "feature_plan.json",
                round_dir / "validation_decisions.csv",
                round_dir / "validation_summary.json",
                round_dir / "merged_feature_code.py",
                merged / "execution_log.txt",
            }
            for path in expected:
                path.write_text("artifact", encoding="utf-8")

            self.assertTrue(expected.issubset(set(list_result_artifacts(root))))

    def test_registry_overrides_plan_and_preserves_design_fields(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            round_dir = root / "round_1"
            round_dir.mkdir()
            (round_dir / "feature_plan.json").write_text(json.dumps({
                "features": [{
                    "name": "network_fragmentation",
                    "description": "Branch loss in a mitochondrial network.",
                    "method": "code",
                    "category": "spatial",
                    "candidate_operators": ["skeletonize", "connected components"],
                }]
            }), encoding="utf-8")
            (root / "feature_registry.json").write_text(json.dumps({
                "entries": [{
                    "feature_id": "f-1",
                    "name": "network_fragmentation",
                    "actual_column_name": "network_fragmentation",
                    "method": "code",
                    "category": "spatial",
                    "description": "Validated branch loss.",
                    "latest_round": 1,
                    "current_status": "retained",
                    "decision_history": [{"validation_score": 0.87, "reason_codes": ["high_cv"]}],
                }]
            }), encoding="utf-8")

            cards = load_feature_cards(root)
            self.assertEqual(len(cards), 1)
            self.assertEqual(cards[0].status, "retained")
            self.assertAlmostEqual(cards[0].validation_score or 0, 0.87)
            self.assertIn("skeletonize", cards[0].candidate_operators)


if __name__ == "__main__":
    unittest.main()
