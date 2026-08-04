from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from morphagent_ui.controller import (
    DynamicEta,
    PipelineWorker,
    ReuseConfig,
    ReuseProgressDetector,
    StageDetector,
    artifact_snapshot,
    estimate_run_seconds,
)
from morphagent_ui.models import RunConfig, scan_dataset


class StageDetectorTests(unittest.TestCase):
    def test_top_level_messages_are_monotonic(self) -> None:
        detector = StageDetector()
        self.assertEqual(detector.feed("Step 1: Read the dataset index"), 0)
        self.assertEqual(detector.feed("Step 2.4: Data segmentation (running)"), 1)
        self.assertEqual(detector.feed("Step 3: Feature planning"), 2)
        self.assertIsNone(detector.feed("Step 3.5: Data preprocessing"))
        self.assertEqual(detector.feed("Step 4: Batch feature extraction"), 3)
        self.assertEqual(detector.feed("Step 6: Deterministic feature validation"), 4)
        self.assertIsNone(detector.feed("[OK] Round 1 complete!"))
        self.assertEqual(detector.feed("[DONE] All 2 rounds complete!"), 5)
        self.assertIsNone(detector.feed("Step 2: Dataset understanding"))

    def test_indented_code_route_steps_do_not_change_stage(self) -> None:
        detector = StageDetector()
        detector.feed("Step 3: Feature planning")
        self.assertIsNone(detector.feed("  Step 4: Organizing feature results and saving to CSV..."))
        self.assertEqual(detector.index, 2)


class ArtifactSnapshotTests(unittest.TestCase):
    def test_completed_round_and_files_are_detected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "features.csv").write_text("sample_id,f1\na,1\n", encoding="utf-8")
            round_dir = root / "round_2"
            round_dir.mkdir()
            (round_dir / "round_results.json").write_text(json.dumps({"round": 2}), encoding="utf-8")
            snapshot = artifact_snapshot(root)
            self.assertTrue(snapshot["files"]["features.csv"])
            self.assertEqual(snapshot["completed_rounds"], [2])
            self.assertEqual(snapshot["artifact_count"], 2)


class RunEstimateTests(unittest.TestCase):
    def _dataset(self, image_count: int):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for index in range(image_count):
                sample = root / f"sample_{index}"
                sample.mkdir()
                (sample / "image.png").touch()
            return scan_dataset(root)

    def test_estimate_scales_with_rounds_features_and_images(self) -> None:
        small = RunConfig(
            num_rounds=1,
            features_per_iteration=5,
            method="both",
            code_parallel_workers=1,
            vlm_online_concurrency=1,
        )
        large = RunConfig(
            num_rounds=3,
            features_per_iteration=10,
            method="both",
            code_parallel_workers=1,
            vlm_online_concurrency=1,
        )
        self.assertGreater(
            estimate_run_seconds(large, self._dataset(12)),
            estimate_run_seconds(small, self._dataset(3)),
        )

    def test_dynamic_eta_uses_stage_and_completed_rounds(self) -> None:
        eta = DynamicEta(initial_total_seconds=900, num_rounds=3)
        initial_remaining = eta.remaining_seconds(60)
        eta.update_progress(40)
        after_plan = eta.remaining_seconds(60)
        self.assertLess(after_plan, initial_remaining)

        eta.update_completed_rounds(2)
        self.assertEqual(eta.effective_progress, 68)
        after_rounds = eta.remaining_seconds(60)
        self.assertLess(after_rounds, after_plan)

    def test_dynamic_eta_adjusts_to_observed_speed(self) -> None:
        fast = DynamicEta(initial_total_seconds=600, num_rounds=1, progress_percent=50)
        slow = DynamicEta(initial_total_seconds=600, num_rounds=1, progress_percent=50)
        fast_elapsed = 100
        slow_elapsed = 400
        self.assertGreater(
            slow_elapsed + slow.remaining_seconds(slow_elapsed),
            fast_elapsed + fast.remaining_seconds(fast_elapsed),
        )


class PipelineWorkerTests(unittest.TestCase):
    def test_subprocess_stream_manifest_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repository = root / "repo"
            dataset = root / "dataset"
            results = root / "results"
            repository.mkdir()
            sample = dataset / "sample_1"
            sample.mkdir(parents=True)
            (sample / "image.png").touch()
            (repository / "main.py").write_text(
                "import sys\n"
                "from pathlib import Path\n"
                "out = Path(sys.argv[sys.argv.index('--results-dir') + 1])\n"
                "out.mkdir(parents=True, exist_ok=True)\n"
                "print('Step 1: Read the dataset index', flush=True)\n"
                "print('Step 2.4: Data segmentation (disabled)', flush=True)\n"
                "print('Step 3: Feature planning', flush=True)\n"
                "(out / 'round_1').mkdir(exist_ok=True)\n"
                "print('Step 4: Batch feature extraction', flush=True)\n"
                "(out / 'features.csv').write_text('sample_id,f1\\nsample_1,0.5\\n')\n"
                "print('Step 6: Deterministic feature validation', flush=True)\n"
                "(out / 'round_1' / 'round_results.json').write_text('{}')\n"
                "print('[OK] Round 1 complete!', flush=True)\n"
                "print('[DONE] All 1 rounds of feature extraction complete!', flush=True)\n",
                encoding="utf-8",
            )
            config = RunConfig(
                data_root=str(dataset),
                query="Profile cells",
                results_dir=str(results),
                python_executable=sys.executable,
                repository_root=str(repository),
                enable_segmentation=False,
            )
            worker = PipelineWorker(config, scan_dataset(dataset))
            logs: list[str] = []
            stages: list[int] = []
            successes: list[str] = []
            worker.log_line.connect(logs.append)
            worker.stage_changed.connect(lambda index, _key, _title: stages.append(index))
            worker.run_succeeded.connect(lambda _code, path: successes.append(path))

            worker.run()

            self.assertTrue((results / "ui_run_manifest.json").is_file())
            self.assertTrue((results / "ui_console.log").is_file())
            self.assertTrue((results / "features.csv").is_file())
            self.assertEqual(stages[-1], 5)
            self.assertTrue(any("Feature planning" in line for line in logs))
            self.assertEqual(successes, [str(results.resolve())])


class ReuseControllerTests(unittest.TestCase):
    def test_progress_detector_tracks_rounds(self) -> None:
        detector = ReuseProgressDetector()
        self.assertEqual(detector.feed("[Reuse] Round 1/2 · round_1 · 2 code feature(s)"), 10)
        self.assertEqual(detector.feed("[Reuse] Round 2/2 · round_2 · 1 code feature(s)"), 50)
        self.assertEqual(detector.feed("[Reuse] [DONE] Wrote 3 code feature column(s)"), 100)

    def test_reuse_worker_streams_and_writes_log(self) -> None:
        from morphagent_ui.controller import ReuseWorker

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repository = root / "repo"
            dataset = root / "dataset" / "sample_1"
            source = root / "history"
            results = root / "results"
            repository.mkdir()
            dataset.mkdir(parents=True)
            (dataset / "image.png").touch()
            source.mkdir()
            (repository / "reuse_code.py").write_text(
                "import sys\n"
                "from pathlib import Path\n"
                "out = Path(sys.argv[sys.argv.index('--results-dir') + 1])\n"
                "out.mkdir(parents=True, exist_ok=True)\n"
                "print('[Reuse] Round 1/1 · round_1 · 1 code feature(s)', flush=True)\n"
                "(out / 'features.csv').write_text('sample_id,f1\\nsample_1,0.5\\n')\n"
                "(out / 'round_1').mkdir(exist_ok=True)\n"
                "(out / 'round_1' / 'round_results.json').write_text('{}')\n"
                "print('[Reuse] [DONE] Wrote 1 code feature column(s)', flush=True)\n"
                "print(f'Final feature file: {out / \"features.csv\"}', flush=True)\n",
                encoding="utf-8",
            )
            config = ReuseConfig(
                repository_root=str(repository),
                source_results=str(source),
                data_root=str(root / "dataset"),
                results_dir=str(results),
                python_executable=sys.executable,
            )
            worker = ReuseWorker(config)
            logs: list[str] = []
            progress: list[int] = []
            successes: list[str] = []
            worker.log_line.connect(logs.append)
            worker.progress_changed.connect(progress.append)
            worker.run_succeeded.connect(lambda _code, path: successes.append(path))

            worker.run()

            self.assertTrue((results / "ui_console.log").is_file())
            self.assertTrue((results / "features.csv").is_file())
            self.assertTrue(any("[Reuse] Round 1/1" in line for line in logs))
            self.assertIn(100, progress)
            self.assertEqual(successes, [str(results.resolve())])


if __name__ == "__main__":
    unittest.main()
