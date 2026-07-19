from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from morphagent_ui.controller import PipelineWorker, StageDetector, artifact_snapshot
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
        self.assertEqual(detector.feed("✅ Round 1 complete!"), 5)
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
                "print('✅ Round 1 complete!', flush=True)\n",
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


if __name__ == "__main__":
    unittest.main()
