from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from tools.code_executor import ExtractionResult
from tools.code_reuse import (
    diagnose_reuse_inputs,
    discover_reuse_rounds,
    run_code_reuse,
    summarize_source_results,
)


MERGED_CODE_ROUND_1 = '''
def extract_all(img, seg):
    return {
        "kept_feature": 1.5,
        "dropped_feature": 2.5,
    }
'''

MERGED_CODE_ROUND_2 = '''
def extract_all(img, seg):
    return {
        "round_two_feature": 3.5,
    }
'''


def _write_source_results(root: Path) -> Path:
    source = root / "history_run"
    for round_number, code, features in (
        (
            1,
            MERGED_CODE_ROUND_1,
            [
                {
                    "name": "kept_feature",
                    "method": "code",
                    "category": "intensity",
                    "description": "kept",
                },
                {
                    "name": "dropped_feature",
                    "method": "code",
                    "category": "morphology",
                    "description": "dropped later",
                },
                {
                    "name": "vlm_skip_me",
                    "method": "vlm",
                    "category": "other",
                    "description": "should be skipped",
                },
            ],
        ),
        (
            2,
            MERGED_CODE_ROUND_2,
            [
                {
                    "name": "round_two_feature",
                    "method": "code",
                    "category": "texture",
                    "description": "second round",
                },
            ],
        ),
    ):
        round_dir = source / f"round_{round_number}"
        merged_dir = round_dir / "merged_features"
        merged_dir.mkdir(parents=True)
        (merged_dir / "extract_all.py").write_text(code, encoding="utf-8")
        (round_dir / "feature_plan.json").write_text(
            json.dumps({"features": features}),
            encoding="utf-8",
        )
        (round_dir / "round_results.json").write_text(
            json.dumps({"round": round_number}),
            encoding="utf-8",
        )

    registry = {
        "version": 1,
        "entries": [
            {
                "feature_id": "round_1:kept_feature",
                "name": "kept_feature",
                "method": "code",
                "category": "intensity",
                "description": "kept",
                "first_round": 1,
                "latest_round": 1,
                "current_status": "retained",
                "live": True,
                "actual_column_name": "kept_feature",
                "decision_history": [{"status": "retained", "reason_codes": ["passed_hard_filters"]}],
            },
            {
                "feature_id": "round_1:dropped_feature",
                "name": "dropped_feature",
                "method": "code",
                "category": "morphology",
                "description": "dropped later",
                "first_round": 1,
                "latest_round": 1,
                "current_status": "dropped",
                "live": False,
                "actual_column_name": "dropped_feature",
                "decision_history": [{"status": "dropped", "reason_codes": ["low_variability"]}],
            },
            {
                "feature_id": "round_1:vlm_skip_me",
                "name": "vlm_skip_me",
                "method": "vlm",
                "category": "other",
                "description": "skip",
                "first_round": 1,
                "latest_round": 1,
                "current_status": "retained",
                "live": True,
                "actual_column_name": "vlm_skip_me",
                "decision_history": [{"status": "retained", "reason_codes": ["passed_hard_filters"]}],
            },
            {
                "feature_id": "round_2:round_two_feature",
                "name": "round_two_feature",
                "method": "code",
                "category": "texture",
                "description": "second round",
                "first_round": 2,
                "latest_round": 2,
                "current_status": "retained",
                "live": True,
                "actual_column_name": "round_two_feature",
                "decision_history": [{"status": "retained", "reason_codes": ["passed_hard_filters"]}],
            },
        ],
        "live_feature_ids": ["round_1:kept_feature", "round_2:round_two_feature"],
        "feature_id_to_column": {
            "round_1:kept_feature": "kept_feature",
            "round_2:round_two_feature": "round_two_feature",
        },
    }
    (source / "feature_registry.json").write_text(json.dumps(registry), encoding="utf-8")
    return source


def _write_dataset(root: Path) -> Path:
    data_root = root / "project"
    for sample_id in ("sample_a", "sample_b"):
        sample = data_root / "dataset" / sample_id
        sample.mkdir(parents=True)
        (sample / "image.tif").write_bytes(b"II*\x00")  # tiny placeholder
        seg = sample / "segmentation"
        seg.mkdir()
        (seg / "mask_cell.tif").write_bytes(b"II*\x00")
    return data_root


class CodeReuseDiscoveryTests(unittest.TestCase):
    def test_discover_rounds_keeps_all_code_and_skips_vlm(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            source = _write_source_results(Path(raw))
            rounds, skipped = discover_reuse_rounds(source)
            self.assertEqual([item.round_number for item in rounds], [1, 2])
            self.assertEqual(rounds[0].feature_names, ("kept_feature", "dropped_feature"))
            self.assertEqual(rounds[0].skipped_vlm_names, ("vlm_skip_me",))
            self.assertEqual(rounds[1].feature_names, ("round_two_feature",))
            self.assertEqual(skipped, [])
            summary = summarize_source_results(source)
            self.assertEqual(summary["code_feature_count"], 3)

    def test_diagnose_requires_usable_source_and_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            blockers = diagnose_reuse_inputs("", "")
            self.assertTrue(any("results" in item.lower() for item in blockers))
            self.assertTrue(any("dataset" in item.lower() for item in blockers))
            source = _write_source_results(root)
            data = _write_dataset(root)
            self.assertEqual(diagnose_reuse_inputs(source, data), [])


class CodeReuseExecutionTests(unittest.TestCase):
    def test_run_code_reuse_writes_matrix_registry_and_skips_models(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = _write_source_results(root)
            data = _write_dataset(root)
            output = root / "reuse_out"
            progress: list[str] = []

            def fake_execute(merged_code, feature_names, sample_ids, *_args, **_kwargs):
                values = {}
                for sample_id in sample_ids:
                    payload = {}
                    for name in feature_names:
                        payload[name] = 10.0 if sample_id.endswith("a") else 20.0
                    values[sample_id] = payload
                # Force a sample-level failure to verify isolation.
                if "dropped_feature" in feature_names:
                    values.pop("sample_b", None)
                    return ExtractionResult(
                        values=values,
                        errors={"sample_b": "synthetic failure"},
                    )
                return ExtractionResult(values=values, errors={})

            with mock.patch("tools.code_reuse.execute_reused_merged_code", side_effect=fake_execute) as mocked:
                with mock.patch.dict("sys.modules", {"openai": None, "langchain_openai": None}):
                    summary = run_code_reuse(
                        source_results=source,
                        data_root=data,
                        output_dir=output,
                        progress_callback=progress.append,
                    )

            self.assertEqual(mocked.call_count, 2)
            self.assertTrue((output / "features.csv").is_file())
            frame = pd.read_csv(output / "features.csv")
            self.assertEqual(list(frame["sample_id"]), ["sample_a", "sample_b"])
            self.assertIn("kept_feature", frame.columns)
            self.assertIn("dropped_feature", frame.columns)
            self.assertIn("round_two_feature", frame.columns)
            self.assertNotIn("vlm_skip_me", frame.columns)
            # Round order preserved in column accumulation.
            self.assertLess(
                list(frame.columns).index("kept_feature"),
                list(frame.columns).index("round_two_feature"),
            )
            self.assertTrue(pd.isna(frame.loc[frame["sample_id"] == "sample_b", "dropped_feature"]).iloc[0])

            registry = json.loads((output / "feature_registry.json").read_text(encoding="utf-8"))
            methods = {entry["name"]: entry["method"] for entry in registry["entries"]}
            statuses = {entry["name"]: entry["current_status"] for entry in registry["entries"]}
            self.assertEqual(methods.get("kept_feature"), "code")
            self.assertEqual(methods.get("dropped_feature"), "code")
            self.assertNotIn("vlm_skip_me", methods)
            self.assertEqual(statuses.get("dropped_feature"), "dropped")
            self.assertEqual(statuses.get("kept_feature"), "retained")

            manifest = json.loads((output / "reuse_manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["llm_calls"])
            self.assertFalse(manifest["vlm_calls"])
            self.assertEqual(manifest["feature_columns"], summary.feature_columns)
            self.assertTrue(any("[Reuse] Round 1/2" in line for line in progress))
            self.assertTrue((output / "round_1" / "round_results.json").is_file())
            self.assertTrue((output / "round_2" / "merged_features" / "extract_all.py").is_file())
            # Source tree must remain untouched.
            self.assertFalse((source / "reuse_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
