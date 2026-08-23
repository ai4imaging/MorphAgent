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
    extract_required_mask_stems,
    history_mask_stems,
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


def _mask_order_description(stems: tuple[str, ...]) -> str:
    """Mirror the block tools/data_statistics.py writes into a run summary."""

    lines = [
        "\n**CRITICAL: Segmentation is passed as a dict `seg` (key = filename stem)**",
        "Access masks by key only. Available keys and semantics:\n",
    ]
    for stem in stems:
        lines.append(f'  - **seg["{stem}"]**: `{stem}.tif` — **segmentation mask from `{stem}.tif`**')
    lines.append('\n**Important:** Use `seg.get("key")` or `seg["key"]`; do NOT use position or index.')
    lines.append("")
    return "\n".join(lines)


def _write_segmentation_summary(
    source: Path,
    stems: tuple[str, ...],
    *,
    enabled: bool = True,
) -> None:
    (source / "segmentation_summary.json").write_text(
        json.dumps(
            {
                "total_samples": 2,
                "successful": 0,
                "skipped_user_seg": 2,
                "failed": 0,
                "results": {},
                "mask_order_description": _mask_order_description(stems) if enabled else "",
                "segmentation_enabled": enabled,
            }
        ),
        encoding="utf-8",
    )


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
    def test_extracts_direct_aliased_and_helper_mask_keys(self) -> None:
        code = '''
def extract_all(img, seg):
    seg_dict = seg if isinstance(seg, dict) else {}
    def get_mask(key):
        return seg.get(key)
    return {
        "a": seg["mask_cell"],
        "b": seg_dict.get("mask_nucleus"),
        "c": get_mask("mask_filament"),
    }
'''
        self.assertEqual(
            extract_required_mask_stems(code),
            ("mask_cell", "mask_filament", "mask_nucleus"),
        )

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

            (data / "dataset" / "sample_b" / "image.tif").unlink()
            blockers = diagnose_reuse_inputs(source, data)
            self.assertTrue(
                any("sample_b" in item and "primary images" in item.lower() for item in blockers)
            )

    def test_diagnose_requires_every_mask_for_every_sample(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = _write_source_results(root)
            merged = source / "round_1" / "merged_features" / "extract_all.py"
            merged.write_text(
                '''
def extract_all(img, seg):
    return {
        "kept_feature": float(seg.get("mask_cell") is not None),
        "dropped_feature": float(seg.get("mask_nucleus") is not None),
    }
''',
                encoding="utf-8",
            )
            data = _write_dataset(root)
            blockers = diagnose_reuse_inputs(source, data)
            self.assertEqual(len(blockers), 1)
            self.assertIn("Round 1", blockers[0])
            self.assertIn("mask_cell, mask_nucleus", blockers[0])
            self.assertIn("sample_a, sample_b", blockers[0])
            self.assertEqual(
                summarize_source_results(source)["required_masks"],
                ["mask_cell", "mask_nucleus"],
            )

            output = root / "must_not_start"
            with self.assertRaisesRegex(ValueError, "mask_nucleus"):
                run_code_reuse(source, data, output_dir=output)
            self.assertFalse(output.exists())

            for sample_id in ("sample_a", "sample_b"):
                (data / "dataset" / sample_id / "segmentation" / "mask_nucleus.tif").write_bytes(
                    b"II*\x00"
                )
            self.assertEqual(diagnose_reuse_inputs(source, data), [])


CANDIDATE_MASK_CODE = '''
def extract_all(img, seg):
    mask = None
    for key in ("mask_neurite", "mask_axon", "mask_cell"):
        if key in seg:
            mask = seg.get(key)
            break
    return {
        "kept_feature": 1.0 if mask is not None else 0.0,
        "dropped_feature": 2.0,
    }
'''

TWO_MASK_CODE = '''
def extract_all(img, seg):
    return {
        "kept_feature": float(seg.get("mask_cell") is not None),
        "dropped_feature": float(seg.get("mask_nucleus") is not None),
    }
'''


class HistoryMaskInventoryTests(unittest.TestCase):
    """The source run's recorded mask inventory is the primary requirement source."""

    def _source_with_code(self, root: Path, code: str) -> Path:
        source = _write_source_results(root)
        (source / "round_1" / "merged_features" / "extract_all.py").write_text(
            code, encoding="utf-8"
        )
        return source

    def test_reads_stems_from_summary_without_matching_the_usage_hint(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            source = _write_source_results(Path(raw))
            _write_segmentation_summary(source, ("mask_cell", "mask_nucleus"))
            # The trailing 'Use seg["key"]' hint must not be mistaken for a mask.
            self.assertEqual(history_mask_stems(source), ("mask_cell", "mask_nucleus"))

    def test_unknown_inventory_when_summary_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            source = _write_source_results(Path(raw))
            self.assertIsNone(history_mask_stems(source))

    def test_disabled_segmentation_reports_an_empty_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            source = _write_source_results(Path(raw))
            _write_segmentation_summary(source, (), enabled=False)
            self.assertEqual(history_mask_stems(source), ())

    def test_optional_candidate_keys_do_not_block_when_history_is_known(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = self._source_with_code(root, CANDIDATE_MASK_CODE)
            data = _write_dataset(root)

            # Without the inventory, every probed candidate looks mandatory.
            blockers = diagnose_reuse_inputs(source, data)
            self.assertTrue(any("mask_axon" in item for item in blockers))

            _write_segmentation_summary(source, ("mask_cell",))
            self.assertEqual(diagnose_reuse_inputs(source, data), [])
            self.assertEqual(summarize_source_results(source)["required_masks"], ["mask_cell"])

    def test_masks_the_history_run_had_are_still_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = self._source_with_code(root, TWO_MASK_CODE)
            _write_segmentation_summary(source, ("mask_cell", "mask_nucleus"))
            data = _write_dataset(root)

            blockers = diagnose_reuse_inputs(source, data)
            self.assertEqual(len(blockers), 1)
            self.assertIn("that the history run used", blockers[0])
            self.assertIn("mask_nucleus missing in sample_a, sample_b", blockers[0])

            for sample_id in ("sample_a", "sample_b"):
                (data / "dataset" / sample_id / "segmentation" / "mask_nucleus.tif").write_bytes(
                    b"II*\x00"
                )
            self.assertEqual(diagnose_reuse_inputs(source, data), [])

    def test_history_run_without_masks_never_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = self._source_with_code(root, TWO_MASK_CODE)
            _write_segmentation_summary(source, (), enabled=False)
            data = _write_dataset(root)
            for sample_id in ("sample_a", "sample_b"):
                (data / "dataset" / sample_id / "segmentation" / "mask_cell.tif").unlink()

            self.assertEqual(diagnose_reuse_inputs(source, data), [])
            self.assertEqual(summarize_source_results(source)["required_masks"], [])


class CodeReuseExecutionTests(unittest.TestCase):
    def test_run_code_reuse_writes_matrix_registry_and_skips_models(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = _write_source_results(root)
            _write_segmentation_summary(source, ("mask_cell",))
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
            # Reuse output must stay reusable: it inherits the mask inventory.
            self.assertEqual(history_mask_stems(output), ("mask_cell",))
            # Source tree must remain untouched.
            self.assertFalse((source / "reuse_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
