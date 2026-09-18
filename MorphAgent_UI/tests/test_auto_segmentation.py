from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import tifffile

from tools.auto_segmentation import (
    AutoSegmentationOutcome,
    dataset_has_segmentation,
    run_auto_segmentation,
)
from tools.auto_segmentation_runner import resolve_keys, to_display_plane
from tools.data_statistics import _generate_mask_order_description
from tools.segmentation import list_segmentation_files

SAMPLE_IDS = ["sample_a", "sample_b", "sample_c"]

PLAN_RESPONSE = json.dumps(
    {
        "strategy": "Threshold the DNA channel, then grow cell bodies from the nuclei.",
        "masks": [
            {
                "name": "mask_nucleus",
                "target": "nuclei",
                "semantics": "nucleus (instance labels)",
                "channel_hint": "channel 0, DNA",
                "rationale": "per-nucleus morphology is the readout",
            },
            {
                "name": "mask_cell",
                "target": "whole cells",
                "semantics": "cell body (whole cell)",
                "channel_hint": "channel 1",
                "rationale": "needed to normalise intensities per cell",
            },
        ],
    }
)

GOOD_CODE = '''```python
import numpy as np


def segment(img):
    from scipy import ndimage as ndi

    arr = np.asarray(img)
    dna = arr[0].astype(np.float64)
    nuclei, _ = ndi.label(dna > dna.mean() + dna.std())
    body = arr[1].astype(np.float64)
    cells, _ = ndi.label(body > body.mean())
    return {"mask_nucleus": nuclei, "mask_cell": cells}
```'''

CRASHING_CODE = '''```python
def segment(img):
    raise RuntimeError("wrong channel index")
```'''

EMPTY_MASK_CODE = '''```python
import numpy as np


def segment(img):
    shape = np.asarray(img).shape[1:]
    return {"mask_nucleus": np.zeros(shape, dtype=np.uint8),
            "mask_cell": np.zeros(shape, dtype=np.uint8)}
```'''


def _vlm_reply(passed: bool, feedback: str = "") -> str:
    return json.dumps(
        {
            "masks": {
                "mask_nucleus": {"passed": passed, "reason": "outlines follow the nuclei"},
                "mask_cell": {"passed": passed, "reason": "outlines follow the cells"},
            },
            "passed": passed,
            "feedback": feedback,
        }
    )


class AutoSegmentationTests(unittest.TestCase):
    def _dataset(self, root: Path) -> Path:
        """Three samples of synthetic two-channel data with separable blobs."""
        data_root = root / "dataset"
        rng = np.random.default_rng(7)
        for index, sample_id in enumerate(SAMPLE_IDS):
            sample = data_root / sample_id
            sample.mkdir(parents=True)
            image = rng.integers(0, 200, (2, 48, 48), dtype=np.uint16)
            image[0, 8:20, 8:20] = 4000
            image[0, 28:40, 28:40] = 4000
            image[1, 4:24, 4:24] = 3000
            image[1, 24:44, 24:44] = 3000
            tifffile.imwrite(sample / "image.tif", image)
        return data_root

    def _llm(self, responses: list[str]) -> Mock:
        llm = Mock()
        llm.invoke.side_effect = [SimpleNamespace(content=text) for text in responses]
        return llm

    def _run(self, data_root: Path, results_dir: Path, llm: Mock, vlm) -> AutoSegmentationOutcome:
        with patch("config.make_chat_llm", return_value=llm), patch(
            "tools.auto_segmentation._vlm_freeform", side_effect=vlm
        ):
            return run_auto_segmentation(
                sample_ids=SAMPLE_IDS,
                data_root=data_root,
                user_query="How does the compound change nuclear shape?",
                dataset_description="Two channels: 0 DNA, 1 cell body.",
                results_dir=results_dir,
                max_rounds=3,
            )

    # -- happy path ---------------------------------------------------------

    def test_masks_reach_every_sample_and_carry_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            data_root = self._dataset(root)
            results = root / "results"

            outcome = self._run(
                data_root,
                results,
                self._llm([PLAN_RESPONSE, GOOD_CODE]),
                lambda *args, **kwargs: _vlm_reply(True),
            )

            self.assertTrue(outcome.applied)
            self.assertEqual(outcome.vlm_status, "passed")
            self.assertEqual(outcome.rounds_used, 1)
            self.assertEqual([mask.name for mask in outcome.masks], ["mask_nucleus", "mask_cell"])

            # Every sample must expose the masks through the pipeline's own scan,
            # otherwise the seg dict would be missing keys for some samples.
            for sample_id in SAMPLE_IDS:
                found = dict(list_segmentation_files(data_root / sample_id))
                self.assertEqual(sorted(found), ["mask_cell.tif", "mask_nucleus.tif"])
                self.assertEqual(outcome.results[sample_id], "success")

            # The planner's wording must survive into the prompt block that the
            # feature planner and code generator read.
            seg_files = list_segmentation_files(data_root / SAMPLE_IDS[0])
            description = _generate_mask_order_description(
                [
                    {"index": index, "name": name, "stem": path.stem}
                    for index, (name, path) in enumerate(seg_files, start=1)
                ],
                semantics_override=outcome.semantics,
            )
            self.assertIn('seg["mask_nucleus"]', description)
            self.assertIn("nucleus (instance labels)", description)
            self.assertIn("cell body (whole cell)", description)

    def test_readme_semantics_are_parsed_back_out(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            data_root = self._dataset(root)

            outcome = self._run(
                data_root,
                root / "results",
                self._llm([PLAN_RESPONSE, GOOD_CODE]),
                lambda *args, **kwargs: _vlm_reply(True),
            )
            self.assertTrue(outcome.applied)

            from tools.data_statistics import _parse_segmentation_semantics

            semantics = _parse_segmentation_semantics(None, data_root / SAMPLE_IDS[0])
            self.assertEqual(semantics.get("mask_nucleus"), "nucleus (instance labels)")
            self.assertEqual(semantics.get("mask_cell"), "cell body (whole cell)")

    def test_artifacts_record_the_decision(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            data_root = self._dataset(root)
            results = root / "results"

            self._run(
                data_root,
                results,
                self._llm([PLAN_RESPONSE, GOOD_CODE]),
                lambda *args, **kwargs: _vlm_reply(True),
            )

            artifacts = results / "auto_segmentation"
            self.assertTrue((artifacts / "mask_plan.json").is_file())
            self.assertTrue((artifacts / "segment_code.py").is_file())
            summary = json.loads((artifacts / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["applied"])
            self.assertEqual(summary["vlm_status"], "passed")
            self.assertEqual(len(summary["masks"]), 2)
            self.assertTrue((artifacts / "round_1" / "preview" / "original.png").is_file())
            self.assertTrue((artifacts / "round_1" / "preview" / "overlay_mask_nucleus.png").is_file())

    # -- retry behaviour ----------------------------------------------------

    def test_crashing_code_is_retried_with_the_error(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            data_root = self._dataset(root)
            llm = self._llm([PLAN_RESPONSE, CRASHING_CODE, GOOD_CODE])

            outcome = self._run(
                data_root, root / "results", llm, lambda *args, **kwargs: _vlm_reply(True)
            )

            self.assertTrue(outcome.applied)
            self.assertEqual(outcome.rounds_used, 2)
            # The failing code and its error must be replayed to the LLM.
            retry_prompt = llm.invoke.call_args_list[2][0][0][0].content
            self.assertIn("wrong channel index", retry_prompt)
            self.assertIn("Previous attempts", retry_prompt)

    def test_degenerate_masks_never_reach_the_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            data_root = self._dataset(root)
            llm = self._llm([PLAN_RESPONSE, EMPTY_MASK_CODE, GOOD_CODE])

            outcome = self._run(
                data_root, root / "results", llm, lambda *args, **kwargs: _vlm_reply(True)
            )

            self.assertTrue(outcome.applied)
            self.assertEqual(outcome.rounds_used, 2)
            retry_prompt = llm.invoke.call_args_list[2][0][0][0].content
            self.assertIn("is empty", retry_prompt)

    def test_rejected_masks_are_applied_after_the_last_round(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            data_root = self._dataset(root)
            llm = self._llm([PLAN_RESPONSE, GOOD_CODE, GOOD_CODE, GOOD_CODE])

            outcome = self._run(
                data_root,
                root / "results",
                llm,
                lambda *args, **kwargs: _vlm_reply(False, "the nuclei outlines are too tight"),
            )

            # "Not passing the check still uses the current code": three rounds are
            # spent, then the last runnable segmentation is applied regardless.
            self.assertTrue(outcome.applied)
            self.assertEqual(outcome.rounds_used, 3)
            self.assertEqual(outcome.vlm_status, "rejected")
            for sample_id in SAMPLE_IDS:
                self.assertTrue(list_segmentation_files(data_root / sample_id))

            feedback_prompt = llm.invoke.call_args_list[3][0][0][0].content
            self.assertIn("too tight", feedback_prompt)

    def test_unreachable_vlm_does_not_burn_rounds(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            data_root = self._dataset(root)

            outcome = self._run(
                data_root,
                root / "results",
                self._llm([PLAN_RESPONSE, GOOD_CODE]),
                Mock(side_effect=RuntimeError("no VLM credentials")),
            )

            self.assertTrue(outcome.applied)
            self.assertEqual(outcome.rounds_used, 1)
            self.assertEqual(outcome.vlm_status, "unverified")

    def test_unparsable_plan_falls_back_to_a_foreground_mask(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            data_root = self._dataset(root)
            single_mask_code = '''```python
import numpy as np


def segment(img):
    arr = np.asarray(img)[0].astype(np.float64)
    return {"mask_foreground": arr > arr.mean() + arr.std()}
```'''

            outcome = self._run(
                data_root,
                root / "results",
                self._llm(["I am afraid I cannot help with that.", single_mask_code]),
                lambda *args, **kwargs: json.dumps({"passed": True, "masks": {}}),
            )

            self.assertTrue(outcome.applied)
            self.assertEqual([mask.name for mask in outcome.masks], ["mask_foreground"])

    # -- give up cleanly ----------------------------------------------------

    def test_dataset_is_untouched_when_no_code_can_be_produced(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            data_root = self._dataset(root)

            outcome = self._run(
                data_root,
                root / "results",
                self._llm([PLAN_RESPONSE, "no code here", "still none", "sorry"]),
                lambda *args, **kwargs: _vlm_reply(True),
            )

            self.assertFalse(outcome.applied)
            self.assertEqual(outcome.rounds_used, 3)
            for sample_id in SAMPLE_IDS:
                self.assertEqual(list_segmentation_files(data_root / sample_id), [])

    def test_llm_failure_is_swallowed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            data_root = self._dataset(root)
            llm = Mock()
            llm.invoke.side_effect = RuntimeError("API down")

            outcome = self._run(
                data_root, root / "results", llm, lambda *args, **kwargs: _vlm_reply(True)
            )

            self.assertFalse(outcome.applied)
            for sample_id in SAMPLE_IDS:
                self.assertEqual(list_segmentation_files(data_root / sample_id), [])

    def test_missing_images_are_reported_not_raised(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            empty_root = root / "dataset"
            for sample_id in SAMPLE_IDS:
                (empty_root / sample_id).mkdir(parents=True)

            outcome = run_auto_segmentation(
                sample_ids=SAMPLE_IDS,
                data_root=empty_root,
                user_query="anything",
                results_dir=root / "results",
            )
            self.assertFalse(outcome.applied)
            self.assertIn("no readable image", outcome.reason)

    def test_no_samples_returns_cleanly(self) -> None:
        outcome = run_auto_segmentation(sample_ids=[], data_root=Path("/nonexistent"), user_query="x")
        self.assertFalse(outcome.applied)

    # -- gate ---------------------------------------------------------------

    def test_existing_masks_short_circuit_the_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            data_root = self._dataset(root)
            self.assertFalse(dataset_has_segmentation(SAMPLE_IDS, data_root))

            seg = data_root / SAMPLE_IDS[1] / "segmentation"
            seg.mkdir()
            tifffile.imwrite(seg / "mask_user.tif", np.zeros((4, 4), dtype=np.uint8))
            self.assertTrue(dataset_has_segmentation(SAMPLE_IDS, data_root))

    def test_preview_only_names_do_not_count_as_masks(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            data_root = self._dataset(root)
            seg = data_root / SAMPLE_IDS[0] / "segmentation"
            seg.mkdir()
            tifffile.imwrite(seg / "segmentation_visualization.tif", np.zeros((4, 4), dtype=np.uint8))
            self.assertFalse(dataset_has_segmentation(SAMPLE_IDS, data_root))


class RunnerHelperTests(unittest.TestCase):
    def test_plane_reduction_matches_each_layout(self) -> None:
        self.assertEqual(to_display_plane(np.zeros((32, 48))).shape, (32, 48))
        self.assertEqual(to_display_plane(np.zeros((3, 32, 48))).shape, (32, 48))
        self.assertEqual(to_display_plane(np.zeros((32, 48, 3))).shape, (32, 48))
        self.assertEqual(to_display_plane(np.zeros((40, 32, 48))).shape, (32, 48))
        self.assertEqual(to_display_plane(np.zeros((1, 32, 48))).shape, (32, 48))
        self.assertEqual(to_display_plane(np.zeros((2, 3, 32, 48))).shape, (32, 48))

    def test_name_similarity_beats_return_order(self) -> None:
        nucleus = np.array([[1]])
        cell = np.array([[2]])
        resolved, _ = resolve_keys(
            {"nuclei": nucleus, "cells": cell}, ["mask_cell", "mask_nucleus"]
        )
        self.assertIs(resolved["mask_nucleus"], nucleus)
        self.assertIs(resolved["mask_cell"], cell)

    def test_ambiguous_names_stay_unmatched(self) -> None:
        resolved, _ = resolve_keys(
            {"alpha": np.array([[1]]), "beta": np.array([[2]])},
            ["mask_cell", "mask_nucleus"],
        )
        self.assertEqual(resolved, {})


if __name__ == "__main__":
    unittest.main()
