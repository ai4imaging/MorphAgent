from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.segmentation import segment_all_samples


class SegmentAllSamplesTests(unittest.TestCase):
    @staticmethod
    def _sample_with_complete_masks(root: Path) -> tuple[Path, Path]:
        sample = root / "sample_1"
        segmentation = sample / "segmentation"
        segmentation.mkdir(parents=True)
        image = sample / "image.tif"
        image.touch()
        for name in ("cyto.tif", "nuclei.tif", "cytoplasm.tif"):
            (segmentation / name).touch()
        return sample, image

    @patch("tools.segmentation._segmentation_backend", return_value="cellpose")
    @patch("tools.segmentation.segment_image_with_cellpose", return_value=True)
    @patch("utils_helpers.find_image_paths")
    def test_recreate_policy_reruns_even_with_complete_cellpose_masks(
        self,
        find_image_paths,
        segment_image,
        _backend,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _sample, image = self._sample_with_complete_masks(root)
            find_image_paths.return_value = [image]

            result = segment_all_samples(
                ["sample_1"],
                root,
                skip_if_any_segmentation_exists=False,
            )

            self.assertEqual(result, {"sample_1": "success"})
            segment_image.assert_called_once()

    @patch("tools.segmentation.segment_image_with_allen", return_value=True)
    @patch("tools.segmentation.segment_image_with_cellpose", return_value=True)
    def test_reuse_policy_keeps_complete_existing_masks(self, segment_cellpose, segment_allen) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self._sample_with_complete_masks(root)

            result = segment_all_samples(
                ["sample_1"],
                root,
                skip_if_any_segmentation_exists=True,
            )

            self.assertEqual(result, {"sample_1": "skipped_user_seg"})
            segment_cellpose.assert_not_called()
            segment_allen.assert_not_called()

    @patch("tools.segmentation._segmentation_backend", return_value="allen")
    @patch("tools.segmentation.segment_image_with_allen", return_value=False)
    @patch("utils_helpers.find_image_paths")
    def test_allen_failure_skips_sample_without_abort(
        self,
        find_image_paths,
        segment_allen,
        _backend,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            sample = root / "sample_1"
            sample.mkdir()
            image = sample / "image.tif"
            image.touch()
            find_image_paths.return_value = [image]

            result = segment_all_samples(
                ["sample_1"],
                root,
                skip_if_any_segmentation_exists=True,
            )

            self.assertEqual(result, {"sample_1": "skipped_allen_unavailable"})
            segment_allen.assert_called_once()


if __name__ == "__main__":
    unittest.main()
