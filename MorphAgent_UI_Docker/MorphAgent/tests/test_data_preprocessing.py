from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import mrcfile
import tifffile
from PIL import Image

from tools.code_executor import CodeExecutor
from tools.image_io import load_image_array
from utils_modules.data_preprocessing import ensure_slices_directory


class TwoDimensionalImagePreprocessingTests(unittest.TestCase):
    def test_opaque_rgba_png_generates_one_slice_per_signal_channel(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            sample = Path(raw) / "sample"
            sample.mkdir()
            image = np.zeros((16, 20, 4), dtype=np.uint8)
            image[..., 0] = 11
            image[..., 1] = 22
            image[..., 2] = 33
            image[..., 3] = 255
            Image.fromarray(image, mode="RGBA").save(sample / "image.png")

            existed, paths = ensure_slices_directory(sample)

            self.assertFalse(existed)
            self.assertEqual(
                [path.name for path in paths],
                [
                    "slice_0000_R.png",
                    "slice_0001_G.png",
                    "slice_0002_B.png",
                ],
            )
            for index, expected in enumerate((11, 22, 33)):
                with Image.open(paths[index]) as channel:
                    self.assertEqual(channel.mode, "L")
                    self.assertEqual(channel.size, (20, 16))
                    self.assertTrue(np.all(np.asarray(channel) == expected))

            mapping = json.loads(
                (sample / "slices" / "channel_mapping.json").read_text(encoding="utf-8")
            )
            self.assertEqual(mapping["num_channels"], 3)
            self.assertEqual(mapping["channel_mapping"]["2"]["marker"], "B")

            existed_again, existing_paths = ensure_slices_directory(sample)
            self.assertTrue(existed_again)
            self.assertEqual(len(existing_paths), 3)

    def test_nonconstant_alpha_is_preserved_as_a_slice(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            sample = Path(raw) / "sample"
            sample.mkdir()
            image = np.zeros((16, 20, 4), dtype=np.uint8)
            image[..., 3] = np.arange(20, dtype=np.uint8)
            Image.fromarray(image, mode="RGBA").save(sample / "image.png")

            _existed, paths = ensure_slices_directory(sample)

            self.assertEqual(paths[-1].name, "slice_0003_A.png")
            self.assertEqual(len(paths), 4)

    def test_grayscale_png_generates_one_slice(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            sample = Path(raw) / "sample"
            sample.mkdir()
            image = np.arange(16 * 20, dtype=np.uint8).reshape(16, 20)
            Image.fromarray(image, mode="L").save(sample / "image.png")

            existed, paths = ensure_slices_directory(sample)

            self.assertFalse(existed)
            self.assertEqual([path.name for path in paths], ["slice_0000.png"])
            with Image.open(paths[0]) as channel:
                self.assertEqual(channel.mode, "L")
                np.testing.assert_array_equal(np.asarray(channel), image)

    def test_mrc_volume_generates_one_slice_per_z_plane(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            sample = Path(raw) / "sample"
            sample.mkdir()
            volume = np.arange(3 * 16 * 20, dtype=np.float32).reshape(3, 16, 20)
            with mrcfile.new(str(sample / "volume.mrc"), overwrite=True) as handle:
                handle.set_data(volume)

            existed, paths = ensure_slices_directory(sample)

            self.assertFalse(existed)
            self.assertEqual(
                [path.name for path in paths],
                ["slice_z0000.png", "slice_z0001.png", "slice_z0002.png"],
            )
            manifest = json.loads(
                (sample / "slices" / "slice_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["axes"], "ZYX")
            self.assertEqual(manifest["source_shape"], [3, 16, 20])
            self.assertEqual(manifest["frame_count"], 3)
            np.testing.assert_array_equal(load_image_array(sample / "volume.mrc"), volume)
            extract = sample / "extract.py"
            extract.write_text(
                "def extract(img):\n    return float(img.shape[0])\n",
                encoding="utf-8",
            )
            success, value, error = CodeExecutor(
                sample,
                conda_env=None,
            ).execute_single_sample(extract, sample / "volume.mrc", [])
            self.assertTrue(success, error)
            self.assertEqual(value, 3.0)

    def test_ome_tiff_preserves_z_and_channel_axes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            sample = Path(raw) / "sample"
            sample.mkdir()
            volume = np.arange(2 * 3 * 16 * 20, dtype=np.uint16).reshape(2, 3, 16, 20)
            tifffile.imwrite(
                sample / "volume.ome.tif",
                volume,
                metadata={"axes": "ZCYX"},
                photometric="minisblack",
            )

            _existed, paths = ensure_slices_directory(sample)

            self.assertEqual(len(paths), 6)
            self.assertEqual(paths[0].name, "slice_z0000_c0000_Channel_0.png")
            self.assertEqual(paths[-1].name, "slice_z0001_c0002_Channel_2.png")
            manifest = json.loads(
                (sample / "slices" / "slice_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["axes"], "ZCYX")
            self.assertEqual(manifest["frame_count"], 6)
            mapping = json.loads(
                (sample / "slices" / "channel_mapping.json").read_text(encoding="utf-8")
            )
            self.assertEqual(mapping["num_channels"], 3)
            self.assertEqual(len(mapping["slice_files"]["0"]), 2)

    def test_plain_tiff_stack_is_not_mistaken_for_channels(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            sample = Path(raw) / "sample"
            sample.mkdir()
            volume = np.arange(3 * 16 * 20, dtype=np.uint16).reshape(3, 16, 20)
            tifffile.imwrite(
                sample / "stack.tif",
                volume,
                photometric="minisblack",
            )

            _existed, paths = ensure_slices_directory(sample)

            self.assertEqual(
                [path.name for path in paths],
                ["slice_z0000.png", "slice_z0001.png", "slice_z0002.png"],
            )
            manifest = json.loads(
                (sample / "slices" / "slice_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["axes"], "ZYX")

    def test_animated_gif_expands_time_and_color_channels(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            sample = Path(raw) / "sample"
            sample.mkdir()
            first = Image.fromarray(
                np.full((16, 20, 3), (10, 20, 30), dtype=np.uint8),
                mode="RGB",
            )
            second = Image.fromarray(
                np.full((16, 20, 3), (40, 50, 60), dtype=np.uint8),
                mode="RGB",
            )
            first.save(
                sample / "movie.gif",
                save_all=True,
                append_images=[second],
                duration=100,
                loop=0,
            )

            _existed, paths = ensure_slices_directory(sample)

            self.assertEqual(len(paths), 6)
            self.assertEqual(paths[0].name, "slice_t0000_c0000_Channel_0.png")
            self.assertEqual(paths[-1].name, "slice_t0001_c0002_Channel_2.png")
            manifest = json.loads(
                (sample / "slices" / "slice_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["axes"], "TCYX")
            self.assertEqual(manifest["frame_count"], 6)


if __name__ == "__main__":
    unittest.main()
