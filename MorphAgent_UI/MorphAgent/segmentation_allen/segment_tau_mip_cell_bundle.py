#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Single-channel Tau MIP: generate two kinds of segmentation masks consistent with data_test_18.
- mask_cell.tif: Allen MO threshold (large targets) + largest connected component (solid single-cell region)
- mask_bundle.tif: Allen filament_2d coarse scale (same bundle parameters as build_final_seg_WT1)

Must be run in the conda allen environment (depends on aicssegmentation / aicsimageio).
"""

import sys
from pathlib import Path

import numpy as np
from aicsimageio.writers import OmeTiffWriter
from skimage.measure import label

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grid_search_WT1_all_targets import load_mip, segment_mo, segment_filament


def keep_largest_component(mask):
    """Keep only the largest connected component of a binary mask, uint8 0/255."""
    m = np.squeeze(mask)
    if m.ndim != 2:
        return mask
    binary = (m > 0).astype(np.uint8)
    labeled = label(binary, connectivity=1)
    if labeled.max() == 0:
        return (binary * 255).astype(np.uint8)
    areas = np.bincount(labeled.ravel())[1:]
    largest_id = 1 + int(np.argmax(areas))
    out = (labeled == largest_id).astype(np.uint8) * 255
    if mask.ndim == 3:
        return out[np.newaxis, ...]
    return out


def ensure_3d(mask):
    if mask.ndim == 2:
        return mask[np.newaxis, ...]
    return mask


# Consistent with the bundle section in build_final_seg_WT1
PARAMS_BUNDLE = {
    "intensity_norm_param": [0.5, 15],
    "sigma": 1.0,
    "f2_param": [[2.0, 0.02], [2.5, 0.02]],
    "min_size": 600,
}

# Whole-cell MO: a moderately strong setting from the "cell" grid in grid_search
PARAMS_CELL_MO = {
    "intensity_norm_param": [0.5, 15],
    "sigma": 1.5,
    "object_minArea": 800,
    "min_size": 600,
    "global_thresh_method": "tri",
}


def write_mask_cell_bundle(image_tif, seg_dir, verbose=True):
    """Read image_tif and write seg_dir/mask_cell.tif and mask_dir/mask_bundle.tif."""
    image_tif = Path(image_tif)
    seg_dir = Path(seg_dir)
    seg_dir.mkdir(parents=True, exist_ok=True)
    if not image_tif.exists():
        raise FileNotFoundError(image_tif)

    struct_img = load_mip(image_tif)
    cell_raw = segment_mo(struct_img, PARAMS_CELL_MO)
    cell_mask = keep_largest_component(cell_raw)
    bundle_mask = segment_filament(struct_img, PARAMS_BUNDLE)

    for path, arr in (
        (seg_dir / "mask_cell.tif", cell_mask),
        (seg_dir / "mask_bundle.tif", bundle_mask),
    ):
        if path.exists():
            path.unlink()
        arr3 = ensure_3d(np.asarray(arr))
        with OmeTiffWriter(str(path)) as w:
            w.save(arr3)
        if verbose:
            print(f"  wrote {path.name}")


def main():
    import argparse

    p = argparse.ArgumentParser(description="Generate mask_cell / mask_bundle for a single Tau MIP")
    p.add_argument("image_tif", type=Path)
    p.add_argument(
        "-o",
        "--seg-dir",
        type=Path,
        default=None,
        help="Output directory (default: segmentation/ next to the tif)",
    )
    args = p.parse_args()
    seg_dir = args.seg_dir
    if seg_dir is None:
        seg_dir = args.image_tif.parent / "segmentation"
    write_mask_cell_bundle(args.image_tif, seg_dir, verbose=True)
    print("done:", seg_dir)


if __name__ == "__main__":
    main()
