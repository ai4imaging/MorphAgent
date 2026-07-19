#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mitochondria segmentation: small punctate structures, each object kept separate, each object shown in a different color in the visualization.

See the PARAMS comments inside the script for tunable parameters and their meaning; see PARAMETERS_REFERENCE.md for details.
"""

import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.morphology import remove_small_objects, dilation, ball, disk, binary_closing
from skimage.measure import label
from skimage.color import label2rgb
from skimage.feature import peak_local_max
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from skimage.segmentation import watershed

from aicsimageio import AICSImage
from aicssegmentation.core.pre_processing_utils import (
    intensity_normalization,
    image_smoothing_gaussian_slice_by_slice,
)
from aicssegmentation.core.seg_dot import dot_2d_slice_by_slice_wrapper
from aicssegmentation.core.MO_threshold import MO
from aicsimageio.writers import OmeTiffWriter


# Input/output are taken from CLI args (see main()).


# ========== Tunable parameters ==========
# mode="blocks": first ensure large bright regions are segmented (MO threshold + large min_size)
# mode="dots": punctate structures (dot filter), easily fragmented into small dots
# See PARAMETERS_REFERENCE.md for details

# Final parameters: derived from run_008,012,044,048,056,060,068,072 in grid_search_blocks (see mitochondria_final_params.md)
PARAMS = {
    "mode": "blocks",
    "intensity_norm_param": [0.5, 15],
    "gaussian_smoothing_sigma": 1.5,
    "global_thresh_method": "tri",
    "object_minArea": 100,
    "min_size": 150,
    "use_watershed": True,
    "watershed_min_distance": 8,
    # fallback for dots mode
    "s2_param": [[1.0, 0.08], [1.5, 0.06]],
}


def load_tif_channel(image_path):
    """Load a single-channel TIFF and return (Z,Y,X) float32 in [0,1]."""
    print(f"Loading: {image_path}")
    img = AICSImage(str(image_path))
    data = img.get_image_data()
    while data.ndim > 3 and data.shape[0] == 1:
        data = data.squeeze(0)
    if data.ndim == 3:
        data = data[np.newaxis, ...]
    elif data.ndim == 2:
        data = data[np.newaxis, np.newaxis, ...]
    if data.ndim == 4:
        data = data[0]
    ch = data.astype(np.float32)
    minv, maxv = ch.min(), ch.max()
    if maxv > minv:
        ch = (ch - minv) / (maxv - minv)
    else:
        ch = np.zeros_like(ch)
    if ch.ndim == 2:
        ch = ch[np.newaxis, ...]
    print(f"  Shape: {ch.shape}")
    return ch


def _instance_from_bw(bw, use_watershed, watershed_min_distance):
    """Derive instance labels from binary bw (optionally use watershed to split touching objects)."""
    if use_watershed and bw.any():
        bw_2d = bw[0] if bw.ndim == 3 else bw
        distance = distance_transform_edt(bw_2d)
        peaks = peak_local_max(
            distance,
            labels=bw_2d,
            min_distance=watershed_min_distance,
            indices=False,
        )
        selem = disk(1) if peaks.ndim == 2 else ball(1)
        markers = label(dilation(peaks, selem))
        if markers.ndim == 3:
            markers = markers[0]
        ws = watershed(-distance, markers, mask=bw_2d, watershed_line=True)
        ws = np.where(ws > 0, ws, 0)
        if bw.ndim == 3:
            instance_map = np.zeros_like(bw, dtype=np.int32)
            instance_map[0] = ws
        else:
            instance_map = ws.astype(np.int32)
    else:
        instance_map = label(bw, connectivity=1)
        if instance_map.ndim == 3 and instance_map.shape[0] == 1:
            instance_map = instance_map[0]
    return instance_map


def segment_mitochondria_blocks(struct_img, params):
    """Block segmentation: MO threshold + large min_size, first ensuring large bright regions are segmented."""
    intensity_norm_param = params["intensity_norm_param"]
    sigma = params["gaussian_smoothing_sigma"]
    global_thresh_method = params.get("global_thresh_method", "tri")
    object_minArea = params.get("object_minArea", 200)
    min_size = params.get("min_size", 300)
    use_watershed = params.get("use_watershed", False)
    watershed_min_distance = params.get("watershed_min_distance", 5)

    print("\n[1] Intensity normalization...")
    norm = intensity_normalization(struct_img, scaling_param=intensity_norm_param)
    print("[2] Gaussian smoothing...")
    smooth = image_smoothing_gaussian_slice_by_slice(norm, sigma=sigma)
    print("[3] MO threshold (blocks)...")
    bw = MO(
        smooth,
        global_thresh_method=global_thresh_method,
        object_minArea=object_minArea,
        return_object=False,
    )
    bw = bw > 0
    print("[4] Keep only large blocks (min_size=%d), fill holes..." % min_size)
    bw = remove_small_objects(bw, min_size=min_size, connectivity=1, in_place=False)
    if bw.ndim == 3:
        bw = np.array([binary_fill_holes(s) for s in bw])
    else:
        bw = binary_fill_holes(bw)

    print("[5] Instance labeling (optionally use watershed to split touching blocks)...")
    instance_map = _instance_from_bw(bw, use_watershed, watershed_min_distance)
    n_objects = int(np.max(instance_map))
    print(f"  Detected {n_objects} block objects")
    bw_final = (instance_map > 0).astype(np.uint8) * 255
    if bw_final.ndim == 2:
        bw_final = bw_final[np.newaxis, ...]
    if instance_map.ndim == 2:
        instance_map = instance_map[np.newaxis, ...]
    return bw_final, instance_map, n_objects


def segment_mitochondria(struct_img, params):
    """Depending on mode, choose block segmentation (blocks) or punctate segmentation (dots); returns binary and instance labels."""
    mode = params.get("mode", "blocks")
    if mode == "blocks":
        return segment_mitochondria_blocks(struct_img, params)

    # dots mode: punctate structures
    intensity_norm_param = params["intensity_norm_param"]
    sigma = params["gaussian_smoothing_sigma"]
    s2_param = params["s2_param"]
    min_size = params["min_size"]
    use_watershed = params["use_watershed"]
    watershed_min_distance = params.get("watershed_min_distance", 2)

    print("\n[1] Intensity normalization...")
    norm = intensity_normalization(struct_img, scaling_param=intensity_norm_param)
    print("[2] Gaussian smoothing...")
    smooth = image_smoothing_gaussian_slice_by_slice(norm, sigma=sigma)
    print("[3] Dot filter (dot_2d_slice_by_slice_wrapper)...")
    bw = dot_2d_slice_by_slice_wrapper(smooth, s2_param)
    bw = bw > 0
    print("[4] Fill holes, remove small objects...")
    if bw.ndim == 3:
        bw = np.array([binary_fill_holes(s) for s in bw])
    else:
        bw = binary_fill_holes(bw)
    bw = remove_small_objects(bw, min_size=min_size, connectivity=1, in_place=False)
    if params.get("use_closing") and params.get("closing_radius", 1) >= 1:
        r = int(params.get("closing_radius", 1))
        if bw.ndim == 3:
            bw = np.array([binary_closing(s, selem=disk(r)) for s in bw])
        else:
            bw = binary_closing(bw, selem=disk(r))

    print("[5] Instance labeling...")
    instance_map = _instance_from_bw(bw, use_watershed, watershed_min_distance)
    n_objects = int(np.max(instance_map))
    print(f"  Detected {n_objects} separate objects")
    bw_final = (instance_map > 0).astype(np.uint8) * 255
    if bw_final.ndim == 2:
        bw_final = bw_final[np.newaxis, ...]
    if instance_map.ndim == 2:
        instance_map = instance_map[np.newaxis, ...]
    return bw_final, instance_map, n_objects


def visualize_instances(original_2d, instance_map_2d, output_path):
    """Show each object in a different color."""
    print("\n[6] Generating per-object colored visualization...")
    n = int(np.max(instance_map_2d))
    if n <= 0:
        # If there are no objects, draw the original image
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(original_2d, cmap="gray")
        ax.set_title("No objects detected")
        ax.axis("off")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return
    np.random.seed(42)
    colors = np.zeros((n + 1, 3))
    colors[0] = 0, 0, 0
    for i in range(1, n + 1):
        colors[i] = np.random.rand(3)
    rgb = label2rgb(instance_map_2d, colors=colors, bg_label=0)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(original_2d, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(rgb)
    axes[1].set_title(f"Instances (n={n}), each color = one object")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Allen segmenter for mitochondria / punctate structures from a TIFF."
    )
    parser.add_argument("image", type=str, help="Input TIFF path")
    parser.add_argument("-o", "--output-dir", type=str, default="allen_mito_output",
                        help="Output directory (default: ./allen_mito_output)")
    args = parser.parse_args()
    IMAGE_PATH = Path(args.image).resolve()
    OUTPUT_DIR = Path(args.output_dir).resolve()
    if not IMAGE_PATH.exists():
        print(f"Error: file does not exist {IMAGE_PATH}")
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Mitochondria / small punctate structure segmentation (each object separate, different colors)")
    print("=" * 60)
    struct_img = load_tif_channel(IMAGE_PATH)
    bw_final, instance_map, n_objects = segment_mitochondria(struct_img, PARAMS)

    # Save the binary segmentation
    out_binary = OUTPUT_DIR / "mitochondria_binary.tiff"
    if out_binary.exists():
        out_binary.unlink()
    with OmeTiffWriter(str(out_binary)) as w:
        w.save(bw_final)
    print(f"\nBinary segmentation: {out_binary}")

    # Save the instance labels (one integer per object)
    out_labels = OUTPUT_DIR / "mitochondria_labels.tiff"
    if out_labels.exists():
        out_labels.unlink()
    with OmeTiffWriter(str(out_labels)) as w:
        w.save(instance_map.astype(np.int32))
    print(f"Instance labels: {out_labels}")

    # Visualization: each object in a different color
    orig_2d = struct_img[0] if struct_img.ndim == 3 else struct_img
    inst_2d = instance_map[0] if instance_map.ndim == 3 else instance_map
    vis_path = OUTPUT_DIR / "mitochondria_instances_visualization.png"
    visualize_instances(orig_2d, inst_2d, vis_path)

    print("\n" + "=" * 60)
    print(f"Done, {n_objects} objects in total, results directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
