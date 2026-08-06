#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Segment image.tif using the Allen Cell Segmenter
Supports single-channel or multi-channel TIFF (STCZYX and similar dimensions).
Single-channel: the same channel is used for nucleus/cytoplasm segmentation;
Multi-channel: the order is assumed to be [DAPI, Actin, Tubulin] or specified by channel_indices.

I/O uses tifffile (not aicsimageio) so morphagent_allen does not need aicspylibczi /
CZI native builds that fail on modern macOS toolchains.
"""

import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.morphology import remove_small_objects, dilation, ball
from skimage.measure import label
from skimage.color import label2rgb
from scipy.ndimage import binary_fill_holes
import tifffile

from aicssegmentation.core.pre_processing_utils import (
    intensity_normalization,
    image_smoothing_gaussian_slice_by_slice,
)
from aicssegmentation.core.MO_threshold import MO


# Default channel indices [dapi, actin, tubulin] (0-based); None = auto-infer.
CHANNEL_INDICES = None  # e.g. [2, 0, 1] means channel2=DAPI, 0=Actin, 1=Tubulin


def load_tif_image(image_path, channel_indices=None):
    """
    Load a TIFF/PNG/JPEG (etc.) and return (Z,Y,X) for each channel plus a composite for viz.
    For single-channel input, dapi and actin use the same channel.
    """
    image_path = Path(image_path)
    print(f"Loading: {image_path}")
    suffix = image_path.suffix.lower()
    data = None
    if suffix in {".tif", ".tiff"}:
        try:
            data = tifffile.imread(str(image_path))
        except Exception as exc:
            print(f"  tifffile failed ({exc}); trying PIL…")
    if data is None:
        from PIL import Image
        arr = np.asarray(Image.open(str(image_path)))
        # Drop alpha if present
        if arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]
        data = arr

    # Normalize to (C, Z, Y, X)
    if data.ndim == 2:
        data = data[np.newaxis, np.newaxis, ...]  # (1, 1, Y, X)
    elif data.ndim == 3:
        # (Z, Y, X), (C, Y, X), or (Y, X, C)
        if data.shape[-1] in (1, 2, 3, 4) and data.shape[-1] < min(data.shape[0], data.shape[1]):
            data = np.moveaxis(data, -1, 0)  # (C, Y, X)
            data = data[:, np.newaxis, ...]  # (C, 1, Y, X)
        elif data.shape[0] <= 4 and data.shape[0] < min(data.shape[1], data.shape[2]):
            data = data[:, np.newaxis, ...]  # (C, 1, Y, X)
        else:
            data = data[np.newaxis, ...]  # (1, Z, Y, X)
    elif data.ndim == 4:
        # Already (C, Z, Y, X) or (Z, Y, X, C)
        if data.shape[-1] in (1, 2, 3, 4) and data.shape[-1] < data.shape[0]:
            data = np.moveaxis(data, -1, 0)
    elif data.ndim > 4:
        while data.ndim > 4 and data.shape[0] == 1:
            data = data.squeeze(0)
        if data.ndim > 4:
            data = data.reshape((-1,) + data.shape[-3:])

    nch = data.shape[0]

    def norm_ch(c):
        c = c.astype(np.float32)
        minv, maxv = c.min(), c.max()
        if maxv > minv:
            c = (c - minv) / (maxv - minv)
        else:
            c = np.zeros_like(c)
        return c

    channels_list = [norm_ch(data[i]) for i in range(nch)]

    if nch == 1:
        dapi = channels_list[0]
        actin = channels_list[0]
        tubulin = channels_list[0]
        print("Single-channel TIFF; the same channel is used for nucleus and cytoplasm segmentation")
    else:
        if channel_indices is not None:
            idx_d, idx_a, idx_t = channel_indices[0], channel_indices[1], channel_indices[2]
        else:
            if nch >= 3:
                idx_d, idx_a, idx_t = 2, 0, 1
            elif nch == 2:
                idx_d, idx_a, idx_t = 0, 1, 0
            else:
                idx_d = idx_a = idx_t = 0
        dapi = channels_list[idx_d]
        actin = channels_list[idx_a]
        tubulin = channels_list[idx_t]
        print(f"Multi-channel TIFF (C={nch}); DAPI/Actin/Tubulin channel indices: {idx_d},{idx_a},{idx_t}")

    if dapi.ndim == 2:
        dapi = dapi[np.newaxis, ...]
        actin = actin[np.newaxis, ...]
        tubulin = tubulin[np.newaxis, ...]

    channels = {"dapi": dapi, "actin": actin, "tubulin": tubulin}
    rgb = np.stack([actin[0], tubulin[0], dapi[0]], axis=-1)
    if rgb.ndim == 3:
        rgb = rgb[np.newaxis, ...]
    return channels, rgb


def segment_nucleus(dapi_channel, intensity_norm_param=None, sigma=1.0, object_min_area=10, min_size=50):
    """Nucleus segmentation (DAPI)."""
    if intensity_norm_param is None:
        intensity_norm_param = [0.5, 15]
    print("\n[1/2] Segmenting nucleus (DAPI)...")
    dapi_norm = intensity_normalization(dapi_channel, scaling_param=intensity_norm_param)
    dapi_smooth = image_smoothing_gaussian_slice_by_slice(dapi_norm, sigma=sigma)
    nucleus_bw = MO(
        dapi_smooth,
        global_thresh_method="tri",
        object_minArea=object_min_area,
        return_object=False,
    )
    nucleus_bw = remove_small_objects(nucleus_bw > 0, min_size=min_size, connectivity=1, in_place=False)
    if nucleus_bw.ndim == 3:
        nucleus_bw = np.array([binary_fill_holes(s) for s in nucleus_bw])
    else:
        nucleus_bw = binary_fill_holes(nucleus_bw)
    nucleus_bw = nucleus_bw.astype(np.uint8) * 255
    print(f"  Detected {np.max(label(nucleus_bw > 0))} nuclei")
    return nucleus_bw


def segment_cytoplasm(actin_channel, nucleus_bw, intensity_norm_param=None, sigma=1.5, object_min_area=100, min_size=200, dilation_r=5):
    """Cytoplasm segmentation (Actin), expanded by dilating the nuclei."""
    if intensity_norm_param is None:
        intensity_norm_param = [0.5, 15]
    print("\n[2/2] Segmenting cytoplasm (Actin)...")
    actin_norm = intensity_normalization(actin_channel, scaling_param=intensity_norm_param)
    actin_smooth = image_smoothing_gaussian_slice_by_slice(actin_norm, sigma=sigma)
    cytoplasm_bw = MO(
        actin_smooth,
        global_thresh_method="tri",
        object_minArea=object_min_area,
        return_object=False,
    )
    cytoplasm_bw = remove_small_objects(cytoplasm_bw > 0, min_size=min_size, connectivity=1, in_place=False)
    if cytoplasm_bw.ndim == 3:
        cytoplasm_bw = np.array([binary_fill_holes(s) for s in cytoplasm_bw])
    else:
        cytoplasm_bw = binary_fill_holes(cytoplasm_bw)

    nucleus_labeled = label(nucleus_bw > 0)
    num_nuclei = np.max(nucleus_labeled)
    if num_nuclei > 0:
        expanded = np.zeros_like(cytoplasm_bw)
        for i in range(1, num_nuclei + 1):
            single = nucleus_labeled == i
            expanded = np.logical_or(expanded, dilation(single, selem=ball(dilation_r)))
        cytoplasm_bw = np.logical_or(cytoplasm_bw, expanded)
        if cytoplasm_bw.ndim == 3:
            cytoplasm_bw = np.array([binary_fill_holes(s) for s in cytoplasm_bw])
        else:
            cytoplasm_bw = binary_fill_holes(cytoplasm_bw)
    cytoplasm_bw = cytoplasm_bw.astype(np.uint8) * 255
    print(f"  Detected {np.max(label(cytoplasm_bw > 0))} cytoplasm regions")
    return cytoplasm_bw


def _instance_colors(n, seed=42):
    """Generate distinct colors for n objects (background 0 is black)."""
    np.random.seed(seed)
    colors = np.zeros((n + 1, 3))
    colors[0] = 0, 0, 0
    for i in range(1, n + 1):
        colors[i] = np.random.rand(3)
    return colors


def visualize_results(original_rgb, nucleus_bw, cytoplasm_bw, output_path):
    """Save a 2x2 visualization; each nucleus/cytoplasm object is shown in a distinct color."""
    print("\n[3] Generating visualization (each object in a different color)...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    if original_rgb.ndim == 4:
        axes[0, 0].imshow(original_rgb[0])
    else:
        axes[0, 0].imshow(original_rgb)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    n2d = nucleus_bw[0] if nucleus_bw.ndim == 3 else nucleus_bw
    c2d = cytoplasm_bw[0] if cytoplasm_bw.ndim == 3 else cytoplasm_bw
    nucleus_labels = label(nucleus_bw > 0)
    cytoplasm_labels = label(cytoplasm_bw > 0)
    if nucleus_labels.ndim == 3:
        nucleus_labels_2d = nucleus_labels[0]
    else:
        nucleus_labels_2d = nucleus_labels
    if cytoplasm_labels.ndim == 3:
        cytoplasm_labels_2d = cytoplasm_labels[0]
    else:
        cytoplasm_labels_2d = cytoplasm_labels

    n_nuc = int(np.max(nucleus_labels_2d))
    n_cyt = int(np.max(cytoplasm_labels_2d))
    colors_nuc = _instance_colors(n_nuc, seed=42)
    colors_cyt = _instance_colors(n_cyt, seed=123)
    rgb_nuc = label2rgb(nucleus_labels_2d, colors=colors_nuc, bg_label=0)
    rgb_cyt = label2rgb(cytoplasm_labels_2d, colors=colors_cyt, bg_label=0)
    axes[0, 1].imshow(rgb_nuc)
    axes[0, 1].set_title(f"Nucleus ({n_nuc} objects, each color = one object)")
    axes[0, 1].axis("off")
    axes[1, 0].imshow(rgb_cyt)
    axes[1, 0].set_title(f"Cytoplasm ({n_cyt} objects, each color = one object)")
    axes[1, 0].axis("off")
    overlay = np.zeros((*n2d.shape, 3))
    overlay[..., 0] = n2d / 255.0
    overlay[..., 1] = c2d / 255.0
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title("Overlay (R: Nucleus, G: Cytoplasm)")
    axes[1, 1].axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Allen Cell Segmenter — nucleus + cytoplasm masks from TIFF/PNG/JPEG."
    )
    parser.add_argument("image", type=str, help="Input image path (TIFF/PNG/JPEG/…)")
    parser.add_argument("-o", "--output-dir", type=str, default="allen_seg_output",
                        help="Output directory (default: ./allen_seg_output)")
    parser.add_argument("-c", "--channels", type=int, nargs="+", default=CHANNEL_INDICES,
                        help="Channel indices [DAPI ACTIN TUBULIN] (0-based); omit to auto-infer")
    parser.add_argument(
        "--save-visualization",
        action="store_true",
        help="Also write segmentation_visualization.png (off by default; masks only)",
    )
    args = parser.parse_args()

    IMAGE_PATH = Path(args.image).resolve()
    OUTPUT_DIR = Path(args.output_dir).resolve()
    if not IMAGE_PATH.exists():
        print(f"Error: file does not exist {IMAGE_PATH}")
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Allen Segmenter — nucleus + cytoplasm")
    print("=" * 60)

    channels, rgb = load_tif_image(IMAGE_PATH, channel_indices=args.channels)
    nucleus_bw = segment_nucleus(channels["dapi"])
    cytoplasm_bw = segment_cytoplasm(channels["actin"], nucleus_bw)

    # Save TIFF masks only by default (no preview PNG in segmentation/).
    nucleus_out = OUTPUT_DIR / "nucleus_segmentation.tiff"
    cytoplasm_out = OUTPUT_DIR / "cytoplasm_segmentation.tiff"
    for p in (nucleus_out, cytoplasm_out):
        if p.exists():
            p.unlink()
    n3d = np.expand_dims(nucleus_bw, axis=0) if nucleus_bw.ndim == 2 else nucleus_bw
    c3d = np.expand_dims(cytoplasm_bw, axis=0) if cytoplasm_bw.ndim == 2 else cytoplasm_bw
    tifffile.imwrite(str(nucleus_out), n3d.astype(np.uint8))
    tifffile.imwrite(str(cytoplasm_out), c3d.astype(np.uint8))
    print(f"\n  Nucleus: {nucleus_out}")
    print(f"  Cytoplasm: {cytoplasm_out}")

    if args.save_visualization:
        vis_path = OUTPUT_DIR / "segmentation_visualization.png"
        visualize_results(rgb, nucleus_bw, cytoplasm_bw, vis_path)
    else:
        leftover_vis = OUTPUT_DIR / "segmentation_visualization.png"
        if leftover_vis.exists():
            leftover_vis.unlink()

    print("\n" + "=" * 60)
    print(f"Segmentation complete, results directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
