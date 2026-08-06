#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test segmentation on a single image
Use the Allen Cell Segmenter to segment original.png
"""

import numpy as np
import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from skimage.morphology import remove_small_objects, dilation, ball
from skimage.measure import label
from scipy.ndimage import binary_fill_holes

# Allen segmentation imports
from aicssegmentation.core.pre_processing_utils import intensity_normalization, image_smoothing_gaussian_slice_by_slice
from aicssegmentation.core.MO_threshold import MO
from aicsimageio.writers import OmeTiffWriter


def load_png_image(image_path):
    """
    Load a PNG image and convert it into a 3D array format (Allen tools require 3D format).

    Args:
        image_path: Path to the PNG image

    Returns:
        img_3d: Array of shape (1, H, W, C), with C=3 (RGB)
        channels: Dictionary containing per-channel data
    """
    print(f"Loading image: {image_path}")
    img = Image.open(image_path)
    img_array = np.array(img)
    
    print(f"Original image shape: {img_array.shape}")
    
    # If RGBA, keep only RGB
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
        print("Converted to RGB format")
    
    # If grayscale, convert to RGB
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array, img_array, img_array], axis=2)
        print("Grayscale image converted to RGB format")
    
    # Convert to float32 and normalize to [0, 1]
    img_array = img_array.astype(np.float32) / 255.0
    
    # Allen tools require 3D format (Z, Y, X, C); here Z=1
    img_3d = np.expand_dims(img_array, axis=0)  # (1, H, W, 3)
    
    # Extract each channel
    channels = {
        'actin': img_3d[:, :, :, 0],      # cytoplasm (R channel)
        'tubulin': img_3d[:, :, :, 1],    # microtubules (G channel)
        'dapi': img_3d[:, :, :, 2]        # nucleus (B channel)
    }
    
    print(f"Processed image shape: {img_3d.shape}")
    print(f"Channel info: R(Actin/cytoplasm), G(Tubulin/microtubules), B(DAPI/nucleus)")
    
    return img_3d, channels


def segment_nucleus(dapi_channel):
    """
    Segment the nucleus (DAPI channel)
    """
    print("\n[Step 1] Segmenting nucleus (DAPI channel)...")
    
    # Preprocessing: intensity normalization
    print("- Intensity normalization...")
    intensity_norm_param = [0.5, 15]  # Auto-contrast normalization
    dapi_norm = intensity_normalization(dapi_channel, scaling_param=intensity_norm_param)
    
    # Smoothing
    print("- Gaussian smoothing...")
    dapi_smooth = image_smoothing_gaussian_slice_by_slice(dapi_norm, sigma=1.0)
    
    # Segment the nucleus with the MO (Masked-Object) thresholding method
    print("- MO thresholding...")
    nucleus_bw = MO(
        dapi_smooth, 
        global_thresh_method='tri',  # Triangle method
        object_minArea=10,  # minimum nucleus area
        return_object=False
    )
    
    # Postprocessing: remove small objects, fill holes
    print("- Postprocessing (remove small objects, fill holes)...")
    nucleus_bw = remove_small_objects(nucleus_bw > 0, min_size=50, connectivity=1, in_place=False)
    
    # For 2D images, use scipy's binary_fill_holes
    if len(nucleus_bw.shape) == 3:
        nucleus_bw = np.array([binary_fill_holes(slice) for slice in nucleus_bw])
    else:
        nucleus_bw = binary_fill_holes(nucleus_bw)
    
    nucleus_bw = nucleus_bw.astype(np.uint8) * 255
    num_nuclei = np.max(label(nucleus_bw > 0))
    print(f"   Detected {num_nuclei} nuclei")
    
    return nucleus_bw


def segment_cytoplasm(actin_channel, nucleus_bw):
    """
    Segment the cytoplasm (Actin channel)
    """
    print("\n[Step 2] Segmenting cytoplasm (Actin channel)...")
    
    # Preprocessing
    print("- Intensity normalization...")
    intensity_norm_param = [0.5, 15]
    actin_norm = intensity_normalization(actin_channel, scaling_param=intensity_norm_param)
    
    print("- Gaussian smoothing...")
    actin_smooth = image_smoothing_gaussian_slice_by_slice(actin_norm, sigma=1.5)
    
    # Segment the cytoplasm with the MO thresholding method
    print("- MO thresholding...")
    cytoplasm_bw = MO(
        actin_smooth,
        global_thresh_method='tri',
        object_minArea=100,  # the cytoplasm is usually larger than the nucleus
        return_object=False
    )
    
    # Postprocessing
    print("- Postprocessing (remove small objects, fill holes)...")
    cytoplasm_bw = remove_small_objects(cytoplasm_bw > 0, min_size=200, connectivity=1, in_place=False)
    
    if len(cytoplasm_bw.shape) == 3:
        cytoplasm_bw = np.array([binary_fill_holes(slice) for slice in cytoplasm_bw])
    else:
        cytoplasm_bw = binary_fill_holes(cytoplasm_bw)
    
    # Use the nuclei as seeds to expand the cytoplasm segmentation
    print("- Using nuclei as seeds to expand the cytoplasm region...")
    nucleus_labeled = label(nucleus_bw > 0)
    num_nuclei = np.max(nucleus_labeled)
    
    if num_nuclei > 0:
        # Dilate each nucleus to ensure the cytoplasm covers the nucleus region
        expanded_cytoplasm = np.zeros_like(cytoplasm_bw)
        for i in range(1, num_nuclei + 1):
            single_nucleus = (nucleus_labeled == i)
            # Dilate the nucleus region
            expanded_nucleus = dilation(single_nucleus, selem=ball(5))
            # Merge into the cytoplasm
            expanded_cytoplasm = np.logical_or(expanded_cytoplasm, expanded_nucleus)
        
        # Merge the original cytoplasm and the expanded region
        cytoplasm_bw = np.logical_or(cytoplasm_bw, expanded_cytoplasm)
        if len(cytoplasm_bw.shape) == 3:
            cytoplasm_bw = np.array([binary_fill_holes(slice) for slice in cytoplasm_bw])
        else:
            cytoplasm_bw = binary_fill_holes(cytoplasm_bw)
    
    cytoplasm_bw = cytoplasm_bw.astype(np.uint8) * 255
    num_cytoplasm = np.max(label(cytoplasm_bw > 0))
    print(f"   Detected {num_cytoplasm} cytoplasm regions")
    
    return cytoplasm_bw


def visualize_results(original_img, nucleus_bw, cytoplasm_bw, output_path):
    """
    Visualize the segmentation results
    """
    print("\n[Step 3] Generating visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # Original image (RGB composite)
    if len(original_img.shape) == 4:
        axes[0, 0].imshow(original_img[0, :, :, :])
    else:
        axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('Original image (RGB)', fontsize=14)
    axes[0, 0].axis('off')
    
    # Nucleus segmentation
    if len(nucleus_bw.shape) == 3:
        axes[0, 1].imshow(nucleus_bw[0], cmap='gray')
    else:
        axes[0, 1].imshow(nucleus_bw, cmap='gray')
    axes[0, 1].set_title(f'Nucleus segmentation ({np.max(label(nucleus_bw > 0))} detected)', fontsize=14)
    axes[0, 1].axis('off')
    
    # Cytoplasm segmentation
    if len(cytoplasm_bw.shape) == 3:
        axes[1, 0].imshow(cytoplasm_bw[0], cmap='gray')
    else:
        axes[1, 0].imshow(cytoplasm_bw, cmap='gray')
    axes[1, 0].set_title(f'Cytoplasm segmentation ({np.max(label(cytoplasm_bw > 0))} detected)', fontsize=14)
    axes[1, 0].axis('off')
    
    # Overlay display
    if len(nucleus_bw.shape) == 3:
        nucleus_2d = nucleus_bw[0]
        cytoplasm_2d = cytoplasm_bw[0]
    else:
        nucleus_2d = nucleus_bw
        cytoplasm_2d = cytoplasm_bw
    
    overlay = np.zeros((*nucleus_2d.shape, 3))
    overlay[:, :, 0] = nucleus_2d / 255.0  # red: nucleus
    overlay[:, :, 1] = cytoplasm_2d / 255.0  # green: cytoplasm
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Overlay (Red: Nucleus, Green: Cytoplasm)', fontsize=14)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   Visualization saved: {output_path}")


def main():
    """Segment a single image (smoke test)."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Allen Cell Segmenter — single image smoke test."
    )
    parser.add_argument("image", type=str, help="Input image path (PNG/TIFF)")
    parser.add_argument("-o", "--output-dir", type=str, default="test_output",
                        help="Output directory (default: ./test_output)")
    args = parser.parse_args()

    print("="*60)
    print("Allen Cell Segmenter - single image test")
    print("="*60)

    image_path = Path(args.image).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check whether the image exists
    if not image_path.exists():
        print(f"Error: image file does not exist: {image_path}")
        return
    
    try:
        # Load the image
        img_3d, channels = load_png_image(image_path)
        
        # Segment the nucleus
        nucleus_bw = segment_nucleus(channels['dapi'])
        
        # Segment the cytoplasm
        cytoplasm_bw = segment_cytoplasm(channels['actin'], nucleus_bw)
        
        # Save the results
        print("\n[Step 4] Saving segmentation results...")
        
        # Save in TIFF format
        nucleus_output = output_dir / "nucleus_segmentation.tiff"
        cytoplasm_output = output_dir / "cytoplasm_segmentation.tiff"
        
        # Convert to 3D format for saving (Z, Y, X)
        if len(nucleus_bw.shape) == 2:
            nucleus_3d = np.expand_dims(nucleus_bw, axis=0)
        else:
            nucleus_3d = nucleus_bw
        
        if len(cytoplasm_bw.shape) == 2:
            cytoplasm_3d = np.expand_dims(cytoplasm_bw, axis=0)
        else:
            cytoplasm_3d = cytoplasm_bw
        
        # Delete existing files (if any)
        if nucleus_output.exists():
            nucleus_output.unlink()
        if cytoplasm_output.exists():
            cytoplasm_output.unlink()
        
        # Save using the correct API
        with OmeTiffWriter(str(nucleus_output)) as writer:
            writer.save(nucleus_3d)
        
        with OmeTiffWriter(str(cytoplasm_output)) as writer:
            writer.save(cytoplasm_3d)
        
        print(f"   Nucleus segmentation result: {nucleus_output}")
        print(f"   Cytoplasm segmentation result: {cytoplasm_output}")
        
        # Visualization
        vis_output = output_dir / "segmentation_visualization.png"
        visualize_results(img_3d, nucleus_bw, cytoplasm_bw, vis_output)
        
        print("\n" + "="*60)
        print("Segmentation complete!")
        print(f"Results saved in: {output_dir}")
        print("="*60)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
