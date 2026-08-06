#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Allen Cell Segmentation Pipeline for BBBC021 Dataset
Workflow 1: Allen classic segmentation method
Workflow 2: Allen-ml deep learning method

Data format:
- Channel 0 (R): Actin (F-actin) - actin cytoskeleton, marks the cytoplasm
- Channel 1 (G): Tubulin (beta-tubulin) - tubulin
- Channel 2 (B): DAPI - nucleus (DNA)
"""

import numpy as np
import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from skimage.morphology import remove_small_objects, erosion, ball, dilation, binary_closing, binary_opening
from skimage.filters import threshold_otsu, threshold_triangle
from skimage.measure import label
from scipy.ndimage import binary_fill_holes

# Allen segmentation imports
from aicssegmentation.core.pre_processing_utils import intensity_normalization, image_smoothing_gaussian_slice_by_slice
from aicssegmentation.core.seg_dot import dot_2d_slice_by_slice_wrapper
from aicssegmentation.core.MO_threshold import MO
from aicssegmentation.core.output_utils import save_segmentation
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
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # If RGBA, keep only RGB
    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    
    # Convert to float32 and normalize to [0, 1]
    img_array = img_array.astype(np.float32) / 255.0
    
    # Allen tools require 3D format (Z, Y, X); here Z=1
    img_3d = np.expand_dims(img_array, axis=0)  # (1, H, W, 3)
    
    # Extract each channel
    channels = {
        'actin': img_3d[:, :, :, 0],      # cytoplasm
        'tubulin': img_3d[:, :, :, 1],    # microtubules
        'dapi': img_3d[:, :, :, 2]        # nucleus
    }
    
    return img_3d, channels


def workflow1_classic_segmentation(channels, output_dir, image_name):
    """
    Workflow 1: Allen classic segmentation method
    Segments the nucleus (DAPI) and the cytoplasm (Actin)

    Args:
        channels: Dictionary containing each channel
        output_dir: Output directory
        image_name: Image name (without extension)
    """
    print(f"\n{'='*60}")
    print(f"Workflow 1: Allen classic segmentation method - {image_name}")
    print(f"{'='*60}")
    
    results = {}
    
    # ========== 1. Nucleus segmentation (DAPI channel) ==========
    print("\n[1/2] Segmenting nucleus (DAPI channel)...")
    dapi_channel = channels['dapi']
    
    # Preprocessing: intensity normalization
    intensity_norm_param = [0.5, 15]  # Auto-contrast normalization
    dapi_norm = intensity_normalization(dapi_channel, scaling_param=intensity_norm_param)
    
    # Smoothing
    dapi_smooth = image_smoothing_gaussian_slice_by_slice(dapi_norm, sigma=1.0)
    
    # Segment the nucleus with the MO (Masked-Object) thresholding method
    # Nuclei are usually punctate/round structures
    nucleus_bw = MO(
        dapi_smooth, 
        global_thresh_method='tri',  # Triangle method
        object_minArea=10,  # minimum nucleus area
        return_object=False
    )
    
    # Postprocessing: remove small objects, fill holes
    nucleus_bw = remove_small_objects(nucleus_bw > 0, min_size=50, connectivity=1, in_place=False)
    # For 2D images, use scipy's binary_fill_holes
    if len(nucleus_bw.shape) == 3:
        nucleus_bw = np.array([binary_fill_holes(slice) for slice in nucleus_bw])
    else:
        nucleus_bw = binary_fill_holes(nucleus_bw)
    
    results['nucleus'] = nucleus_bw.astype(np.uint8) * 255
    print(f"  Detected {np.max(label(nucleus_bw))} nuclei")
    
    # ========== 2. Cytoplasm segmentation (Actin channel) ==========
    print("\n[2/2] Segmenting cytoplasm (Actin channel)...")
    actin_channel = channels['actin']
    
    # Preprocessing
    actin_norm = intensity_normalization(actin_channel, scaling_param=intensity_norm_param)
    actin_smooth = image_smoothing_gaussian_slice_by_slice(actin_norm, sigma=1.5)
    
    # Segment the cytoplasm with the MO thresholding method
    # The cytoplasm is usually a larger continuous region
    cytoplasm_bw = MO(
        actin_smooth,
        global_thresh_method='tri',
        object_minArea=100,  # the cytoplasm is usually larger than the nucleus
        return_object=False
    )
    
    # Postprocessing
    cytoplasm_bw = remove_small_objects(cytoplasm_bw > 0, min_size=200, connectivity=1, in_place=False)
    # For 2D images, use scipy's binary_fill_holes
    if len(cytoplasm_bw.shape) == 3:
        cytoplasm_bw = np.array([binary_fill_holes(slice) for slice in cytoplasm_bw])
    else:
        cytoplasm_bw = binary_fill_holes(cytoplasm_bw)
    
    # Use the nuclei as seeds to expand the cytoplasm segmentation
    # Ensure each nucleus has a corresponding cytoplasm region around it
    nucleus_labeled = label(nucleus_bw)
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
    
    results['cytoplasm'] = cytoplasm_bw.astype(np.uint8) * 255
    print(f"  Detected {np.max(label(cytoplasm_bw))} cytoplasm regions")
    
    # ========== 3. Save results ==========
    output_path = Path(output_dir) / f"{image_name}_classic"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save in TIFF format
    nucleus_output = output_path / f"{image_name}_nucleus.tiff"
    cytoplasm_output = output_path / f"{image_name}_cytoplasm.tiff"
    
    # Convert to 3D format for saving
    nucleus_3d = np.expand_dims(results['nucleus'], axis=0)
    cytoplasm_3d = np.expand_dims(results['cytoplasm'], axis=0)
    
    OmeTiffWriter.save(nucleus_3d, nucleus_output, dim_order="CZYX")
    OmeTiffWriter.save(cytoplasm_3d, cytoplasm_output, dim_order="CZYX")
    
    print(f"\nResults saved to: {output_path}")
    
    return results


def workflow2_ml_segmentation(channels, output_dir, image_name, model_path=None):
    """
    Workflow 2: Allen-ml deep learning method
    Requires a pretrained model

    Args:
        channels: Dictionary containing each channel
        output_dir: Output directory
        image_name: Image name (without extension)
        model_path: Path to the pretrained model (if None, the user will be prompted)
    """
    print(f"\n{'='*60}")
    print(f"Workflow 2: Allen-ml deep learning method - {image_name}")
    print(f"{'='*60}")
    
    if model_path is None or not os.path.exists(model_path):
        print("\n[WARN]  Warning: pretrained model not found!")
        print("\nTo use the Allen-ml deep learning method, you need to:")
        print("1. Train your own model, or")
        print("2. Download a pretrained model from the Allen Institute")
        print("\nPretrained model download information:")
        print("- Visit: https://www.allencell.org/segmenter.html")
        print("- Or contact: forum.allencell.org")
        print("- Model format: .pth or .pytorch file")
        print("\nIf you have a model, please:")
        print("1. Place the model file in a suitable location")
        print("2. Edit the configuration file: aics-ml-segmentation/configs/predict_file_config.yaml")
        print("3. Set the model_path parameter")
        print("\nThe ML segmentation workflow will now be skipped...")
        return None
    
    try:
        from aicsmlsegment.bin.predict import main as predict_main
        import yaml
        import tempfile
        
        print(f"\nUsing pretrained model: {model_path}")
        
        # Create a temporary configuration file
        config = {
            'model': {
                'name': 'unet_xy',
                'zoom_ratio': 1
            },
            'model_path': model_path,
            'nchannel': 1,
            'nclass': [2],
            'OutputCh': [0],
            'size_in': [1, 512, 512],
            'size_out': [1, 256, 256],
            'OutputDir': str(Path(output_dir) / f"{image_name}_ml"),
            'InputCh': [0],  # use the DAPI channel
            'ResizeRatio': [1.0, 1.0, 1.0],
            'Normalization': 10,
            'Threshold': 0.6,
            'RuntimeAug': False,
            'mode': {
                'name': 'file',
                'InputFile': '',  # needs to be converted to TIFF format
                'timelapse': False
            }
        }
        
        # The image must first be converted to TIFF format here
        # Since the ML workflow is fairly complex, users are advised to follow the official docs
        
        print("\n[WARN]  ML segmentation requires a complete configuration process")
        print("Please refer to: aics-ml-segmentation/docs/doc_pred_yaml.md")
        
        return None
        
    except ImportError:
        print("\n[WARN]  Unable to import the Allen-ml module; make sure the allen_ml environment is installed and activated")
        return None


def visualize_results(original_img, results, output_path, image_name):
    """
    Visualize the segmentation results
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Original image (RGB composite)
    axes[0, 0].imshow(original_img[0, :, :, :])
    axes[0, 0].set_title('Original Image (RGB)')
    axes[0, 0].axis('off')
    
    # Nucleus segmentation
    if 'nucleus' in results:
        axes[0, 1].imshow(results['nucleus'], cmap='gray')
        axes[0, 1].set_title('Nucleus Segmentation')
        axes[0, 1].axis('off')
    
    # Cytoplasm segmentation
    if 'cytoplasm' in results:
        axes[1, 0].imshow(results['cytoplasm'], cmap='gray')
        axes[1, 0].set_title('Cytoplasm Segmentation')
        axes[1, 0].axis('off')
    
    # Overlay display
    if 'nucleus' in results and 'cytoplasm' in results:
        overlay = np.zeros((*results['nucleus'].shape, 3))
        overlay[:, :, 0] = results['nucleus'] / 255.0  # red: nucleus
        overlay[:, :, 1] = results['cytoplasm'] / 255.0  # green: cytoplasm
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('Overlay (Red: Nucleus, Green: Cytoplasm)')
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path / f"{image_name}_visualization.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved: {output_path / f'{image_name}_visualization.png'}")


def main():
    """Batch-segment all PNG images in an input directory."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Allen classic pipeline: batch nucleus+cytoplasm segmentation of PNGs."
    )
    parser.add_argument("-d", "--data-dir", type=str, default="test_data",
                        help="Directory of input images named image_*.png (default: ./test_data)")
    parser.add_argument("-o", "--output-dir", type=str, default="segmentation_results",
                        help="Output directory (default: ./segmentation_results)")
    args = parser.parse_args()
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all PNG images
    image_files = sorted(data_dir.glob("image_*.png"))
    
    if len(image_files) == 0:
        print(f"Error: no image files found in {data_dir}")
        return
    
    print(f"Found {len(image_files)} image files")
    
    # Process each image
    for img_path in image_files:
        image_name = img_path.stem
        print(f"\n{'#'*60}")
        print(f"Processing image: {image_name}")
        print(f"{'#'*60}")
        
        try:
            # Load the image
            img_3d, channels = load_png_image(img_path)
            print(f"Image size: {img_3d.shape}")
            
            # Workflow 1: classic segmentation
            results = workflow1_classic_segmentation(channels, output_dir, image_name)
            
            # Visualization
            result_path = output_dir / f"{image_name}_classic"
            visualize_results(img_3d, results, result_path, image_name)
            
            # Workflow 2: ML segmentation (requires a pretrained model)
            # workflow2_ml_segmentation(channels, output_dir, image_name, model_path=None)
            
        except Exception as e:
            print(f"Error while processing {image_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("All images processed!")
    print(f"Results saved in: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
