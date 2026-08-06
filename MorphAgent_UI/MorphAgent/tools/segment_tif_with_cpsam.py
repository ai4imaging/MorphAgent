#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script for segmenting TIFF images with cellpose-SAM

Features:
- Read single- or multi-channel TIFF images
- Use cellpose-SAM to segment both cell bodies and nuclei
- Compute the cytoplasm (cell body - nucleus)
- Save three segmentation results: cyto.tif, nuclei.tif, cytoplasm.tif

Usage:
    python segment_tif_with_cpsam.py <input_tif_file> [options]
"""

import numpy as np
from cellpose import models, core, io, plot
from pathlib import Path
import argparse
import sys


def check_gpu():
    """Check whether a GPU is available"""
    if not core.use_gpu():
        raise ImportError("No GPU access, please check your GPU setup")
    print("GPU is available and will be used")


def load_cyto_model():
    """Load the cellpose-SAM cell body model"""
    print("Loading cellpose-SAM cyto model...")
    model = models.CellposeModel(model_type='cyto2', gpu=True)
    print("Cyto model loaded successfully")
    return model


def load_nuclei_model():
    """Load the cellpose-SAM nucleus model"""
    print("Loading cellpose-SAM nuclei model...")
    model = models.CellposeModel(model_type='nuclei', gpu=True)
    print("Nuclei model loaded successfully")
    return model


def select_channels(img, channels):
    """
    Select image channels
    
    Args:
        img: numpy array with shape (H, W) or (H, W, C)
        channels: list of channels to select, e.g. [0, 1], or None (use all channels)
    
    Returns:
        The image after channel selection
    
    Note:
        If channels are specified, an array the same size as the original is created,
        the selected channels are copied to the front of the new array, and the
        remaining channels stay 0 (consistent with the logic in the notebook)
    """
    if len(img.shape) == 2:
        # Single-channel image
        return img
    elif len(img.shape) == 3:
        # Multi-channel image
        if channels is None or len(channels) == 0:
            # Use all channels
            return img
        else:
            # Select the specified channels
            selected_channels = []
            for c in channels:
                if c >= img.shape[-1]:
                    raise ValueError(f'Invalid channel index {c}, image has {img.shape[-1]} channels')
                selected_channels.append(int(c))
            
            # Following the notebook's logic: create an array the same size as the original and put the selected channels at the front
            # This keeps the array shape unchanged so cellpose can process it correctly
            img_selected = np.zeros_like(img)
            img_selected[:, :, :len(selected_channels)] = img[:, :, selected_channels]
            return img_selected
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")


def segment_image(model, img, flow_threshold=0.4, cellprob_threshold=0.0, 
                  tile_norm_blocksize=0, batch_size=32):
    """
    Segment the image
    
    Args:
        model: cellpose model
        img: input image
        flow_threshold: flow threshold, default 0.4
        cellprob_threshold: cell probability threshold, default 0.0
        tile_norm_blocksize: normalization block size, default 0 (whole-image normalization)
        batch_size: batch size, default 32
    
    Returns:
        masks: segmentation masks
        flows: flow information
        styles: style information
    """
    print(f"Segmenting image with shape: {img.shape}")
    masks, flows, styles = model.eval(
        img, 
        batch_size=batch_size, 
        flow_threshold=flow_threshold, 
        cellprob_threshold=cellprob_threshold,
        normalize={"tile_norm_blocksize": tile_norm_blocksize}
    )
    print(f"Segmentation completed. Found {len(np.unique(masks)) - 1} objects")
    return masks, flows, styles


def compute_cytoplasm(cyto_masks, nuclei_masks):
    """
    Compute the cytoplasm mask (cell body - nucleus)
    
    Args:
        cyto_masks: cell body masks
        nuclei_masks: nucleus masks
    
    Returns:
        cytoplasm_masks: cytoplasm masks
    """
    # Create the cytoplasm mask, initialized to the cell body mask
    cytoplasm_masks = cyto_masks.copy()
    
    # Ensure the two masks have the same shape
    if cyto_masks.shape != nuclei_masks.shape:
        raise ValueError(f"Mask shapes don't match: cyto {cyto_masks.shape} vs nuclei {nuclei_masks.shape}")
    
    # For each cell body, subtract all overlapping nuclei from the cell body
    unique_cyto = np.unique(cyto_masks)
    unique_cyto = unique_cyto[unique_cyto > 0]  # Exclude the background
    
    print(f"Processing {len(unique_cyto)} cells...")
    
    for cell_id in unique_cyto:
        # Find the region of this cell body
        cyto_region = (cyto_masks == cell_id)
        
        # Find all nuclei within this region (there may be multiple nuclei in one cell body)
        nuclei_in_cell = nuclei_masks[cyto_region]
        unique_nuclei = np.unique(nuclei_in_cell)
        unique_nuclei = unique_nuclei[unique_nuclei > 0]  # Exclude the background
        
        # Subtract all overlapping nuclei from the cell body
        # Use a more direct approach: within the cell body region, set all non-zero nucleus regions to 0
        for nuc_id in unique_nuclei:
            nuc_region = (nuclei_masks == nuc_id)
            # Subtract the nucleus within the cell body region
            cytoplasm_masks[cyto_region & nuc_region] = 0
    
    # Validate the result: count the number of objects in the cytoplasm
    unique_cytoplasm = np.unique(cytoplasm_masks)
    unique_cytoplasm = unique_cytoplasm[unique_cytoplasm > 0]
    print(f"Result: {len(unique_cytoplasm)} cytoplasm regions (should be <= {len(unique_cyto)} cells)")
    
    return cytoplasm_masks


def save_results(masks, output_path):
    """
    Save the segmentation result
    
    Args:
        masks: segmentation masks
        output_path: output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    io.imsave(output_path, masks)
    print(f"Masks saved to: {output_path}")


def save_all_masks(cyto_masks, nuclei_masks, cytoplasm_masks, output_dir):
    """
    Save all three mask files
    
    Args:
        cyto_masks: cell body masks
        nuclei_masks: nucleus masks
        cytoplasm_masks: cytoplasm masks
        output_dir: output directory path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the cell body
    cyto_path = output_dir / "cyto.tif"
    io.imsave(str(cyto_path), cyto_masks)
    print(f"Cyto masks saved to: {cyto_path}")
    
    # Save the nucleus
    nuclei_path = output_dir / "nuclei.tif"
    io.imsave(str(nuclei_path), nuclei_masks)
    print(f"Nuclei masks saved to: {nuclei_path}")
    
    # Save the cytoplasm
    cytoplasm_path = output_dir / "cytoplasm.tif"
    io.imsave(str(cytoplasm_path), cytoplasm_masks)
    print(f"Cytoplasm masks saved to: {cytoplasm_path}")
    
    return cyto_path, nuclei_path, cytoplasm_path


def main():
    parser = argparse.ArgumentParser(
        description='Segment TIFF images with cellpose-SAM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python segment_tif_with_cpsam.py input.tif
  
  # Specify the output file
  python segment_tif_with_cpsam.py input.tif -o output_masks.tif
  
  # Select specific channels (e.g. use channels 0 and 1)
  python segment_tif_with_cpsam.py input.tif -c 0 1
  
  # Adjust the segmentation parameters
  python segment_tif_with_cpsam.py input.tif --flow_threshold 0.5 --cellprob_threshold -0.5
        """
    )
    
    parser.add_argument('input_file', type=str, help='Path to the input TIFF file')
    parser.add_argument('-o', '--output', type=str, default=None, 
                       help='Output file path (defaults to <input filename>_masks.tif)')
    parser.add_argument('-c', '--channels', type=int, nargs='+', default=None,
                       help='Channel indices to use (0-based), e.g. -c 0 1 means use channels 0 and 1')
    parser.add_argument('--flow_threshold', type=float, default=0.4,
                       help='Flow threshold, default 0.4. Increasing this reduces the number of returned masks')
    parser.add_argument('--cellprob_threshold', type=float, default=0.0,
                       help='Cell probability threshold, default 0.0. Lowering this increases the number of returned masks')
    parser.add_argument('--tile_norm_blocksize', type=int, default=0,
                       help='Normalization block size, default 0 (whole-image normalization). For images with uneven brightness, set it to 100-200')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size, default 32')
    parser.add_argument('--show_plot', action='store_true',
                       help='Display a visualization of the segmentation result')
    
    args = parser.parse_args()
    
    # Set up logging
    io.logger_setup()
    
    # Check the GPU
    check_gpu()
    
    # Load the models
    cyto_model = load_cyto_model()
    nuclei_model = load_nuclei_model()
    
    # Read the input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    print(f"Reading image from: {input_path}")
    img = io.imread(str(input_path))
    print(f"Image shape: {img.shape}")
    
    # Determine the output directory
    if args.output is None:
        # By default, output to a segmentation subdirectory of the input file's directory
        output_dir = input_path.parent / "segmentation"
    else:
        output_path = Path(args.output)
        # If a file path is given, use its parent directory; if a directory, use it directly
        if output_path.suffix:
            output_dir = output_path.parent
        else:
            output_dir = output_path
    
    # Prepare the images for segmentation
    # For cell body segmentation: use the cytoplasm channel (e.g. Actin, channel 0) and the nucleus channel (e.g. DAPI, channel 2)
    # For nucleus segmentation: use only the nucleus channel (e.g. DAPI, channel 2)
    if args.channels is not None and len(args.channels) > 0:
        print(f"User specified channels: {args.channels}")
        # If the user specified channels, assume:
        # - For cyto: use all specified channels (usually [0, 2] meaning Actin+DAPI)
        # - For nuclei: use only the last channel (usually DAPI, channel 2)
        cyto_channels = args.channels
        # Try to find the DAPI channel (usually the last channel, or channel 2)
        if len(args.channels) >= 2:
            # Assume the last channel is DAPI
            nuclei_channels = [args.channels[-1]]
        else:
            # If only one channel is specified, both models use it
            nuclei_channels = args.channels
        print(f"  Cyto model will use channels: {cyto_channels}")
        print(f"  Nuclei model will use channels: {nuclei_channels}")
    else:
        # If no channels are specified, auto-detect
        # Assume the image is RGB or 3-channel: channel 0=Actin, channel 1=Tubulin, channel 2=DAPI
        if len(img.shape) == 3 and img.shape[2] >= 3:
            # Use channel 0 (Actin) and channel 2 (DAPI) for cell body segmentation
            cyto_channels = [0, 2]
            # Use only channel 2 (DAPI) for nucleus segmentation
            nuclei_channels = [2]
            print(f"Auto-detected channels (assuming 3-channel image):")
            print(f"  Cyto model will use channels: {cyto_channels} (Actin + DAPI)")
            print(f"  Nuclei model will use channels: {nuclei_channels} (DAPI only)")
        elif len(img.shape) == 3 and img.shape[2] == 2:
            # 2-channel image: assume channel 0=cytoplasm, channel 1=nucleus
            cyto_channels = [0, 1]
            nuclei_channels = [1]
            print(f"Auto-detected channels (2-channel image):")
            print(f"  Cyto model will use channels: {cyto_channels}")
            print(f"  Nuclei model will use channels: {nuclei_channels}")
        else:
            # Single-channel image; both models use it
            cyto_channels = None
            nuclei_channels = None
            print(f"Single channel image, both models will use the same channel")
    
    # Prepare the cell body segmentation image
    if cyto_channels is not None:
        img_cyto = select_channels(img, cyto_channels)
    else:
        img_cyto = img
    
    # Prepare the nucleus segmentation image
    if nuclei_channels is not None:
        img_nuclei = select_channels(img, nuclei_channels)
    else:
        img_nuclei = img
    
    # Perform cell body segmentation
    print("\n" + "="*60)
    print("Segmenting cytoplasm (cell body)...")
    print("="*60)
    cyto_masks, cyto_flows, cyto_styles = segment_image(
        cyto_model, 
        img_cyto, 
        flow_threshold=args.flow_threshold,
        cellprob_threshold=args.cellprob_threshold,
        tile_norm_blocksize=args.tile_norm_blocksize,
        batch_size=args.batch_size
    )
    
    # Perform nucleus segmentation
    print("\n" + "="*60)
    print("Segmenting nuclei...")
    print("="*60)
    nuclei_masks, nuclei_flows, nuclei_styles = segment_image(
        nuclei_model, 
        img_nuclei, 
        flow_threshold=args.flow_threshold,
        cellprob_threshold=args.cellprob_threshold,
        tile_norm_blocksize=args.tile_norm_blocksize,
        batch_size=args.batch_size
    )
    
    # Compute the cytoplasm (cell body - nucleus)
    print("\n" + "="*60)
    print("Computing cytoplasm (cyto - nuclei)...")
    print("="*60)
    cytoplasm_masks = compute_cytoplasm(cyto_masks, nuclei_masks)
    print(f"Cytoplasm computation completed. Found {len(np.unique(cytoplasm_masks)) - 1} objects")
    
    # Save all three masks
    print("\n" + "="*60)
    print("Saving all masks...")
    print("="*60)
    cyto_path, nuclei_path, cytoplasm_path = save_all_masks(
        cyto_masks, nuclei_masks, cytoplasm_masks, output_dir
    )
    
    # Optional: display the visualization
    if args.show_plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        plot.show_segmentation(fig, img, cyto_masks, cyto_flows[0], ax=axes[0])
        axes[0].set_title('Cyto Masks')
        
        plot.show_segmentation(fig, img, nuclei_masks, nuclei_flows[0], ax=axes[1])
        axes[1].set_title('Nuclei Masks')
        
        # Cytoplasm visualization (using the cyto flow)
        plot.show_segmentation(fig, img, cytoplasm_masks, cyto_flows[0], ax=axes[2])
        axes[2].set_title('Cytoplasm Masks')
        
        plt.tight_layout()
        plt.show()
    
    print("\nDone! All masks saved successfully.")


if __name__ == "__main__":
    main()

