def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage

    # 1. Input Validation and Preprocessing
    # Ensure image is a numpy array
    img = np.asarray(img)
    
    # Check dimensionality and extract Actin channel (Channel 0)
    # Expected shape: (512, 512, 3) -> (H, W, C)
    if img.ndim == 3 and img.shape[2] == 3:
        # Standard (H, W, C) format
        actin_channel = img[:, :, 0]
    elif img.ndim == 3 and img.shape[0] == 3:
        # Channel-first (C, H, W) format - handle just in case
        actin_channel = img[0, :, :]
    elif img.ndim == 2:
        # If 2D, assume it's a single channel image. 
        # However, dataset spec says 3-channel. If this happens, it's ambiguous.
        # We'll assume the input is the relevant data.
        actin_channel = img
    else:
        # Unexpected format
        return 0.0

    # Convert to float64 to prevent overflow during summation
    # Input is uint8 (0-255). Summing thousands of pixels can easily exceed uint16.
    actin_channel = actin_channel.astype(np.float64)

    # 2. Handle Segmentation Masks
    # We need at least one mask to define "cells".
    if not segmentation_masks:
        # Fallback: If no segmentation is provided, we cannot compute "per cell" metrics.
        # We could treat the whole image as one cell, or return 0.
        # Given the feature definition implies cellular analysis, returning 0.0 is safer 
        # than returning a massive whole-image sum which would be an outlier.
        return 0.0

    # Select the appropriate mask.
    # If multiple masks are present (e.g., Nuclei, Cells), we prefer the one that covers the cytoplasm
    # because Actin is cytoskeletal. Usually, the larger/later mask is the whole-cell mask.
    # We will use the last mask in the list as a heuristic for the most inclusive mask (Cell mask).
    target_mask = segmentation_masks[-1]
    
    # Ensure mask shape matches image shape (handle potential 2D vs 3D mismatch)
    if target_mask.shape != actin_channel.shape:
        # If shapes don't match exactly, try to squeeze or check compatibility
        # This is a basic safety check.
        if target_mask.shape[-2:] == actin_channel.shape[-2:]:
             # Handle case where mask might have a singleton channel dim
             target_mask = target_mask.squeeze()
        else:
             return 0.0

    # 3. Feature Computation: Mean Total Intensity per Cell
    
    # Get unique labels (excluding background 0)
    # np.unique is safe but can be slow for large images. 
    # ndimage.sum allows passing an index, but we need to know which indices exist.
    labels = np.unique(target_mask)
    labels = labels[labels > 0] # Remove background

    if len(labels) == 0:
        return 0.0

    # Calculate the sum of pixel intensities for each labeled region
    # ndimage.sum(input, labels, index) returns a list of sums corresponding to the indices
    total_intensities = ndimage.sum(actin_channel, target_mask, labels)

    # 4. Aggregation
    # Compute the mean of these total intensities across the population of cells
    mean_total_intensity = np.mean(total_intensities)

    return float(mean_total_intensity)
