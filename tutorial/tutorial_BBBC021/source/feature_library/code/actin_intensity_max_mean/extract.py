def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage

    # 1. Input Validation and Preprocessing
    # Ensure image is a numpy array
    img = np.asarray(img)

    # Check dimensionality: Expecting (H, W, C) where C=3
    if img.ndim != 3 or img.shape[2] < 1:
        return 0.0

    # Extract Actin Channel (Channel 0 based on dataset description)
    # Channel 0 = Red = Actin
    actin_channel = img[:, :, 0]

    # Normalize to [0, 1] range for feature extraction
    # The input is uint8 (0-255), so we divide by 255.0
    actin_normalized = actin_channel.astype(np.float32) / 255.0

    # 2. Handle Segmentation Masks
    # We need at least one mask to define "cells".
    # If multiple masks are provided, we prioritize the first one (usually the primary object mask).
    if not segmentation_masks:
        # Fallback: If no mask is provided, we cannot compute "per cell" statistics.
        # We could treat the whole image as one cell, or return NaN.
        # Given the instruction to return a float and handle missing data gracefully:
        # We will treat the entire image as background/undefined and return 0.0
        return 0.0
    
    # Use the first available mask
    mask = segmentation_masks[0]
    
    # Ensure mask matches image spatial dimensions
    if mask.shape != actin_normalized.shape:
        # If dimensions mismatch, we cannot proceed reliably
        return 0.0

    # 3. Feature Computation: Mean of Max Intensity per Cell
    
    # Get unique labels from the mask (0 is background)
    unique_labels = np.unique(mask)
    
    # Filter out background (label 0)
    cell_labels = unique_labels[unique_labels > 0]
    
    # If no cells are detected in the mask, return 0.0
    if len(cell_labels) == 0:
        return 0.0

    # Calculate the maximum intensity value within each labeled region
    # scipy.ndimage.maximum is efficient for this "group by label" operation
    # It returns a list/array of max values corresponding to the index labels provided
    max_intensities = ndimage.maximum(actin_normalized, labels=mask, index=cell_labels)

    # 4. Aggregation
    # Compute the mean of these maximum values across the population of cells
    # If max_intensities is a scalar (single cell), np.mean handles it correctly
    result = np.mean(max_intensities)

    return float(result)
