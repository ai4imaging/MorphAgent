def extract(img, *segmentation_masks):
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    from skimage.measure import label

    # 1. Input Validation and Preprocessing
    # Ensure image is a numpy array
    img = np.asarray(img)

    # Check dimensionality: Expecting (H, W, 3) for BBBC021
    if img.ndim != 3 or img.shape[2] < 2:
        # If we don't have at least 2 channels (Actin and Tubulin), we can't compute the ratio
        return 0.0

    # Extract Channels
    # Channel 0: Actin (Cytoskeleton)
    # Channel 1: Tubulin (Microtubules)
    actin_channel = img[..., 0].astype(np.float64)
    tubulin_channel = img[..., 1].astype(np.float64)

    # 2. Background Subtraction (Simple estimation)
    # Subtract the 1st percentile to remove sensor noise/background offset
    # This ensures the ratio reflects actual signal, not background levels
    actin_bg = np.percentile(actin_channel, 1)
    tubulin_bg = np.percentile(tubulin_channel, 1)
    
    actin_channel = np.maximum(actin_channel - actin_bg, 0)
    tubulin_channel = np.maximum(tubulin_channel - tubulin_bg, 0)

    # 3. Mask Handling
    # We need a mask to define "cells".
    # If segmentation masks are provided, use the first one (assuming it's a cell mask).
    # If not, generate a fallback mask using the combined intensity of Actin and Tubulin.
    
    cell_mask = None
    
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use provided mask
        input_mask = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions
        if input_mask.shape == img.shape[:2]:
            cell_mask = input_mask
    
    if cell_mask is None:
        # Fallback: Create a foreground mask based on structural channels
        # Combine Actin and Tubulin for segmentation
        combined_structure = actin_channel + tubulin_channel
        
        # Check if image is empty/black
        if combined_structure.max() == 0:
            return 0.0
            
        try:
            thresh = threshold_otsu(combined_structure)
            binary_mask = combined_structure > thresh
            # Label connected components
            cell_mask = label(binary_mask)
        except Exception:
            # Fallback for extremely low contrast images where otsu might fail
            return 0.0

    # 4. Compute Intensities per Cell
    # Get unique labels (excluding background 0)
    labels = np.unique(cell_mask)
    labels = labels[labels > 0]
    
    if len(labels) == 0:
        return 0.0

    # Use scipy.ndimage.sum to sum intensities over labeled regions efficiently
    # This is much faster than looping through labels manually
    actin_sums = ndimage.sum(actin_channel, labels=cell_mask, index=labels)
    tubulin_sums = ndimage.sum(tubulin_channel, labels=cell_mask, index=labels)

    # 5. Calculate Ratios
    # Add epsilon to denominator to avoid division by zero
    epsilon = 1e-9
    ratios = actin_sums / (tubulin_sums + epsilon)

    # Handle potential edge cases (infinity or NaN)
    # If tubulin sum is effectively 0, the ratio is huge. Clip it to avoid skewing stats.
    # A ratio of > 100 means Actin is 100x stronger than Tubulin, which is biologically extreme.
    ratios = np.nan_to_num(ratios, nan=0.0, posinf=100.0, neginf=0.0)

    # 6. Aggregate
    # We use the median to be robust against outliers (e.g., small debris classified as cells)
    if len(ratios) > 0:
        final_score = np.median(ratios)
    else:
        final_score = 0.0

    return float(final_score)
