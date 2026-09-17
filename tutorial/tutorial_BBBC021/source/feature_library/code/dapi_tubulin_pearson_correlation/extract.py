def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    
    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Expected shape: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract relevant channels based on dataset description
    # Channel 1: Tubulin (Green)
    # Channel 2: DAPI (Blue)
    tubulin = arr[:, :, 1]
    dapi = arr[:, :, 2]

    # Normalize intensities to [0, 1] for stability, though Pearson is scale-invariant
    # We do this to help with thresholding if needed
    def normalize_channel(ch):
        vmax = np.percentile(ch, 99.5) if ch.size > 0 else 1.0
        if vmax <= 0: vmax = 1.0
        return np.clip(ch / vmax, 0.0, 1.0)

    tubulin_norm = normalize_channel(tubulin)
    dapi_norm = normalize_channel(dapi)

    # Determine the region of interest (ROI) mask
    # We want to calculate correlation per cell to avoid background bias
    
    labels = None
    
    # Check if segmentation masks are provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        mask_input = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions
        if mask_input.shape == arr.shape[:2]:
            labels = mask_input
    
    # Fallback: If no segmentation mask, generate a simple foreground mask
    # Correlation on pure background (0,0) is undefined or misleading
    if labels is None:
        # Create a combined foreground mask from both channels
        # Use a low threshold to capture cell bodies
        try:
            t_thresh = threshold_otsu(tubulin_norm)
            d_thresh = threshold_otsu(dapi_norm)
            foreground = (tubulin_norm > t_thresh) | (dapi_norm > d_thresh)
            
            # Label connected components to treat distinct blobs as "cells"
            labels, num_features = ndimage.label(foreground)
        except Exception:
            # If otsu fails (e.g. empty image), return 0
            return 0.0

    # Get unique cell IDs (excluding background 0)
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]

    if len(unique_labels) == 0:
        return 0.0

    correlations = []

    # Iterate over each cell to compute local Pearson correlation
    for label_id in unique_labels:
        # Extract pixels for this cell
        # Using boolean indexing creates a 1D array of pixel values
        cell_mask = (labels == label_id)
        
        # Skip very small fragments
        if np.sum(cell_mask) < 10:
            continue
            
        val_tubulin = tubulin[cell_mask]
        val_dapi = dapi[cell_mask]

        # Calculate Pearson Correlation
        # Formula: cov(x,y) / (std(x) * std(y))
        
        # Check for zero variance (flat signal) to avoid division by zero
        if np.std(val_tubulin) == 0 or np.std(val_dapi) == 0:
            # If one channel is flat and the other isn't, correlation is 0.
            # If both are flat, it's undefined, but 0 is a safe fallback for "no correlation".
            correlations.append(0.0)
        else:
            # np.corrcoef returns the correlation matrix: [[1, r], [r, 1]]
            # We want the off-diagonal element [0, 1]
            r = np.corrcoef(val_tubulin, val_dapi)[0, 1]
            
            # Handle potential NaNs from numerical instability
            if not np.isnan(r):
                correlations.append(r)

    # Aggregate results
    if not correlations:
        return 0.0
    
    # Return the mean correlation across all cells in the image
    result = np.mean(correlations)

    return float(result)
