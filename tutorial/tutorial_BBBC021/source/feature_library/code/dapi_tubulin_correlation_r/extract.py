def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.filters import threshold_otsu

    # Convert to appropriate array type (float32 to avoid overflow during stats)
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Expected shape: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract relevant channels
    # Channel 1: Tubulin (Green)
    # Channel 2: DAPI (Blue)
    tubulin = arr[:, :, 1]
    dapi = arr[:, :, 2]

    # Define Region of Interest (ROI)
    # We only want to compute correlation on "cellular" pixels, excluding the vast black background.
    # Including background (0,0) pixels artificially inflates positive correlation.
    
    roi_mask = None

    # 1. Try to use provided segmentation masks
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use the first available mask (assuming it covers cells/nuclei)
        # Masks are typically labeled integers (0=bg, 1..N=cells)
        mask = segmentation_masks[0]
        
        # Ensure mask shape matches image spatial dimensions
        if mask.shape == dapi.shape:
            roi_mask = mask > 0
    
    # 2. Fallback: Generate a mask from signal intensity if no external mask is provided
    if roi_mask is None:
        # We define the "cell" area as regions where EITHER DAPI or Tubulin has signal.
        # This captures the union of the nucleus and the cytoplasm.
        
        # Calculate thresholds safely
        try:
            # Use a small epsilon to handle completely black images
            if dapi.max() > dapi.min():
                t_dapi = threshold_otsu(dapi)
            else:
                t_dapi = dapi.min()
                
            if tubulin.max() > tubulin.min():
                t_tubulin = threshold_otsu(tubulin)
            else:
                t_tubulin = tubulin.min()
                
            roi_mask = (dapi > t_dapi) | (tubulin > t_tubulin)
        except Exception:
            # Fallback for extremely low signal or errors
            roi_mask = (dapi > 0) | (tubulin > 0)

    # Extract pixels within the ROI
    # Flatten the arrays to 1D vectors based on the mask
    if roi_mask.sum() == 0:
        return 0.0

    dapi_pixels = dapi[roi_mask]
    tubulin_pixels = tubulin[roi_mask]

    # Compute Pearson Correlation Coefficient
    # Check for zero variance to avoid division by zero
    if np.std(dapi_pixels) == 0 or np.std(tubulin_pixels) == 0:
        return 0.0

    # np.corrcoef returns the correlation matrix: [[1.0, r], [r, 1.0]]
    # We want the off-diagonal element [0, 1]
    correlation_matrix = np.corrcoef(dapi_pixels, tubulin_pixels)
    
    if np.isnan(correlation_matrix).any():
        return 0.0
        
    r = correlation_matrix[0, 1]

    return float(r)
