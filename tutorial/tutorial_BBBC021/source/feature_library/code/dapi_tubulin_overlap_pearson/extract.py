def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    
    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset Description: (Height, Width, Channels) = (512, 512, 3)
    # Channel 0: Actin (Red)
    # Channel 1: Tubulin (Green) -> Target
    # Channel 2: DAPI (Blue) -> Target
    
    if arr.ndim != 3 or arr.shape[2] != 3:
        # If dimensions don't match expected (H, W, 3), try to adapt or return 0
        if arr.ndim == 2:
            # Cannot compute overlap between channels if only 1 channel exists
            return 0.0
        return 0.0

    # Extract relevant channels
    # Channel 1: Tubulin (Green)
    # Channel 2: DAPI (Blue)
    tubulin = arr[:, :, 1]
    dapi = arr[:, :, 2]

    # Determine the mask to use for defining biological regions
    # We want to calculate correlation ONLY within cells/nuclei, not on the background.
    # Including background (0,0) pixels artificially inflates correlation.
    
    mask = None
    
    # Check if segmentation masks are provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use the first provided mask
        mask = segmentation_masks[0]
        # Ensure mask shape matches image spatial dimensions
        if mask.shape != tubulin.shape:
            # If mask is 3D (e.g. labeled volume) but image is 2D, or mismatch
            # Try to project or resize? For safety, if shapes mismatch significantly, ignore mask
            if mask.ndim == 3 and mask.shape[:2] == tubulin.shape:
                mask = np.max(mask, axis=2) # MIP projection
            elif mask.shape != tubulin.shape:
                mask = None

    # If no valid mask provided, generate a simple foreground mask based on DAPI
    if mask is None:
        # Simple thresholding on DAPI to find biological material
        # DAPI is usually high contrast.
        # Use a safe low threshold to capture most cell area
        threshold_val = np.percentile(dapi, 90) * 0.2
        if threshold_val < 5: threshold_val = 5 # Minimum noise floor
        mask = (dapi > threshold_val).astype(np.int32)
        # Label connected components to treat them as instances
        mask, _ = ndimage.label(mask)

    # Ensure mask is integer type for labeling
    mask = mask.astype(np.int32)
    
    # Get unique labels (excluding background 0)
    unique_labels = np.unique(mask)
    if unique_labels.size > 0 and unique_labels[0] == 0:
        unique_labels = unique_labels[1:]
        
    if len(unique_labels) == 0:
        return 0.0

    correlations = []

    # Iterate over each cell/object to compute local correlation
    # Averaging per-cell correlation is more robust than global correlation
    # which can be dominated by intensity differences between cells.
    for label in unique_labels:
        # Create boolean mask for current cell
        cell_mask = (mask == label)
        
        # Extract pixel values for this cell
        dapi_vals = dapi[cell_mask]
        tubulin_vals = tubulin[cell_mask]
        
        # Check for sufficient pixels
        if len(dapi_vals) < 5:
            continue
            
        # Check for variance (if constant, correlation is undefined)
        dapi_std = np.std(dapi_vals)
        tubulin_std = np.std(tubulin_vals)
        
        if dapi_std < 1e-6 or tubulin_std < 1e-6:
            # If one channel is flat, correlation is technically 0 or undefined.
            # In biological context, if signal is missing in one, no correlation.
            correlations.append(0.0)
            continue
            
        # Calculate Pearson correlation coefficient
        # np.corrcoef returns a matrix [[1, r], [r, 1]]
        r = np.corrcoef(dapi_vals, tubulin_vals)[0, 1]
        
        if not np.isnan(r):
            correlations.append(r)

    # Aggregate results
    if len(correlations) == 0:
        result = 0.0
    else:
        result = np.mean(correlations)

    return float(result)
