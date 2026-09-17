def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.filters import threshold_otsu

    # Convert to appropriate array type and handle dimensions
    # Image shape is expected to be (512, 512, 3)
    arr = np.asarray(img, dtype=np.float64) # Use float64 for accumulation to avoid overflow

    if arr.ndim != 3 or arr.shape[2] != 3:
        # If dimensions are unexpected (e.g., not 3 channels), return 0.0
        return 0.0

    # Extract specific channels based on dataset description
    # Channel 0: Actin (Red) - Cytoskeleton
    # Channel 2: DAPI (Blue) - Nucleus
    actin_channel = arr[:, :, 0]
    dapi_channel = arr[:, :, 2]

    # Determine the Region of Interest (ROI) / Foreground Mask
    # We want to sum intensities only within cellular regions to avoid background noise skewing the ratio.
    
    mask = None
    
    # Check if external segmentation masks are provided
    if len(segmentation_masks) > 0:
        # Combine all provided masks. Assuming masks are labeled (0=bg, >0=cell)
        # We create a binary mask where any provided mask indicates a cell
        combined_mask = np.zeros(actin_channel.shape, dtype=bool)
        for seg in segmentation_masks:
            if seg is not None and seg.shape == actin_channel.shape:
                combined_mask = combined_mask | (seg > 0)
        
        if np.any(combined_mask):
            mask = combined_mask

    # Fallback: If no valid segmentation provided, generate a mask using Otsu thresholding
    if mask is None:
        # We need a mask that covers both the nucleus and the cytoplasm.
        # Calculate thresholds for both channels.
        try:
            # Check if image is not empty/constant
            if np.min(actin_channel) == np.max(actin_channel):
                thresh_actin = np.min(actin_channel)
            else:
                thresh_actin = threshold_otsu(actin_channel)
                
            if np.min(dapi_channel) == np.max(dapi_channel):
                thresh_dapi = np.min(dapi_channel)
            else:
                thresh_dapi = threshold_otsu(dapi_channel)
            
            # Create a union mask: pixel is valid if it's significant in EITHER channel
            mask = (actin_channel > thresh_actin) | (dapi_channel > thresh_dapi)
        except Exception:
            # Fallback for extremely low signal or errors in thresholding
            mask = np.ones(actin_channel.shape, dtype=bool)

    # Ensure mask is not empty to avoid division by zero later
    if not np.any(mask):
        return 0.0

    # Calculate Total Intensity within the ROI
    # We sum the raw intensities. Normalization to [0,1] isn't strictly necessary for a ratio 
    # of the same image, but using the raw float values is standard for intensity ratios.
    total_actin_intensity = np.sum(actin_channel[mask])
    total_dapi_intensity = np.sum(dapi_channel[mask])

    # Compute Ratio: Actin / DAPI
    # Avoid division by zero
    if total_dapi_intensity <= 1e-9:
        return 0.0
    
    ratio = total_actin_intensity / total_dapi_intensity

    return float(ratio)
