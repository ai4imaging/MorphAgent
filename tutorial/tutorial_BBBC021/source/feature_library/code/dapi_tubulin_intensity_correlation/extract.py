def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    
    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Expected shape: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract relevant channels based on dataset description
    # Channel 1: Green (Tubulin)
    # Channel 2: Blue (DAPI)
    tubulin_channel = arr[:, :, 1]
    dapi_channel = arr[:, :, 2]

    # Define Region of Interest (ROI)
    # Correlation should ideally be calculated only on the cellular foreground to avoid 
    # the large background (0,0) correlation inflating the result.
    
    mask = None
    
    # 1. Try to use provided segmentation masks
    if len(segmentation_masks) > 0:
        # Combine all available masks to get a general "cellular area" mask
        # Masks are typically labeled integers (0=bg, 1..N=cells)
        combined_mask = np.zeros(arr.shape[:2], dtype=bool)
        for seg in segmentation_masks:
            if seg is not None and seg.shape == arr.shape[:2]:
                combined_mask = np.logical_or(combined_mask, seg > 0)
        
        if np.any(combined_mask):
            mask = combined_mask

    # 2. Fallback: Intensity-based thresholding if no valid mask provided
    if mask is None:
        # Create a simple foreground mask where there is signal in either channel
        # Use a low threshold to separate background noise from signal
        # Assuming uint8 input [0, 255], a threshold of ~10-15 is usually safe for background
        # Since we converted to float32 but didn't normalize to [0,1] yet, values are likely 0-255.
        # If the input was already normalized, we need to check range.
        
        # Check value range to set appropriate threshold
        max_val = np.max(arr)
        threshold = 10.0 if max_val > 1.0 else 0.04  # 10/255 approx 0.04
        
        mask = (tubulin_channel > threshold) | (dapi_channel > threshold)

    # Flatten arrays and apply mask
    # If mask is empty (no signal), return 0.0
    if not np.any(mask):
        return 0.0

    dapi_pixels = dapi_channel[mask]
    tubulin_pixels = tubulin_channel[mask]

    # Check for sufficient variance to calculate correlation
    # If a channel is constant (std=0), correlation is undefined (division by zero)
    if np.std(dapi_pixels) == 0 or np.std(tubulin_pixels) == 0:
        return 0.0

    # Calculate Pearson Correlation Coefficient
    # Returns a matrix [[1.0, r], [r, 1.0]]
    correlation_matrix = np.corrcoef(dapi_pixels, tubulin_pixels)
    
    # Extract the correlation value
    if correlation_matrix.size > 1:
        pcc = correlation_matrix[0, 1]
    else:
        return 0.0

    # Handle NaN results (can happen if variance is effectively zero due to precision)
    if np.isnan(pcc):
        return 0.0

    return float(pcc)
