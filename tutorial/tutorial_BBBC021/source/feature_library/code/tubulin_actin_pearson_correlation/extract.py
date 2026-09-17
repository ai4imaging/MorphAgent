def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    
    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset description: (512, 512, 3) - 2D composite image
    # Channel 0: Actin (Red)
    # Channel 1: Tubulin (Green)
    # Channel 2: DAPI (Blue)
    
    # Check for valid shape
    if arr.ndim != 3 or arr.shape[2] < 2:
        # If not enough channels or wrong dimensions, return 0.0
        return 0.0

    # Extract relevant channels
    # Feature: tubulin_actin_pearson_correlation
    actin_channel = arr[:, :, 0]   # Red
    tubulin_channel = arr[:, :, 1] # Green

    # Define the Region of Interest (ROI)
    # Correlation should be calculated on biological foreground to avoid 
    # high correlation driven by background noise (where both channels are near 0).
    
    mask = None
    
    # Check if segmentation masks are provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use the first available mask (assuming it covers cells/tissue)
        # Masks are typically labeled integers (0=bg, 1..N=cells)
        seg = segmentation_masks[0]
        
        # Ensure mask matches image spatial dimensions
        if seg.shape == actin_channel.shape:
            mask = seg > 0
        elif seg.ndim == 3 and seg.shape[:2] == actin_channel.shape:
             # Handle case where mask might be 3D (e.g. one-hot or same as image)
             mask = np.any(seg > 0, axis=2)
    
    # Fallback: If no valid segmentation mask, generate a foreground mask based on intensity
    if mask is None:
        # Simple thresholding to identify foreground
        # We use a low threshold to exclude pure background noise
        # Calculate a robust background level (e.g., slightly above min or mean of lower percentile)
        # Here we use a fixed low threshold relative to data range or a simple Otsu-like approximation
        # Since data is uint8 (0-255), a value like 10-15 is usually safe for background exclusion in fluorescence
        # Or use mean as a dynamic threshold
        
        threshold_actin = np.mean(actin_channel) * 0.5
        threshold_tubulin = np.mean(tubulin_channel) * 0.5
        
        # Pixel is foreground if it has signal in EITHER channel
        mask = (actin_channel > threshold_actin) | (tubulin_channel > threshold_tubulin)

    # Flatten the arrays based on the mask
    # We only care about the pixel-wise correspondence
    if np.sum(mask) < 2:
        # Not enough foreground pixels to calculate correlation
        return 0.0

    flat_actin = actin_channel[mask]
    flat_tubulin = tubulin_channel[mask]

    # Calculate Pearson Correlation
    # Formula: cov(X, Y) / (std(X) * std(Y))
    
    # Check for zero variance (constant signal) which causes division by zero
    if np.std(flat_actin) == 0 or np.std(flat_tubulin) == 0:
        return 0.0

    # Use numpy's correlation coefficient function
    # Returns a matrix [[1.0, r], [r, 1.0]]
    corr_matrix = np.corrcoef(flat_actin, flat_tubulin)
    
    # Extract the correlation coefficient
    pearson_r = corr_matrix[0, 1]

    # Handle NaN result (can happen if inputs are invalid despite checks)
    if np.isnan(pearson_r):
        return 0.0
        
    # Clip to valid range [-1, 1] to handle floating point errors
    pearson_r = np.clip(pearson_r, -1.0, 1.0)

    return float(pearson_r)
