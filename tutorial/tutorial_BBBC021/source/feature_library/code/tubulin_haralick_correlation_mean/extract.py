def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.feature import graycomatrix, graycoprops
    from skimage import img_as_ubyte
    import warnings

    # Suppress warnings that might arise from constant images (divide by zero in correlation)
    warnings.filterwarnings("ignore")

    # 1. Input Validation and Channel Extraction
    # Ensure input is a numpy array
    img = np.asarray(img)

    # Check dimensionality. We expect (H, W, C) = (512, 512, 3)
    # If it's 2D (H, W), we can't reliably guess the tubulin channel, so we return 0.0
    if img.ndim != 3:
        return 0.0
    
    # Extract the Tubulin channel (Channel 1 - Green)
    # Channel 0: Actin, Channel 1: Tubulin, Channel 2: DAPI
    try:
        tubulin_channel = img[:, :, 1]
    except IndexError:
        return 0.0

    # 2. Preprocessing and Masking
    # The GLCM calculation requires integer types. The dataset is uint8.
    # If the input is float, we need to convert it safely.
    if tubulin_channel.dtype.kind == 'f':
        # Normalize to 0-1 if not already, then convert to uint8
        if tubulin_channel.max() > 1.0:
            tubulin_channel = tubulin_channel / 255.0
        tubulin_channel = img_as_ubyte(np.clip(tubulin_channel, 0, 1))
    else:
        # Ensure it is uint8
        tubulin_channel = tubulin_channel.astype(np.uint8)

    # Handle segmentation masks to define the Region of Interest (ROI)
    # If masks are provided, we only want to analyze texture within the cells/tissue
    # to avoid the background (zeros) dominating the correlation statistics.
    roi_mask = None
    if len(segmentation_masks) > 0:
        # Combine all available masks
        combined_mask = np.zeros(tubulin_channel.shape, dtype=bool)
        for mask in segmentation_masks:
            if mask is not None:
                # Ensure mask matches image shape (handle potential squeezing issues)
                if mask.shape == tubulin_channel.shape:
                    combined_mask = np.logical_or(combined_mask, mask > 0)
        
        if np.any(combined_mask):
            roi_mask = combined_mask

    # If we have a mask, we have a choice:
    # 1. Mask out background pixels to 0.
    # 2. Compute GLCM only on masked pixels (hard with rectangular matrix requirement).
    # Standard approach: Set background to 0. Note that the large number of 0-0 transitions
    # in the background will affect the matrix, but 'correlation' handles this better than 'contrast'.
    # However, a better approach for biological texture is often to just crop or zero out background.
    
    if roi_mask is not None:
        # Apply mask: keep original values where mask is True, 0 elsewhere
        tubulin_channel = tubulin_channel * roi_mask.astype(np.uint8)

    # 3. GLCM Computation
    # Parameters:
    # - distances: [1] (pixel immediately next to current pixel)
    # - angles: [0, 45, 90, 135] degrees (in radians) for rotational invariance
    # - levels: 256 (for uint8)
    # - symmetric: True
    # - normed: True (required for correlation)
    
    # Check if image is empty or constant to avoid errors
    if np.max(tubulin_channel) == np.min(tubulin_channel):
        return 0.0

    try:
        glcm = graycomatrix(
            tubulin_channel, 
            distances=[1], 
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
            levels=256, 
            symmetric=True, 
            normed=True
        )

        # 4. Feature Extraction
        # Calculate Haralick Correlation
        # Returns a 1x4 array (one value for each angle)
        correlation_values = graycoprops(glcm, 'correlation')

        # 5. Aggregation
        # Average across the 4 directions to get a rotation-invariant feature
        mean_correlation = np.mean(correlation_values)

        # Handle NaN (can happen if variance is 0, though we checked for constant image above)
        if np.isnan(mean_correlation):
            return 0.0
            
        return float(mean_correlation)

    except Exception:
        return 0.0
