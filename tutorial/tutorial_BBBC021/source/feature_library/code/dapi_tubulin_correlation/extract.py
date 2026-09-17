def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    
    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3) - Height, Width, Channels
    if arr.ndim != 3 or arr.shape[2] != 3:
        # If dimensions don't match expected (H, W, C), return 0.0
        return 0.0

    # Extract relevant channels based on dataset description
    # Channel 1: Green (Tubulin)
    # Channel 2: Blue (DAPI)
    tubulin_channel = arr[:, :, 1]
    dapi_channel = arr[:, :, 2]

    # Define Region of Interest (ROI)
    # We want to compute correlation only on the "biological foreground" to avoid 
    # high correlation driven by the large black background.
    
    roi_mask = None

    # 1. Try to use segmentation masks if available
    if len(segmentation_masks) > 0:
        # Combine all available masks
        combined_mask = np.zeros(dapi_channel.shape, dtype=bool)
        for mask in segmentation_masks:
            if mask is not None:
                # Ensure mask matches image spatial dimensions (handle potential 2D vs 3D issues)
                if mask.shape == dapi_channel.shape:
                    combined_mask = np.logical_or(combined_mask, mask > 0)
                elif mask.ndim == 3 and mask.shape[:2] == dapi_channel.shape:
                    # If mask is 3D (e.g. labeled volume), project or slice? 
                    # Assuming masks passed here match the 2D image plane.
                    # If mask is (H, W, 1), squeeze it.
                    combined_mask = np.logical_or(combined_mask, mask.squeeze() > 0)
        
        if np.any(combined_mask):
            roi_mask = combined_mask

    # 2. Fallback: Create a foreground mask based on intensity if no valid segmentation provided
    if roi_mask is None:
        # Use DAPI channel to define cellular regions (nuclei are usually distinct)
        # A simple threshold is often robust enough for correlation analysis
        # Calculate a background threshold (e.g., mean + std or a fixed low value for uint8)
        # Since we converted to float but didn't normalize to [0,1] yet, values are 0-255.
        # Let's use a safe lower bound threshold to exclude pure background.
        threshold = np.mean(dapi_channel) + 0.5 * np.std(dapi_channel)
        # Ensure threshold is at least somewhat above 0 to exclude empty background
        threshold = max(threshold, 5.0) 
        roi_mask = (dapi_channel > threshold) | (tubulin_channel > threshold)

    # Flatten the arrays based on the mask
    # This selects only the pixels within the ROI
    dapi_pixels = dapi_channel[roi_mask]
    tubulin_pixels = tubulin_channel[roi_mask]

    # Check for sufficient data points
    if dapi_pixels.size < 2:
        return 0.0

    # Check for zero variance (flat signals)
    # If a channel is constant in the ROI, correlation is undefined (division by zero)
    dapi_std = np.std(dapi_pixels)
    tubulin_std = np.std(tubulin_pixels)

    if dapi_std == 0 or tubulin_std == 0:
        return 0.0

    # Compute Pearson Correlation Coefficient
    # np.corrcoef returns a matrix [[1, r], [r, 1]]
    correlation_matrix = np.corrcoef(dapi_pixels, tubulin_pixels)
    
    # Extract the off-diagonal element
    result = correlation_matrix[0, 1]

    # Handle potential NaN results (though std check usually prevents this)
    if np.isnan(result):
        return 0.0

    return float(result)
