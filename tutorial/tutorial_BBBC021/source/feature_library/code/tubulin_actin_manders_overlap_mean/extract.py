def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.filters import threshold_otsu

    # Convert to float32 for calculations to avoid overflow and allow precise division
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality
    # Expected shape: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract relevant channels
    # Channel 0: Actin (Red)
    # Channel 1: Tubulin (Green)
    actin = arr[..., 0]
    tubulin = arr[..., 1]

    # Normalize to [0, 1] based on uint8 range
    actin = actin / 255.0
    tubulin = tubulin / 255.0

    # Define Region of Interest (ROI)
    # If segmentation masks are provided, use them to define the cellular area.
    # If not, generate a foreground mask based on signal intensity.
    roi_mask = None
    
    if len(segmentation_masks) > 0:
        # Combine all available masks (logical OR)
        # Masks are typically labeled integers (0=bg, 1+=cells)
        combined_mask = np.zeros(actin.shape, dtype=bool)
        for mask in segmentation_masks:
            if mask is not None:
                # Ensure mask matches image spatial dimensions
                if mask.shape == actin.shape:
                    combined_mask = combined_mask | (mask > 0)
        
        if np.any(combined_mask):
            roi_mask = combined_mask

    # If no valid mask provided or mask is empty, generate one from image content
    if roi_mask is None:
        # Create a rough foreground mask using the sum of channels
        # This avoids analyzing empty background space
        total_signal = actin + tubulin
        if np.max(total_signal) > 0:
            try:
                # Use Otsu to separate foreground from background
                bg_thresh = threshold_otsu(total_signal)
                roi_mask = total_signal > bg_thresh
            except Exception:
                # Fallback if Otsu fails (e.g., uniform image)
                roi_mask = total_signal > 0.05
        else:
            return 0.0

    # Apply ROI mask: Zero out background pixels in both channels
    # This ensures we only calculate statistics within the cells
    actin_roi = actin * roi_mask
    tubulin_roi = tubulin * roi_mask

    # Determine signal thresholds for Manders' coefficients
    # Manders' requires defining what is "signal" vs "noise" for each channel independently.
    # We calculate thresholds based on the pixel values within the ROI.
    
    # Get pixels within ROI for threshold calculation
    valid_pixels_actin = actin_roi[roi_mask]
    valid_pixels_tubulin = tubulin_roi[roi_mask]

    if valid_pixels_actin.size == 0 or valid_pixels_tubulin.size == 0:
        return 0.0

    # Calculate thresholds (Otsu is standard for this)
    # If the signal is too uniform or zero, handle gracefully
    try:
        t_actin = threshold_otsu(valid_pixels_actin) if np.max(valid_pixels_actin) > 0 else 0.0
    except:
        t_actin = 0.0
        
    try:
        t_tubulin = threshold_otsu(valid_pixels_tubulin) if np.max(valid_pixels_tubulin) > 0 else 0.0
    except:
        t_tubulin = 0.0

    # Calculate Manders' Coefficients
    # M1: Fraction of Actin overlapping Tubulin
    # Sum of Actin pixels where Tubulin is present (above threshold)
    # divided by total sum of Actin pixels.
    
    actin_sum = np.sum(actin_roi)
    if actin_sum > 0:
        # Mask where Tubulin is considered "present"
        tubulin_present_mask = tubulin_roi > t_tubulin
        # Sum actin intensity only in those locations
        actin_overlapping = np.sum(actin_roi[tubulin_present_mask])
        m1 = actin_overlapping / actin_sum
    else:
        m1 = 0.0

    # M2: Fraction of Tubulin overlapping Actin
    # Sum of Tubulin pixels where Actin is present (above threshold)
    # divided by total sum of Tubulin pixels.
    
    tubulin_sum = np.sum(tubulin_roi)
    if tubulin_sum > 0:
        # Mask where Actin is considered "present"
        actin_present_mask = actin_roi > t_actin
        # Sum tubulin intensity only in those locations
        tubulin_overlapping = np.sum(tubulin_roi[actin_present_mask])
        m2 = tubulin_overlapping / tubulin_sum
    else:
        m2 = 0.0

    # The feature is the mean of the two coefficients
    result = (m1 + m2) / 2.0

    return float(result)
