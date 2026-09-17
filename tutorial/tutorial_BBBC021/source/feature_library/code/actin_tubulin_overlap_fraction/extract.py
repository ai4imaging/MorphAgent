def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.filters import threshold_otsu, gaussian
    
    # Convert to appropriate array type
    # Image is (512, 512, 3) uint8
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Expected shape: (H, W, C) = (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        # If not the expected 3-channel image, return 0.0
        return 0.0

    # Extract relevant channels
    # Channel 0: Actin (Cytoskeleton)
    # Channel 1: Tubulin (Microtubules)
    actin_channel = arr[:, :, 0]
    tubulin_channel = arr[:, :, 1]

    # Preprocessing: Mild smoothing to reduce noise
    # Using a small sigma to preserve structure but remove pixel-level noise
    actin_smooth = gaussian(actin_channel, sigma=1.0)
    tubulin_smooth = gaussian(tubulin_channel, sigma=1.0)

    # Define Region of Interest (ROI) - The "Cell Area"
    # This serves as the denominator for the fraction
    roi_mask = None

    # Check if segmentation masks are provided
    if len(segmentation_masks) > 0:
        # Use the first available mask
        # Assuming masks are labeled (0=bg, 1..N=cells)
        # We treat any non-zero value as cell area
        mask = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions
        if mask.shape == arr.shape[:2]:
            roi_mask = mask > 0
    
    # Fallback if no mask provided or mask was invalid shape
    if roi_mask is None:
        # Create a tissue mask based on combined intensity
        # Combine actin and tubulin as they define the cell body better than DAPI
        combined_intensity = actin_smooth + tubulin_smooth
        
        # Check if image is empty or near empty
        if np.max(combined_intensity) < 1e-6:
            return 0.0
            
        try:
            # Calculate global threshold for background separation
            bg_thresh = threshold_otsu(combined_intensity)
            roi_mask = combined_intensity > bg_thresh
        except Exception:
            # Fallback for extremely low contrast images where Otsu might fail
            roi_mask = combined_intensity > np.mean(combined_intensity)

    # Calculate ROI area (denominator)
    roi_area = np.sum(roi_mask)
    
    # If no cell area detected, return 0.0
    if roi_area == 0:
        return 0.0

    # Determine thresholds for Actin and Tubulin positivity
    # We calculate thresholds based on pixels WITHIN the ROI to be adaptive
    # to the specific cell population's intensity profile
    
    # Get pixel values within the ROI
    actin_pixels = actin_smooth[roi_mask]
    tubulin_pixels = tubulin_smooth[roi_mask]
    
    # Calculate adaptive thresholds
    # If the signal is uniform (std dev ~ 0), Otsu fails or is meaningless.
    # We add checks to ensure valid thresholding.
    
    try:
        if np.std(actin_pixels) > 1e-6:
            thresh_actin = threshold_otsu(actin_pixels)
        else:
            thresh_actin = np.min(actin_pixels) # All pixels considered positive if uniform > 0? Or use mean.
            
        if np.std(tubulin_pixels) > 1e-6:
            thresh_tubulin = threshold_otsu(tubulin_pixels)
        else:
            thresh_tubulin = np.min(tubulin_pixels)
    except Exception:
        # Fallback if Otsu fails (e.g. all pixels same value)
        thresh_actin = np.mean(actin_pixels)
        thresh_tubulin = np.mean(tubulin_pixels)

    # Create binary masks for high intensity regions
    # We apply the threshold to the whole smoothed image, then mask with ROI
    actin_binary = (actin_smooth > thresh_actin) & roi_mask
    tubulin_binary = (tubulin_smooth > thresh_tubulin) & roi_mask

    # Calculate Overlap (Colocalization)
    # Intersection of both binary masks
    overlap_mask = actin_binary & tubulin_binary
    
    overlap_area = np.sum(overlap_mask)

    # Calculate Fraction
    # Fraction of the total cell area (ROI) where both signals are high
    fraction = overlap_area / roi_area

    return float(fraction)
