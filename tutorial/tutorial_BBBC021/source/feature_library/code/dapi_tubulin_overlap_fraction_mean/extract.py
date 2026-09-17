def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.filters import threshold_otsu
    from scipy import ndimage

    # Convert to appropriate array type
    img_arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3)
    if img_arr.ndim != 3 or img_arr.shape[2] < 3:
        return 0.0

    # Channel Mapping based on dataset description:
    # Channel 0: Actin (Red)
    # Channel 1: Tubulin (Green) -> Target Signal
    # Channel 2: DAPI (Blue) -> Nuclear Mask
    tubulin_channel = img_arr[..., 1]
    dapi_channel = img_arr[..., 2]

    # Helper function for normalization
    def normalize_minmax(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn <= 1e-7:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    # Normalize channels to [0, 1] for consistent thresholding
    tubulin_norm = normalize_minmax(tubulin_channel)
    dapi_norm = normalize_minmax(dapi_channel)

    # --- 1. Generate Nuclear Mask (DAPI) ---
    # Apply Gaussian blur to reduce noise and smooth boundaries
    dapi_smooth = ndimage.gaussian_filter(dapi_norm, sigma=2.0)
    
    try:
        # Use Otsu's method to find a global threshold for nuclei
        thresh_dapi = threshold_otsu(dapi_smooth)
        mask_nuclei = dapi_smooth > thresh_dapi
        
        # Fill holes to ensure solid nuclear regions
        mask_nuclei = ndimage.binary_fill_holes(mask_nuclei)
        
        # CRITICAL FIX based on feedback: "Check if nuclear mask is overly inclusive"
        # Erode the mask slightly to ensure we are strictly inside the nucleus and 
        # avoid the perinuclear region where tubulin (centrosome) is dense.
        mask_nuclei = ndimage.binary_erosion(mask_nuclei, iterations=1)
        
        # Sanity check: If nuclei cover > 60% of image, thresholding likely failed (background noise)
        if np.mean(mask_nuclei) > 0.6:
            # Fallback to a stricter percentile-based threshold
            mask_nuclei = dapi_smooth > np.percentile(dapi_smooth, 95)
            
    except Exception:
        # Fallback for empty or uniform images
        return 0.0

    # --- 2. Generate Tubulin Signal Mask ---
    # Tubulin is fibrous; use lighter smoothing
    tubulin_smooth = ndimage.gaussian_filter(tubulin_norm, sigma=1.0)
    
    try:
        # Use Otsu to separate tubulin structure from background
        thresh_tubulin = threshold_otsu(tubulin_smooth)
        mask_tubulin = tubulin_smooth > thresh_tubulin
    except Exception:
        return 0.0

    # --- 3. Compute Overlap Fraction ---
    # Feature: Fraction of the Tubulin signal that spatially overlaps with the nuclear mask.
    # We use intensity-weighted calculation to represent the "amount of signal".
    
    # Identify pixels that are both Tubulin-positive and Nucleus-positive
    overlap_mask = mask_nuclei & mask_tubulin
    
    # Sum of Tubulin intensity in the overlap region
    overlap_intensity = np.sum(tubulin_norm[overlap_mask])
    
    # Total Tubulin intensity (in the Tubulin mask, excluding background)
    total_tubulin_intensity = np.sum(tubulin_norm[mask_tubulin])
    
    # Avoid division by zero
    if total_tubulin_intensity == 0:
        return 0.0
        
    result = overlap_intensity / total_tubulin_intensity

    return float(result)
