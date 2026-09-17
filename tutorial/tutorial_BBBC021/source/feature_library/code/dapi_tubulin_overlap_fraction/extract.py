def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_erosion, disk, binary_opening
    from scipy import ndimage

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Check dimensionality
    # Expected: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Channel mapping based on dataset description:
    # Channel 0: Actin (Red)
    # Channel 1: Tubulin (Green)
    # Channel 2: DAPI (Blue)
    
    tubulin_ch = arr[:, :, 1]
    dapi_ch = arr[:, :, 2]

    # --- 1. Define Nuclear Mask ---
    # Strategy: Use provided segmentation if available, otherwise compute from DAPI channel
    nuclear_mask = None
    
    # Check if segmentation masks are provided
    # We look for a mask that likely corresponds to nuclei. 
    # Usually, if multiple masks are provided, one is nuclei and one is cells/cytoplasm.
    # Without specific metadata on mask order, we can try to guess or fallback to computing it.
    # However, the prompt implies we should use them if available.
    # Let's prioritize computing it fresh from DAPI to ensure we strictly follow the "DAPI is blue" rule 
    # and avoid potential misalignment or mislabeling in external masks for this specific overlap metric.
    # But if a mask is passed, we can use it to refine the region of interest.
    
    # Given the critic feedback about "Verifying that the segmentation mask... is correctly identifying nuclei",
    # it is safer to derive the nuclear mask directly from the DAPI channel using a robust method, 
    # as we know exactly which channel is DAPI (Ch 2).
    
    # Normalize DAPI
    dapi_norm = dapi_ch / 255.0
    
    # Threshold DAPI
    try:
        thresh_dapi = threshold_otsu(dapi_norm)
        # Make threshold slightly more stringent to avoid cytoplasmic haze
        thresh_dapi = max(thresh_dapi, 0.1) 
        nuclear_mask = dapi_norm > thresh_dapi
        
        # Fill holes to ensure solid nuclei
        nuclear_mask = ndimage.binary_fill_holes(nuclear_mask)
        
        # Remove small artifacts
        nuclear_mask = binary_opening(nuclear_mask, footprint=disk(2))
    except Exception:
        return 0.0

    if np.sum(nuclear_mask) == 0:
        return 0.0

    # --- 2. Define Tubulin Mask ---
    # Normalize Tubulin
    tubulin_norm = tubulin_ch / 255.0
    
    # Threshold Tubulin
    # Critic feedback: "Use a more stringent threshold for tubulin"
    try:
        thresh_tub = threshold_otsu(tubulin_norm)
        # Increase threshold significantly to capture only real structures, not background noise
        # Multiplier 1.2 or adding a fixed offset helps reduce false positives in the nucleus
        thresh_tub = max(thresh_tub * 1.2, 0.15) 
        
        tubulin_mask = tubulin_norm > thresh_tub
        
        # Critic feedback: "Applying morphological operations (like erosion) to remove small tubulin specks"
        # Remove small noise specks
        tubulin_mask = binary_opening(tubulin_mask, footprint=disk(1))
        # Erode slightly to thin the structures and detach them from boundaries
        tubulin_mask = binary_erosion(tubulin_mask, footprint=disk(1))
        
    except Exception:
        return 0.0

    # --- 3. Compute Overlap ---
    
    # We are interested in Tubulin signal *inside* the Nucleus.
    # Intersection
    overlap_mask = np.logical_and(nuclear_mask, tubulin_mask)
    
    overlap_area = np.sum(overlap_mask)
    nuclear_area = np.sum(nuclear_mask)
    
    if nuclear_area == 0:
        return 0.0
        
    fraction = overlap_area / nuclear_area
    
    # Critic feedback: "Add a sanity check to return 0 if the overlap fraction exceeds a biological threshold (e.g., >0.1)"
    # In healthy interphase cells, tubulin is excluded from the nucleus. 
    # High overlap usually means segmentation failure (tubulin bleeding into DAPI channel or vice versa) 
    # or a specific mitotic state (prometaphase) which is rare in a general population.
    # However, if the feature is meant to detect this breakdown, we shouldn't zero it out completely if it's real.
    # But for "unreasonable" results (like > 50% overlap which is physically unlikely even in breakdown unless the cell is flat),
    # we might want to be careful. 
    # The critic suggested > 0.1 as a flag for potential errors. 
    # We will return the raw fraction but ensure the calculation was conservative.
    
    return float(fraction)
