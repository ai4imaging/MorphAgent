def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.filters import threshold_otsu
    
    # Convert to appropriate array type
    # The dataset is uint8, but we need float for statistical calculations
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Expected shape: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Channel Mapping based on dataset description:
    # Channel 0: Actin (Red) - Cytoskeleton
    # Channel 1: Tubulin (Green) - Microtubules (TARGET)
    # Channel 2: DAPI (Blue) - Nucleus
    actin_channel = arr[:, :, 0]
    tubulin_channel = arr[:, :, 1]
    dapi_channel = arr[:, :, 2]

    # --- Segmentation Logic ---
    # Goal: Define the Cytoplasmic Region (Cell Body excluding Nucleus)
    
    # 1. Define Cell Body Mask (Outer Boundary)
    cell_mask = None
    
    # Strategy: Use provided masks if available, otherwise fallback to Actin channel
    if len(segmentation_masks) > 0:
        # If masks are provided, combine them to define the cellular region.
        # We assume any labeled region (value > 0) in any mask belongs to a cell.
        combined_mask = np.zeros(arr.shape[:2], dtype=bool)
        for mask in segmentation_masks:
            if mask is not None and mask.shape == arr.shape[:2]:
                combined_mask = np.logical_or(combined_mask, mask > 0)
        
        if np.any(combined_mask):
            cell_mask = combined_mask

    # Fallback: If no valid masks provided, use Actin channel (Otsu thresholding)
    if cell_mask is None:
        try:
            # Check if image has content
            if np.max(actin_channel) > np.min(actin_channel):
                thresh_actin = threshold_otsu(actin_channel)
                cell_mask = actin_channel > thresh_actin
            else:
                cell_mask = np.zeros(actin_channel.shape, dtype=bool)
        except Exception:
            cell_mask = np.zeros(actin_channel.shape, dtype=bool)

    # 2. Define Nuclei Mask (Inner Boundary to exclude)
    # We always use the DAPI channel for this as it's the specific nuclear marker
    try:
        if np.max(dapi_channel) > np.min(dapi_channel):
            thresh_dapi = threshold_otsu(dapi_channel)
            nuclei_mask = dapi_channel > thresh_dapi
        else:
            nuclei_mask = np.zeros(dapi_channel.shape, dtype=bool)
    except Exception:
        nuclei_mask = np.zeros(dapi_channel.shape, dtype=bool)

    # 3. Define Cytoplasm Mask
    # Cytoplasm is the cell body region that is NOT the nucleus
    cytoplasm_mask = np.logical_and(cell_mask, np.logical_not(nuclei_mask))

    # --- Feature Calculation ---
    # Feature: tubulin_intensity_heterogeneity
    # Metric: Coefficient of Variation (std / mean) of Tubulin intensity in Cytoplasm
    
    # Extract pixels belonging to the cytoplasm
    valid_pixels = tubulin_channel[cytoplasm_mask]

    # Handle edge case: No cytoplasm detected
    if valid_pixels.size == 0:
        return 0.0

    # Compute statistics
    mean_val = np.mean(valid_pixels)
    std_val = np.std(valid_pixels)

    # Compute Coefficient of Variation (CV)
    # Add small epsilon to prevent division by zero if mean is 0 (e.g., pure black image)
    epsilon = 1e-7
    cv = std_val / (mean_val + epsilon)

    return float(cv)
