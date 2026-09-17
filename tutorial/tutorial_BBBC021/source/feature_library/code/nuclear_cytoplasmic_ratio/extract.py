def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_closing, binary_opening, disk
    
    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Expected shape: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        # If not the expected 3-channel image, return 0.0
        return 0.0

    # Channel Mapping based on dataset description:
    # Channel 0: Actin (Cytoskeleton) -> Used for total cell body / cytoplasm
    # Channel 2: DAPI (Nucleus) -> Used for nucleus
    actin_channel = arr[..., 0]
    dapi_channel = arr[..., 2]

    # Preprocessing: Gaussian blur to reduce noise before thresholding
    # Sigma=2.0 is appropriate for 512x512 images to smooth out texture
    actin_smooth = ndimage.gaussian_filter(actin_channel, sigma=2.0)
    dapi_smooth = ndimage.gaussian_filter(dapi_channel, sigma=2.0)

    # --- Segmentation Logic ---
    
    # 1. Nuclear Mask
    # Check if image has content (avoid error in threshold_otsu on empty images)
    if np.max(dapi_smooth) > np.min(dapi_smooth):
        thresh_nuc = threshold_otsu(dapi_smooth)
        mask_nuc = dapi_smooth > thresh_nuc
        # Refine mask: fill holes and remove small noise
        mask_nuc = binary_closing(mask_nuc, disk(2))
        mask_nuc = binary_opening(mask_nuc, disk(1))
    else:
        mask_nuc = np.zeros_like(dapi_smooth, dtype=bool)

    # 2. Total Cell / Cytoplasm Mask
    # We use Actin to define the extent of the cell body
    if np.max(actin_smooth) > np.min(actin_smooth):
        thresh_actin = threshold_otsu(actin_smooth)
        mask_total_cell = actin_smooth > thresh_actin
        # Refine mask
        mask_total_cell = binary_closing(mask_total_cell, disk(3))
        mask_total_cell = binary_opening(mask_total_cell, disk(1))
    else:
        mask_total_cell = np.zeros_like(actin_smooth, dtype=bool)

    # --- Feature Computation: Nuclear-Cytoplasmic Ratio ---
    
    # Definition: Ratio of Nuclear Area to Cytoplasmic Area
    # Cytoplasmic Area is defined as (Total Cell Area - Nuclear Area)
    # Note: Sometimes Actin stain is weak over the nucleus, so we take the union 
    # of Actin and DAPI as the "Total Cell" area to be safe, or strictly use Actin.
    # Given the prompt description "Cytoplasmic area (Actin - DAPI)", we interpret this
    # as the area positive for Actin but NOT positive for DAPI, or simply the non-nuclear part of the cell.
    
    # A robust approach for "Cytoplasm":
    # Ideally, the cell body contains the nucleus. 
    # We ensure the total cell mask includes the nuclear mask (biology constraint).
    mask_total_cell = np.logical_or(mask_total_cell, mask_nuc)
    
    # Define Cytoplasm strictly as Total Cell Area excluding the Nucleus
    mask_cyto = np.logical_and(mask_total_cell, ~mask_nuc)

    # Calculate Areas (pixel counts)
    area_nuc = np.sum(mask_nuc)
    area_cyto = np.sum(mask_cyto)

    # Compute Ratio
    # Handle division by zero if no cytoplasm is detected
    if area_cyto == 0:
        if area_nuc > 0:
            # If there are nuclei but no cytoplasm detected (unlikely but possible in artifacts),
            # the ratio is effectively infinite. We return a high value or 0 depending on convention.
            # Returning 0.0 is safer for stability, or we could return the raw nuclear area.
            # However, usually this indicates a segmentation failure.
            return 0.0 
        else:
            # Empty image
            return 0.0
            
    ratio = area_nuc / area_cyto

    return float(ratio)
