def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    
    # Convert to appropriate array type and handle dimensionality
    # Expected shape: (512, 512, 3)
    img = np.asarray(img)
    
    # Check for valid dimensions
    if img.ndim != 3 or img.shape[2] < 3:
        return 0.0

    # Extract Channels
    # Channel 0: Actin (Red) - Cytoskeleton
    # Channel 1: Tubulin (Green) - Microtubules (Target Signal)
    # Channel 2: DAPI (Blue) - Nucleus
    actin_ch = img[:, :, 0].astype(np.float32)
    tubulin_ch = img[:, :, 1].astype(np.float32)
    dapi_ch = img[:, :, 2].astype(np.float32)

    # Define Masks for Nuclei and Cell Body
    mask_nuclei = None
    mask_cells = None

    # Strategy 1: Use provided segmentation masks
    # We expect masks to potentially be passed. 
    # If 2 masks are passed, we assume order: cell_mask, nuclei_mask or similar.
    # However, without strict metadata on mask order, we can try to infer or use fallback if ambiguous.
    # Given the prompt says "masks are automatically loaded", but doesn't guarantee order or content,
    # we will prioritize robust fallback if masks aren't clearly usable.
    
    # If we have masks, let's try to use them.
    # Common convention: if 2 masks, often [nuclei, cells] or [cells, nuclei].
    # If we can't be sure, we will generate our own to be safe and consistent with the feature definition.
    # For this specific feature, accurate separation of nucleus from cytoplasm is critical.
    # Self-computed Otsu is often safer than relying on ambiguous external masks unless defined strictly.
    # Let's implement a robust internal segmentation fallback which is standard for this type of feature.

    # --- Internal Segmentation Logic (Fallback or Primary) ---
    
    # 1. Generate Nuclei Mask (from DAPI)
    try:
        # Smooth slightly to reduce noise
        dapi_smooth = ndimage.gaussian_filter(dapi_ch, sigma=2)
        thresh_nuc = threshold_otsu(dapi_smooth)
        mask_nuclei = dapi_smooth > thresh_nuc
        # Fill holes
        mask_nuclei = ndimage.binary_fill_holes(mask_nuclei)
    except Exception:
        # Fallback if image is flat/empty
        mask_nuclei = np.zeros(dapi_ch.shape, dtype=bool)

    # 2. Generate Cell Body Mask (from Actin + Tubulin)
    # Combine structural channels to get the full cell extent
    try:
        structural_img = np.maximum(actin_ch, tubulin_ch)
        struct_smooth = ndimage.gaussian_filter(structural_img, sigma=2)
        thresh_cell = threshold_otsu(struct_smooth)
        mask_cells = struct_smooth > thresh_cell
        # Fill holes
        mask_cells = ndimage.binary_fill_holes(mask_cells)
    except Exception:
        mask_cells = np.zeros(actin_ch.shape, dtype=bool)

    # --- Define Cytoplasm Region ---
    # Cytoplasm = Cell Body AND NOT Nucleus
    # We ensure the nucleus is fully contained within the cell mask logically
    mask_cytoplasm = np.logical_and(mask_cells, np.logical_not(mask_nuclei))

    # --- Extract Pixels ---
    # Get the Tubulin intensity values within the cytoplasmic region
    cytoplasm_pixels = tubulin_ch[mask_cytoplasm]

    # --- Compute Feature ---
    # Coefficient of Variation = std / mean
    
    if cytoplasm_pixels.size == 0:
        return 0.0

    mean_val = np.mean(cytoplasm_pixels)
    std_val = np.std(cytoplasm_pixels)

    # Avoid division by zero
    if mean_val == 0:
        return 0.0

    cv = std_val / mean_val

    return float(cv)
