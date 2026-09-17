def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.filters import threshold_otsu
    from scipy import ndimage

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3) uint8
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Intensity normalization
    # Normalize to [0, 1] for stable thresholding
    arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)

    # Check for empty image
    if np.max(arr) == 0:
        return 0.0

    # Channel Extraction
    # Channel 0: Actin (Red) -> Cytoskeleton
    # Channel 1: Tubulin (Green) -> Cytoskeleton
    # Channel 2: DAPI (Blue) -> Nucleus
    
    # Nuclear signal
    nuc_signal = arr[..., 2]
    
    # Cytoplasmic signal (combine Actin and Tubulin for robust cell body detection)
    # Using maximum projection to capture structure from either marker
    cyto_signal = np.maximum(arr[..., 0], arr[..., 1])

    # --- Segmentation Logic ---
    # We calculate the Global Area Ratio (Total Nuclear Area / Total Cytoplasmic Area)
    # This is robust to cell clumping common in MCF-7 lines.

    # 1. Segment Nuclei
    try:
        # Check if signal is sufficient for Otsu
        if np.max(nuc_signal) - np.min(nuc_signal) < 0.05:
            thresh_nuc = 0.1 # Fallback low threshold
        else:
            thresh_nuc = threshold_otsu(nuc_signal)
        
        mask_nuc = nuc_signal > thresh_nuc
        
        # Fill holes in nuclei to get accurate area
        mask_nuc = ndimage.binary_fill_holes(mask_nuc)
    except Exception:
        mask_nuc = np.zeros_like(nuc_signal, dtype=bool)

    # 2. Segment Cell Body (Cytoplasm + Nucleus)
    try:
        # Check if signal is sufficient for Otsu
        if np.max(cyto_signal) - np.min(cyto_signal) < 0.05:
            thresh_cyto = 0.1 # Fallback low threshold
        else:
            thresh_cyto = threshold_otsu(cyto_signal)
            
        mask_body = cyto_signal > thresh_cyto
        
        # Fill holes in cell bodies
        mask_body = ndimage.binary_fill_holes(mask_body)
    except Exception:
        mask_body = np.zeros_like(cyto_signal, dtype=bool)

    # 3. Define Cytoplasm Area
    # Cytoplasm is the cell body area excluding the nuclear area.
    # We use logical AND NOT to subtract the nucleus from the body mask.
    # Note: Sometimes the body mask might not perfectly overlap the nuclear mask due to staining differences.
    # However, biologically, cytoplasm is the area outside the nucleus.
    # A robust definition for ratio is: Area(Nucleus) / Area(Cell Body - Nucleus)
    
    # Ensure masks are boolean
    mask_nuc = mask_nuc.astype(bool)
    mask_body = mask_body.astype(bool)
    
    # Often, the "body" stain (actin/tubulin) is present throughout the cell, but sometimes dim over the nucleus.
    # To be safe, we define the total cellular footprint as the union of both masks.
    mask_total_cell = np.logical_or(mask_nuc, mask_body)
    
    # Cytoplasm is total footprint minus nucleus
    mask_cyto_only = np.logical_and(mask_total_cell, ~mask_nuc)

    # --- Feature Calculation ---
    area_nuc = np.sum(mask_nuc)
    area_cyto = np.sum(mask_cyto_only)

    # Handle edge cases
    if area_cyto == 0:
        if area_nuc > 0:
            return 100.0 # Arbitrary high value for "all nucleus"
        else:
            return 0.0   # Empty image

    ratio = area_nuc / area_cyto

    return float(ratio)
