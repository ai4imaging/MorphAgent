def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_opening, disk, binary_dilation
    
    # Convert to appropriate array type and normalize to float
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality and validate input
    # Expected shape: (H, W, 3) or (H, W) if single channel passed erroneously
    if arr.ndim == 3:
        # Channel 0: Actin (Target Intensity)
        # Channel 1: Tubulin (Cell Body helper)
        # Channel 2: DAPI (Nucleus helper)
        actin_channel = arr[..., 0]
        tubulin_channel = arr[..., 1]
        dapi_channel = arr[..., 2]
    elif arr.ndim == 2:
        # Fallback: assume single channel is actin, but we can't segment well without DAPI
        # Returning 0.0 as feature cannot be reliably computed
        return 0.0
    else:
        return 0.0

    # Normalize intensity for thresholding calculations (0-1 range)
    # We keep the original float values for the final ratio calculation to preserve dynamic range
    # but use normalized copies for mask generation
    def normalize_ch(ch):
        vmax = np.max(ch)
        return ch / vmax if vmax > 0 else ch

    norm_dapi = normalize_ch(dapi_channel)
    norm_actin = normalize_ch(actin_channel)
    norm_tubulin = normalize_ch(tubulin_channel)

    # --- Segmentation Logic ---
    
    nucleus_mask = None
    cytoplasm_mask = None

    # Strategy 1: Use provided segmentation masks
    # We expect masks to be passed in order, typically Nuclei first, then Cells/Cytoplasm
    if len(segmentation_masks) > 0:
        # Check first mask (Nuclei)
        mask0 = segmentation_masks[0]
        if mask0 is not None and mask0.shape == actin_channel.shape:
            nucleus_mask = mask0 > 0

        # Check second mask (Cells or Cytoplasm)
        if len(segmentation_masks) > 1:
            mask1 = segmentation_masks[1]
            if mask1 is not None and mask1.shape == actin_channel.shape:
                # If mask1 is "Cells" (entire cell), Cytoplasm = Cell - Nucleus
                # If mask1 is "Cytoplasm" (ring), use directly
                # We assume it's a cell mask (common in BBBC021)
                cell_mask = mask1 > 0
                if nucleus_mask is not None:
                    cytoplasm_mask = np.logical_and(cell_mask, ~nucleus_mask)
                else:
                    cytoplasm_mask = cell_mask # Fallback if no nucleus mask

    # Strategy 2: Compute masks on the fly if not provided
    if nucleus_mask is None:
        # Segment Nucleus from DAPI (Channel 2)
        # Smooth slightly to reduce noise
        smooth_dapi = ndimage.gaussian_filter(norm_dapi, sigma=2)
        try:
            thresh_nuc = threshold_otsu(smooth_dapi)
            nucleus_mask = smooth_dapi > thresh_nuc
            # Clean up mask
            nucleus_mask = binary_opening(nucleus_mask, disk(2))
        except Exception:
            # Fallback for very low contrast/empty images
            nucleus_mask = np.zeros_like(dapi_channel, dtype=bool)

    if cytoplasm_mask is None:
        # Segment Cell Body from Actin (Ch0) + Tubulin (Ch1)
        # Combine channels to get full cell extent
        cell_signal = np.maximum(norm_actin, norm_tubulin)
        smooth_cell = ndimage.gaussian_filter(cell_signal, sigma=2)
        try:
            thresh_cell = threshold_otsu(smooth_cell)
            cell_mask = smooth_cell > thresh_cell
            # Ensure cell mask covers nucleus mask (morphological consistency)
            cell_mask = np.logical_or(cell_mask, nucleus_mask)
            # Define Cytoplasm as Cell Body excluding Nucleus
            cytoplasm_mask = np.logical_and(cell_mask, ~nucleus_mask)
        except Exception:
            cytoplasm_mask = np.zeros_like(actin_channel, dtype=bool)

    # --- Feature Calculation ---

    # Ensure masks are boolean
    nucleus_mask = nucleus_mask.astype(bool)
    cytoplasm_mask = cytoplasm_mask.astype(bool)

    # Count pixels to ensure validity
    n_nuc_pixels = np.sum(nucleus_mask)
    n_cyto_pixels = np.sum(cytoplasm_mask)

    if n_nuc_pixels == 0 or n_cyto_pixels == 0:
        return 0.0

    # Calculate Mean Intensity of Actin (Channel 0) in both regions
    # We use the original float array 'actin_channel' for accurate intensity measurement
    mean_actin_nucleus = np.mean(actin_channel[nucleus_mask])
    mean_actin_cytoplasm = np.mean(actin_channel[cytoplasm_mask])

    # Avoid division by zero
    if mean_actin_cytoplasm == 0:
        return 0.0

    # Calculate Ratio
    ratio = mean_actin_nucleus / mean_actin_cytoplasm

    return float(ratio)
