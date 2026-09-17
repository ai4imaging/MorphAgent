def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.filters import threshold_otsu
    
    # Convert to appropriate array type (float32 for calculations)
    # Image is (512, 512, 3) uint8
    arr = np.asarray(img, dtype=np.float32)
    
    # Handle dimensionality
    # If image is (C, H, W) instead of (H, W, C), transpose it
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[2] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    
    # Check for valid shape
    if arr.ndim != 3 or arr.shape[2] < 3:
        return 0.0

    # Channel Mapping based on dataset description:
    # Channel 0: Actin (Red)
    # Channel 1: Tubulin (Green) -> Target for intensity measurement
    # Channel 2: DAPI (Blue) -> Nucleus
    tubulin_channel = arr[..., 1]
    
    # Initialize masks
    mask_nuclei = None
    mask_cells = None
    
    # Logic to define Cytoplasm Mask
    # Cytoplasm is defined as the Cell region minus the Nuclear region.
    
    if len(segmentation_masks) >= 2:
        # Strategy: If multiple masks are provided, we distinguish them by area.
        # Biologically, the Whole Cell mask must be larger than or equal to the Nuclei mask.
        masks = []
        for m in segmentation_masks:
            # Ensure mask is boolean/binary
            m_bool = np.asarray(m, dtype=bool)
            # Handle dimensions if mask is 3D (e.g. 1, H, W)
            if m_bool.ndim == 3:
                m_bool = np.max(m_bool, axis=0)
            masks.append(m_bool)
            
        # Sort by area (number of True pixels)
        # Smallest area -> Nuclei (usually)
        # Largest area -> Whole Cell
        masks_sorted = sorted(masks, key=lambda x: np.sum(x))
        
        # Assign based on size hierarchy
        mask_nuclei = masks_sorted[0]
        mask_cells = masks_sorted[-1]
        
    elif len(segmentation_masks) == 1:
        # If only one mask, we assume it's the cell mask if it's large, or we can't do subtraction.
        # However, to be safe, we will treat it as the ROI mask and use it directly.
        # But for "cytoplasm" specifically, we ideally need to subtract nuclei.
        # We will fallback to generating a nuclei mask from the image if possible.
        mask_cells = np.asarray(segmentation_masks[0], dtype=bool)
        if mask_cells.ndim == 3:
            mask_cells = np.max(mask_cells, axis=0)
            
        # Generate nuclei mask from DAPI channel to subtract
        try:
            dapi_channel = arr[..., 2]
            # Simple robust thresholding if variance is sufficient
            if np.std(dapi_channel) > 1e-5:
                thresh = threshold_otsu(dapi_channel)
                mask_nuclei = dapi_channel > thresh
            else:
                mask_nuclei = np.zeros_like(mask_cells, dtype=bool)
        except Exception:
            mask_nuclei = np.zeros_like(mask_cells, dtype=bool)

    else:
        # Fallback: No segmentation masks provided.
        # Generate masks using Otsu thresholding on relevant channels.
        try:
            # 1. Nuclei from DAPI (Channel 2)
            dapi_channel = arr[..., 2]
            if np.std(dapi_channel) > 1e-5:
                thresh_nuc = threshold_otsu(dapi_channel)
                mask_nuclei = dapi_channel > thresh_nuc
            else:
                mask_nuclei = np.zeros(dapi_channel.shape, dtype=bool)

            # 2. Cells from Tubulin (Ch1) + Actin (Ch0) combined
            # This provides a robust cell body signal
            cell_signal = arr[..., 0] + arr[..., 1]
            if np.std(cell_signal) > 1e-5:
                thresh_cell = threshold_otsu(cell_signal)
                mask_cells = cell_signal > thresh_cell
            else:
                mask_cells = np.zeros(cell_signal.shape, dtype=bool)
                
        except Exception:
            return 0.0

    # Ensure masks are same shape as image spatial dims
    if mask_cells.shape != tubulin_channel.shape:
        # Resize or crop would be complex, return 0.0 to be safe
        return 0.0

    if mask_nuclei is None:
        mask_nuclei = np.zeros_like(mask_cells, dtype=bool)

    # Define Cytoplasm: Cell Mask AND NOT Nuclei Mask
    mask_cytoplasm = np.logical_and(mask_cells, np.logical_not(mask_nuclei))
    
    # Compute Feature
    # Extract pixels belonging to the cytoplasm
    cytoplasmic_pixels = tubulin_channel[mask_cytoplasm]
    
    if cytoplasmic_pixels.size == 0:
        return 0.0
        
    # Calculate Mean Intensity
    # We use the raw intensity values (float32 converted)
    mean_intensity = np.mean(cytoplasmic_pixels)

    return float(mean_intensity)
