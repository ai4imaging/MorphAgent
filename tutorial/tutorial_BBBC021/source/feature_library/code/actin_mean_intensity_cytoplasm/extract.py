def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    
    # Convert to appropriate array type and handle potential float inputs
    # We keep the original scale (0-255) for intensity measurements unless normalization is required
    # But for calculation precision, we use float32
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Expected shape: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        # If not standard 3-channel image, return 0.0
        return 0.0

    # Extract Channels
    # Channel 0: Actin (Red) - Target for intensity
    # Channel 2: DAPI (Blue) - Marker for Nucleus (exclusion zone)
    actin_channel = arr[:, :, 0]
    dapi_channel = arr[:, :, 2]

    # Initialize masks
    nucleus_mask = None
    cell_mask = None
    
    # Strategy to define Cytoplasm: Cell Mask - Nucleus Mask
    
    # 1. Parse Segmentation Masks if available
    if len(segmentation_masks) >= 2:
        # Heuristic: If 2 masks, the one with smaller area is likely nucleus, larger is cell
        mask1 = np.asarray(segmentation_masks[0], dtype=bool)
        mask2 = np.asarray(segmentation_masks[1], dtype=bool)
        
        area1 = np.sum(mask1)
        area2 = np.sum(mask2)
        
        if area1 < area2:
            nucleus_mask = mask1
            cell_mask = mask2
        else:
            nucleus_mask = mask2
            cell_mask = mask1
            
    elif len(segmentation_masks) == 1:
        # Heuristic: If 1 mask, it's usually the primary object (Nucleus in many HCS datasets)
        # But we need to check coverage. If it covers most of the image, it might be cell.
        # Usually, nuclei are smaller. Let's assume it's nucleus and generate cell mask from Actin.
        nucleus_mask = np.asarray(segmentation_masks[0], dtype=bool)
        
        # Generate Cell Mask from Actin signal
        try:
            thresh_actin = threshold_otsu(actin_channel)
            cell_mask = actin_channel > thresh_actin
        except Exception:
            # Fallback if otsu fails (e.g. uniform image)
            cell_mask = actin_channel > np.mean(actin_channel)

    else:
        # No masks provided: Generate both from channels
        try:
            # Nucleus from DAPI
            thresh_dapi = threshold_otsu(dapi_channel)
            nucleus_mask = dapi_channel > thresh_dapi
            
            # Cell from Actin
            thresh_actin = threshold_otsu(actin_channel)
            cell_mask = actin_channel > thresh_actin
        except Exception:
            # Fallback for very low contrast/empty images
            return 0.0

    # Ensure masks are same shape as image (handle potential 2D vs 3D issues)
    if nucleus_mask.shape != actin_channel.shape:
        return 0.0
    if cell_mask.shape != actin_channel.shape:
        return 0.0

    # 2. Define Cytoplasm Mask
    # Cytoplasm = Cell Body AND NOT Nucleus
    cytoplasm_mask = np.logical_and(cell_mask, np.logical_not(nucleus_mask))

    # 3. Calculate Mean Intensity
    # Select pixels in the cytoplasm
    cytoplasm_pixels = actin_channel[cytoplasm_mask]

    # Check if we have any pixels
    if cytoplasm_pixels.size == 0:
        return 0.0

    # Compute mean
    mean_intensity = np.mean(cytoplasm_pixels)

    return float(mean_intensity)
