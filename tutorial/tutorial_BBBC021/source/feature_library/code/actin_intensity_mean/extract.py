def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.filters import threshold_otsu

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3) where Channel 0 is Actin
    if arr.ndim != 3:
        # If not 3D, we can't reliably identify the Actin channel by index 0
        return 0.0
    
    if arr.shape[-1] != 3:
        # Expecting 3 channels
        return 0.0

    # Extract Actin Channel (Channel 0)
    # According to dataset info: Channel 0 = Red (Actin)
    actin_channel = arr[:, :, 0]

    # Determine the region of interest (ROI) mask
    # We want to measure intensity "within the cell".
    mask = None

    # 1. Try to use provided segmentation masks
    if len(segmentation_masks) > 0:
        # Heuristic: If multiple masks are provided, the one with the larger area 
        # is typically the whole-cell/cytoplasm mask, while the smaller is the nucleus.
        # If only one is provided, we use it.
        
        best_mask = None
        max_area = -1
        
        for m in segmentation_masks:
            if m is None:
                continue
            
            # Ensure mask matches spatial dimensions of the image
            # Image is (H, W, C), mask should be (H, W)
            if m.shape != actin_channel.shape:
                continue
                
            current_area = np.sum(m > 0)
            if current_area > max_area:
                max_area = current_area
                best_mask = m
        
        if best_mask is not None:
            mask = best_mask > 0

    # 2. Fallback: If no valid mask provided, generate a foreground mask using Otsu thresholding
    # This prevents the large black background from artificially lowering the mean intensity.
    if mask is None:
        try:
            # Calculate threshold on the actin channel itself
            # Check if image is not uniform (std > 0) to avoid Otsu errors
            if np.std(actin_channel) > 1e-5:
                thresh = threshold_otsu(actin_channel)
                mask = actin_channel > thresh
            else:
                # If image is uniform, treat whole image as ROI (or empty)
                mask = np.ones_like(actin_channel, dtype=bool)
        except Exception:
            # Fallback for any thresholding errors
            mask = np.ones_like(actin_channel, dtype=bool)

    # 3. Compute Mean Intensity within the Mask
    # Check if mask is empty
    if np.sum(mask) == 0:
        return 0.0

    # Extract pixels belonging to the cells
    cell_pixels = actin_channel[mask]
    
    # Compute mean
    mean_intensity = np.mean(cell_pixels)

    return float(mean_intensity)
