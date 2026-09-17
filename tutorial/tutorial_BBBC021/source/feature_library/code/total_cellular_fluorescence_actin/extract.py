def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.filters import threshold_otsu

    # Convert to float64 to prevent overflow during summation of pixel values
    # The input is uint8 (0-255), but the sum over 512x512 pixels can easily exceed 255
    img_float = np.asarray(img, dtype=np.float64)

    # Handle dimensionality according to dataset format
    # Dataset Description: (Height, Width, Channels) = (512, 512, 3)
    # Channel 0 = Red (Actin/Cytoskeleton)
    # Channel 1 = Green (Tubulin)
    # Channel 2 = Blue (DAPI/Nucleus)
    
    # We specifically want the Actin channel (Channel 0)
    if img_float.ndim == 3 and img_float.shape[2] >= 1:
        actin_channel = img_float[:, :, 0]
    elif img_float.ndim == 2:
        # Fallback: if image is 2D, assume it is the relevant channel or a projection
        actin_channel = img_float
    else:
        # Unexpected format
        return 0.0

    # Determine the Region of Interest (ROI) / Mask
    # We want to sum fluorescence only within cellular regions.
    mask = None
    
    # Filter out any None values that might be passed (e.g. placeholders)
    valid_masks = [m for m in segmentation_masks if m is not None and hasattr(m, 'shape')]

    if len(valid_masks) > 0:
        # Strategy: If segmentation masks are provided, use the union of all masks.
        # This ensures we capture the entire cellular area (whether the mask is for nuclei, cytoplasm, or both).
        combined_mask = np.zeros(actin_channel.shape, dtype=bool)
        for seg in valid_masks:
            # Ensure mask matches image spatial dimensions (handle potential 2D vs 3D mismatches)
            if seg.shape[:2] == actin_channel.shape[:2]:
                combined_mask = np.logical_or(combined_mask, seg > 0)
        
        if np.any(combined_mask):
            mask = combined_mask

    # Fallback: If no valid masks provided, generate one from the Actin channel itself
    # Actin defines the cell shape/cytoskeleton, so it is good for auto-segmentation.
    if mask is None:
        data_min = np.min(actin_channel)
        data_max = np.max(actin_channel)
        
        # Check if image has contrast
        if data_max > data_min:
            try:
                # Use Otsu's method to separate foreground (cells) from background
                thresh = threshold_otsu(actin_channel)
                mask = actin_channel > thresh
            except Exception:
                # Fallback for extremely low contrast or uniform images
                mask = actin_channel > data_min
        else:
            # Image is flat (e.g., all zeros), return 0
            return 0.0

    # Compute Total Cellular Fluorescence
    # Sum of pixel intensities within the masked region
    if mask is not None:
        # Apply mask and sum
        total_fluorescence = np.sum(actin_channel[mask])
    else:
        total_fluorescence = 0.0

    return float(total_fluorescence)
