def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.morphology import skeletonize
    from skimage.filters import threshold_otsu
    from scipy import ndimage

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset: (512, 512, 3), Channel 0 = Actin (Red)
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Extract Actin channel (Channel 0)
        actin_channel = arr[:, :, 0]
    elif arr.ndim == 2:
        # Fallback if passed a single channel image, assume it's the relevant one
        actin_channel = arr
    else:
        # Unexpected format
        return 0.0

    # Check for empty image
    if np.max(actin_channel) == 0:
        return 0.0

    # Normalize intensity to [0, 1] for processing
    # Using robust max to avoid hot pixel issues
    vmax = np.percentile(actin_channel, 99.5)
    if vmax > 0:
        actin_norm = actin_channel / vmax
    else:
        actin_norm = actin_channel
    actin_norm = np.clip(actin_norm, 0.0, 1.0)

    # Determine Region of Interest (ROI)
    # If segmentation masks are provided, use them to restrict analysis to cellular regions
    # This prevents background noise from being skeletonized
    roi_mask = None
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use the first mask (usually cells or nuclei)
        # Combine all masks if multiple are provided to get total cellular area
        combined_mask = np.zeros(actin_channel.shape, dtype=bool)
        for mask in segmentation_masks:
            if mask is not None:
                # Ensure mask matches image spatial dimensions
                if mask.shape[:2] == actin_channel.shape[:2]:
                    combined_mask = np.logical_or(combined_mask, mask > 0)
        
        if np.any(combined_mask):
            roi_mask = combined_mask

    # Preprocessing: Gaussian smoothing to reduce noise artifacts in skeletonization
    # Small sigma to preserve thin filaments
    actin_smooth = ndimage.gaussian_filter(actin_norm, sigma=1.0)

    # Binarization
    # We need to segment the actin filaments from the background cytoplasm
    if roi_mask is not None:
        # Calculate threshold only on ROI pixels
        roi_pixels = actin_smooth[roi_mask]
        if roi_pixels.size > 0:
            try:
                thresh = threshold_otsu(roi_pixels)
            except ValueError:
                # Fallback if ROI is uniform
                thresh = 0.1
            binary_actin = (actin_smooth > thresh) & roi_mask
        else:
            binary_actin = np.zeros_like(actin_smooth, dtype=bool)
    else:
        # Global threshold if no mask
        try:
            thresh = threshold_otsu(actin_smooth)
        except ValueError:
            thresh = 0.1
        binary_actin = actin_smooth > thresh

    # Skeletonization
    # Reduces binary objects to 1-pixel wide lines
    # This represents the topological structure of the actin network
    skeleton = skeletonize(binary_actin)

    # Quantification
    # Sum the number of pixels in the skeleton
    # This is a proxy for the total length/complexity of the network
    total_length = np.sum(skeleton)

    return float(total_length)
