def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.filters import threshold_otsu

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3) where Channel 1 is Tubulin (Green)
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Extract Tubulin channel (Index 1)
        tubulin_channel = arr[..., 1]
    elif arr.ndim == 2:
        # Fallback if single channel passed (unlikely based on description but safe)
        tubulin_channel = arr
    else:
        # Unexpected format
        return 0.0

    # Intensity normalization
    # Data is uint8 (0-255). Normalize to [0, 1] for standard variance interpretation
    tubulin_channel = tubulin_channel / 255.0
    tubulin_channel = np.clip(tubulin_channel, 0.0, 1.0)

    # Determine Region of Interest (ROI)
    # We want to calculate variance only on the cells, not the background.
    # Including the background (lots of zeros) would artificially inflate variance.
    
    roi_mask = None

    # Check if segmentation masks are available
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use the first available mask (assuming it covers cells/nuclei)
        # Masks are typically labeled integers. Convert to boolean foreground mask.
        mask = segmentation_masks[0]
        
        # Ensure mask shape matches image shape (handle potential 2D mask for 3D image or vice versa)
        if mask.shape == tubulin_channel.shape:
            roi_mask = mask > 0
        else:
            # If shapes don't match exactly, try to fallback to thresholding
            pass

    # Fallback: If no valid mask provided, generate one using Otsu thresholding on the Tubulin channel
    if roi_mask is None:
        try:
            # Calculate threshold. If image is empty/uniform, this might fail or warn
            if np.min(tubulin_channel) == np.max(tubulin_channel):
                return 0.0
            thresh = threshold_otsu(tubulin_channel)
            roi_mask = tubulin_channel > thresh
        except Exception:
            # If thresholding fails (e.g., extremely low contrast), use the whole image
            roi_mask = np.ones_like(tubulin_channel, dtype=bool)

    # Extract pixels belonging to the ROI
    foreground_pixels = tubulin_channel[roi_mask]

    # Calculate Variance
    # If no foreground pixels found (empty mask), return 0.0
    if foreground_pixels.size == 0:
        return 0.0

    # Compute variance of the intensity values
    # High variance -> Distinct structures (bundles, fibers)
    # Low variance -> Diffuse signal (depolymerization)
    intensity_variance = np.var(foreground_pixels)

    return float(intensity_variance)
