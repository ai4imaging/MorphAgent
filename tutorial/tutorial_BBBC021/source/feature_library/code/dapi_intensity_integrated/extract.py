def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.filters import threshold_otsu
    from scipy import ndimage

    # 1. Input Validation and Channel Extraction
    # Convert to appropriate array type (keep as float for calculations to avoid overflow)
    # Image shape is expected to be (512, 512, 3) based on dataset description
    img_arr = np.asarray(img)
    
    # Check dimensionality
    if img_arr.ndim != 3 or img_arr.shape[2] != 3:
        # If not the expected 3-channel image, return 0.0
        return 0.0

    # Extract DAPI channel (Channel 2 / Blue based on dataset description)
    # Channel 0: Actin, Channel 1: Tubulin, Channel 2: DAPI
    dapi_channel = img_arr[:, :, 2].astype(np.float64)

    # 2. Determine Mask
    # The feature requires summing intensity *across all nuclei*.
    # We prioritize using the provided segmentation mask if available.
    
    mask = None
    
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use the first provided mask (assumed to be nuclei or primary objects)
        provided_mask = segmentation_masks[0]
        
        # Ensure mask matches image spatial dimensions
        if provided_mask.shape[:2] == dapi_channel.shape:
            # Create binary mask (values > 0 are considered objects)
            mask = provided_mask > 0
    
    # Fallback: If no valid mask provided, generate one using Otsu thresholding
    if mask is None:
        # Check if image has content (std dev > 0)
        if np.std(dapi_channel) > 0:
            # Apply slight smoothing to reduce noise before thresholding
            blurred = ndimage.gaussian_filter(dapi_channel, sigma=2.0)
            try:
                thresh = threshold_otsu(blurred)
                mask = dapi_channel > thresh
            except Exception:
                # Fallback if otsu fails (e.g. uniform image)
                mask = np.ones_like(dapi_channel, dtype=bool)
        else:
            # If image is flat/empty, mask is all false
            mask = np.zeros_like(dapi_channel, dtype=bool)

    # 3. Compute Integrated Intensity
    # Select pixels belonging to the mask
    masked_pixels = dapi_channel[mask]
    
    # Sum the intensities
    # If mask is empty, sum is 0.0
    if masked_pixels.size == 0:
        result = 0.0
    else:
        result = np.sum(masked_pixels)

    return float(result)
