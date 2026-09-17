def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.filters import threshold_otsu
    from scipy import ndimage

    # Convert to appropriate array type (float64 to prevent overflow during summation)
    arr = np.asarray(img, dtype=np.float64)

    # Handle dimensionality according to dataset format
    # Dataset: (512, 512, 3), Channel 2 is DAPI (Nucleus)
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Extract DAPI channel (Channel 2 - Blue)
        dapi_channel = arr[:, :, 2]
    elif arr.ndim == 2:
        # Fallback if single channel passed (though unlikely given dataset desc)
        dapi_channel = arr
    else:
        # Unexpected format
        return 0.0

    # Determine the nuclear mask
    # Strategy: Use provided mask if available, otherwise perform on-the-fly segmentation
    binary_mask = None

    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use the first provided mask (assuming it's the primary/nuclear mask)
        # Ensure it matches the spatial dimensions of the image
        mask_input = segmentation_masks[0]
        if mask_input.shape == dapi_channel.shape:
            binary_mask = mask_input > 0
        elif mask_input.ndim == 3 and mask_input.shape[:2] == dapi_channel.shape:
             # Handle case where mask might be 3D (e.g. label matrix)
             binary_mask = mask_input[:, :, 0] > 0
    
    # Fallback: Self-segmentation if no valid mask provided
    if binary_mask is None:
        # 1. Gaussian blur to reduce noise
        blurred = ndimage.gaussian_filter(dapi_channel, sigma=2)
        
        # 2. Check if image is not empty/constant to avoid Otsu errors
        if np.min(blurred) == np.max(blurred):
            return 0.0
            
        # 3. Otsu thresholding
        try:
            thresh = threshold_otsu(blurred)
            binary_mask = blurred > thresh
        except Exception:
            # Fallback for extremely low contrast or empty images
            return 0.0

    # Ensure mask is boolean
    binary_mask = binary_mask.astype(bool)

    # Check if any nuclei were detected
    if not np.any(binary_mask):
        return 0.0

    # Calculate Background Intensity
    # We estimate background from pixels outside the mask to correct the integrated intensity.
    # This makes the measure robust to exposure time differences.
    background_pixels = dapi_channel[~binary_mask]
    
    if background_pixels.size > 0:
        # Use median for robust background estimation
        bg_intensity = np.median(background_pixels)
    else:
        # If mask covers entire image (unlikely), assume 0 background
        bg_intensity = 0.0

    # Compute Integrated Intensity
    # Formula: Sum of (Pixel Intensity - Background Intensity) for all pixels in the mask
    # We clip at 0 to avoid negative intensity contributions
    foreground_pixels = dapi_channel[binary_mask]
    corrected_pixels = np.maximum(0, foreground_pixels - bg_intensity)
    
    integrated_intensity_sum = np.sum(corrected_pixels)

    return float(integrated_intensity_sum)
