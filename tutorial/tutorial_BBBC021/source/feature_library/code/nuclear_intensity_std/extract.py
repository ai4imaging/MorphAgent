def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import remove_small_objects
    from scipy import ndimage

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset: (512, 512, 3), Channel 2 is DAPI (Nucleus)
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Extract DAPI channel (Channel 2)
        dapi_channel = arr[:, :, 2]
    elif arr.ndim == 2:
        # Fallback: if single channel, assume it's the relevant one (though unlikely given dataset spec)
        dapi_channel = arr
    else:
        # Unexpected format
        return 0.0

    # Intensity normalization
    # Normalize to [0, 1] for consistent calculation regardless of bit depth
    # Dataset is uint8 (0-255)
    dapi_channel = dapi_channel / 255.0
    dapi_channel = np.clip(dapi_channel, 0.0, 1.0)

    # Handle segmentation masks
    labeled_mask = None
    
    # Check if valid segmentation masks are provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        mask_input = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions
        if mask_input.shape[:2] == dapi_channel.shape[:2]:
            # If mask is already labeled (int > 1), use it. If binary, label it.
            if np.max(mask_input) > 1:
                labeled_mask = mask_input.astype(int)
            else:
                labeled_mask = label(mask_input > 0)
    
    # Fallback: Generate segmentation if no valid mask provided
    if labeled_mask is None:
        # Simple preprocessing to reduce noise
        blurred = ndimage.gaussian_filter(dapi_channel, sigma=2.0)
        
        # Check for valid range before thresholding
        if np.max(blurred) > np.min(blurred):
            try:
                thresh = threshold_otsu(blurred)
                binary_mask = blurred > thresh
                # Remove small artifacts (noise)
                binary_mask = remove_small_objects(binary_mask, min_size=50)
                labeled_mask = label(binary_mask)
            except Exception:
                # Fallback for extremely low contrast or empty images
                return 0.0
        else:
            return 0.0

    # Feature Computation: Standard Deviation of Intensity per Nucleus
    # We compute the std dev for each nucleus individually, then average across the population.
    # This avoids confounding inter-cellular intensity differences with intra-cellular texture.
    
    regions = regionprops(labeled_mask, intensity_image=dapi_channel)
    
    if not regions:
        return 0.0

    std_devs = []
    for region in regions:
        # region.image is the binary mask for the bounding box
        # region.intensity_image is the intensity values for the bounding box
        # Select only pixels belonging to the object
        intensities = region.intensity_image[region.image]
        
        if intensities.size > 1:
            std_devs.append(np.std(intensities))
        else:
            std_devs.append(0.0)

    if not std_devs:
        return 0.0

    # Return the mean of the standard deviations across all cells
    result = np.mean(std_devs)

    return float(result)
