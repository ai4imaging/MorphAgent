def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import remove_small_objects

    # Convert to appropriate array type and handle dimensionality
    # Dataset is (512, 512, 3), uint8. Channel 2 is DAPI (Nucleus).
    # We need float precision for intensity summation to avoid overflow.
    img_arr = np.asarray(img, dtype=np.float32)

    # Check for valid dimensions
    if img_arr.ndim != 3 or img_arr.shape[2] < 3:
        return 0.0

    # Extract the Nuclear Channel (Channel 2 - Blue/DAPI)
    # Based on dataset description: Channel 2 = DAPI = Nucleus
    dapi_channel = img_arr[:, :, 2]

    # Determine Segmentation Mask
    labeled_mask = None

    # 1. Try to use provided segmentation masks
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        mask_input = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions
        if mask_input.shape[:2] == dapi_channel.shape:
            # If mask is already labeled (integer labels), use it directly
            if np.issubdtype(mask_input.dtype, np.integer) and mask_input.max() > 1:
                labeled_mask = mask_input
            else:
                # If binary mask, label connected components
                labeled_mask = label(mask_input > 0)
    
    # 2. Fallback: Compute segmentation on DAPI channel if no valid mask provided
    if labeled_mask is None:
        # Simple intensity-based segmentation
        # Check if image is not empty/black
        if dapi_channel.max() > dapi_channel.min():
            try:
                thresh = threshold_otsu(dapi_channel)
                binary_mask = dapi_channel > thresh
                # Remove small noise (e.g., < 50 pixels)
                binary_mask = remove_small_objects(binary_mask, min_size=50)
                labeled_mask = label(binary_mask)
            except Exception:
                # Fallback if otsu fails (e.g. uniform image)
                return 0.0
        else:
            return 0.0

    # Compute Feature: Integrated Intensity CV
    # Integrated Intensity = Sum of pixel values within the nucleus region
    
    # Get properties for all labeled regions
    # We pass dapi_channel as intensity_image to calculate intensity-based props
    regions = regionprops(labeled_mask, intensity_image=dapi_channel)

    if not regions:
        return 0.0

    # Collect integrated intensities
    # region.intensity_image is the crop of the intensity image for that region
    # We sum it to get total DNA content proxy
    integrated_intensities = []
    for region in regions:
        # Only consider regions with valid area to avoid division by zero artifacts
        if region.area > 0:
            # Sum of intensity values in the region
            # region.image is the binary mask for the bounding box
            # region.intensity_image contains intensity values inside bounding box
            # We must only sum pixels where the mask is True
            val = np.sum(region.intensity_image[region.image])
            integrated_intensities.append(val)

    integrated_intensities = np.array(integrated_intensities)

    # Calculate Coefficient of Variation (CV) = StdDev / Mean
    if integrated_intensities.size == 0:
        return 0.0
    
    mean_val = np.mean(integrated_intensities)
    
    if mean_val == 0:
        return 0.0
        
    std_val = np.std(integrated_intensities)
    
    cv = std_val / mean_val

    return float(cv)
