def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_opening, disk

    # Convert to appropriate array type
    arr = np.asarray(img)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3) where Channel 2 (index 2) is DAPI (Nucleus)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        # Extract DAPI channel (Blue channel, index 2)
        dapi_channel = arr[:, :, 2]
    elif arr.ndim == 2:
        # Fallback if passed a single channel image
        dapi_channel = arr
    else:
        # Unexpected format
        return 0.0

    # Ensure we have a float array for processing if needed, though uint8 is fine for thresholding
    # We will work with the raw intensity for thresholding to be robust
    
    labeled_mask = None

    # 1. Check if segmentation masks are provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use the first provided mask. Assuming it corresponds to nuclei or cells.
        # If the mask is binary (0/1 or 0/255), we need to label it to get instances.
        # If it's already instance-labeled (1, 2, 3...), label() will just re-index it which is safe.
        mask_input = segmentation_masks[0]
        
        # Handle potential dimension mismatch in masks (e.g. if mask is 3D but image is 2D projected)
        if mask_input.ndim == 3:
            mask_input = np.max(mask_input, axis=2) # Project if needed, though unlikely for masks
            
        # Ensure mask matches image spatial dimensions
        if mask_input.shape[:2] == dapi_channel.shape[:2]:
            labeled_mask = label(mask_input)

    # 2. Fallback: On-the-fly segmentation if no valid mask provided
    if labeled_mask is None:
        # Preprocessing: Gaussian blur to reduce noise
        # Normalize to 0-1 for consistent processing
        img_norm = dapi_channel.astype(np.float32) / 255.0
        
        # Smooth slightly
        img_smooth = ndimage.gaussian_filter(img_norm, sigma=2)
        
        # Thresholding
        try:
            thresh = threshold_otsu(img_smooth)
            binary_mask = img_smooth > thresh
        except Exception:
            # Fallback if image is uniform (e.g. all black)
            return 0.0
            
        # Morphological cleanup: remove small noise
        # Using a small disk for opening
        clean_mask = binary_opening(binary_mask, footprint=disk(2))
        
        # Label connected components
        labeled_mask = label(clean_mask)

    # 3. Compute Feature: Median Area
    regions = regionprops(labeled_mask)
    
    areas = []
    for props in regions:
        # Filter out very small artifacts (e.g., < 50 pixels)
        # This is important for noise robustness
        if props.area >= 50:
            areas.append(props.area)
            
    if not areas:
        return 0.0
        
    # Calculate median
    median_area = np.median(areas)
    
    return float(median_area)
