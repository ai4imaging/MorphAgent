def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    
    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Expected shape: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract the Nuclear Channel (Channel 2: DAPI/Blue)
    # Based on dataset info: Channel 0=Actin, 1=Tubulin, 2=DAPI
    nuclear_channel = arr[:, :, 2]

    # Determine the labeled mask
    labeled_mask = None

    # Check if segmentation masks are provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use the first provided mask
        input_mask = segmentation_masks[0]
        
        # Ensure mask matches image spatial dimensions
        if input_mask.shape[:2] == nuclear_channel.shape:
            # If the mask is already labeled (integers > 1), use it directly
            if np.issubdtype(input_mask.dtype, np.integer) and input_mask.max() > 1:
                labeled_mask = input_mask
            else:
                # If binary, label it
                labeled_mask = label(input_mask > 0)
    
    # Fallback: Perform on-the-fly segmentation if no valid mask provided
    if labeled_mask is None:
        # Normalize for segmentation
        norm_nuc = nuclear_channel.copy()
        vmax = np.percentile(norm_nuc, 99.5) if norm_nuc.size > 0 else 1.0
        if vmax > 0:
            norm_nuc = norm_nuc / vmax
        norm_nuc = np.clip(norm_nuc, 0.0, 1.0)

        # Smooth to reduce noise
        smoothed = ndimage.gaussian_filter(norm_nuc, sigma=2.0)

        # Threshold
        try:
            thresh = threshold_otsu(smoothed)
            binary_mask = smoothed > thresh
        except Exception:
            # Fallback if image is uniform (e.g., all black)
            binary_mask = smoothed > 0.1

        # Clean up binary mask (remove small holes/objects)
        # Simple morphological opening to separate touching nuclei slightly and remove noise
        binary_mask = ndimage.binary_opening(binary_mask, structure=np.ones((3,3)))
        
        # Label connected components
        labeled_mask = label(binary_mask)

    # Calculate properties
    # We only need 'area'
    regions = regionprops(labeled_mask)
    
    # Extract areas
    areas = []
    for props in regions:
        # Filter out very small artifacts (e.g., < 20 pixels) to ensure we are measuring nuclei
        if props.area >= 20:
            areas.append(props.area)
            
    areas = np.array(areas, dtype=np.float64)

    # Compute Coefficient of Variation (CV)
    # CV = Standard Deviation / Mean
    if len(areas) < 2:
        # If 0 or 1 nucleus, variation is undefined or zero. Return 0.0.
        return 0.0
    
    mean_area = np.mean(areas)
    std_area = np.std(areas, ddof=1) # Use sample standard deviation

    if mean_area == 0:
        return 0.0
        
    cv = std_area / mean_area

    return float(cv)
