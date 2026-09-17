def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import remove_small_objects, binary_closing, disk
    from scipy import ndimage

    # 1. Data Preparation & Channel Selection
    # The dataset description specifies Channel 2 (Blue) is DAPI/Nucleus.
    # Input shape is (512, 512, 3).
    # We need to extract the nuclear channel for analysis.
    
    # Ensure input is an array
    img = np.asarray(img)
    
    # Handle dimensionality
    if img.ndim == 3 and img.shape[-1] == 3:
        # Standard (H, W, C) format -> Extract Blue channel (Index 2)
        nuclear_channel = img[..., 2]
    elif img.ndim == 2:
        # If already 2D, assume it's the relevant channel (fallback)
        nuclear_channel = img
    else:
        # Unexpected format, return 0.0
        return 0.0

    # 2. Segmentation Logic
    # We prioritize provided masks, but fallback to on-the-fly segmentation if needed.
    
    labeled_nuclei = None

    # Check if valid segmentation masks are provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Assume the first mask is the nuclear mask based on typical ordering
        mask_input = segmentation_masks[0]
        
        # Ensure mask matches image spatial dimensions
        if mask_input.shape[:2] == nuclear_channel.shape[:2]:
            # If mask is already labeled (int), use it. If boolean/binary, label it.
            if np.issubdtype(mask_input.dtype, np.integer) and mask_input.max() > 1:
                labeled_nuclei = mask_input
            else:
                labeled_nuclei = label(mask_input > 0)

    # Fallback: On-the-fly segmentation if no mask provided
    if labeled_nuclei is None:
        # Preprocessing: Gaussian blur to reduce noise
        # Normalize to float for processing
        nuc_float = nuclear_channel.astype(np.float32)
        nuc_smooth = ndimage.gaussian_filter(nuc_float, sigma=2.0)
        
        # Thresholding
        try:
            thresh_val = threshold_otsu(nuc_smooth)
            binary_mask = nuc_smooth > thresh_val
        except ValueError:
            # Handle case where image is uniform (e.g., all black)
            return 0.0
            
        # Refinement: Close gaps and remove small artifacts
        binary_mask = binary_closing(binary_mask, disk(2))
        binary_mask = remove_small_objects(binary_mask, min_size=50)
        
        # Labeling
        labeled_nuclei = label(binary_mask)

    # 3. Feature Computation
    # We need to calculate the eccentricity for each nucleus.
    # Eccentricity is a property of the ellipse with the same second-moments as the region.
    # 0 = circle, -> 1 = line segment.
    
    regions = regionprops(labeled_nuclei)
    
    if not regions:
        return 0.0
        
    eccentricities = []
    for prop in regions:
        # Filter out extremely small regions that might be noise artifacts
        # (though remove_small_objects helps, labeled masks passed in might be raw)
        if prop.area < 30:
            continue
        eccentricities.append(prop.eccentricity)
    
    # 4. Statistical Aggregation
    # Compute the standard deviation of the eccentricities.
    # High std dev -> Heterogeneous population (mix of round interphase and elongated mitotic/damaged cells).
    
    if len(eccentricities) < 2:
        # Standard deviation requires at least two data points to be meaningful regarding population variance,
        # though numpy returns 0.0 for a single item, which is mathematically acceptable here.
        return 0.0
        
    result = np.std(eccentricities)
    
    return float(result)
