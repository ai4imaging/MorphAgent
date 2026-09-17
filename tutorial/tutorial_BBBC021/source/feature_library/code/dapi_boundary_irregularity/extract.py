def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_closing, disk
    from scipy import ndimage

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3) RGB TIFF
    # Channel 2 is DAPI (Blue), which targets the Nucleus
    if arr.ndim == 3 and arr.shape[2] == 3:
        dapi_channel = arr[:, :, 2]
    elif arr.ndim == 2:
        # Fallback if single channel passed
        dapi_channel = arr
    else:
        return 0.0

    # Determine the segmentation mask to use
    labeled_mask = None
    
    # Check if segmentation masks are provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use the first provided mask (assuming it corresponds to nuclei or cells)
        mask_input = segmentation_masks[0]
        
        # Ensure mask matches image spatial dimensions
        if mask_input.shape[:2] == dapi_channel.shape[:2]:
            # If the mask is already labeled (int type with values > 1), use it directly
            if np.issubdtype(mask_input.dtype, np.integer) and mask_input.max() > 1:
                labeled_mask = mask_input
            else:
                # If binary, label it
                labeled_mask = label(mask_input > 0)
    
    # Fallback: Generate mask on the fly if no valid mask provided
    if labeled_mask is None:
        # Normalize DAPI channel for thresholding
        if dapi_channel.max() > dapi_channel.min():
            norm_dapi = (dapi_channel - dapi_channel.min()) / (dapi_channel.max() - dapi_channel.min())
        else:
            norm_dapi = dapi_channel
            
        # Smooth to reduce noise before thresholding
        smooth_dapi = ndimage.gaussian_filter(norm_dapi, sigma=2.0)
        
        try:
            thresh = threshold_otsu(smooth_dapi)
            binary_mask = smooth_dapi > thresh
            # Close small holes/gaps to ensure solid objects
            binary_mask = binary_closing(binary_mask, disk(2))
            labeled_mask = label(binary_mask)
        except Exception:
            return 0.0

    # Compute Feature: Boundary Irregularity
    # Metric: (Perimeter^2) / (4 * pi * Area)
    # A perfect circle has value 1.0. Higher values indicate irregularity/blebbing.
    
    regions = regionprops(labeled_mask)
    irregularity_scores = []
    
    for region in regions:
        # Filter out small noise artifacts
        if region.area < 50:
            continue
            
        area = region.area
        perimeter = region.perimeter
        
        if area == 0:
            continue
            
        # Calculate circularity-based irregularity
        # Formula: P^2 / (4 * pi * A)
        # Value >= 1.0
        metric = (perimeter ** 2) / (4 * np.pi * area)
        
        # We subtract 1.0 so that a perfect circle is 0.0, making it a measure of "deviation"
        # However, the prompt asks for "irregularity", and P^2/A is standard. 
        # Let's keep the raw ratio but ensure it's at least 1.0 theoretically.
        # To make it a "feature" where 0 is baseline, we can return metric - 1.0.
        # Let's return the raw ratio as it's more standard in literature (Form Factor inverse).
        irregularity_scores.append(metric)

    # Aggregate results into a single scalar for the image
    if not irregularity_scores:
        return 0.0
        
    # Use median to be robust against segmentation artifacts (e.g. under-segmentation creating dumbbells)
    result = np.median(irregularity_scores)

    return float(result)
