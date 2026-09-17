def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.segmentation import clear_border
    from skimage.morphology import binary_opening, disk
    import math

    # Convert to appropriate array type
    # The input is (512, 512, 3) uint8
    arr = np.asarray(img)
    
    # Handle dimensionality and extract Nuclear Channel (DAPI)
    # Dataset info: Channel 2 is DAPI (Blue)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        nuclear_channel = arr[..., 2]
    elif arr.ndim == 2:
        # Fallback if single channel passed (unlikely based on spec but safe)
        nuclear_channel = arr
    else:
        return 0.0

    # Determine Segmentation Mask
    # If segmentation masks are provided, use the first one (assuming it's the primary object mask)
    # Otherwise, generate one using Otsu thresholding
    labeled_mask = None
    
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        mask_input = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions
        if mask_input.shape[:2] == nuclear_channel.shape[:2]:
            if mask_input.dtype == bool:
                labeled_mask = label(mask_input)
            else:
                # Assume it's already labeled or integer mask
                labeled_mask = mask_input.astype(int)
    
    # Fallback: Generate mask if not provided or invalid
    if labeled_mask is None:
        # Normalize for thresholding
        norm_img = nuclear_channel.astype(np.float32)
        
        # Check for empty image
        if np.max(norm_img) == np.min(norm_img):
            return 0.0
            
        thresh = threshold_otsu(norm_img)
        binary_mask = norm_img > thresh
        
        # Clean up noise
        binary_mask = binary_opening(binary_mask, disk(2))
        
        # Label connected components
        labeled_mask = label(binary_mask)

    # Clear objects touching the border
    # Incomplete objects have incorrect perimeters, distorting the form factor
    labeled_mask = clear_border(labeled_mask)

    # Compute Form Factor for each nucleus
    # Form Factor = (4 * pi * Area) / (Perimeter^2)
    regions = regionprops(labeled_mask)
    
    form_factors = []
    
    for region in regions:
        area = region.area
        perimeter = region.perimeter
        
        # Filter small artifacts
        if area < 50:
            continue
            
        # Avoid division by zero
        if perimeter == 0:
            continue
            
        # Calculate Form Factor
        ff = (4 * math.pi * area) / (perimeter ** 2)
        form_factors.append(ff)

    # Aggregation
    # Return the mean form factor of the population
    if not form_factors:
        return 0.0
        
    result = np.mean(form_factors)

    return float(result)
