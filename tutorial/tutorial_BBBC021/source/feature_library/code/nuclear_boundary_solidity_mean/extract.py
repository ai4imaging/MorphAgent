def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import remove_small_objects

    # Convert to appropriate array type
    # The dataset is uint8, but we convert to float32 for any intensity math if needed,
    # though for this specific morphology feature, we mostly need the mask.
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Expected shape: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        # If unexpected dimensions, return 0.0
        return 0.0

    # Identify the Nuclei Channel (Channel 2 / Blue based on dataset description)
    # Channel 0: Actin, Channel 1: Tubulin, Channel 2: DAPI (Nuclei)
    nuclei_channel = arr[..., 2]

    # Determine the segmentation mask to use
    labeled_mask = None

    # Check if valid segmentation masks are provided
    # We prioritize provided masks. If multiple are provided, we assume the order matches 
    # the channel order or the first one is the primary object (nuclei).
    # Given the context of "nuclear_boundary_solidity", we look for a mask.
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        candidate_mask = segmentation_masks[0]
        # Ensure it matches image spatial dimensions
        if candidate_mask.shape[:2] == arr.shape[:2]:
            # If the mask is not integer labeled (e.g. binary), label it
            if candidate_mask.dtype == bool or candidate_mask.max() == 1:
                labeled_mask = label(candidate_mask)
            else:
                labeled_mask = candidate_mask.astype(int)

    # Fallback: Perform on-the-fly segmentation if no valid mask is provided
    if labeled_mask is None:
        # 1. Smooth the image to reduce noise
        # Normalize to 0-1 for thresholding stability
        norm_nuclei = nuclei_channel / 255.0
        blurred = ndimage.gaussian_filter(norm_nuclei, sigma=2.0)
        
        # 2. Thresholding (Otsu)
        # Check if image is not empty/black
        if np.max(blurred) > 0:
            try:
                thresh = threshold_otsu(blurred)
                binary_mask = blurred > thresh
            except ValueError:
                # Handle case where image is uniform (e.g. all black)
                binary_mask = np.zeros_like(blurred, dtype=bool)
        else:
            binary_mask = np.zeros_like(blurred, dtype=bool)

        # 3. Clean up mask
        # Remove small artifacts (e.g., < 50 pixels)
        binary_mask = remove_small_objects(binary_mask, min_size=50)
        
        # 4. Label connected components
        labeled_mask = label(binary_mask)

    # Compute Feature: Mean Solidity
    # Solidity = Area / Convex Hull Area
    # regionprops calculates this automatically as .solidity
    
    regions = regionprops(labeled_mask)
    
    solidity_values = []
    
    for region in regions:
        # Filter out extremely small regions that might still exist or be noise
        # (e.g., single pixel dots that survived or were in the provided mask)
        if region.area >= 50:
            solidity_values.append(region.solidity)
            
    # Aggregate results
    if len(solidity_values) == 0:
        result = 0.0
    else:
        result = np.mean(solidity_values)

    return float(result)
