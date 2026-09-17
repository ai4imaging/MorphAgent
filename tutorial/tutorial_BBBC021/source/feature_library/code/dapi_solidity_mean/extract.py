def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import remove_small_objects
    from scipy import ndimage

    # Convert to appropriate array type
    arr = np.asarray(img)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3) uint8.
    # Channel 2 is DAPI (Nucleus).
    if arr.ndim == 3 and arr.shape[2] == 3:
        dapi_channel = arr[:, :, 2]
    elif arr.ndim == 2:
        # Fallback if a single channel image is passed (unlikely given spec, but safe)
        dapi_channel = arr
    else:
        return 0.0

    # Determine the labeled mask to use
    labeled_mask = None

    # 1. Try to use provided segmentation masks
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Assume the first mask corresponds to nuclei or cells
        mask_input = segmentation_masks[0]
        
        # Ensure mask matches image spatial dimensions
        if mask_input.shape[:2] == dapi_channel.shape[:2]:
            # If the mask is already labeled (int type with values > 1), use it directly
            if np.issubdtype(mask_input.dtype, np.integer) and mask_input.max() > 1:
                labeled_mask = mask_input
            else:
                # If binary mask, label it
                labeled_mask = label(mask_input > 0)

    # 2. Fallback: Perform on-the-fly segmentation if no valid mask provided
    if labeled_mask is None:
        # Preprocessing: Gaussian blur to reduce noise
        # Normalize to float for processing
        dapi_float = dapi_channel.astype(np.float32)
        
        # Check for empty image
        if dapi_float.max() == 0:
            return 0.0
            
        blurred = ndimage.gaussian_filter(dapi_float, sigma=2.0)
        
        # Thresholding (Otsu)
        try:
            thresh = threshold_otsu(blurred)
            binary_mask = blurred > thresh
        except ValueError:
            # Handle case where image is uniform (e.g., all black)
            return 0.0

        # Post-processing: Remove small artifacts (noise)
        # Min size 50 pixels is a conservative estimate for a nucleus at 512x512
        cleaned_mask = remove_small_objects(binary_mask, min_size=50)
        
        # Label connected components
        labeled_mask = label(cleaned_mask)

    # Compute Feature: Mean Solidity
    # Solidity = Area / ConvexHullArea
    props = regionprops(labeled_mask)
    
    if not props:
        return 0.0

    solidity_values = [p.solidity for p in props]
    
    if not solidity_values:
        return 0.0
        
    result = np.mean(solidity_values)

    return float(result)
