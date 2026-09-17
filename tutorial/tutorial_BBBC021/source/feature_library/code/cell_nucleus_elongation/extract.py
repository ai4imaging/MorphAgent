def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import remove_small_objects
    from scipy import ndimage

    # Convert to appropriate array type
    # The input is expected to be (512, 512, 3) uint8 based on dataset description
    arr = np.asarray(img)
    
    # Check dimensionality and extract the nuclear channel (Channel 2 - Blue/DAPI)
    # If the image is 2D (H, W), assume it's already a single channel or grayscale
    # If (H, W, C), extract index 2.
    if arr.ndim == 3 and arr.shape[2] >= 3:
        nuclear_channel = arr[:, :, 2]
    elif arr.ndim == 2:
        nuclear_channel = arr
    else:
        # Unexpected format, return 0.0
        return 0.0

    # Determine the labeled mask
    labeled_mask = None

    # Scenario 1: Use provided segmentation masks
    # We check if any masks were passed. If so, we assume the first one is relevant or iterate to find a nuclear one.
    # Given the prompt structure, if masks are present, we try to use the first one.
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        mask_input = segmentation_masks[0]
        
        # Ensure mask matches image spatial dimensions
        if mask_input.shape[:2] == nuclear_channel.shape[:2]:
            # If the mask is already integer labeled (0=bg, 1,2...=objects), use it directly
            if np.issubdtype(mask_input.dtype, np.integer) and mask_input.max() > 1:
                labeled_mask = mask_input
            else:
                # If binary/boolean, label it
                labeled_mask = label(mask_input > 0)

    # Scenario 2: Fallback - Perform on-the-fly segmentation if no valid mask provided
    if labeled_mask is None:
        # Normalize for thresholding
        img_float = nuclear_channel.astype(np.float32)
        
        # Basic noise reduction
        img_smooth = ndimage.gaussian_filter(img_float, sigma=2.0)
        
        # Determine threshold (handle empty/black images)
        if img_smooth.max() > img_smooth.min():
            try:
                thresh = threshold_otsu(img_smooth)
                binary_mask = img_smooth > thresh
            except Exception:
                # Fallback if otsu fails (e.g. uniform image)
                binary_mask = img_smooth > np.mean(img_smooth)
        else:
            binary_mask = np.zeros_like(img_smooth, dtype=bool)

        # Clean up the mask: remove small artifacts (noise)
        # Assuming 512x512 image of cells, nuclei should be reasonably sized (>30 pixels)
        binary_mask = remove_small_objects(binary_mask, min_size=30)
        
        # Label the connected components
        labeled_mask = label(binary_mask)

    # Compute properties
    # regionprops returns a list of RegionProperties objects
    regions = regionprops(labeled_mask)

    if not regions:
        return 0.0

    # Extract eccentricity for each nucleus
    # Eccentricity is a value between 0 (circle) and 1 (line segment)
    eccentricities = [r.eccentricity for r in regions]

    if not eccentricities:
        return 0.0

    # Calculate the mean
    result = np.mean(eccentricities)

    return float(result)
