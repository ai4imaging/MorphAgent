def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_opening, disk

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset: (512, 512, 3), Channel 2 is DAPI (Nucleus)
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Extract the DAPI channel (Index 2)
        nucleus_channel = arr[:, :, 2]
    elif arr.ndim == 2:
        # Fallback if passed a single channel image (assume it's the relevant one)
        nucleus_channel = arr
    else:
        # Unexpected format
        return 0.0

    # Normalize intensity
    vmax = np.percentile(nucleus_channel, 99.5) if nucleus_channel.size > 0 else 1.0
    if vmax > 0:
        nucleus_channel = nucleus_channel / vmax
    nucleus_channel = np.clip(nucleus_channel, 0.0, 1.0)

    # Handle segmentation masks
    labeled_mask = None
    
    # Check if valid segmentation masks are provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        mask_input = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions
        if mask_input.shape[:2] == nucleus_channel.shape[:2]:
            # If mask is already labeled (int), use it. If binary, label it.
            if np.issubdtype(mask_input.dtype, np.integer) and mask_input.max() > 1:
                labeled_mask = mask_input
            else:
                labeled_mask = label(mask_input > 0)
    
    # Fallback: Perform segmentation if no mask provided
    if labeled_mask is None:
        # Gaussian blur to reduce noise
        blurred = ndimage.gaussian_filter(nucleus_channel, sigma=2)
        # Otsu thresholding
        try:
            thresh = threshold_otsu(blurred)
            binary_mask = blurred > thresh
        except ValueError: # Handle empty/uniform images
            return 0.0
            
        # Morphological opening to separate touching nuclei slightly and remove noise
        binary_mask = binary_opening(binary_mask, disk(3))
        # Label connected components
        labeled_mask = label(binary_mask)

    # Extract properties
    regions = regionprops(labeled_mask)
    
    # Collect orientations
    # regionprops returns orientation in radians, range [-pi/2, pi/2]
    orientations = []
    for props in regions:
        # Filter small artifacts
        if props.area < 50:
            continue
        orientations.append(props.orientation)

    orientations = np.array(orientations)
    n = len(orientations)

    # If fewer than 2 nuclei, standard deviation is undefined/zero
    if n < 2:
        return 0.0

    # Calculate Circular Standard Deviation for Axial Data
    # 1. Map axial data (0 to pi) to circular data (0 to 2pi) by doubling angles
    #    This accounts for the fact that 0 degrees is the same as 180 degrees for an ellipse.
    angles_doubled = 2 * orientations

    # 2. Compute the mean resultant vector length (R)
    #    R ranges from 0 (uniform dispersion) to 1 (perfect concentration)
    sin_sum = np.sum(np.sin(angles_doubled))
    cos_sum = np.sum(np.cos(angles_doubled))
    R = np.sqrt(sin_sum**2 + cos_sum**2) / n

    # 3. Compute circular standard deviation
    #    Formula: sqrt(-2 * ln(R))
    #    Clip R to avoid log(0) or log(negative) due to float precision
    R = np.clip(R, 1e-9, 1.0)
    
    # If R is 1.0, log(1) is 0, std is 0.
    circ_std_doubled = np.sqrt(-2 * np.log(R))

    # 4. Rescale back to original domain
    circ_std = circ_std_doubled / 2.0

    return float(circ_std)
