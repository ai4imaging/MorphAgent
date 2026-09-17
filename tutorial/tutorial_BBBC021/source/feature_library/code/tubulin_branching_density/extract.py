def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.morphology import skeletonize, binary_closing, disk
    from skimage.filters import threshold_otsu
    
    # 1. Data Loading and Validation
    # Convert to float32 for processing
    arr = np.asarray(img, dtype=np.float32)
    
    # Check dimensionality: Expecting (H, W, 3) for BBBC021
    if arr.ndim != 3 or arr.shape[2] != 3:
        # If not the expected 3-channel image, return 0.0
        return 0.0

    # 2. Channel Extraction
    # Channel 1 is Tubulin (Green) based on dataset description
    tubulin_channel = arr[..., 1]

    # 3. Preprocessing
    # Normalize intensity to [0, 1]
    vmax = np.percentile(tubulin_channel, 99.5) if tubulin_channel.size > 0 else 1.0
    if vmax <= 0:
        vmax = 1.0
    tubulin_norm = np.clip(tubulin_channel / vmax, 0.0, 1.0)
    
    # Apply Gaussian smoothing to reduce noise before thresholding/skeletonization
    # This prevents single-pixel noise from creating false branches
    tubulin_smooth = ndimage.gaussian_filter(tubulin_norm, sigma=1.0)

    # 4. Foreground Segmentation (Region of Interest)
    # Determine the area occupied by the cytoskeleton
    try:
        thresh = threshold_otsu(tubulin_smooth)
        binary_mask = tubulin_smooth > thresh
    except Exception:
        # Fallback if image is uniform
        return 0.0

    # If segmentation masks are provided, intersect with them to focus on valid cell areas
    if len(segmentation_masks) > 0:
        # Combine all masks into a single boolean mask
        combined_seg = np.zeros(binary_mask.shape, dtype=bool)
        for mask in segmentation_masks:
            if mask is not None:
                # Ensure mask matches image spatial dimensions
                if mask.shape[:2] == binary_mask.shape:
                    combined_seg = np.logical_or(combined_seg, mask > 0)
        
        # Only apply intersection if we actually found valid masks
        if np.any(combined_seg):
            binary_mask = np.logical_and(binary_mask, combined_seg)

    # Clean up the binary mask
    # Close small gaps to ensure filaments are continuous
    binary_mask = binary_closing(binary_mask, disk(1))
    
    # Calculate the total area of the cytoskeleton (denominator for density)
    foreground_area = np.sum(binary_mask)
    
    # If no foreground detected, return 0
    if foreground_area < 10:  # Arbitrary small threshold to avoid division by zero/noise
        return 0.0

    # 5. Skeletonization
    # Reduce the binary structures to 1-pixel wide lines
    skeleton = skeletonize(binary_mask)

    # 6. Branch Point Detection
    # A branch point in a skeleton is a pixel with > 2 neighbors
    # We use a convolution to count neighbors for every pixel
    
    # Define 3x3 kernel for neighbor counting (8-connectivity)
    # Center is 0 because we only want to count neighbors, not the pixel itself
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    
    # Convert skeleton to uint8 for convolution
    skeleton_uint = skeleton.astype(np.uint8)
    
    # Convolve to count neighbors
    neighbor_count = ndimage.convolve(skeleton_uint, kernel, mode='constant', cval=0)
    
    # Identify branch points:
    # 1. Must be part of the skeleton (skeleton_uint == 1)
    # 2. Must have > 2 neighbors
    branch_points = (skeleton_uint == 1) & (neighbor_count > 2)
    
    num_branches = np.sum(branch_points)

    # 7. Compute Density
    # Branch points per unit of cytoskeleton area
    # We multiply by a scaling factor (e.g., 1000) to make the numbers more readable/manageable
    # or return raw density. Here we return raw density.
    density = num_branches / foreground_area

    return float(density)
