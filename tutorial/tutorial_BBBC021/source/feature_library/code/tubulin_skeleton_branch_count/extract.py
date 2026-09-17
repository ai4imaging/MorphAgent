def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.morphology import skeletonize, white_tophat, disk
    from skimage.filters import threshold_otsu, gaussian
    
    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3) where Channel 1 is Tubulin (Green)
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Extract Tubulin channel (Index 1)
        tubulin = arr[..., 1]
    elif arr.ndim == 2:
        # Fallback if single channel passed (unlikely based on desc, but safe)
        tubulin = arr
    else:
        return 0.0

    # Intensity normalization
    # Normalize to [0, 1] range for processing
    vmax = np.percentile(tubulin, 99.5) if tubulin.size > 0 else 1.0
    if vmax > 0:
        tubulin = tubulin / vmax
    tubulin = np.clip(tubulin, 0.0, 1.0)

    # Preprocessing to enhance filaments
    # 1. Gaussian blur to reduce noise
    tubulin_smooth = gaussian(tubulin, sigma=1.0)
    
    # 2. White Top-Hat transform to enhance bright filament structures against background
    # This is crucial for separating microtubules from diffuse background fluorescence
    # Using a small disk structuring element (radius 2-3 pixels) matches filament width
    tubulin_enhanced = white_tophat(tubulin_smooth, footprint=disk(2))

    # Binarization
    # Use Otsu's method to find a global threshold for the enhanced structures
    try:
        thresh = threshold_otsu(tubulin_enhanced)
        binary_tubulin = tubulin_enhanced > thresh
    except Exception:
        # Fallback if image is uniform (e.g. empty)
        return 0.0

    # Handle segmentation masks if available to restrict analysis to cells
    # If masks are provided, use the union of all masks as the ROI
    roi_mask = None
    if len(segmentation_masks) > 0:
        roi_mask = np.zeros(binary_tubulin.shape, dtype=bool)
        for mask in segmentation_masks:
            if mask is not None:
                # Ensure mask matches image dimensions (handle potential 3D vs 2D mismatch)
                if mask.shape == binary_tubulin.shape:
                    roi_mask = roi_mask | (mask > 0)
                elif mask.ndim == 3 and mask.shape[:2] == binary_tubulin.shape:
                     # If mask is 3D (e.g. labeled volume), project or slice
                     roi_mask = roi_mask | (np.max(mask, axis=2) > 0)
        
        # Apply ROI mask
        if roi_mask.any():
            binary_tubulin = binary_tubulin & roi_mask
        else:
            # If masks were provided but empty/invalid, return 0
            return 0.0

    # Skeletonization
    # Reduce the binary filaments to 1-pixel wide lines
    skeleton = skeletonize(binary_tubulin)

    # Branch Point Detection
    # A branch point in a skeleton is a pixel with > 2 neighbors
    # We use convolution to count neighbors for every pixel
    
    # Kernel for 8-connectivity neighbor counting
    # [[1, 1, 1],
    #  [1, 0, 1],
    #  [1, 1, 1]]
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)

    # Convert skeleton to uint8 for convolution
    skeleton_int = skeleton.astype(np.uint8)
    
    # Count neighbors
    neighbor_count = ndimage.convolve(skeleton_int, kernel, mode='constant', cval=0)
    
    # Identify branch points:
    # 1. Must be part of the skeleton (skeleton_int == 1)
    # 2. Must have more than 2 neighbors
    branch_points_mask = (skeleton_int == 1) & (neighbor_count > 2)
    
    # Calculate the total number of branch points
    result = np.sum(branch_points_mask)

    return float(result)
