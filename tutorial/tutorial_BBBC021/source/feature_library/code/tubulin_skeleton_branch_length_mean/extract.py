def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.morphology import skeletonize
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3), Channel 1 is Tubulin (Green)
    if arr.ndim == 3 and arr.shape[2] >= 2:
        tubulin = arr[..., 1]
    elif arr.ndim == 2:
        tubulin = arr
    else:
        # Unexpected format
        return 0.0

    # Intensity normalization
    # Image is uint8, so range is 0-255. Normalize to 0-1.
    # We use robust min/max to handle potential pre-normalized inputs or float inputs
    t_min, t_max = np.min(tubulin), np.max(tubulin)
    if t_max > t_min:
        tubulin = (tubulin - t_min) / (t_max - t_min)
    else:
        return 0.0

    # Preprocessing: Gaussian blur to smooth filaments and reduce noise spurs
    # This is critical for skeletonization to avoid creating tiny branches from pixel noise
    tubulin_smooth = ndimage.gaussian_filter(tubulin, sigma=1.0)

    # Segmentation (Binarization)
    try:
        thresh = threshold_otsu(tubulin_smooth)
        binary = tubulin_smooth > thresh
    except Exception:
        return 0.0

    # Handle segmentation masks if available
    # If masks are provided, we restrict the analysis to the cellular regions
    # to avoid skeletonizing background noise/debris.
    if len(segmentation_masks) > 0:
        combined_mask = np.zeros(tubulin.shape, dtype=bool)
        has_valid_mask = False
        for mask in segmentation_masks:
            # Ensure mask matches image shape (handle potential 2D/3D mismatch)
            if mask.shape == tubulin.shape:
                combined_mask = combined_mask | (mask > 0)
                has_valid_mask = True
            elif mask.ndim == 2 and tubulin.ndim == 2 and mask.shape == tubulin.shape:
                combined_mask = combined_mask | (mask > 0)
                has_valid_mask = True
        
        if has_valid_mask and np.any(combined_mask):
            binary = binary & combined_mask

    # Skeletonization
    # Reduces the binary filaments to 1-pixel wide lines
    skeleton = skeletonize(binary)
    
    if not np.any(skeleton):
        return 0.0

    # Branch Analysis
    # To measure branch lengths, we identify junctions (pixels with > 2 neighbors)
    # and remove them, breaking the skeleton into isolated segments.
    
    # 1. Identify Junctions
    # Kernel to count neighbors (8-connectivity)
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)

    skeleton_int = skeleton.astype(np.uint8)
    
    # Count neighbors for every pixel in the skeleton
    neighbor_count = ndimage.convolve(skeleton_int, kernel, mode='constant', cval=0)
    
    # A pixel is a junction if it is part of the skeleton AND has > 2 neighbors
    # (Endpoints have 1, lines have 2, junctions have 3+)
    junctions = (skeleton_int > 0) & (neighbor_count > 2)
    
    # 2. Isolate Branches
    # Subtract junctions from the skeleton
    branches = skeleton & ~junctions
    
    # 3. Label Segments
    # Use 8-connectivity (connectivity=2) to keep diagonal lines connected
    labeled_branches, num_branches = label(branches, return_num=True, connectivity=2)
    
    if num_branches == 0:
        return 0.0

    # 4. Measure Lengths
    props = regionprops(labeled_branches)
    
    # For 1-pixel wide skeletons, Area is a robust proxy for Length in pixels
    lengths = [p.area for p in props]
    
    # Filter out tiny spurs (artifacts of skeletonization)
    # Branches < 3 pixels are often noise or artifacts near junctions
    min_branch_length = 3
    valid_lengths = [l for l in lengths if l >= min_branch_length]
    
    if not valid_lengths:
        return 0.0
        
    # Calculate Mean Branch Length
    result = np.mean(valid_lengths)

    return float(result)
