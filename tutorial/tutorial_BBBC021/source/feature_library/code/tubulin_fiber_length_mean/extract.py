def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.morphology import skeletonize, white_tophat, disk
    from skimage.filters import threshold_otsu, gaussian
    from skimage.measure import label, regionprops

    # 1. Input Validation and Preprocessing
    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Check for valid dimensions (512, 512, 3) expected
    if arr.ndim != 3 or arr.shape[2] < 2:
        return 0.0

    # Extract Tubulin channel (Channel 1 - Green)
    # Channel mapping: 0=Actin, 1=Tubulin, 2=DAPI
    tubulin = arr[..., 1]

    # Normalize intensity to [0, 1]
    # Use robust max to avoid hot pixels skewing normalization
    vmax = np.percentile(tubulin, 99.5) if tubulin.size > 0 else 1.0
    if vmax > 0:
        tubulin = tubulin / vmax
    tubulin = np.clip(tubulin, 0.0, 1.0)

    # 2. Mask Handling
    # Create a region of interest (ROI) mask
    roi_mask = np.ones(tubulin.shape, dtype=bool)
    
    # If segmentation masks are provided, combine them to focus on cellular regions
    # We prefer a cytoplasm or whole-cell mask if available
    if len(segmentation_masks) > 0:
        combined_mask = np.zeros(tubulin.shape, dtype=bool)
        for mask in segmentation_masks:
            if mask is not None and mask.shape == tubulin.shape:
                combined_mask = np.logical_or(combined_mask, mask > 0)
        
        # Only apply if we actually got a valid mask out of it
        if np.any(combined_mask):
            roi_mask = combined_mask

    # 3. Fiber Enhancement
    # Use White Top-Hat transform to enhance bright ridge-like structures (fibers)
    # against the background. This is effective for fluorescent filaments.
    # A disk of radius 2-3 pixels covers typical fiber widths.
    selem = disk(2)
    enhanced_tubulin = white_tophat(tubulin, footprint=selem)
    
    # Apply slight Gaussian smoothing to reduce pixel noise before thresholding
    enhanced_tubulin = gaussian(enhanced_tubulin, sigma=1.0)

    # 4. Binarization
    # Calculate threshold only within the ROI to avoid background bias
    valid_pixels = enhanced_tubulin[roi_mask]
    if valid_pixels.size == 0 or np.max(valid_pixels) == 0:
        return 0.0
        
    try:
        thresh = threshold_otsu(valid_pixels)
    except ValueError:
        # Fallback if image is uniform
        thresh = 0.0
        
    binary_fibers = (enhanced_tubulin > thresh) & roi_mask

    # 5. Skeletonization
    # Reduce fibers to 1-pixel wide centerlines
    skeleton = skeletonize(binary_fibers)
    
    if not np.any(skeleton):
        return 0.0

    # 6. Branch Decomposition and Length Measurement
    # To measure lengths of individual segments, we need to break the skeleton
    # at junction points.
    
    # A pixel is a junction if it has > 2 neighbors in the 8-connected neighborhood
    # Convolve with a ring filter to count neighbors
    # Kernel for 8-connectivity neighbor counting (center is 0)
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    
    skeleton_int = skeleton.astype(np.uint8)
    neighbor_count = ndimage.convolve(skeleton_int, kernel, mode='constant', cval=0)
    
    # Junctions are skeleton pixels with > 2 neighbors
    junctions = (skeleton_int > 0) & (neighbor_count > 2)
    
    # Subtract junctions from the skeleton to break it into segments
    # We use logical XOR or subtraction. 
    # Note: Removing junctions might remove pixels, slightly underestimating length,
    # but it allows unique labeling of branches.
    segments = (skeleton_int > 0) & (~junctions)
    
    # Label the disconnected segments
    labeled_segments, num_segments = label(segments, connectivity=2, return_num=True)
    
    if num_segments == 0:
        return 0.0

    # Measure properties of segments
    props = regionprops(labeled_segments)
    
    # Filter out very small segments (noise/spurs)
    # A minimal fiber segment should be at least ~3 pixels
    min_length = 3
    lengths = []
    
    for prop in props:
        # Area is the number of pixels in the skeleton branch
        # For a 1-pixel wide line, Area approx Length
        # We can improve accuracy by considering diagonal connections, 
        # but pixel count is a standard robust proxy for this type of feature.
        segment_len = prop.area
        if segment_len >= min_length:
            lengths.append(segment_len)

    # 7. Compute Statistics
    if not lengths:
        return 0.0
        
    mean_length = np.mean(lengths)
    
    return float(mean_length)
