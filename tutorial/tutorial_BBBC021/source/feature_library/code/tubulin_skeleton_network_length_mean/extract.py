def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.morphology import skeletonize, remove_small_objects
    from skimage.filters import threshold_otsu
    from skimage.measure import label, regionprops
    from scipy import ndimage

    # 1. Input Validation and Preprocessing
    # Convert to appropriate array type if needed, but keep uint8 for thresholding if possible
    # The dataset description says (512, 512, 3) uint8.
    
    # Check for valid image shape and content
    if img is None or img.size == 0:
        return 0.0
    
    # Handle dimensionality
    # We expect (H, W, C). If 2D (H, W), we assume it's a single channel or projection.
    # If 3D (H, W, C), we extract the Tubulin channel.
    
    tubulin_img = None
    
    if img.ndim == 3 and img.shape[2] == 3:
        # Channel 1 is Tubulin (Green) based on dataset description
        tubulin_img = img[:, :, 1]
    elif img.ndim == 2:
        # Fallback: if only 2D provided, assume it's the relevant channel
        tubulin_img = img
    else:
        # Unexpected format
        return 0.0

    # Ensure we have data to work with
    if tubulin_img.max() == 0:
        return 0.0

    # 2. Global Segmentation of Tubulin Structure
    # We need to convert the grayscale tubulin image into a binary mask of the cytoskeleton structure.
    # Smooth slightly to reduce noise before thresholding
    smoothed = ndimage.gaussian_filter(tubulin_img, sigma=1.0)
    
    try:
        thresh = threshold_otsu(smoothed)
        binary_tubulin = smoothed > thresh
    except Exception:
        # Fallback if otsu fails (e.g. uniform image)
        binary_tubulin = smoothed > np.mean(smoothed)

    # Remove small noise specks that would create tiny skeletons
    # A microtubule fragment should be reasonably sized.
    binary_tubulin = remove_small_objects(binary_tubulin, min_size=20)

    # 3. Determine Cell Regions
    # We need to calculate the mean length *per cell*.
    # We look for a segmentation mask.
    
    cell_mask = None
    
    # Heuristic to find the best mask:
    # If masks are provided, we prefer the one with the largest coverage (likely cytoplasm/whole cell)
    # over smaller ones (likely nuclei).
    if segmentation_masks and len(segmentation_masks) > 0:
        best_mask_idx = -1
        max_coverage = -1
        
        for i, mask in enumerate(segmentation_masks):
            if mask is None:
                continue
            # Ensure mask shape matches image shape (ignoring channels)
            if mask.shape != tubulin_img.shape:
                continue
                
            coverage = np.sum(mask > 0)
            if coverage > max_coverage:
                max_coverage = coverage
                best_mask_idx = i
        
        if best_mask_idx != -1:
            cell_mask = segmentation_masks[best_mask_idx]

    # If no valid mask found, we can treat the whole image as one "cell" or 
    # try to infer objects from the binary tubulin itself (though this risks merging cells).
    # Given the feature definition "per cell", if we have no mask, we might be limited.
    # However, to be robust, if no mask is provided, we treat connected components of the 
    # binary tubulin as "objects" or simply return the total length if we can't separate.
    # Let's try to label the binary tubulin itself if no mask is present, 
    # assuming distinct cell colonies.
    if cell_mask is None:
        cell_mask = label(binary_tubulin)

    # 4. Per-Cell Skeletonization and Measurement
    props = regionprops(cell_mask)
    
    if not props:
        return 0.0

    skeleton_lengths = []

    for prop in props:
        # Optimization: Crop to the bounding box of the cell
        minr, minc, maxr, maxc = prop.bbox
        
        # Extract the binary tubulin within this bounding box
        tubulin_crop = binary_tubulin[minr:maxr, minc:maxc]
        
        # Extract the cell mask within this bounding box
        mask_crop = cell_mask[minr:maxr, minc:maxc]
        
        # Create a specific mask for THIS cell (handle overlapping bounding boxes)
        # We only want to skeletonize the tubulin belonging to the current label
        current_cell_bool = (mask_crop == prop.label)
        
        # Intersect: Tubulin signal MUST be inside the current cell region
        cell_specific_tubulin = tubulin_crop & current_cell_bool
        
        if not np.any(cell_specific_tubulin):
            skeleton_lengths.append(0.0)
            continue

        # Skeletonize
        # This reduces the tubulin network to 1-pixel wide lines
        skeleton = skeletonize(cell_specific_tubulin)
        
        # Calculate length
        # Simple pixel count is a standard approximation for skeleton length
        length = np.sum(skeleton)
        skeleton_lengths.append(length)

    # 5. Aggregate Results
    if not skeleton_lengths:
        return 0.0
        
    result = np.mean(skeleton_lengths)

    return float(result)
