def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.filters.rank import entropy
    from skimage.morphology import disk
    from skimage.filters import threshold_otsu
    from scipy import ndimage

    # 1. Input Validation and Channel Extraction
    # The dataset description specifies (512, 512, 3) images.
    # Channel 0: Actin (Red) - Cytoskeleton
    # Channel 1: Tubulin (Green) - Microtubules (Target for this feature)
    # Channel 2: DAPI (Blue) - Nucleus
    
    img = np.asarray(img)
    
    # Handle dimensionality
    if img.ndim == 3 and img.shape[2] == 3:
        # Standard case: (H, W, C)
        tubulin_channel = img[:, :, 1]
        actin_channel = img[:, :, 0]
    elif img.ndim == 2:
        # Fallback: Single channel image, assume it's the relevant one
        tubulin_channel = img
        actin_channel = img
    else:
        # Unexpected format
        return 0.0

    # 2. Data Type Preparation for Entropy Calculation
    # skimage.filters.rank.entropy requires integer input (uint8 or uint16) to build histograms.
    # The dataset is uint8. If it's float, we must convert it.
    if tubulin_channel.dtype.kind == 'f':
        # Normalize to 0-255 and cast
        norm_img = (tubulin_channel - tubulin_channel.min())
        if norm_img.max() > 0:
            norm_img = norm_img / norm_img.max()
        tubulin_uint8 = (norm_img * 255).astype(np.uint8)
    else:
        # Assume it's already in a suitable integer range (e.g. uint8)
        tubulin_uint8 = tubulin_channel.astype(np.uint8)

    # 3. Mask Generation (ROI Definition)
    # We need to calculate entropy only within the cell body to avoid background noise (which has 0 entropy).
    
    mask = None
    
    # Strategy A: Use provided segmentation masks
    if len(segmentation_masks) > 0:
        # Check masks for a suitable cell/cytoplasm mask.
        # Often the first mask is cells or cytoplasm, second might be nuclei.
        # We prefer a mask that covers the cytoplasm.
        # We'll iterate and pick the largest coverage mask that isn't the whole image.
        best_mask = None
        max_area = 0
        
        for m in segmentation_masks:
            if m is None: continue
            m_arr = np.asarray(m)
            if m_arr.shape != tubulin_channel.shape:
                continue
            
            # Binarize
            bin_mask = m_arr > 0
            area = np.sum(bin_mask)
            
            # Heuristic: We want the largest mask (likely cell body) 
            if area > max_area:
                max_area = area
                best_mask = bin_mask
        
        if best_mask is not None:
            mask = best_mask

    # Strategy B: Fallback to intensity-based thresholding if no valid mask found
    if mask is None:
        # Use Actin channel for cell body segmentation as it usually defines the shape well.
        # If Actin is empty/flat, fallback to Tubulin.
        target_for_mask = actin_channel if np.std(actin_channel) > 1e-5 else tubulin_channel
        
        try:
            thresh = threshold_otsu(target_for_mask)
            mask = target_for_mask > thresh
            # Fill holes to get a solid ROI
            mask = ndimage.binary_fill_holes(mask)
        except Exception:
            # Fallback if otsu fails (e.g. uniform image)
            mask = np.ones_like(tubulin_channel, dtype=bool)

    # Ensure mask is boolean
    mask = mask.astype(bool)
    
    # If mask is empty (no cells), return 0.0
    if np.sum(mask) == 0:
        return 0.0

    # 4. Local Entropy Calculation
    # We use a disk footprint to define the local neighborhood.
    # A radius of 3-5 is typical for texture analysis at this resolution (512x512).
    # High entropy = disordered fibers. Low entropy = bundles or smooth areas.
    footprint = disk(4)
    
    # Compute entropy map
    # This returns a float array where each pixel is the entropy of its neighborhood
    entropy_map = entropy(tubulin_uint8, footprint)

    # 5. Aggregation
    # Extract entropy values only from the masked region (the cells)
    masked_entropy = entropy_map[mask]
    
    if masked_entropy.size == 0:
        return 0.0
        
    # Calculate the mean entropy of the tubulin texture within the cells
    result = np.mean(masked_entropy)

    return float(result)
