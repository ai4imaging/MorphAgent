def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import label as skimage_label
    
    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Dataset Dimensions: (512, 512, 3)
    # Channel 2 is DAPI (Nucleus)
    
    # Check input dimensions
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract DAPI channel (Channel 2)
    dapi_channel = arr[:, :, 2]

    # Normalize DAPI channel
    # Data is uint8 (0-255), but we converted to float32
    # Robust normalization
    p99 = np.percentile(dapi_channel, 99.9)
    if p99 > 0:
        dapi_norm = dapi_channel / p99
    else:
        dapi_norm = dapi_channel # Should be all zeros if p99 is 0
    dapi_norm = np.clip(dapi_norm, 0.0, 1.0)

    # Determine which mask to use
    # The prompt mentions masks might be passed as *segmentation_masks
    # We need to identify the nuclei mask.
    # Since we don't have the filenames associated with the *args tuple inside the function,
    # we have to rely on heuristics or the order.
    # The error guidance mentions: "multiple segmentation masks are available (cyto, cytoplasm, nuclei). 
    # The code should prioritize using the mask named 'nuclei' if available... rather than just taking the first one."
    # However, inside the function `extract(img, *segmentation_masks)`, we only get the arrays, not the names.
    # We will assume a standard order or try to infer based on overlap with the DAPI channel if multiple are present.
    # If no masks are provided, we must segment it ourselves.

    labeled_mask = None

    if len(segmentation_masks) > 0:
        # Heuristic: The nuclei mask should have the highest intensity correlation with the DAPI channel
        # or we simply check if we can distinguish them. 
        # Often, nuclei masks are smaller and more numerous than cytoplasm masks (which might be one large blob or cover the whole cell).
        # Without metadata, a safe bet is to check overlap with high DAPI intensity.
        
        best_mask = None
        best_overlap_score = -1.0

        # Create a binary threshold of DAPI to check overlap
        try:
            thresh = threshold_otsu(dapi_norm)
        except ValueError: # If image is uniform
            thresh = 0.5
        dapi_binary = dapi_norm > thresh

        for mask in segmentation_masks:
            # Ensure mask is 2D and matches image shape
            if mask.ndim == 2 and mask.shape == dapi_channel.shape:
                # Check overlap with DAPI signal
                # Calculate Intersection over Union (IoU) or just Intersection with the bright DAPI spots
                mask_binary = mask > 0
                intersection = np.logical_and(mask_binary, dapi_binary).sum()
                union = np.logical_or(mask_binary, dapi_binary).sum()
                
                if union > 0:
                    score = intersection / union
                else:
                    score = 0.0
                
                if score > best_overlap_score:
                    best_overlap_score = score
                    best_mask = mask
        
        if best_mask is not None:
            labeled_mask = best_mask.astype(int)

    # Fallback: If no masks provided or selection failed, perform segmentation on DAPI channel
    if labeled_mask is None:
        # Gaussian blur to reduce noise
        smooth_dapi = ndimage.gaussian_filter(dapi_norm, sigma=2)
        
        # Thresholding
        try:
            thresh = threshold_otsu(smooth_dapi)
            binary_mask = smooth_dapi > thresh
        except Exception:
            return 0.0

        # Fill holes
        # Note: binary_fill_holes is in scipy.ndimage
        binary_mask = ndimage.binary_fill_holes(binary_mask)

        # Label connected components
        labeled_mask = skimage_label(binary_mask)

    # Calculate properties
    regions = regionprops(labeled_mask)

    if not regions:
        return 0.0

    # Extract areas
    areas = [r.area for r in regions]

    # Calculate mean area
    if len(areas) > 0:
        mean_area = np.mean(areas)
    else:
        mean_area = 0.0

    return float(mean_area)
