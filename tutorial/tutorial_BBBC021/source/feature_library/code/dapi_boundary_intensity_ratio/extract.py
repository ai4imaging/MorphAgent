def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.morphology import binary_erosion, disk
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu

    # Convert to appropriate array type and handle dimensionality
    # Dataset is (512, 512, 3), Channel 2 is DAPI (Blue)
    arr = np.asarray(img, dtype=np.float32)
    
    # Check dimensionality and extract DAPI channel
    if arr.ndim == 3 and arr.shape[2] == 3:
        dapi_channel = arr[:, :, 2]
    elif arr.ndim == 2:
        # Fallback if only single channel passed (unlikely based on spec but safe)
        dapi_channel = arr
    else:
        return 0.0

    # Normalize DAPI channel for consistent thresholding/intensity calculations
    # Although ratio is scale-invariant, normalization helps with thresholding stability
    vmax = np.percentile(dapi_channel, 99.5) if dapi_channel.size > 0 else 1.0
    if vmax > 0:
        dapi_norm = dapi_channel / vmax
    else:
        dapi_norm = dapi_channel
    dapi_norm = np.clip(dapi_norm, 0.0, 1.0)

    # Determine Segmentation Mask
    # Priority: Provided mask -> Fallback Otsu
    nuclear_mask = None
    
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use the first provided mask. Assuming it's a label mask or binary mask.
        # If it's a label mask (int), convert to binary for morphology, then relabel if needed.
        mask_input = segmentation_masks[0]
        if mask_input.shape == dapi_channel.shape:
            nuclear_mask = mask_input > 0
    
    if nuclear_mask is None:
        # Fallback: Otsu thresholding on DAPI
        try:
            thresh = threshold_otsu(dapi_norm)
            nuclear_mask = dapi_norm > thresh
        except Exception:
            return 0.0

    # Label the nuclei to process individual cells
    # This is crucial because a global ratio would be dominated by the brightest or largest cells
    labeled_nuclei = label(nuclear_mask)
    
    # If no nuclei found, return 0
    if labeled_nuclei.max() == 0:
        return 0.0

    # Define morphological operation for boundary extraction
    # Erosion radius: 3 pixels is reasonable for 512x512 images of MCF-7 cells
    # to separate the rim from the core.
    erosion_radius = 3
    selem = disk(erosion_radius)

    # We can vectorize the erosion over the whole mask to define global regions,
    # but we need to calculate ratios per cell.
    
    # Strategy:
    # 1. Erode the full binary mask to get "centers".
    # 2. "Boundaries" are the original mask minus the "centers".
    # 3. Use the labeled image to aggregate intensities for "centers" and "boundaries" separately per label.
    
    binary_mask = labeled_nuclei > 0
    centers_mask = binary_erosion(binary_mask, footprint=selem)
    boundaries_mask = binary_mask & (~centers_mask)

    # Calculate mean intensity per label for centers
    # We use the labeled image masked by centers_mask
    # Labels present in centers_mask:
    labels_with_centers = np.unique(labeled_nuclei[centers_mask])
    labels_with_centers = labels_with_centers[labels_with_centers > 0] # remove background

    if len(labels_with_centers) == 0:
        return 0.0

    # Calculate sums and counts to derive means
    # ndimage.mean is convenient: mean(input, labels, index)
    
    # We need to be careful: a label might exist in the boundary but not the center (if cell is very small)
    # We only consider cells large enough to have a center.
    
    # Create specific label maps for center and boundary regions
    # This ensures we measure the intensity of label X's center and label X's boundary
    
    # Mask the intensity image
    intensity_centers = dapi_channel.copy()
    intensity_boundaries = dapi_channel.copy()
    
    # We need to compute mean intensity of label L in the center region
    # and mean intensity of label L in the boundary region.
    
    # Get mean intensities for centers
    # We pass the full label image, but the intensity image is effectively masked? 
    # No, ndimage.mean takes (values, labels, index). We need to mask the values or the labels.
    
    # Better approach:
    # Create a label image that only contains labels in the center region
    label_img_centers = np.where(centers_mask, labeled_nuclei, 0)
    # Create a label image that only contains labels in the boundary region
    label_img_boundaries = np.where(boundaries_mask, labeled_nuclei, 0)
    
    # Calculate means for the valid labels
    center_means = ndimage.mean(dapi_channel, labels=label_img_centers, index=labels_with_centers)
    boundary_means = ndimage.mean(dapi_channel, labels=label_img_boundaries, index=labels_with_centers)
    
    # Calculate ratios
    # Avoid division by zero
    valid_indices = center_means > 1e-6
    
    if np.sum(valid_indices) == 0:
        return 0.0
        
    ratios = boundary_means[valid_indices] / center_means[valid_indices]
    
    # Return the average ratio across the population
    # High ratio (>1) indicates chromatin margination (apoptosis)
    result = np.mean(ratios)

    return float(result)
