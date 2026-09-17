def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.segmentation import watershed, clear_border
    from skimage.morphology import binary_opening, disk

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Expected: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        # Fallback for potential unexpected shapes, though dataset spec is strict
        return 0.0

    # Extract channels
    # Channel 0: Actin (Cytoskeleton)
    # Channel 1: Tubulin
    # Channel 2: DAPI (Nucleus)
    actin_ch = arr[..., 0]
    dapi_ch = arr[..., 2]

    # Intensity normalization (per channel)
    def normalize(ch):
        vmax = np.percentile(ch, 99.5) if ch.size > 0 else 1.0
        if vmax > 0:
            ch = ch / vmax
        return np.clip(ch, 0.0, 1.0)

    actin_norm = normalize(actin_ch)
    dapi_norm = normalize(dapi_ch)

    # --- Step 1: Nucleus Segmentation (Seeds) ---
    # Gaussian blur to reduce noise
    dapi_smooth = ndimage.gaussian_filter(dapi_norm, sigma=2)
    
    # Thresholding
    try:
        thresh_nuc = threshold_otsu(dapi_smooth)
    except ValueError: # Handle empty images
        return 0.0
        
    nuc_mask = dapi_smooth > thresh_nuc
    
    # Morphological cleanup
    nuc_mask = binary_opening(nuc_mask, footprint=disk(2))
    
    # Label nuclei
    nuc_labels = label(nuc_mask)
    
    # Filter small nuclei (debris)
    sizes = ndimage.sum(nuc_mask, nuc_labels, range(nuc_labels.max() + 1))
    mask_size = sizes < 50 # Minimum area filter
    remove_pixel = mask_size[nuc_labels]
    nuc_labels[remove_pixel] = 0
    
    # Re-label to ensure contiguous indices
    nuc_labels, _ = ndimage.label(nuc_labels > 0)

    if nuc_labels.max() == 0:
        return 0.0

    # --- Step 2: Cell Body Segmentation (Watershed) ---
    # Define cell foreground: Union of Actin and Nucleus signals
    # We use a combined signal to ensure the cell body covers the nucleus
    combined_signal = np.maximum(actin_norm, dapi_norm)
    combined_smooth = ndimage.gaussian_filter(combined_signal, sigma=2)
    
    try:
        thresh_cell = threshold_otsu(combined_smooth)
    except ValueError:
        thresh_cell = 0.1
        
    # Create a mask for where cells generally are
    cell_mask = combined_smooth > thresh_cell
    
    # Watershed
    # Basin: Inverted intensity (grow from bright to dark)
    # Markers: Nuclei
    # Mask: Cell foreground
    cell_labels = watershed(-combined_smooth, nuc_labels, mask=cell_mask)

    # --- Step 3: Clear Border ---
    # Remove cells touching the border as their area ratio would be inaccurate
    # We clear based on the cell body labels
    cell_labels = clear_border(cell_labels)
    
    # Update nuc_labels to match cleared cell_labels
    # (Pixels where cell_labels is 0 should also be 0 in nuc_labels for consistency in this logic,
    # though strictly we just iterate over the remaining unique labels in cell_labels)
    valid_labels = np.unique(cell_labels)
    valid_labels = valid_labels[valid_labels > 0]

    if len(valid_labels) == 0:
        return 0.0

    # --- Step 4: Calculate Ratios ---
    ratios = []
    
    # We can use regionprops on the cell_labels to get total area
    # And we need to measure the nuclear area for the *same* label index
    
    # Pre-calculate areas for efficiency
    # bincount is faster than iterating regionprops for just area
    # cell_labels max value might be higher than len(valid_labels) if gaps exist, so we size accordingly
    max_label = cell_labels.max()
    
    cell_areas = np.bincount(cell_labels.ravel())
    
    # We need to mask nuc_labels by the valid cell_labels to ensure we measure the correct corresponding nucleus
    # (In watershed, the label ID is propagated, so nuc_labels inside cell_labels==k should be k)
    # However, nuc_labels might be slightly different if watershed expanded into background not covered by nuc_mask originally.
    # The strict definition: Nucleus Area is the count of pixels that were originally nuclei AND belong to the cell.
    
    # Efficient way: count pixels where (nuc_labels == k)
    # But since we cleared borders on cell_labels, we must ensure we only count nuclei for valid cells.
    # A simple way: mask nuc_labels with (cell_labels > 0) and count.
    # Actually, since watershed preserves seed labels, we can just count nuc_labels where cell_labels is valid.
    
    # Let's use a masked array approach or just iterate valid labels for clarity and safety
    
    for label_id in valid_labels:
        # Area of the whole cell (Nucleus + Cytoplasm)
        area_total = cell_areas[label_id]
        
        # Area of the nucleus for this specific cell
        # Since watershed grew FROM nuc_labels, the region in cell_labels labeled 'label_id'
        # contains the original nucleus seed 'label_id'.
        # We intersect the original nuc_mask with the specific cell region to get the nuclear area.
        # Optimization: The nuc_labels == label_id is the nuclear area.
        # We just need to check if it was cleared.
        
        # Count pixels where nuc_labels == label_id
        # Since we cleared borders on cell_labels, we should check if the nucleus is still valid.
        # If cell_labels[y,x] == 0 (cleared), we ignore it.
        # But we are iterating `valid_labels` which are present in `cell_labels`.
        
        # So, area_nucleus is simply the count of pixels in the original nuc_labels equal to label_id
        # (Assuming the nucleus is fully contained in the cell, which watershed guarantees).
        area_nucleus = np.sum(nuc_labels == label_id)
        
        if area_nucleus > 0:
            # Cytoplasm area = Total Cell Area - Nucleus Area
            area_cyto = area_total - area_nucleus
            
            # Clamp to 0 just in case
            if area_cyto < 0: 
                area_cyto = 0.0
                
            ratio = area_cyto / area_nucleus
            ratios.append(ratio)

    if not ratios:
        return 0.0

    result = np.mean(ratios)
    return float(result)
