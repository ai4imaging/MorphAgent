def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu, threshold_li
    from skimage.morphology import binary_opening, disk, binary_closing
    from skimage.segmentation import watershed

    # Convert to appropriate array type and normalize
    # Image shape is (512, 512, 3), uint8
    arr = np.asarray(img, dtype=np.float32)
    
    # Handle potential dimensionality issues
    if arr.ndim == 2:
        # If grayscale, we can't distinguish nucleus/cyto easily without masks
        # But dataset says (512, 512, 3), so we proceed assuming channels exist
        return 0.0
    elif arr.ndim == 3:
        if arr.shape[2] == 3:
            # Standard HWC format
            pass
        elif arr.shape[0] == 3:
            # CHW format, transpose to HWC
            arr = np.transpose(arr, (1, 2, 0))
        else:
            return 0.0
    else:
        return 0.0

    # Normalize to [0, 1]
    # Robust max to avoid outliers affecting normalization too much
    vmax = np.percentile(arr, 99.5) if arr.size > 0 else 255.0
    if vmax <= 0: vmax = 1.0
    arr = np.clip(arr / vmax, 0.0, 1.0)

    # Channel mapping based on dataset description
    # Ch 0: Actin (Red) - Cytoskeleton
    # Ch 1: Tubulin (Green) - Cytoskeleton
    # Ch 2: DAPI (Blue) - Nucleus
    ch_actin = arr[..., 0]
    ch_tubulin = arr[..., 1]
    ch_dapi = arr[..., 2]

    # --- Segmentation Logic ---
    
    # 1. Nucleus Segmentation (Seeds)
    # Smooth DAPI channel
    dapi_smooth = ndimage.gaussian_filter(ch_dapi, sigma=2)
    
    # Thresholding
    try:
        thresh_nuc = threshold_otsu(dapi_smooth)
    except Exception:
        thresh_nuc = 0.1
        
    mask_nuc = dapi_smooth > thresh_nuc
    
    # Clean up nucleus mask
    mask_nuc = binary_opening(mask_nuc, footprint=disk(2))
    # Fill holes to ensure solid objects
    mask_nuc = ndimage.binary_fill_holes(mask_nuc)

    # Label nuclei to use as markers
    markers_nuc, num_nuc = label(mask_nuc, return_num=True)

    # If no nuclei found, return 0.0
    if num_nuc == 0:
        return 0.0

    # 2. Cell Body Segmentation (Mask)
    # Combine Actin and Tubulin for a robust cell body signal
    # Use maximum projection of the two cytoskeletal channels
    cyto_signal = np.maximum(ch_actin, ch_tubulin)
    # Also include DAPI because the nucleus is part of the cell
    total_signal = np.maximum(cyto_signal, ch_dapi)
    
    total_smooth = ndimage.gaussian_filter(total_signal, sigma=2)
    
    try:
        # Li threshold is often better for the faint cytoplasmic tails
        thresh_cell = threshold_li(total_smooth)
    except Exception:
        thresh_cell = 0.05
        
    mask_cell = total_smooth > thresh_cell
    
    # Ensure the cell mask covers the nucleus mask (biological constraint)
    mask_cell = np.logical_or(mask_cell, mask_nuc)
    
    # Clean up cell mask
    mask_cell = binary_closing(mask_cell, footprint=disk(3))
    mask_cell = ndimage.binary_fill_holes(mask_cell)

    # 3. Instance Segmentation via Watershed
    # We propagate the nucleus labels into the cell body mask
    # We use the inverse intensity as the "elevation" map for watershed
    elevation_map = -total_smooth
    
    # Run watershed
    # mask=mask_cell ensures we only segment within the foreground
    segmentation = watershed(elevation_map, markers_nuc, mask=mask_cell)

    # --- Feature Calculation ---
    
    # We need to calculate Area(Nucleus) / Area(Cell) for each cell
    # The 'segmentation' array contains the cell labels (1 to N)
    # The 'markers_nuc' array contains the nucleus labels (1 to N)
    # Note: Watershed preserves the seed labels. So label 'k' in 'segmentation'
    # corresponds to the cell growing from nucleus label 'k'.
    
    props_cell = regionprops(segmentation)
    
    ratios = []
    
    for prop in props_cell:
        label_id = prop.label
        
        # Total cell area (from watershed result)
        area_cell = prop.area
        
        # Nucleus area
        # We look at the original nucleus mask where the label matches
        # Because markers_nuc might have been slightly different before watershed,
        # strictly speaking, the nucleus area is the intersection of the nucleus mask
        # and the specific cell region.
        # However, since markers_nuc were seeds, we can just count pixels in markers_nuc
        # that have this label_id.
        area_nucleus = np.sum(markers_nuc == label_id)
        
        # Filter small artifacts
        if area_cell < 50 or area_nucleus < 10:
            continue
            
        # Calculate ratio
        # Biologically, nucleus is inside cell, so area_nucleus <= area_cell
        # Clamp to 1.0 just in case of edge pixel weirdness
        if area_cell > 0:
            ratio = area_nucleus / float(area_cell)
            ratio = min(ratio, 1.0)
            ratios.append(ratio)

    if not ratios:
        return 0.0

    # Return the mean ratio
    result = np.mean(ratios)
    
    return float(result)
