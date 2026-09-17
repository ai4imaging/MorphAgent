def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_opening, disk

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Expected: (512, 512, 3)
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Channel 2 is DAPI (Blue) based on dataset description
        dapi_channel = arr[..., 2]
    elif arr.ndim == 2:
        # Fallback for single channel input
        dapi_channel = arr
    else:
        return 0.0

    # Intensity normalization
    # Robust max to avoid hot pixels affecting scaling too much
    vmax = np.percentile(dapi_channel, 99.5) if dapi_channel.size > 0 else 1.0
    if vmax > 0:
        dapi_channel = dapi_channel / vmax
    dapi_channel = np.clip(dapi_channel, 0.0, 1.0)

    # Segmentation Logic
    # 1. Thresholding
    try:
        thresh = threshold_otsu(dapi_channel)
    except Exception:
        # Fallback if image is uniform
        thresh = 0.1
    
    # Create binary mask
    binary_mask = dapi_channel > thresh

    # 2. Morphological cleanup
    # Remove very small noise (speckles) but keep micronuclei candidates
    # A micronucleus is small but usually larger than single-pixel noise.
    # Opening with a small disk (radius 1 or 2) helps separate touching objects slightly
    binary_mask = binary_opening(binary_mask, disk(1))

    # 3. Label connected components
    labeled_image = label(binary_mask)
    regions = regionprops(labeled_image)

    if not regions:
        return 0.0

    # 4. Filter and Classify Objects
    # We need to distinguish "Normal Nuclei" from "Micronuclei".
    # Criteria based on literature and previous error guidance:
    # - Normal Nuclei: Large area.
    # - Micronuclei: Small area (fraction of normal), detached.
    
    areas = [r.area for r in regions]
    
    # Determine reference size for a "normal" nucleus
    # Using percentile (e.g., 90th) is safer than median if there's a lot of debris
    # But if the image is mostly debris, we need a hard floor.
    # Dataset is 512x512. A typical MCF-7 nucleus might be ~20-40 pixels diameter -> ~300-1200 area.
    # Let's set a hard minimum for a "normal" nucleus to avoid classifying debris as normal.
    
    # Heuristic: Normal nuclei are usually the largest objects in the scene.
    # Let's take the median of the top 50% of objects by area as a rough estimator of "normal size",
    # provided they are above a hard threshold.
    
    sorted_areas = sorted(areas)
    if not sorted_areas:
        return 0.0
        
    # Hard threshold for a normal nucleus (e.g., > 150 pixels)
    min_normal_area = 150
    
    normal_nuclei_candidates = [r for r in regions if r.area >= min_normal_area]
    
    if not normal_nuclei_candidates:
        # If no large nuclei found, we can't calculate a ratio relative to normal nuclei.
        # However, if there are *only* small objects, maybe the ratio is infinite? 
        # Usually, we return 0.0 if the baseline is missing to avoid exploding gradients/values.
        return 0.0

    # Calculate statistics of normal nuclei to define micronuclei bounds
    normal_areas = [r.area for r in normal_nuclei_candidates]
    median_normal_area = np.median(normal_areas)
    
    # Micronucleus definition:
    # - Area: Typically 1/3 to 1/100 of the main nucleus.
    # - Shape: Round (high solidity), though we relax this per guidance.
    # - Intensity: Similar to main nucleus (checked via thresholding).
    
    # Upper bound for micronucleus: 1/3 of the median normal size
    # Lower bound: Enough to be a real object (e.g., 15 pixels)
    mn_max_area = median_normal_area * 0.33
    mn_min_area = 15 
    
    micronuclei_count = 0
    normal_nuclei_count = len(normal_nuclei_candidates)

    for r in regions:
        # Check if it falls in the micronucleus size range
        if mn_min_area <= r.area <= mn_max_area:
            # Shape check (relaxed)
            # Micronuclei are generally round-ish. Debris might be irregular.
            # Solidity > 0.8 is a gentle filter.
            if r.solidity > 0.8:
                micronuclei_count += 1
    
    # Calculate Ratio
    # Ratio = (Number of Micronuclei) / (Number of Normal Nuclei)
    if normal_nuclei_count > 0:
        ratio = micronuclei_count / normal_nuclei_count
    else:
        ratio = 0.0

    return float(ratio)
