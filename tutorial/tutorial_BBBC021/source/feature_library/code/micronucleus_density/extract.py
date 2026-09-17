def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import remove_small_objects, binary_opening, disk
    from scipy import ndimage

    # 1. Handle Input and Dimensionality
    # Dataset is (512, 512, 3), uint8.
    # Channel 2 (Blue) is DAPI (Nucleus).
    
    # Convert to float32 for processing
    img_arr = np.asarray(img, dtype=np.float32)

    # Check dimensionality
    if img_arr.ndim == 3 and img_arr.shape[-1] == 3:
        # Standard RGB/Composite: Channel 2 is DAPI
        dapi_channel = img_arr[..., 2]
    elif img_arr.ndim == 2:
        # Grayscale fallback
        dapi_channel = img_arr
    else:
        return 0.0

    # 2. Normalization
    # Robust normalization to handle intensity variations
    p_min, p_max = np.percentile(dapi_channel, (1, 99))
    if p_max > p_min:
        dapi_norm = (dapi_channel - p_min) / (p_max - p_min)
    else:
        dapi_norm = dapi_channel
    dapi_norm = np.clip(dapi_norm, 0, 1)

    # 3. Segmentation Logic
    # We need to identify main nuclei and micronuclei.
    # Micronuclei are small, detached DAPI-positive objects.
    
    # Thresholding
    try:
        thresh = threshold_otsu(dapi_norm)
        # Lower threshold slightly to capture faint micronuclei, but not too much to grab noise
        binary_mask = dapi_norm > (thresh * 0.8) 
    except Exception:
        # Fallback if image is empty or uniform
        return 0.0

    # Clean up noise
    # Remove very small specks (noise)
    binary_mask = remove_small_objects(binary_mask, min_size=5)
    
    # Label objects
    labeled_mask = label(binary_mask)
    regions = regionprops(labeled_mask)

    if not regions:
        return 0.0

    # 4. Feature Extraction: Micronucleus Density
    # Strategy:
    # - Identify "Main Nuclei" based on size (large objects).
    # - Identify "Micronuclei" based on size (small objects) and potentially shape/intensity.
    # - Micronuclei should be distinct from the main nucleus but in the vicinity (though here we count global density).
    
    areas = np.array([r.area for r in regions])
    
    # Dynamic size thresholds based on the distribution of object sizes
    # We assume the largest objects are nuclei.
    # If there are many cells, the median of the larger half might be a good estimator for nucleus size.
    
    # Filter out potential noise (very small)
    valid_areas = areas[areas > 5]
    
    if len(valid_areas) == 0:
        return 0.0

    # Heuristic: Main nuclei are usually significantly larger than micronuclei.
    # Micronuclei are typically 1/100th to 1/10th the area of a main nucleus.
    # Let's try to find a cluster of large objects.
    
    # Sort areas to find the "main" population
    sorted_areas = np.sort(valid_areas)
    
    # Assume the top 10% largest objects are definitely nuclei to estimate "typical nucleus size"
    # If few objects, take the max.
    num_objects = len(sorted_areas)
    if num_objects > 0:
        # Take the median of the top 50% of areas as the reference nucleus size
        # This is robust against debris.
        cutoff_index = int(num_objects * 0.5)
        reference_nucleus_area = np.median(sorted_areas[cutoff_index:])
    else:
        reference_nucleus_area = 100.0 # Fallback, though unlikely to reach here

    # Define thresholds relative to reference size
    # Main Nucleus: > 30% of reference size (allows for some variation/segmentation splits)
    # Micronucleus: < 20% of reference size AND > 5 pixels (to avoid single pixel noise)
    
    # Absolute fallbacks are important if the image contains ONLY micronuclei or ONLY debris
    # A typical MCF-7 nucleus at 20x/40x is roughly 200-1000 pixels depending on binning.
    # Let's enforce some absolute bounds to be safe.
    # If reference area is tiny (e.g. < 50), the image might be empty or just noise.
    
    if reference_nucleus_area < 50:
        # If the "large" objects are small, maybe we are zoomed out or they are just debris.
        # Use absolute thresholds.
        min_nucleus_area = 100
        max_micronucleus_area = 80
    else:
        min_nucleus_area = reference_nucleus_area * 0.3
        max_micronucleus_area = reference_nucleus_area * 0.2

    main_nuclei_count = 0
    micronuclei_count = 0
    
    for r in regions:
        area = r.area
        if area >= min_nucleus_area:
            main_nuclei_count += 1
        elif 5 <= area <= max_micronucleus_area:
            # Additional check: Circularity/Solidity?
            # Micronuclei are usually round-ish. Debris might be irregular.
            # But simple size is often the primary definition in high-throughput screens.
            micronuclei_count += 1

    # Calculate Density
    # Definition: Micronuclei per cell (or per main nucleus)
    # If no main nuclei, density is technically undefined or infinite, but for feature stability we return 0 or count/area.
    
    if main_nuclei_count > 0:
        density = micronuclei_count / main_nuclei_count
    else:
        # If no main nuclei found, but micronuclei candidates exist, 
        # it might be a field of debris or very sparse. 
        # Return 0 to avoid skewing statistics with infinity.
        density = 0.0

    return float(density)
