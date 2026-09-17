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
    # Dataset is (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract DAPI channel (Channel 2)
    dapi_channel = arr[..., 2]

    # Intensity normalization
    # Robust max to handle outliers/hot pixels
    vmax = np.percentile(dapi_channel, 99.5) if dapi_channel.size > 0 else 1.0
    if vmax > 0:
        dapi_norm = dapi_channel / vmax
    else:
        dapi_norm = dapi_channel
    dapi_norm = np.clip(dapi_norm, 0.0, 1.0)

    # Preprocessing: Mild smoothing to reduce noise
    dapi_smooth = ndimage.gaussian_filter(dapi_norm, sigma=1.0)

    # Determine Threshold
    try:
        thresh = threshold_otsu(dapi_smooth)
    except Exception:
        # Fallback if image is empty or uniform
        thresh = 0.1

    # Create binary mask of all DNA material
    dna_mask = dapi_smooth > thresh

    # Clean up mask (remove tiny noise specks)
    dna_mask = binary_opening(dna_mask, footprint=disk(1))

    # Label all objects
    labeled_dna, num_features = label(dna_mask, return_num=True)
    
    if num_features == 0:
        return 0.0

    props = regionprops(labeled_dna, intensity_image=dapi_norm)

    # Analyze size distribution to distinguish Main Nuclei from Micronuclei
    areas = np.array([p.area for p in props])
    
    if len(areas) == 0:
        return 0.0

    # Heuristic: Main nuclei are the dominant large objects.
    # We use the median of the larger half of objects to estimate "typical nucleus size"
    # to be robust against debris.
    median_area = np.median(areas)
    
    # If the image is very sparse, median might be small (debris). 
    # Let's assume a minimum valid nucleus size for MCF-7 at 512x512.
    # A typical MCF-7 nucleus is roughly 20-40 pixels diameter -> ~300-1200 pixels area.
    # Let's set a hard lower bound for a "Main Nucleus" to avoid classifying debris as main nuclei.
    # And a range for micronuclei relative to the detected main nuclei.
    
    # Filter for Main Nuclei candidates to establish a baseline size
    # Assume main nuclei are at least 150 pixels in area
    main_nuclei_candidates = [p.area for p in props if p.area > 150]
    
    if not main_nuclei_candidates:
        # If no substantial objects found, return 0
        return 0.0
        
    avg_nucleus_area = np.median(main_nuclei_candidates)

    # Definition of Micronucleus based on literature:
    # 1. Size: Typically 1/100 to 1/3 of the main nucleus area.
    # 2. Shape: Round/Oval (eccentricity check).
    # 3. Intensity: Similar to main nucleus (not faint background noise).
    # 4. Separation: Distinct object (already handled by labeling).

    min_mn_area = avg_nucleus_area * 0.02  # Lower bound (e.g., ~1/50th)
    max_mn_area = avg_nucleus_area * 0.30  # Upper bound (e.g., ~1/3rd)
    
    # Absolute minimum pixel count to avoid single-pixel noise
    min_mn_area = max(min_mn_area, 5.0) 

    micronuclei_count = 0

    # Identify Main Nuclei Mask for proximity check
    # We create a mask of "Main Nuclei" to ensure micronuclei are somewhat near cells
    # but not part of them.
    main_nuclei_mask = np.zeros_like(dna_mask, dtype=bool)
    for p in props:
        if p.area > max_mn_area:
            # Fill the bounding box in the mask (approximation for distance calc speed)
            # Better: use the coords or slice
            r_min, c_min, r_max, c_max = p.bbox
            main_nuclei_mask[r_min:r_max, c_min:c_max] |= p.image

    # Distance transform from main nuclei
    # distance_to_nucleus[i, j] is distance to nearest True pixel in main_nuclei_mask
    # We invert the mask because distance_transform_edt calculates distance to zero (background)
    if np.any(main_nuclei_mask):
        dist_map = ndimage.distance_transform_edt(~main_nuclei_mask)
    else:
        # If no main nuclei, we can't validate proximity, but maybe we just count small objects?
        # Stricter approach: if no main nuclei, context is lost.
        dist_map = np.zeros_like(dna_mask, dtype=np.float32) + 9999

    for p in props:
        # Criteria 1: Size
        if not (min_mn_area <= p.area <= max_mn_area):
            continue

        # Criteria 2: Shape (Eccentricity)
        # Micronuclei are usually round. Eccentricity 0 is circle, 1 is line.
        # Allow some deformation but filter out linear streaks/artifacts.
        if p.eccentricity > 0.85:
            continue

        # Criteria 3: Intensity
        # Should be relatively bright (DNA content), not faint smear.
        # Compare max intensity of object to global max (1.0) or local stats.
        if p.max_intensity < 0.2: # Threshold relative to normalized image
            continue

        # Criteria 4: Proximity
        # Micronuclei should be associated with a cell, typically within a short distance 
        # of a main nucleus (or cytoplasm).
        # We check the distance of the centroid to the nearest main nucleus.
        cy, cx = int(p.centroid[0]), int(p.centroid[1])
        
        # Ensure coordinates are within bounds
        cy = min(max(cy, 0), dist_map.shape[0]-1)
        cx = min(max(cx, 0), dist_map.shape[1]-1)
        
        dist_to_main = dist_map[cy, cx]
        
        # Allow a reasonable distance (e.g., 50-80 pixels) representing cytoplasmic radius
        if dist_to_main > 100: 
            continue
            
        # If passed all checks
        micronuclei_count += 1

    return float(micronuclei_count)
