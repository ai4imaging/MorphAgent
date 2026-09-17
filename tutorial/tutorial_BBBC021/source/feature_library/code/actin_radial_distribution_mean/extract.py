def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    from skimage.measure import label, regionprops
    from skimage.segmentation import watershed
    from skimage.morphology import binary_opening, disk

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Expected: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract channels
    # Channel 0: Actin (Red) - Target Intensity
    # Channel 1: Tubulin (Green) - Helper for cell body
    # Channel 2: DAPI (Blue) - Nucleus seed
    actin = arr[:, :, 0]
    tubulin = arr[:, :, 1]
    dapi = arr[:, :, 2]

    # Normalize intensities to 0-1 range for processing
    def normalize_ch(ch):
        vmax = np.percentile(ch, 99.5) if ch.size > 0 else 1.0
        if vmax <= 0: return ch
        return np.clip(ch / vmax, 0.0, 1.0)

    actin_norm = normalize_ch(actin)
    dapi_norm = normalize_ch(dapi)
    # Combine actin and tubulin for a robust cell body signal
    cell_signal = np.maximum(actin_norm, normalize_ch(tubulin))

    # --- Segmentation Logic ---
    # Since masks are optional and currently None, we perform internal segmentation
    # to define "Nucleus" (center) and "Cell Body" (boundary).
    
    # 1. Nuclei Segmentation
    try:
        thresh_nuc = threshold_otsu(dapi_norm)
        mask_nuc = dapi_norm > thresh_nuc
        mask_nuc = binary_opening(mask_nuc, footprint=disk(2))
        markers_nuc = label(mask_nuc)
    except Exception:
        return 0.0 # Fallback if thresholding fails (e.g. empty image)

    if markers_nuc.max() == 0:
        return 0.0

    # 2. Cell Body Segmentation (Watershed)
    try:
        thresh_cell = threshold_otsu(cell_signal)
        mask_cell = cell_signal > thresh_cell
        # Ensure nuclei are inside cells
        mask_cell = np.logical_or(mask_cell, mask_nuc)
        
        # Watershed to separate touching cells based on nuclei markers
        # We use the inverse intensity as the "elevation map"
        elevation_map = -cell_signal
        segmentation = watershed(elevation_map, markers_nuc, mask=mask_cell)
    except Exception:
        return 0.0

    # --- Feature Computation: Radial Distribution ---
    # We want the ratio of Actin intensity at the periphery vs perinuclear region.
    
    ratios = []
    
    props = regionprops(segmentation, intensity_image=actin)
    
    # Pre-calculate distance transform from nuclei
    # We calculate distance FROM the nuclei mask (0 inside nucleus, increasing outside)
    # Invert mask_nuc so nuclei are 0, background is 1
    # distance_transform_edt calculates distance to nearest zero pixel
    # So we want distance to the nearest nucleus pixel.
    # We create a mask where nuclei are 0 and everything else is 1.
    # However, standard EDT computes distance to background (0).
    # So input to EDT: 0 at nuclei, 1 elsewhere.
    dist_map_input = np.ones_like(segmentation, dtype=np.float32)
    dist_map_input[mask_nuc] = 0
    
    # Calculate Euclidean distance from the nearest nucleus pixel
    dist_from_nuc = ndimage.distance_transform_edt(dist_map_input)

    for prop in props:
        # Skip small artifacts
        if prop.area < 100:
            continue
            
        # Get the bounding box slice for efficiency
        minr, minc, maxr, maxc = prop.bbox
        
        # Extract local masks and maps
        # mask_local is the binary mask of the current cell within the bbox
        mask_local = prop.image 
        
        # Local distance map
        dist_local = dist_from_nuc[minr:maxr, minc:maxc]
        
        # Local intensity map
        intensity_local = prop.intensity_image
        
        # We only care about pixels belonging to THIS cell
        valid_pixels = mask_local
        
        if not np.any(valid_pixels):
            continue
            
        # Get distances and intensities for valid pixels
        dists = dist_local[valid_pixels]
        ints = intensity_local[valid_pixels]
        
        # Normalize distances for this specific cell
        # 0.0 = at nucleus boundary, 1.0 = at cell periphery (furthest point)
        max_d = np.max(dists)
        
        if max_d == 0:
            # Cell is just the nucleus, no cytoplasm to measure radial distribution
            continue
            
        norm_dists = dists / max_d
        
        # Define Zones
        # Perinuclear: 0.0 to 0.5
        # Peripheral: 0.5 to 1.0
        perinuclear_mask = norm_dists < 0.5
        peripheral_mask = norm_dists >= 0.5
        
        if np.sum(perinuclear_mask) == 0 or np.sum(peripheral_mask) == 0:
            continue
            
        mean_inner = np.mean(ints[perinuclear_mask])
        mean_outer = np.mean(ints[peripheral_mask])
        
        # Compute Ratio: Peripheral / Perinuclear
        # High ratio (>1) means cortical actin (ring)
        # Low ratio (<1) means perinuclear accumulation
        if mean_inner > 0:
            ratio = mean_outer / mean_inner
            ratios.append(ratio)
        else:
            # If inner is 0 but outer is > 0, ratio is infinite (very high cortical)
            if mean_outer > 0:
                ratios.append(5.0) # Cap at a reasonable high value
            else:
                ratios.append(1.0) # Flat

    if not ratios:
        return 0.0

    # Return the mean ratio across all valid cells
    result = np.mean(ratios)
    
    return float(result)
