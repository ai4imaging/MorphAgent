def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.morphology import disk, white_tophat, binary_dilation
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu

    # 1. Data Loading and Preprocessing
    # Convert to float32 for processing
    arr = np.asarray(img, dtype=np.float32)
    
    # Check dimensionality and extract DAPI channel (Channel 2)
    # Expected shape: (512, 512, 3)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        dapi = arr[:, :, 2]
    elif arr.ndim == 2:
        # Fallback if single channel passed (unlikely based on spec but safe)
        dapi = arr
    else:
        return 0.0

    # Normalize DAPI to [0, 1]
    # Use robust max to avoid hot pixel scaling issues
    p99 = np.percentile(dapi, 99.9)
    if p99 > 0:
        dapi_norm = dapi / p99
    else:
        dapi_norm = dapi  # Empty image
    dapi_norm = np.clip(dapi_norm, 0.0, 1.0)

    # 2. Define "Main Nuclei" (Parent Objects)
    # We need to identify the main nuclei to normalize the count and define search regions.
    
    main_nuclei_mask = None
    
    # Try to use provided segmentation masks first
    # We look for a mask that likely represents nuclei
    if len(segmentation_masks) > 0:
        # Heuristic: Iterate through masks. If we find one, use it.
        # Often the second mask is nuclei in some datasets, but we can't be sure of order.
        # We'll assume the provided masks are valid cell/nuclei labels.
        # If multiple are provided, we might need to guess which is nuclei. 
        # Without specific metadata, we'll use the first available mask as a proxy for cellular objects.
        # Ideally, we check if the mask overlaps significantly with DAPI signal.
        
        # Let's try to find a mask that correlates with DAPI intensity
        best_overlap = -1
        for mask in segmentation_masks:
            if mask is None: continue
            # Ensure mask is 2D
            if mask.ndim == 3:
                mask = np.max(mask, axis=0) # Project if 3D
            if mask.shape != dapi.shape:
                continue
                
            binary_mask = mask > 0
            # Calculate overlap with high DAPI intensity
            overlap = np.sum(binary_mask & (dapi_norm > 0.2))
            if overlap > best_overlap:
                best_overlap = overlap
                main_nuclei_mask = binary_mask

    # Fallback: Generate nuclei mask if none provided or valid
    if main_nuclei_mask is None:
        # Gaussian blur to smooth noise
        dapi_smooth = ndimage.gaussian_filter(dapi_norm, sigma=2.0)
        try:
            thresh = threshold_otsu(dapi_smooth)
        except ValueError: # Handle empty images
            thresh = 0.5
            
        binary_thresh = dapi_smooth > thresh
        
        # Remove small objects (noise) and fill holes
        labeled_nuclei = label(binary_thresh)
        main_nuclei_mask = np.zeros_like(binary_thresh, dtype=bool)
        
        for prop in regionprops(labeled_nuclei):
            # Filter: Main nuclei are typically large (e.g., > 100 pixels)
            if prop.area > 100:
                # Add to mask
                coords = prop.coords
                main_nuclei_mask[coords[:, 0], coords[:, 1]] = True
                
        # Fill holes in the nuclei mask
        main_nuclei_mask = ndimage.binary_fill_holes(main_nuclei_mask)

    # Count main nuclei for normalization
    labeled_main, num_main_nuclei = label(main_nuclei_mask, return_num=True)
    
    if num_main_nuclei == 0:
        return 0.0

    # 3. Detect Micronuclei Candidates
    # Micronuclei are small, bright spots.
    # Strategy: White Top-Hat transform to enhance small bright spots on dark background,
    # effectively removing the large, slowly varying background (and potentially the main nuclei glow).
    
    # Radius for top-hat: slightly larger than expected micronucleus (radius ~3-5 px)
    # A disk of radius 8 covers objects up to ~16px wide.
    wth = white_tophat(dapi_norm, footprint=disk(8))
    
    # Threshold the top-hat image
    # Since background is removed, we can use a statistical threshold
    # Micronuclei are bright outliers in the top-hat image
    wth_mean = np.mean(wth)
    wth_std = np.std(wth)
    # Threshold: mean + k * std. k=3 is a standard outlier detection.
    # Also enforce a minimum absolute intensity to avoid noise in empty backgrounds.
    mn_thresh = max(wth_mean + 3 * wth_std, 0.05) 
    
    mn_candidates_mask = wth > mn_thresh
    
    # 4. Filter and Associate Candidates
    labeled_candidates = label(mn_candidates_mask)
    
    micronuclei_count = 0
    
    # Define search zone: Perinuclear region
    # Dilate main nuclei mask to capture cytoplasm area where micronuclei reside
    # Radius ~15-20 pixels
    search_zone = binary_dilation(main_nuclei_mask, footprint=disk(15))
    
    # Exclusion zone: The main nucleus itself
    # Micronuclei must be DETACHED from the main nucleus
    # We subtract the main nuclei mask from the search zone
    valid_search_area = search_zone & (~main_nuclei_mask)
    
    for prop in regionprops(labeled_candidates):
        # Filter 1: Area
        # Micronuclei are small but not single-pixel noise.
        # Range: 3 to 60 pixels (approximate for 512x512 image of cells)
        if prop.area < 3 or prop.area > 60:
            continue
            
        # Filter 2: Shape (Optional but good)
        # Micronuclei are usually roughly circular.
        if prop.eccentricity > 0.95: # Skip very elongated streaks
            continue
            
        # Filter 3: Location
        # Check if the candidate centroid falls within the valid search area
        y, x = map(int, prop.centroid)
        
        # Boundary check
        if y < 0 or y >= valid_search_area.shape[0] or x < 0 or x >= valid_search_area.shape[1]:
            continue
            
        if valid_search_area[y, x]:
            micronuclei_count += 1

    # 5. Compute Feature
    # Average number of micronuclei per cell
    result = micronuclei_count / float(num_main_nuclei)
    
    return float(result)
