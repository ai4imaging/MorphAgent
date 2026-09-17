def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_dilation, disk, binary_opening, binary_closing
    
    # --- 1. Data Loading and Preprocessing ---
    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality
    # Expected shape: (512, 512, 3)
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Extract DAPI channel (Channel 2) for nuclear analysis
        dapi_channel = arr[:, :, 2]
    elif arr.ndim == 2:
        # Fallback if single channel passed (unlikely based on spec but safe)
        dapi_channel = arr
    else:
        return 0.0

    # Normalize DAPI channel to [0, 1]
    # Use robust max to handle potential outliers/hot pixels
    vmax = np.percentile(dapi_channel, 99.9) if dapi_channel.size > 0 else 1.0
    if vmax > 0:
        dapi_norm = dapi_channel / vmax
    else:
        dapi_norm = dapi_channel
    dapi_norm = np.clip(dapi_norm, 0.0, 1.0)

    # --- 2. Define Main Nuclei (The Denominator) ---
    # We need to identify the main nuclei to count cells and define exclusion zones.
    
    mask_main_nuclei = None
    
    # Check if segmentation masks are provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Assuming the first mask is the nuclear mask (standard convention)
        # Ensure it's boolean
        mask_main_nuclei = segmentation_masks[0] > 0
    else:
        # Fallback: Generate nuclear mask on the fly
        # Apply Gaussian blur to reduce noise
        dapi_smooth = ndimage.gaussian_filter(dapi_norm, sigma=2.0)
        
        # Otsu thresholding
        try:
            thresh = threshold_otsu(dapi_smooth)
            mask_main_nuclei = dapi_smooth > thresh
        except Exception:
            # Fallback for empty/flat images
            mask_main_nuclei = np.zeros_like(dapi_smooth, dtype=bool)
            
        # Morphological cleanup
        mask_main_nuclei = binary_opening(mask_main_nuclei, footprint=disk(2))
        mask_main_nuclei = binary_closing(mask_main_nuclei, footprint=disk(2))
        
        # Filter small objects (debris) to keep only main nuclei
        # At 512x512, a nucleus is typically > 100 pixels
        labeled_nuclei = label(mask_main_nuclei)
        regions = regionprops(labeled_nuclei)
        mask_main_nuclei = np.zeros_like(mask_main_nuclei, dtype=bool)
        for region in regions:
            if region.area > 100:  # Minimum area for a main nucleus
                # Add region back to mask
                coords = region.coords
                mask_main_nuclei[coords[:, 0], coords[:, 1]] = True

    # Count valid cells (Denominator)
    labeled_main_nuclei = label(mask_main_nuclei)
    num_cells = np.max(labeled_main_nuclei) if labeled_main_nuclei.size > 0 else 0
    
    if num_cells == 0:
        return 0.0

    # --- 3. Detect Micronuclei Candidates (The Numerator) ---
    # Micronuclei are small, bright spots distinct from the main nucleus.
    
    # Enhance small spots using a Top-Hat filter equivalent
    # (Original - Morphological Opening) highlights small bright structures
    # We use a large structural element for opening to estimate background
    background = ndimage.grey_opening(dapi_norm, structure=disk(15))
    tophat = dapi_norm - background
    tophat = np.clip(tophat, 0, None)
    
    # Threshold for micronuclei
    # We use a statistical threshold on the tophat image
    mn_thresh = np.mean(tophat) + 2.0 * np.std(tophat)
    mask_candidates = tophat > mn_thresh
    
    # Label candidates
    labeled_candidates = label(mask_candidates)
    candidate_regions = regionprops(labeled_candidates)
    
    # --- 4. Filter and Validate Micronuclei ---
    micronuclei_count = 0
    
    # Define search zone: Perinuclear region
    # Dilate the main nuclei mask to define the "cytoplasmic" area where micronuclei are expected
    # Radius of 30 pixels covers the immediate vicinity of the nucleus
    search_zone = binary_dilation(mask_main_nuclei, footprint=disk(30))
    
    # Exclusion zone: The main nucleus itself
    # Micronuclei must be detached
    exclusion_zone = mask_main_nuclei
    
    for region in candidate_regions:
        # 1. Size Filter
        # Micronuclei are small: typically 5 to 80 pixels area
        if not (5 <= region.area <= 80):
            continue
            
        # 2. Shape Filter (Optional but good for robustness)
        # Micronuclei are usually somewhat circular
        if region.eccentricity > 0.95:
            continue
            
        # 3. Spatial Filter
        # Check centroid location
        r, c = map(int, region.centroid)
        
        # Must be OUTSIDE the main nucleus
        if exclusion_zone[r, c]:
            continue
            
        # Must be INSIDE the search zone (near a nucleus)
        if not search_zone[r, c]:
            continue
            
        # If all checks pass, it's a micronucleus
        micronuclei_count += 1

    # --- 5. Compute Feature ---
    result = micronuclei_count / float(num_cells)

    return float(result)
