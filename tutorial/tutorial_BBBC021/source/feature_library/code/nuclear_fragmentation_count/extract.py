def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.morphology import disk, white_tophat, binary_opening
    from skimage.filters import threshold_otsu

    # 1. Input Validation and Preprocessing
    # Convert to float32 for processing
    arr = np.asarray(img, dtype=np.float32)

    # Check dimensions: Expected (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # 2. Channel Selection
    # Channel 2 is DAPI (Nucleus), which is critical for detecting nuclear fragmentation/micronuclei.
    dapi_channel = arr[..., 2]

    # 3. Normalization
    # Normalize DAPI channel to [0, 1] based on robust range
    p_min, p_max = np.percentile(dapi_channel, (1, 99.5))
    if p_max > p_min:
        dapi_norm = (dapi_channel - p_min) / (p_max - p_min)
    else:
        dapi_norm = dapi_channel  # Fallback if constant
    dapi_norm = np.clip(dapi_norm, 0.0, 1.0)

    # 4. Main Nuclei Segmentation (to exclude from micronuclei search)
    # We need to identify the main nuclei to subtract them or define a search region outside them.
    # If segmentation masks are provided, use them. Otherwise, compute a coarse mask.
    
    main_nuclei_mask = None
    
    # Try to find a nuclei mask in the provided segmentation masks
    # The prompt implies masks might be passed. We look for one that resembles nuclei.
    # Usually, if masks are present, they might be (cells, nuclei).
    # Without specific metadata on which mask is which, we can infer or fallback to computing it.
    
    if len(segmentation_masks) > 0:
        # Heuristic: Check masks. If we have multiple, usually one is nuclei.
        # Often nuclei are smaller than cells. Or we just use the provided masks as "exclusion zones".
        # Let's combine all provided masks into a binary exclusion zone for "main objects".
        combined_mask = np.zeros(dapi_channel.shape, dtype=bool)
        for mask in segmentation_masks:
            if mask.shape == dapi_channel.shape:
                combined_mask = combined_mask | (mask > 0)
        
        if np.any(combined_mask):
            main_nuclei_mask = combined_mask

    # If no valid mask provided, generate a coarse one for main nuclei
    if main_nuclei_mask is None:
        # Gaussian blur to smooth out noise
        smooth_dapi = ndimage.gaussian_filter(dapi_norm, sigma=2.0)
        try:
            thresh = threshold_otsu(smooth_dapi)
        except:
            thresh = 0.2
        # Main nuclei are large and bright
        main_nuclei_mask = smooth_dapi > thresh
        # Fill holes to ensure the whole nucleus is covered
        main_nuclei_mask = ndimage.binary_fill_holes(main_nuclei_mask)
        # Remove small objects that might be micronuclei from this "main" mask
        # We only want to exclude the BIG nuclei.
        labeled_main, _ = label(main_nuclei_mask, return_num=True)
        props = regionprops(labeled_main)
        # Filter: keep only large objects as "main nuclei"
        # Typical nucleus area in 512x512 might be > 200-300 pixels
        mask_cleaned = np.zeros_like(main_nuclei_mask)
        for prop in props:
            if prop.area > 500: # Threshold for a "main" nucleus
                mask_cleaned[labeled_main == prop.label] = True
        main_nuclei_mask = mask_cleaned

    # Dilate the main nuclei mask slightly to ensure we don't count blebs attached to the nucleus
    # Micronuclei should be detached.
    exclusion_mask = ndimage.binary_dilation(main_nuclei_mask, iterations=3)

    # 5. Micronuclei Detection
    # Strategy:
    # A. Use White Top-Hat transform to find small bright spots on a darker background.
    #    Radius should be slightly larger than the expected micronucleus size.
    #    Micronuclei are typically small (e.g., 1/10th to 1/100th of main nucleus).
    #    Let's assume a radius of ~8-10 pixels for the structuring element.
    
    # Refined based on feedback: Smaller structuring element to isolate small objects better.
    wth = white_tophat(dapi_norm, footprint=disk(8))

    # B. Thresholding specific to micronuclei candidates
    # They should be relatively bright in the top-hat image.
    # Use a fixed threshold or a high percentile of the top-hat response.
    # Feedback suggests Otsu on WTH might be too sensitive to noise.
    # Let's use a statistical threshold based on the WTH distribution (e.g., mean + k*std)
    # or a fixed intensity if normalized.
    
    wth_mean = np.mean(wth)
    wth_std = np.std(wth)
    # Threshold: significantly brighter than background noise
    candidate_thresh = wth_mean + 3.0 * wth_std
    # Also enforce a minimum absolute intensity in the original normalized image
    # to avoid picking up background noise in empty areas.
    min_intensity = 0.15 
    
    candidates = (wth > candidate_thresh) & (dapi_norm > min_intensity)

    # C. Exclude regions belonging to main nuclei
    candidates = candidates & (~exclusion_mask)

    # D. Morphological Cleaning
    # Remove single pixel noise
    candidates = binary_opening(candidates, footprint=disk(1))

    # 6. Feature Extraction (Counting)
    labeled_candidates, num_candidates = label(candidates, return_num=True)
    
    if num_candidates == 0:
        return 0.0

    props = regionprops(labeled_candidates, intensity_image=dapi_norm)
    
    micronuclei_count = 0
    
    # Refined constraints based on feedback:
    # - Area: 20 to 150 pixels (too small = noise, too large = debris/apoptotic body)
    # - Circularity/Compactness: Micronuclei are usually round.
    # - Mean Intensity: Must be DNA-bright.
    
    for prop in props:
        # Area filter
        if not (20 <= prop.area <= 150):
            continue
            
        # Circularity filter: 4 * pi * Area / Perimeter^2
        # Perfect circle = 1.0. 
        if prop.perimeter == 0: continue
        circularity = (4 * np.pi * prop.area) / (prop.perimeter ** 2)
        
        # Micronuclei are generally roundish, but can be slightly irregular.
        # Debris is often very irregular (low circularity).
        if circularity < 0.6:
            continue

        # Intensity filter (double check)
        # The max intensity of the spot should be significant
        if prop.max_intensity < 0.2:
            continue

        micronuclei_count += 1

    return float(micronuclei_count)
