def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_opening, disk

    # 1. Data Loading and Validation
    # Convert to float32 for processing
    arr = np.asarray(img, dtype=np.float32)

    # Check dimensionality
    # Expected: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        # If it's 2D (H, W), assume it's a single channel or grayscale. 
        # But dataset spec says (H, W, 3). If unexpected, return 0.0
        if arr.ndim == 2:
            # Fallback: treat as single channel, but this feature needs DAPI specifically.
            # If we can't identify DAPI, we can't count micronuclei.
            # However, for robustness, if it's 2D, we assume it's the relevant channel or a projection.
            dapi_channel = arr
        else:
            return 0.0
    else:
        # Extract DAPI channel (Channel 2: Blue)
        dapi_channel = arr[:, :, 2]

    # 2. Preprocessing & Normalization
    # Robust normalization using percentiles to handle outliers/artifacts
    p_min, p_max = np.percentile(dapi_channel, (1, 99.5))
    if p_max > p_min:
        norm_img = (dapi_channel - p_min) / (p_max - p_min)
    else:
        norm_img = dapi_channel  # Should be 0s
    
    norm_img = np.clip(norm_img, 0.0, 1.0)

    # 3. Segmentation
    # We need to identify main nuclei vs potential micronuclei.
    # Otsu's threshold is a good starting point for nuclei.
    try:
        thresh = threshold_otsu(norm_img)
    except ValueError:
        # Image might be empty or uniform
        return 0.0

    # Create binary mask
    # We use a slightly lower threshold to ensure we capture the full extent of nuclei
    # but not too low to pick up background noise.
    binary_mask = norm_img > thresh

    # Morphological cleaning: remove tiny noise specks
    # Use a small disk for opening
    binary_mask = binary_opening(binary_mask, footprint=disk(2))

    # Label connected components
    labeled_mask, num_features = label(binary_mask, return_num=True)
    
    if num_features == 0:
        return 0.0

    props = regionprops(labeled_mask, intensity_image=norm_img)

    # 4. Feature Extraction Logic
    # We need to distinguish "Main Nuclei" from "Micronuclei".
    # Criteria based on feedback:
    # - Main Nuclei: Large area.
    # - Micronuclei: Small area (but not noise), detached from main nuclei, circular.
    
    # Parameters (tuned based on feedback)
    # Image is 512x512.
    # Main nuclei are typically large (e.g., > 500-1000 pixels depending on magnification, here likely > 400).
    # Micronuclei are small fragments.
    
    MIN_MAIN_NUCLEUS_AREA = 400  # Threshold to define a "main" nucleus
    
    # Micronuclei constraints (tightened based on feedback)
    MIN_MN_AREA = 50      # Increased from 30 to avoid noise
    MAX_MN_AREA = 250     # Upper limit to distinguish from small apoptotic bodies or small nuclei
    MIN_MN_INTENSITY = 0.2 # Minimum mean intensity to avoid background ghosts
    MAX_MN_ECCENTRICITY = 0.85 # Must be relatively circular
    MIN_DISTANCE_TO_MAIN = 50.0 # Distance in pixels to nearest main nucleus (increased from 30)

    # Separate objects into candidates
    main_nuclei_coords = []
    mn_candidates = []

    for prop in props:
        # Identify Main Nuclei
        if prop.area >= MIN_MAIN_NUCLEUS_AREA:
            # Store centroid for distance calculation
            main_nuclei_coords.append(prop.centroid)
        
        # Identify Potential Micronuclei
        elif (MIN_MN_AREA <= prop.area <= MAX_MN_AREA):
            # Check shape and intensity immediately
            if (prop.mean_intensity >= MIN_MN_INTENSITY and 
                prop.eccentricity <= MAX_MN_ECCENTRICITY):
                
                # Circularity check: 4 * pi * Area / Perimeter^2
                # Perfect circle = 1.0. Lower values = less circular.
                if prop.perimeter > 0:
                    circularity = (4 * np.pi * prop.area) / (prop.perimeter ** 2)
                    if circularity > 0.8: # Reasonably circular
                        mn_candidates.append(prop)

    # If no main nuclei found, we can't define "detached", so return 0 (or count all small objects? usually 0)
    if not main_nuclei_coords:
        return 0.0

    main_nuclei_coords = np.array(main_nuclei_coords)
    
    micronuclei_count = 0

    # 5. Distance Filtering
    # For each candidate micronucleus, check distance to the NEAREST main nucleus.
    for mn in mn_candidates:
        mn_centroid = np.array(mn.centroid)
        
        # Calculate Euclidean distances to all main nuclei
        # dists shape: (N_main_nuclei,)
        dists = np.sqrt(np.sum((main_nuclei_coords - mn_centroid)**2, axis=1))
        
        min_dist = np.min(dists)
        
        # Check if it is detached enough
        if min_dist >= MIN_DISTANCE_TO_MAIN:
            micronuclei_count += 1

    return float(micronuclei_count)
