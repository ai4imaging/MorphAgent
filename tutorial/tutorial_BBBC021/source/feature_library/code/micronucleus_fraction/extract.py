def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import remove_small_objects, binary_closing, disk

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset: (512, 512, 3), Channel 2 is DAPI (Nucleus)
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Extract DAPI channel (Channel 2)
        dapi_channel = arr[:, :, 2]
    elif arr.ndim == 2:
        # Fallback if single channel passed (unlikely based on spec but safe)
        dapi_channel = arr
    else:
        return 0.0

    # Intensity normalization
    # Normalize to [0, 1] based on robust range
    v_min, v_max = np.percentile(dapi_channel, (1, 99))
    if v_max > v_min:
        dapi_channel = (dapi_channel - v_min) / (v_max - v_min)
    else:
        dapi_channel = dapi_channel - v_min
    dapi_channel = np.clip(dapi_channel, 0.0, 1.0)

    # Preprocessing: Reduce noise to prevent single-pixel noise being detected as micronuclei
    # Gaussian blur with small sigma
    dapi_smooth = ndimage.gaussian_filter(dapi_channel, sigma=1.0)

    # Segmentation Logic
    # We need to detect both large nuclei and small micronuclei.
    # Global Otsu is generally robust for DAPI in fluorescence images.
    try:
        thresh = threshold_otsu(dapi_smooth)
        binary_mask = dapi_smooth > thresh
    except Exception:
        # Fallback if image is empty or uniform
        return 0.0

    # Morphological cleaning
    # Close small gaps
    binary_mask = binary_closing(binary_mask, disk(2))
    # Remove very small noise (smaller than a tiny micronucleus, e.g., < 5 pixels)
    binary_mask = remove_small_objects(binary_mask, min_size=5)

    # Label connected components
    labeled_mask = label(binary_mask)
    regions = regionprops(labeled_mask, intensity_image=dapi_channel)

    # Feature Extraction Logic: Classify objects based on size and shape
    # Dimensions are 512x512.
    # Typical MCF-7 nucleus diameter ~20-40 pixels -> Area ~300-1200 pixels.
    # Micronucleus diameter ~3-10 pixels -> Area ~10-80 pixels.
    
    # Thresholds (tuned for 512x512 resolution)
    micronucleus_area_min = 10
    micronucleus_area_max = 100
    nucleus_area_min = 150
    nucleus_area_max = 3000 # Allow for some clustering
    
    # Shape constraint: Micronuclei are usually circular.
    # Eccentricity: 0 (circle) to 1 (line).
    max_eccentricity = 0.9 

    micronuclei_count = 0
    nuclei_count = 0

    for props in regions:
        area = props.area
        
        # Check for Normal Nucleus
        if nucleus_area_min <= area <= nucleus_area_max:
            nuclei_count += 1
            
        # Check for Micronucleus
        elif micronucleus_area_min <= area <= micronucleus_area_max:
            # Additional check: Shape and Intensity
            # Micronuclei should not be extremely elongated (which suggests noise or artifacts)
            if props.eccentricity < max_eccentricity:
                # Intensity check: Micronuclei usually have comparable intensity to nuclei
                # Filter out very faint background blobs that passed threshold
                if props.mean_intensity > 0.2: # Relative to normalized [0,1] image
                    micronuclei_count += 1

    # Compute Fraction
    # Ratio of micronuclei objects to normal nuclei objects
    if nuclei_count == 0:
        return 0.0
    
    result = micronuclei_count / nuclei_count

    return float(result)
