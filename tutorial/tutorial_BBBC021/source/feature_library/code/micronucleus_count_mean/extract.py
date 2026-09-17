def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_opening, disk

    # 1. Data Loading and Preprocessing
    # Convert to float32 for processing
    arr = np.asarray(img, dtype=np.float32)

    # Check dimensionality
    # Expected shape: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        # If it's 2D (512, 512), we can't reliably distinguish channels, return 0
        # If it's (C, H, W), transpose it
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))
        elif arr.ndim == 2:
            return 0.0
        else:
            return 0.0

    # Extract DAPI channel (Channel 2)
    dapi = arr[..., 2]

    # Normalize DAPI channel to [0, 1]
    # Use robust max to avoid hot pixels skewing normalization
    p99 = np.percentile(dapi, 99.9)
    if p99 > 0:
        dapi_norm = dapi / p99
    else:
        dapi_norm = dapi  # Should be all zeros
    dapi_norm = np.clip(dapi_norm, 0.0, 1.0)

    # 2. Define Main Nuclei (The Denominator)
    # We need to identify the main nuclei to:
    #   a) Count the number of cells (denominator)
    #   b) Mask them out to find micronuclei (numerator candidates)
    
    main_nuclei_mask = None
    
    # Check if valid segmentation masks are provided
    # We look for a mask that likely represents nuclei
    if len(segmentation_masks) > 0:
        for mask in segmentation_masks:
            if mask is not None and mask.shape == dapi.shape:
                # Heuristic: if a mask exists, use it. 
                # If multiple exist, the first one is often the primary object mask.
                main_nuclei_mask = mask > 0
                break
    
    # Fallback: Compute nuclei mask if none provided
    if main_nuclei_mask is None:
        # Gaussian blur to smooth noise before thresholding
        dapi_smooth = ndimage.gaussian_filter(dapi_norm, sigma=2.0)
        
        try:
            thresh = threshold_otsu(dapi_smooth)
        except ValueError:
            # Handle empty image case
            return 0.0
            
        # Create binary mask
        binary_mask = dapi_smooth > thresh
        
        # Clean up mask: remove small noise, fill holes
        # Opening removes small bright spots that might be noise (but we want to keep micronuclei later)
        # Here we are defining MAIN nuclei, so we can be aggressive with size filtering
        binary_mask = binary_opening(binary_mask, footprint=disk(2))
        binary_mask = ndimage.binary_fill_holes(binary_mask)
        
        # Label connected components
        labeled_nuclei, num_features = label(binary_mask, return_num=True)
        
        # Filter by size to keep only "Main" nuclei
        # A typical nucleus in 512x512 at this magnification is > 100-200 pixels
        # Micronuclei are typically < 100 pixels
        min_nucleus_area = 150
        
        main_nuclei_mask = np.zeros_like(binary_mask, dtype=bool)
        
        # We can use bincount for fast area filtering if we don't need other props yet
        if num_features > 0:
            sizes = np.bincount(labeled_nuclei.ravel())
            # sizes[0] is background
            mask_indices = np.where(sizes > min_nucleus_area)[0]
            # Remove background index 0 if present
            mask_indices = mask_indices[mask_indices > 0]
            if len(mask_indices) > 0:
                main_nuclei_mask = np.isin(labeled_nuclei, mask_indices)

    # Count valid main nuclei
    labeled_main, num_cells = label(main_nuclei_mask, return_num=True)
    
    if num_cells == 0:
        return 0.0

    # 3. Detect Micronuclei (The Numerator)
    # Strategy: Look for small, bright objects in the DAPI channel that are NOT main nuclei
    
    # Mask out the main nuclei from the original normalized image
    # We dilate the main nuclei mask slightly to ensure we don't pick up the edges of the main nucleus as micronuclei
    expanded_nuclei_mask = ndimage.binary_dilation(main_nuclei_mask, iterations=2)
    
    # Create search image: DAPI signal outside of main nuclei
    micronucleus_search_img = dapi_norm.copy()
    micronucleus_search_img[expanded_nuclei_mask] = 0.0
    
    # Threshold for micronuclei
    # We can use a slightly lower threshold than Otsu, or reuse Otsu.
    # Micronuclei are DNA, so they should be relatively bright.
    # Let's re-calculate a threshold on the masked image, ignoring zeros
    valid_pixels = micronucleus_search_img[micronucleus_search_img > 0.05] # Ignore background
    if valid_pixels.size == 0:
        return 0.0
        
    # Use a fixed relative threshold or Otsu on the residual
    # A robust approach is to use a fraction of the main nuclei intensity
    # Calculate mean intensity of main nuclei
    mean_nuclei_intensity = np.mean(dapi_norm[main_nuclei_mask])
    mn_threshold = 0.25 * mean_nuclei_intensity # Micronuclei must be at least 25% as bright as average nucleus
    
    mn_candidates_mask = micronucleus_search_img > mn_threshold
    
    # Label candidates
    labeled_mn = label(mn_candidates_mask)
    props = regionprops(labeled_mn, intensity_image=dapi_norm)
    
    micronucleus_count = 0
    
    # Filter candidates
    # Criteria for Micronuclei:
    # 1. Area: Small but not single-pixel noise. Range [5, 100] pixels roughly.
    # 2. Shape: Generally circular (eccentricity < 0.9).
    # 3. Intensity: Should be distinct.
    
    min_mn_area = 5
    max_mn_area = 120
    max_eccentricity = 0.95
    
    for prop in props:
        if min_mn_area <= prop.area <= max_mn_area:
            if prop.eccentricity < max_eccentricity:
                # Optional: Check proximity? 
                # Strictly speaking, micronuclei should be close to a nucleus, 
                # but without cytoplasm segmentation, we assume objects in the field are cell-associated.
                micronucleus_count += 1
                
    # 4. Compute Feature
    result = micronucleus_count / float(num_cells)
    
    return float(result)
