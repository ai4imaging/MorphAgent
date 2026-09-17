def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_closing, remove_small_objects, disk
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Expected: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        # If not the expected 3-channel image, return 0.0
        return 0.0

    # Extract relevant channels
    # Channel 0: Actin (Cytoskeleton) - Primary source for cell shape
    # Channel 2: DAPI (Nucleus) - Useful for seeding separation of clumped cells
    actin_channel = arr[..., 0]
    dapi_channel = arr[..., 2]

    # Intensity normalization (per channel)
    def normalize_channel(ch):
        vmax = np.percentile(ch, 99.5) if ch.size > 0 else 1.0
        if vmax > 0:
            ch = ch / vmax
        return np.clip(ch, 0.0, 1.0)

    actin_norm = normalize_channel(actin_channel)
    dapi_norm = normalize_channel(dapi_channel)

    # Determine the labeled mask
    labeled_mask = None

    # Strategy 1: Use provided segmentation mask if available
    # We prioritize the first mask if it exists, assuming it's a cell/cytoplasm mask
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        mask_input = segmentation_masks[0]
        # Ensure it matches image spatial dimensions
        if mask_input.shape[:2] == arr.shape[:2]:
            if mask_input.ndim == 2:
                labeled_mask = mask_input.astype(int)
            elif mask_input.ndim == 3:
                # If 3D mask, project or take first channel
                labeled_mask = mask_input[..., 0].astype(int)

    # Strategy 2: Compute segmentation on the fly if no mask provided
    if labeled_mask is None:
        # 1. Threshold Actin to get general cell foreground
        # Smooth to reduce noise
        actin_smooth = ndimage.gaussian_filter(actin_norm, sigma=2)
        try:
            thresh_val = threshold_otsu(actin_smooth)
        except Exception:
            thresh_val = 0.1
        
        binary_mask = actin_smooth > thresh_val
        
        # 2. Clean up binary mask
        # Fill holes and smooth boundaries
        binary_mask = binary_closing(binary_mask, disk(3))
        # Remove small debris (noise)
        binary_mask = remove_small_objects(binary_mask, min_size=100)

        # 3. Separate touching cells using Watershed
        # MCF-7 cells clump; we need to split them to get accurate shape metrics.
        # Use DAPI (Nuclei) as seeds if possible, otherwise use distance transform of Actin.
        
        # Generate seeds from DAPI
        dapi_smooth = ndimage.gaussian_filter(dapi_norm, sigma=2)
        try:
            dapi_thresh = threshold_otsu(dapi_smooth)
        except:
            dapi_thresh = 0.1
        
        nuclei_mask = dapi_smooth > dapi_thresh
        nuclei_mask = remove_small_objects(nuclei_mask, min_size=50)
        
        # Distance transform for watershed
        distance = ndimage.distance_transform_edt(binary_mask)
        
        # If we have good nuclei signals, use them as markers
        if np.sum(nuclei_mask) > 0:
            markers = label(nuclei_mask)
        else:
            # Fallback: use local maxima of distance transform
            local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((20, 20)), labels=binary_mask)
            markers = label(local_maxi)

        # Apply watershed
        # -distance because watershed looks for basins
        labeled_mask = watershed(-distance, markers, mask=binary_mask)

    # Compute Feature: Mean Eccentricity
    # Eccentricity: 0 = circle, 1 = line/ellipse
    props = regionprops(labeled_mask)
    
    if len(props) == 0:
        return 0.0

    eccentricities = [p.eccentricity for p in props]
    
    # Return the mean
    result = np.mean(eccentricities)

    return float(result)
