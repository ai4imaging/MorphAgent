def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_closing, disk
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max

    # Convert to appropriate array type and normalize
    arr = np.asarray(img, dtype=np.float32)
    
    # Check for valid image content
    if arr.size == 0 or np.max(arr) == 0:
        return 0.0

    # Normalize to [0, 1]
    vmax = np.percentile(arr, 99.5) if arr.size > 0 else 1.0
    if vmax > 0:
        arr = arr / vmax
    arr = np.clip(arr, 0.0, 1.0)

    # Determine the labeled mask to use
    labeled_mask = None

    # 1. Try to use provided segmentation masks
    # We prioritize a cell body mask if available. Usually, if multiple masks are provided,
    # one might be nuclei and another cells. Without specific metadata on which is which,
    # we check the size. Cell masks generally cover more area than nuclei masks.
    if len(segmentation_masks) > 0:
        # Heuristic: Pick the mask with the largest total foreground area, assuming it's the cell body mask
        best_mask = None
        max_area = -1
        
        for mask in segmentation_masks:
            if mask is None:
                continue
            # Ensure mask is 2D (handle potential 3D or channel dims if passed incorrectly)
            if mask.ndim > 2:
                mask = np.max(mask, axis=-1) # Project
            
            current_area = np.count_nonzero(mask)
            if current_area > max_area:
                max_area = current_area
                best_mask = mask
        
        if best_mask is not None:
            # Ensure it's labeled (instance segmentation)
            if best_mask.max() <= 1:
                labeled_mask = label(best_mask)
            else:
                labeled_mask = best_mask.astype(int)

    # 2. Fallback: Generate segmentation from image channels if no external mask provided
    if labeled_mask is None:
        # Channel mapping: 0=Actin, 1=Tubulin, 2=DAPI
        # Cell body signal is best represented by Actin + Tubulin
        if arr.ndim == 3 and arr.shape[2] >= 2:
            # Combine Actin (0) and Tubulin (1) for cell body
            cyto_signal = np.maximum(arr[..., 0], arr[..., 1])
            # Nuclei signal for seeding
            nuc_signal = arr[..., 2] if arr.shape[2] > 2 else np.zeros_like(cyto_signal)
        elif arr.ndim == 2:
            cyto_signal = arr
            nuc_signal = arr
        else:
            return 0.0

        # Smooth to reduce noise
        cyto_smooth = ndimage.gaussian_filter(cyto_signal, sigma=2)
        
        # Thresholding for cell body mask
        try:
            thresh_val = threshold_otsu(cyto_smooth)
            binary_mask = cyto_smooth > thresh_val
        except Exception:
            # Fallback if image is uniform
            binary_mask = cyto_smooth > 0.1

        # Morphological cleanup
        binary_mask = binary_closing(binary_mask, disk(3))

        # Instance Segmentation via Watershed
        # 1. Identify markers (seeds) from nuclei or intensity peaks
        # If we have a strong DAPI channel, use it. Otherwise use distance transform peaks.
        use_dapi_seeds = (np.max(nuc_signal) > 0.1)
        
        if use_dapi_seeds:
            try:
                nuc_thresh = threshold_otsu(nuc_signal)
                nuc_mask = nuc_signal > nuc_thresh
                markers = label(nuc_mask)
            except:
                markers = None
        else:
            # Use distance transform on the cell body mask
            distance = ndimage.distance_transform_edt(binary_mask)
            # Find peaks
            coords = peak_local_max(distance, min_distance=20, labels=binary_mask)
            mask_peaks = np.zeros(distance.shape, dtype=bool)
            mask_peaks[tuple(coords.T)] = True
            markers = label(mask_peaks)

        if markers is not None and np.max(markers) > 0:
            # Compute distance map for watershed lines
            distance = ndimage.distance_transform_edt(binary_mask)
            labeled_mask = watershed(-distance, markers, mask=binary_mask)
        else:
            # Fallback to connected components if watershed fails
            labeled_mask = label(binary_mask)

    # 3. Compute Circularity
    # Circularity = (4 * pi * Area) / (Perimeter^2)
    # Range: 0.0 (line) to 1.0 (perfect circle)
    
    props = regionprops(labeled_mask)
    circularity_values = []

    for prop in props:
        # Filter small debris
        if prop.area < 50:
            continue
            
        # Avoid division by zero
        if prop.perimeter == 0:
            continue
            
        circ = (4 * np.pi * prop.area) / (prop.perimeter ** 2)
        
        # Clip to valid range (discrete pixels can sometimes cause slight >1.0 values for tiny perfect circles)
        circ = min(max(circ, 0.0), 1.0)
        
        circularity_values.append(circ)

    # 4. Return Mean
    if not circularity_values:
        return 0.0
        
    result = np.mean(circularity_values)
    return float(result)
