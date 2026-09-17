def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu, gaussian
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max
    from skimage.morphology import remove_small_objects
    
    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Expected: (512, 512, 3)
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Standard format
        pass
    elif arr.ndim == 2:
        # If 2D, treat as grayscale (single channel) but expand for consistency
        arr = np.stack([arr, arr, arr], axis=-1)
    else:
        # Unexpected format, return 0.0
        return 0.0

    # Intensity normalization
    # Normalize to [0, 1] for processing
    vmax = np.percentile(arr, 99.5) if arr.size > 0 else 1.0
    if vmax > 0:
        arr = arr / vmax
    arr = np.clip(arr, 0.0, 1.0)

    # Define labeled_mask variable
    labeled_mask = None

    # Handle segmentation masks if available
    # We prioritize masks that represent the "whole cell" (cytoplasm)
    if len(segmentation_masks) > 0:
        # Check available masks. Usually, if multiple are provided, one is nuclei and one is cells.
        # Without specific metadata on which is which, we often assume the larger coverage is the cell.
        # However, for safety, if we have masks, we check if they are labeled.
        
        # Strategy: Iterate through masks, find one that is valid.
        # If multiple, pick the one with the largest total foreground area (likely whole cell vs nucleus).
        best_mask = None
        max_area = -1
        
        for mask in segmentation_masks:
            if mask is None:
                continue
            
            # Ensure mask is integer for labeling
            current_mask = np.asarray(mask, dtype=np.int32)
            
            # If mask is 2D matching image spatial dims
            if current_mask.shape == arr.shape[:2]:
                foreground_area = np.count_nonzero(current_mask)
                if foreground_area > max_area:
                    max_area = foreground_area
                    best_mask = current_mask
        
        if best_mask is not None:
            # Ensure it's labeled (instance segmentation)
            # If it's binary (0s and 1s), label it. If it has multiple IDs, keep them.
            if best_mask.max() <= 1:
                labeled_mask = label(best_mask)
            else:
                labeled_mask = best_mask

    # Fallback: On-the-fly segmentation if no valid mask provided
    if labeled_mask is None:
        # Channel 0 (Actin) and Channel 1 (Tubulin) define the cell body
        # Channel 2 (DAPI) defines the nucleus (seeds)
        
        # 1. Create Cell Body Signal (Max of Actin and Tubulin)
        # We use max projection of structural channels to get the full extent
        cell_signal = np.maximum(arr[..., 0], arr[..., 1])
        nuc_signal = arr[..., 2]
        
        # Smooth to reduce noise
        cell_smooth = gaussian(cell_signal, sigma=2)
        nuc_smooth = gaussian(nuc_signal, sigma=2)
        
        # 2. Thresholding
        try:
            thresh_cell = threshold_otsu(cell_smooth)
            mask_body = cell_smooth > thresh_cell
            
            thresh_nuc = threshold_otsu(nuc_smooth)
            mask_nuc = nuc_smooth > thresh_nuc
        except ValueError:
            # If image is empty or constant
            return 0.0

        # 3. Clean up masks
        mask_body = remove_small_objects(mask_body, min_size=50)
        mask_nuc = remove_small_objects(mask_nuc, min_size=20)
        
        # 4. Watershed Segmentation
        # Use nuclei as markers to separate touching cells
        markers = label(mask_nuc)
        
        # If no nuclei found, try using local maxima on cell body
        if markers.max() == 0:
            coords = peak_local_max(cell_smooth, min_distance=20, labels=mask_body)
            mask_peaks = np.zeros(mask_body.shape, dtype=bool)
            mask_peaks[tuple(coords.T)] = True
            markers = label(mask_peaks)

        # If still no markers, just label the body mask (connected components)
        if markers.max() == 0:
            labeled_mask = label(mask_body)
        else:
            # Watershed: expand markers into the mask_body using the inverse intensity as elevation
            labeled_mask = watershed(-cell_smooth, markers, mask=mask_body)

    # Feature Computation: Mean Eccentricity
    # Eccentricity is a property of the ellipse that has the same second-moments as the region.
    # 0 = circle, -> 1 = line segment
    
    regions = regionprops(labeled_mask)
    
    eccentricities = []
    for region in regions:
        # Filter out very small artifacts that might skew the mean
        if region.area >= 50:
            eccentricities.append(region.eccentricity)
            
    if not eccentricities:
        return 0.0
        
    result = np.mean(eccentricities)

    return float(result)
