def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu, gaussian
    from skimage.measure import label, regionprops
    from skimage.morphology import remove_small_objects, closing, square
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality
    # Expected shape: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        # Fallback for unexpected shapes, e.g., if channels are first or 2D
        if arr.ndim == 2:
            # Assume single channel is relevant, or just return 0 if ambiguous
            # For this specific feature (Actin), we need channel 0. 
            # If 2D, we can't distinguish channels reliably without metadata.
            # However, if it's a projection, we might treat it as the signal.
            # Let's assume if 2D, it's the relevant signal.
            actin_channel = arr
            dapi_channel = None
        elif arr.ndim == 3 and arr.shape[0] == 3:
            # Channels first (3, H, W)
            actin_channel = arr[0]
            dapi_channel = arr[2]
        else:
            return 0.0
    else:
        # Standard case: (H, W, C) -> (512, 512, 3)
        # Channel 0 = Actin (Red)
        # Channel 2 = DAPI (Blue) - useful for separating touching cells
        actin_channel = arr[..., 0]
        dapi_channel = arr[..., 2]

    # Check if valid segmentation masks are provided
    # We look for a mask that likely represents "cells" or "cytoplasm"
    # If masks are provided, we assume the first one is the primary object mask
    # or we try to find one that matches the image dimensions.
    labeled_cells = None
    
    if len(segmentation_masks) > 0:
        for mask in segmentation_masks:
            if mask is not None and mask.ndim == 2 and mask.shape == actin_channel.shape:
                # Assume this is a valid label mask
                # Check if it's labeled (int) or binary
                if np.issubdtype(mask.dtype, np.integer) and np.max(mask) > 1:
                    labeled_cells = mask
                    break
                elif np.max(mask) > 0:
                    # Binary mask, need to label
                    labeled_cells = label(mask > 0)
                    break
    
    # If no valid mask provided, perform on-the-fly segmentation
    if labeled_cells is None:
        # 1. Preprocessing Actin Channel
        # Normalize Actin
        actin_max = np.percentile(actin_channel, 99.5)
        if actin_max > 0:
            actin_norm = actin_channel / actin_max
        else:
            actin_norm = actin_channel
        actin_norm = np.clip(actin_norm, 0, 1)
        
        # Smooth to merge filaments into a cell body blob
        actin_smooth = gaussian(actin_norm, sigma=2.0)

        # Thresholding (Otsu)
        try:
            thresh_val = threshold_otsu(actin_smooth)
            binary_mask = actin_smooth > thresh_val
        except ValueError:
            # Handle empty images (e.g. all zeros)
            return 0.0

        # Morphological cleanup
        # Close gaps in cytoskeleton
        binary_mask = closing(binary_mask, square(3))
        # Remove small noise (debris)
        binary_mask = remove_small_objects(binary_mask, min_size=100)
        
        # Fill holes to get solid cell bodies
        binary_mask = ndimage.binary_fill_holes(binary_mask)

        # 2. Instance Separation (Watershed)
        # Use DAPI as seeds if available and has signal, otherwise use distance transform of Actin
        use_dapi_seeds = False
        if dapi_channel is not None:
            dapi_max = np.max(dapi_channel)
            if dapi_max > 10: # Check if there is actual signal
                use_dapi_seeds = True
        
        if use_dapi_seeds:
            # Process DAPI to find nuclei seeds
            dapi_smooth = gaussian(dapi_channel, sigma=2.0)
            try:
                dapi_thresh = threshold_otsu(dapi_smooth)
                nuclei_mask = dapi_smooth > dapi_thresh
                nuclei_mask = remove_small_objects(nuclei_mask, min_size=50)
                # Label nuclei
                markers = label(nuclei_mask)
            except:
                use_dapi_seeds = False

        if not use_dapi_seeds:
            # Fallback: Distance transform on Actin mask
            distance = ndimage.distance_transform_edt(binary_mask)
            # Find peaks in distance map
            coords = peak_local_max(distance, min_distance=20, labels=binary_mask)
            mask_peaks = np.zeros(distance.shape, dtype=bool)
            mask_peaks[tuple(coords.T)] = True
            markers = label(mask_peaks)

        # Apply watershed
        # We calculate the gradient or invert the distance map for the "basin"
        # Here we use the inverted distance transform of the binary mask to split touching blobs
        distance = ndimage.distance_transform_edt(binary_mask)
        labeled_cells = watershed(-distance, markers, mask=binary_mask)

    # Compute Feature: Mean Area
    regions = regionprops(labeled_cells)
    
    if len(regions) == 0:
        return 0.0

    areas = [r.area for r in regions]
    mean_area = np.mean(areas)

    return float(mean_area)
