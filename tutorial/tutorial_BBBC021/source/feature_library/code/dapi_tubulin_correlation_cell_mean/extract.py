def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    
    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset: (512, 512, 3), uint8
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract relevant channels
    # Channel 1: Tubulin (Green)
    # Channel 2: DAPI (Blue)
    tubulin = arr[:, :, 1]
    dapi = arr[:, :, 2]

    # Handle segmentation masks
    labels = None
    
    # Check if valid segmentation masks are provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        mask = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions
        if mask.shape == arr.shape[:2]:
            # If mask is boolean or binary, label it
            if mask.dtype == bool or len(np.unique(mask)) <= 2:
                labels, _ = ndimage.label(mask)
            else:
                # Assume it's already an instance segmentation mask
                labels = mask.astype(int)
    
    # Fallback: Generate segmentation if no mask provided
    if labels is None:
        # Use DAPI channel for segmentation as it usually has distinct nuclei
        # Normalize DAPI for thresholding
        dapi_norm = dapi
        if dapi.max() > dapi.min():
            dapi_norm = (dapi - dapi.min()) / (dapi.max() - dapi.min())
        
        try:
            thresh = threshold_otsu(dapi_norm)
            binary_mask = dapi_norm > thresh
            labels, _ = ndimage.label(binary_mask)
        except Exception:
            # If thresholding fails (e.g. flat image), return 0.0
            return 0.0

    # Get unique cell labels (excluding background 0)
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]

    if len(unique_labels) == 0:
        return 0.0

    correlations = []

    # Iterate over each cell to compute correlation
    # Optimization: ndimage.find_objects returns slices, which is faster than boolean indexing the whole array
    slices = ndimage.find_objects(labels)

    for i, sl in enumerate(slices):
        if sl is None:
            continue
            
        # The label index is i + 1 because find_objects returns a list ordered by label index
        label_idx = i + 1
        
        # Extract the bounding box for the current cell
        cell_mask_slice = labels[sl] == label_idx
        
        # Extract pixel values for this cell
        d_vals = dapi[sl][cell_mask_slice]
        t_vals = tubulin[sl][cell_mask_slice]

        # Validity checks
        # 1. Need enough pixels to compute correlation
        if len(d_vals) < 5:
            continue
            
        # 2. Check for zero variance (cannot compute correlation)
        d_std = np.std(d_vals)
        t_std = np.std(t_vals)
        
        if d_std == 0 or t_std == 0:
            continue

        # Compute Pearson correlation coefficient
        # np.corrcoef returns the correlation matrix [[1, r], [r, 1]]
        r = np.corrcoef(d_vals, t_vals)[0, 1]
        
        if not np.isnan(r):
            correlations.append(r)

    # Aggregate results
    if len(correlations) == 0:
        result = 0.0
    else:
        result = np.mean(correlations)

    return float(result)
