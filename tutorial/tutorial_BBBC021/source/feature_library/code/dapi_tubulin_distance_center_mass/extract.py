def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.segmentation import watershed
    from skimage.morphology import binary_closing, disk

    # Convert to appropriate array type and normalize
    # Image is (512, 512, 3) uint8
    arr = np.asarray(img, dtype=np.float32)
    
    # Check dimensions
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract relevant channels based on dataset description
    # Channel 1: Tubulin (Green) -> Cytoskeleton
    # Channel 2: DAPI (Blue) -> Nucleus
    tubulin_ch = arr[..., 1]
    dapi_ch = arr[..., 2]

    # Normalize channels to 0-1 range for processing
    def normalize_channel(ch):
        vmax = np.percentile(ch, 99.5) if ch.size > 0 else 1.0
        if vmax <= 0: vmax = 1.0
        return np.clip(ch / vmax, 0.0, 1.0)

    tubulin_norm = normalize_channel(tubulin_ch)
    dapi_norm = normalize_channel(dapi_ch)

    # Determine Segmentation Mask
    labels = None
    
    # Strategy 1: Use provided segmentation mask
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        mask_input = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions
        if mask_input.shape[:2] == arr.shape[:2]:
            labels = mask_input.astype(np.int32)
            # If the mask is 3D (e.g. one-hot or RGB), collapse it, but usually it's 2D labels
            if labels.ndim > 2:
                labels = labels[..., 0]

    # Strategy 2: Fallback segmentation (Watershed)
    if labels is None:
        # 1. Detect Nuclei (Seeds)
        try:
            thresh_dapi = threshold_otsu(dapi_norm)
        except ValueError: # Handle empty images
            thresh_dapi = 0.1
            
        nuclei_mask = dapi_norm > thresh_dapi
        nuclei_mask = binary_closing(nuclei_mask, disk(2))
        nuclei_markers = label(nuclei_mask)

        # 2. Detect Cell Body (Mask)
        # Use Tubulin for cell body boundary
        try:
            thresh_tubulin = threshold_otsu(tubulin_norm)
        except ValueError:
            thresh_tubulin = 0.1
            
        # Lower threshold slightly to capture faint edges
        cell_mask = tubulin_norm > (thresh_tubulin * 0.7)
        cell_mask = binary_closing(cell_mask, disk(3))
        
        # Ensure nuclei are part of the cell mask
        cell_mask = np.logical_or(cell_mask, nuclei_mask)

        # 3. Watershed
        # Use negative intensity as "elevation" so bright spots are basins
        # However, standard distance transform watershed is often more robust for separation
        # Here we use intensity-based watershed constrained by the cell mask
        labels = watershed(-tubulin_norm, nuclei_markers, mask=cell_mask)

    # Feature Computation: Distance between Centers of Mass
    # We need the intensity-weighted center of mass for both channels per cell
    
    # Get unique cell indices (exclude background 0)
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]

    if len(unique_labels) == 0:
        return 0.0

    # Calculate Center of Mass (CoM) for DAPI (Nucleus center)
    # ndimage.center_of_mass returns a list of tuples [(y1, x1), (y2, x2), ...]
    # We pass the intensity image as 'input' to get weighted centroid
    com_dapi = ndimage.center_of_mass(dapi_norm, labels, unique_labels)
    
    # Calculate Center of Mass (CoM) for Tubulin (Cytoskeleton center)
    com_tubulin = ndimage.center_of_mass(tubulin_norm, labels, unique_labels)

    # Convert to numpy arrays for vectorized distance calculation
    # Shape: (N_cells, 2)
    com_dapi_arr = np.array(com_dapi)
    com_tubulin_arr = np.array(com_tubulin)

    # Handle case where center_of_mass might return NaN (e.g. if a label exists but has 0 intensity sum)
    # This is rare with normalized data but possible
    valid_mask = ~np.isnan(com_dapi_arr).any(axis=1) & ~np.isnan(com_tubulin_arr).any(axis=1)
    
    if np.sum(valid_mask) == 0:
        return 0.0

    com_dapi_arr = com_dapi_arr[valid_mask]
    com_tubulin_arr = com_tubulin_arr[valid_mask]

    # Euclidean distance between corresponding centers
    # sqrt((y2-y1)^2 + (x2-x1)^2)
    distances = np.linalg.norm(com_dapi_arr - com_tubulin_arr, axis=1)

    # Return the mean distance across all cells in the image
    result = np.mean(distances)

    return float(result)
