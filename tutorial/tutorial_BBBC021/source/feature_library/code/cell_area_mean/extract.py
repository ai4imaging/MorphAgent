def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    from skimage.measure import label, regionprops
    from skimage.morphology import remove_small_objects, binary_closing, disk
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max

    # Convert to appropriate array type and normalize
    arr = np.asarray(img, dtype=np.float32)
    
    # Handle dimensionality
    # Expected shape: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        # If not the expected 3-channel image, return 0.0
        return 0.0

    # Normalize intensity to [0, 1] for processing
    # We use a robust max to avoid outliers skewing the normalization
    vmax = np.percentile(arr, 99.5) if arr.size > 0 else 1.0
    if vmax > 0:
        arr = arr / vmax
    arr = np.clip(arr, 0.0, 1.0)

    # Channel mapping based on dataset description
    # Ch0: Actin (Cytoskeleton)
    # Ch1: Tubulin (Microtubules)
    # Ch2: DAPI (Nucleus)
    actin = arr[..., 0]
    tubulin = arr[..., 1]
    dapi = arr[..., 2]

    # --- Segmentation Logic ---
    
    # We need a labeled mask of individual cells to compute cell area.
    # Strategy:
    # 1. If a valid cell/cytoplasm mask is provided in segmentation_masks, use it.
    # 2. If not, generate one using the image channels.
    
    labeled_cells = None

    # Check provided masks
    if len(segmentation_masks) > 0:
        # Heuristic: The largest mask usually corresponds to the whole cell/cytoplasm
        # (Nuclei masks are smaller). If multiple are provided, we might need to guess.
        # Often, masks are passed as (nuclei, cells) or just (cells).
        # We will check the area coverage. A cell mask covers more area than a nuclei mask.
        
        best_mask = None
        max_coverage = -1
        
        for mask in segmentation_masks:
            if mask is None: continue
            # Ensure mask is 2D
            if mask.ndim == 3:
                mask = np.max(mask, axis=0) # MIP if 3D
            
            # Check if it's a label matrix or binary
            if mask.max() > 1:
                # Already labeled
                coverage = np.sum(mask > 0)
            else:
                coverage = np.sum(mask)
                
            if coverage > max_coverage:
                max_coverage = coverage
                best_mask = mask
        
        if best_mask is not None:
            if best_mask.max() > 1:
                labeled_cells = best_mask.astype(int)
            else:
                labeled_cells = label(best_mask > 0)

    # Fallback: Compute segmentation from scratch if no valid mask found
    if labeled_cells is None:
        # 1. Define Cell Body (Foreground)
        # Combine Actin and Tubulin for best cell boundary definition
        cell_signal = actin + tubulin
        
        # Smooth to reduce noise and texture
        cell_signal_smooth = ndimage.gaussian_filter(cell_signal, sigma=2)
        
        # Threshold
        try:
            thresh_val = threshold_otsu(cell_signal_smooth)
            cell_mask = cell_signal_smooth > thresh_val
        except ValueError:
            # Handle case where image is empty or uniform
            return 0.0

        # Clean up mask
        cell_mask = binary_closing(cell_mask, footprint=disk(3))
        cell_mask = remove_small_objects(cell_mask, min_size=100)

        # 2. Define Seeds (Nuclei)
        # Use DAPI channel
        dapi_smooth = ndimage.gaussian_filter(dapi, sigma=2)
        try:
            dapi_thresh = threshold_otsu(dapi_smooth)
            nuclei_mask = dapi_smooth > dapi_thresh
        except ValueError:
            nuclei_mask = np.zeros_like(dapi, dtype=bool)

        nuclei_mask = remove_small_objects(nuclei_mask, min_size=50)
        
        # Generate markers for watershed
        # We use distance transform on nuclei to find centers, ensuring separated seeds
        distance = ndimage.distance_transform_edt(nuclei_mask)
        # Find peaks in distance map to use as markers
        local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((5, 5)), labels=nuclei_mask)
        markers = label(local_maxi)
        
        # If no markers found (no nuclei), fall back to labeling the cell mask directly
        if np.max(markers) == 0:
            labeled_cells = label(cell_mask)
        else:
            # 3. Watershed Segmentation
            # Use negative intensity as "elevation" for watershed, constrained by cell mask
            # We use the inverse of the distance transform of the cell mask to encourage boundaries between cells
            # to be where the cell body is thinnest.
            cell_distance = ndimage.distance_transform_edt(cell_mask)
            labeled_cells = watershed(-cell_distance, markers, mask=cell_mask)

    # --- Feature Extraction ---
    
    # Calculate properties
    regions = regionprops(labeled_cells)
    
    if len(regions) == 0:
        return 0.0
    
    areas = []
    for region in regions:
        # Filter out very small artifacts that might have survived
        if region.area < 50:
            continue
        areas.append(region.area)
    
    if len(areas) == 0:
        return 0.0
        
    # Calculate mean area
    mean_area = np.mean(areas)
    
    return float(mean_area)
