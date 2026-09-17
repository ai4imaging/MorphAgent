def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.morphology import binary_erosion, disk, binary_opening
    from skimage.filters import threshold_otsu
    from skimage.measure import label
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max

    # Convert to appropriate array type and normalize
    # Image is (512, 512, 3), uint8. Channel 0 is Actin.
    arr = np.asarray(img, dtype=np.float32)
    
    # Check dimensionality
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Normalize to [0, 1]
    # Using a robust max to avoid outliers skewing normalization too much, though 255 is standard for uint8
    vmax = 255.0
    arr = arr / vmax
    arr = np.clip(arr, 0.0, 1.0)

    # Extract channels
    actin_img = arr[..., 0]   # Channel 0: Actin (Target intensity)
    tubulin_img = arr[..., 1] # Channel 1: Tubulin (Helper for cell body)
    dapi_img = arr[..., 2]    # Channel 2: Nuclei (Helper for seeding)

    # --- Segmentation Logic ---
    # We need a labeled mask where each integer represents a distinct cell instance.
    
    labeled_cells = None
    
    # 1. Try to use provided segmentation masks
    if len(segmentation_masks) > 0:
        # Iterate through masks to find a suitable cell mask
        # We prefer a 'cell' or 'cytoplasm' mask over a 'nuclei' mask for this feature
        # Since we don't know the order/names, we look for the one with the largest coverage
        # or simply use the first one if it looks like a label mask.
        
        # Heuristic: Use the mask with the largest total area (likely whole cell vs nucleus)
        best_mask = None
        max_area = -1
        
        for mask in segmentation_masks:
            if mask is None:
                continue
            # Ensure mask is 2D and matches image shape
            if mask.shape != actin_img.shape:
                continue
            
            # Check if it's a label mask (integer type)
            if np.issubdtype(mask.dtype, np.integer):
                current_area = np.count_nonzero(mask)
                if current_area > max_area:
                    max_area = current_area
                    best_mask = mask
        
        if best_mask is not None:
            labeled_cells = best_mask

    # 2. Fallback: On-the-fly segmentation if no valid mask provided
    if labeled_cells is None:
        try:
            # Step A: Detect Nuclei (Seeds)
            # Smooth DAPI slightly
            dapi_smooth = ndimage.gaussian_filter(dapi_img, sigma=2)
            try:
                thresh_nuc = threshold_otsu(dapi_smooth)
            except ValueError: # Handle empty images
                thresh_nuc = 0.1
            
            mask_nuc = dapi_smooth > thresh_nuc
            mask_nuc = binary_opening(mask_nuc, footprint=disk(2))
            
            # Generate markers for watershed
            # Use distance transform to find centers of nuclei
            distance = ndimage.distance_transform_edt(mask_nuc)
            # Find peaks in distance map
            coords = peak_local_max(distance, min_distance=7, labels=mask_nuc)
            mask_peaks = np.zeros(distance.shape, dtype=bool)
            mask_peaks[tuple(coords.T)] = True
            markers = label(mask_peaks)
            
            # Step B: Detect Cell Body (Basin)
            # Combine Actin and Tubulin for a robust cell body signal
            cell_signal = actin_img + tubulin_img
            cell_signal = ndimage.gaussian_filter(cell_signal, sigma=2)
            try:
                thresh_cell = threshold_otsu(cell_signal)
            except ValueError:
                thresh_cell = 0.1
            
            mask_cell = cell_signal > thresh_cell
            # Fill holes to ensure solid objects
            mask_cell = ndimage.binary_fill_holes(mask_cell)

            # Step C: Watershed
            # We use the negative intensity as the topographic surface
            labeled_cells = watershed(-cell_signal, markers, mask=mask_cell)
            
        except Exception:
            # If segmentation fails completely, return 0.0
            return 0.0

    # --- Feature Computation: Actin Edge/Interior Ratio ---
    
    # Get unique cell labels (excluding background 0)
    unique_labels = np.unique(labeled_cells)
    unique_labels = unique_labels[unique_labels != 0]
    
    if len(unique_labels) == 0:
        return 0.0

    ratios = []
    
    # Define erosion radius for "Edge" definition (Cortex width)
    # For 512x512 images of cells, a 3-4 pixel rim is a reasonable approximation of the cortex
    erosion_radius = 3
    selem = disk(erosion_radius)

    # Pre-smooth actin image slightly to reduce pixel noise affecting the ratio
    actin_smooth = ndimage.gaussian_filter(actin_img, sigma=1.0)

    for label_id in unique_labels:
        # Create binary mask for current cell
        cell_mask = (labeled_cells == label_id)
        
        # Skip very small artifacts
        if np.sum(cell_mask) < 50:
            continue

        # Define Interior: Erode the cell mask
        # This removes the outer boundary layer
        interior_mask = binary_erosion(cell_mask, footprint=selem)
        
        # Define Edge: Cell Mask XOR Interior Mask
        # This captures the pixels that were removed by erosion (the boundary ring)
        edge_mask = cell_mask ^ interior_mask
        
        # Check if we have valid regions (very small cells might erode completely)
        if np.sum(interior_mask) == 0 or np.sum(edge_mask) == 0:
            continue
            
        # Calculate mean intensities
        mean_edge = np.mean(actin_smooth[edge_mask])
        mean_interior = np.mean(actin_smooth[interior_mask])
        
        # Calculate ratio
        # Add small epsilon to denominator to avoid division by zero
        epsilon = 1e-6
        ratio = mean_edge / (mean_interior + epsilon)
        
        # Sanity check for extreme outliers (e.g., if interior is black)
        if ratio > 10.0: 
            ratio = 10.0 # Cap at reasonable biological limit
            
        ratios.append(ratio)

    # Aggregate results
    if len(ratios) == 0:
        return 0.0
        
    result = np.mean(ratios)
    
    return float(result)
