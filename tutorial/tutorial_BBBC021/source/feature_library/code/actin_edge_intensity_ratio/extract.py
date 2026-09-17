def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.morphology import binary_erosion, disk
    from skimage.filters import threshold_otsu
    from skimage.measure import label

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3) where Channel 0 is Actin
    if arr.ndim == 3 and arr.shape[2] == 3:
        actin_channel = arr[:, :, 0]
    elif arr.ndim == 2:
        # Fallback if single channel passed (unlikely based on spec but safe)
        actin_channel = arr
    else:
        return 0.0

    # Intensity normalization (optional but good practice for stability)
    # We don't strictly need 0-1 for a ratio, but it helps avoid overflow if we were doing other math
    # Here we just use the raw float values for the ratio to preserve relative intensity differences
    
    # Determine Mask Strategy
    # We need to define "cells" to compute the per-cell edge/center ratio
    labeled_mask = None
    
    # Check if segmentation masks are provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Assume the first mask is the relevant cell/cytoplasm mask
        # Masks are typically labeled integers
        input_mask = segmentation_masks[0]
        if input_mask.shape == actin_channel.shape:
            labeled_mask = input_mask.astype(int)
    
    # Fallback: Generate mask if none provided or invalid
    if labeled_mask is None:
        # Simple background segmentation on Actin channel
        try:
            thresh = threshold_otsu(actin_channel)
            binary_mask = actin_channel > thresh
            # Label connected components
            labeled_mask, num_features = label(binary_mask, return_num=True)
        except Exception:
            # If image is uniform (e.g. all black), otsu fails
            return 0.0

    # Get unique labels (excluding background 0)
    unique_labels = np.unique(labeled_mask)
    if len(unique_labels) <= 1: # Only background exists
        return 0.0

    # Parameters for morphological operations
    # Cortex width definition: ~3 pixels for 512x512 image of MCF-7 cells
    erosion_radius = 3
    selem = disk(erosion_radius)
    
    ratios = []

    # Iterate over each cell to compute the ratio locally
    # This is more robust than a global edge/center calculation which might mix bright cells with dim cells
    for lbl in unique_labels:
        if lbl == 0:
            continue
            
        # Extract binary mask for current cell
        cell_mask = (labeled_mask == lbl)
        
        # Skip very small artifacts
        if np.sum(cell_mask) < 50:
            continue

        # Define Center (Cytoplasm) via erosion
        # If cell is too small for the erosion radius, the center will be empty
        center_mask = binary_erosion(cell_mask, footprint=selem)
        
        # Define Edge (Cortex) as Cell Mask minus Center Mask
        # Logical XOR works here because center is a strict subset of cell_mask
        edge_mask = cell_mask ^ center_mask
        
        # Check if we have valid regions
        edge_pixels = actin_channel[edge_mask]
        center_pixels = actin_channel[center_mask]
        
        if center_pixels.size == 0:
            # Cell is too thin/small, it's all "edge". 
            # This implies a very high ratio effectively, or undefined.
            # We skip to avoid skewing with infinity, or could treat as high value.
            # Skipping is safer for robust statistics.
            continue
            
        if edge_pixels.size == 0:
            # Should not happen given the logic (cell_mask > center_mask), but for safety
            continue

        mean_edge = np.mean(edge_pixels)
        mean_center = np.mean(center_pixels)

        # Avoid division by zero
        if mean_center <= 1e-6:
            # If center is black but edge has signal -> High ratio
            if mean_edge > 1e-6:
                ratios.append(10.0) # Cap at a reasonable high number
            else:
                ratios.append(1.0) # Both zero
        else:
            ratios.append(mean_edge / mean_center)

    # Aggregate results
    if not ratios:
        return 0.0
        
    # Return the mean ratio across the cell population
    result = np.mean(ratios)

    return float(result)
