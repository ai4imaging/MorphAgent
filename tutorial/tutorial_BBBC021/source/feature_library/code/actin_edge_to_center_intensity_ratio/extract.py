def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    from skimage.measure import label, regionprops
    from skimage.morphology import binary_erosion, disk

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3) RGB. Channel 0 is Actin (Red).
    # We extract Channel 0 for Actin analysis.
    if arr.ndim == 3 and arr.shape[2] >= 1:
        actin_img = arr[..., 0]
    elif arr.ndim == 2:
        actin_img = arr
    else:
        return 0.0

    # Intensity normalization
    # Normalize uint8 [0, 255] to [0, 1]
    if actin_img.max() > 1.0:
        actin_img /= 255.0
    actin_img = np.clip(actin_img, 0.0, 1.0)

    # Handle segmentation masks
    labeled_cells = None
    
    # Check if valid segmentation masks are provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        input_mask = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions (handle potential extra dims)
        if input_mask.shape[:2] == actin_img.shape[:2]:
            # If mask is already integer labels (instance segmentation), use it
            if np.issubdtype(input_mask.dtype, np.integer) and input_mask.max() > 1:
                labeled_cells = input_mask
            else:
                # Otherwise treat as binary mask and label connected components
                binary_mask = input_mask > 0
                labeled_cells, _ = ndimage.label(binary_mask)

    # Fallback: Auto-segmentation if no mask provided
    if labeled_cells is None:
        # Smooth image to reduce noise for thresholding
        smooth = ndimage.gaussian_filter(actin_img, sigma=2)
        try:
            thresh = threshold_otsu(smooth)
            binary_mask = smooth > thresh
            # Fill holes to ensure solid cell bodies
            binary_mask = ndimage.binary_fill_holes(binary_mask)
            labeled_cells, _ = ndimage.label(binary_mask)
        except:
            # Fallback for empty or uniform images
            return 0.0

    # Feature Computation: Actin Edge to Center Intensity Ratio
    # Logic:
    # 1. For each cell, define "Edge" (cortical) and "Center" (cytoplasmic) regions.
    # 2. Edge is defined by the difference between the cell mask and an eroded cell mask.
    # 3. Compute Mean Intensity(Edge) / Mean Intensity(Center).
    # 4. Aggregate across cells (median).

    ratios = []
    
    # Define erosion radius (approx 4 pixels for cortical ring width)
    selem = disk(4)

    # Use regionprops to iterate over individual cells efficiently
    props = regionprops(labeled_cells, intensity_image=actin_img)
    
    for prop in props:
        # Skip very small artifacts
        if prop.area < 100:
            continue

        # Extract local masks (in the bounding box of the cell)
        cell_mask_local = prop.image  # Binary mask of the cell
        cell_intensity_local = prop.intensity_image # Intensity image of the cell
        
        # Define Center region by eroding the cell mask
        center_mask_local = binary_erosion(cell_mask_local, footprint=selem)
        
        # Define Edge region by subtracting Center from Whole Cell
        edge_mask_local = np.logical_xor(cell_mask_local, center_mask_local)
        
        # Validate regions exist
        if np.sum(center_mask_local) == 0 or np.sum(edge_mask_local) == 0:
            # Cell too small for this erosion radius, skip
            continue

        # Compute mean intensities
        mean_center = np.mean(cell_intensity_local[center_mask_local])
        mean_edge = np.mean(cell_intensity_local[edge_mask_local])
        
        # Compute ratio
        # Add small epsilon to denominator to prevent division by zero
        ratio = mean_edge / (mean_center + 1e-7)
        ratios.append(ratio)

    # Aggregate results
    if not ratios:
        return 0.0
        
    # Return median ratio to be robust against outliers
    result = np.median(ratios)

    return float(result)
