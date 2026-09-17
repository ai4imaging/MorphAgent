def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import convex_hull_image
    from scipy.ndimage import binary_fill_holes  # Correct import location

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Check dimensionality
    # Expected: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # We need to define the "cell" shape.
    # Channel 0 (Actin) and Channel 1 (Tubulin) are good markers for the cell body/cytoplasm.
    # Channel 2 (DAPI) is for the nucleus.
    # To get the full cell shape for solidity calculation, combining Actin and Tubulin is usually best.
    # Let's use the maximum projection of the cytoplasmic channels (0 and 1) to define the cell boundary.
    
    # Extract channels
    actin = arr[..., 0]
    tubulin = arr[..., 1]
    
    # Combine cytoplasmic signals
    cyto_signal = np.maximum(actin, tubulin)

    # Normalize
    vmax = np.percentile(cyto_signal, 99.5) if cyto_signal.size > 0 else 1.0
    if vmax > 0:
        cyto_signal = cyto_signal / vmax
    cyto_signal = np.clip(cyto_signal, 0.0, 1.0)

    # Segmentation logic
    # If segmentation masks are provided, use them.
    # The prompt implies masks might be available. Let's check.
    # If available, we assume they label individual cells.
    
    labeled_cells = None
    
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use the first available mask. 
        # Assuming the mask is 2D (H, W) with integer labels.
        mask = segmentation_masks[0]
        if mask.shape == arr.shape[:2]:
            labeled_cells = mask
    
    # Fallback: If no mask is provided, perform basic segmentation
    if labeled_cells is None:
        try:
            thresh = threshold_otsu(cyto_signal)
            binary = cyto_signal > thresh
            # Fill holes to ensure solid objects
            binary = binary_fill_holes(binary)
            # Label connected components
            labeled_cells = label(binary)
        except Exception:
            return 0.0

    # Calculate Solidity
    # Solidity = Area / Convex Hull Area
    # We compute this for each labeled object and take the mean.
    
    props = regionprops(labeled_cells)
    
    solidities = []
    for prop in props:
        # Filter out very small artifacts
        if prop.area < 50:
            continue
            
        # regionprops calculates solidity directly: prop.solidity
        # However, to be explicit and robust, we can use the property directly provided by skimage
        solidities.append(prop.solidity)

    if not solidities:
        return 0.0

    result = np.mean(solidities)

    return float(result)
