def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu

    # Convert to appropriate array type and handle dimensions
    # Dataset is (512, 512, 3) uint8.
    # Channel 0: Actin (Red)
    # Channel 1: Tubulin (Green)
    # Channel 2: DAPI (Blue)
    
    img = np.asarray(img)
    
    # Validation: Check for correct dimensionality
    if img.ndim != 3 or img.shape[2] < 2:
        # If image is not 3-channel or doesn't have at least 2 channels for correlation, return 0.0
        return 0.0

    # Extract relevant channels as float32 for calculation
    # Channel 0 is Actin, Channel 1 is Tubulin
    actin = img[:, :, 0].astype(np.float32)
    tubulin = img[:, :, 1].astype(np.float32)

    # Determine the segmentation mask to use
    # We need a cell mask (covering the cytoplasm) to define the region of interest for correlation.
    cell_labels = None

    if len(segmentation_masks) > 0:
        # Use the first provided mask. Assuming it's a label mask or binary mask.
        # If multiple masks are provided (e.g., nuclei, cells), usually the cell/cytoplasm mask is preferred for cytoskeleton analysis.
        # Without specific metadata on mask order, we default to the first one available.
        mask_input = segmentation_masks[0]
        
        # Ensure mask shape matches image spatial dimensions
        if mask_input.shape == img.shape[:2]:
            if np.issubdtype(mask_input.dtype, np.integer) and mask_input.max() > 1:
                # It's likely an instance label mask
                cell_labels = mask_input
            else:
                # It's likely a binary mask, label it
                cell_labels = label(mask_input > 0)
    
    # Fallback: If no valid mask provided, generate one from the image
    if cell_labels is None:
        # Create a combined intensity image for thresholding (Actin + Tubulin)
        combined_intensity = actin + tubulin
        
        # Simple background segmentation
        try:
            thresh = threshold_otsu(combined_intensity)
            binary_mask = combined_intensity > thresh
            # Remove small noise and label
            cell_labels = label(binary_mask)
        except Exception:
            # Fallback for extremely low contrast or empty images
            return 0.0

    # Compute Pearson correlation per cell
    correlations = []
    
    # Get properties for each labeled region
    props = regionprops(cell_labels)
    
    for prop in props:
        # Get coordinates for the current cell
        coords = prop.coords
        
        # Extract pixel values for this cell from both channels
        # coords[:, 0] are y-indices, coords[:, 1] are x-indices
        actin_vals = actin[coords[:, 0], coords[:, 1]]
        tubulin_vals = tubulin[coords[:, 0], coords[:, 1]]
        
        # Skip very small regions to avoid statistical noise
        if len(actin_vals) < 10:
            continue
            
        # Check for variance. Correlation is undefined if variance is 0 (flat signal).
        actin_std = np.std(actin_vals)
        tubulin_std = np.std(tubulin_vals)
        
        if actin_std < 1e-6 or tubulin_std < 1e-6:
            # If one channel is flat, correlation is technically undefined. 
            # We skip this cell rather than treating it as 0 correlation.
            continue
            
        # Calculate Pearson correlation coefficient
        # np.corrcoef returns a matrix [[1, r], [r, 1]], we want the off-diagonal element [0, 1]
        r = np.corrcoef(actin_vals, tubulin_vals)[0, 1]
        
        if not np.isnan(r):
            correlations.append(r)

    # Aggregate results
    if not correlations:
        return 0.0
        
    result = np.mean(correlations)
    
    return float(result)
