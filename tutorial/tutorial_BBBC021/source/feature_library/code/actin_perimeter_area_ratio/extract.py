def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_closing, binary_opening, disk

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3) - Channel 0 is Actin (Red)
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Extract Actin channel (Channel 0)
        actin_channel = arr[:, :, 0]
    elif arr.ndim == 2:
        # Fallback if single channel passed (unlikely based on desc, but safe)
        actin_channel = arr
    else:
        return 0.0

    # Intensity normalization for processing
    # Normalize to [0, 1] for consistent thresholding behavior
    vmax = np.percentile(actin_channel, 99.5) if actin_channel.size > 0 else 1.0
    if vmax > 0:
        actin_channel = actin_channel / vmax
    actin_channel = np.clip(actin_channel, 0.0, 1.0)

    # --- Segmentation Logic ---
    # We need a mask defining the cell boundaries to calculate perimeter and area.
    # We prioritize provided masks, but if they are absent or seem to be nuclear masks (small area),
    # we derive a mask from the Actin channel itself, as this feature is specifically about Actin morphology.
    
    labeled_mask = None
    
    # Check if valid segmentation masks are provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        mask_candidate = segmentation_masks[0]
        # Ensure mask is 2D
        if mask_candidate.ndim == 3:
            mask_candidate = np.max(mask_candidate, axis=2) # Project if 3D
        
        # Basic check: if the mask is empty, ignore it
        if np.any(mask_candidate):
            labeled_mask = mask_candidate.astype(int)

    # Fallback: Compute segmentation from Actin channel if no mask provided
    if labeled_mask is None:
        # Smooth image to reduce noise for thresholding
        smoothed = ndimage.gaussian_filter(actin_channel, sigma=2.0)
        
        # Determine threshold
        try:
            thresh = threshold_otsu(smoothed)
        except ValueError:
            # Handle case where image is uniform (e.g., all black)
            return 0.0
            
        binary_mask = smoothed > thresh
        
        # Clean up mask
        # Closing to fill holes inside cells
        binary_mask = binary_closing(binary_mask, disk(2))
        # Opening to remove small noise specks
        binary_mask = binary_opening(binary_mask, disk(1))
        
        # Label connected components
        labeled_mask = label(binary_mask)

    # --- Feature Computation ---
    # Calculate properties for each object
    props = regionprops(labeled_mask)
    
    ratios = []
    
    for prop in props:
        # Filter out very small objects (likely debris/noise)
        if prop.area < 50:
            continue
            
        # Calculate Perimeter / Area ratio
        # Perimeter: length of object boundary
        # Area: number of pixels in object
        # High ratio = complex shape / filopodia / protrusions
        # Low ratio = round / compact
        
        # Avoid division by zero (though area < 50 check handles most cases)
        if prop.area > 0:
            ratio = prop.perimeter / prop.area
            ratios.append(ratio)

    # --- Aggregation ---
    # Return the mean ratio across all valid cells in the image
    if len(ratios) == 0:
        return 0.0
    
    result = np.mean(ratios)

    return float(result)
