def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import closing, square

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3) - Channels: 0=Actin(R), 1=Tubulin(G), 2=DAPI(B)
    if arr.ndim != 3 or arr.shape[2] != 3:
        # If unexpected shape, try to handle or return 0.0
        if arr.ndim == 2:
            # Treat as single channel grayscale
            pass 
        else:
            return 0.0

    # Determine the segmentation mask to use
    labeled_mask = None
    
    # Strategy: Use provided masks if available, otherwise generate one from image
    if len(segmentation_masks) > 0:
        # Check masks to find a suitable one (prefer cell/cytoplasm over nucleus)
        # We don't have metadata on which mask is which, so we use a heuristic:
        # The mask with the largest total foreground area is likely the whole cell mask.
        best_mask = None
        max_area = -1
        
        for mask in segmentation_masks:
            if mask is None:
                continue
            # Ensure mask is 2D
            if mask.ndim == 3:
                curr_mask = np.max(mask, axis=2) if mask.shape[2] < 5 else np.max(mask, axis=0)
            else:
                curr_mask = mask
                
            # Count non-zero pixels
            current_area = np.count_nonzero(curr_mask)
            if current_area > max_area:
                max_area = current_area
                best_mask = curr_mask
        
        if best_mask is not None:
            # Ensure it's labeled (integers)
            if np.max(best_mask) == 1: # Binary mask
                labeled_mask = label(best_mask)
            else:
                labeled_mask = best_mask.astype(int)

    # Fallback: Generate segmentation from image if no valid mask provided
    if labeled_mask is None:
        # Use Actin (Ch0) and Tubulin (Ch1) for cell body segmentation
        # Combine them to get full cell shape
        if arr.ndim == 3 and arr.shape[2] >= 2:
            cell_signal = arr[..., 0] + arr[..., 1] # Sum Actin and Tubulin
        elif arr.ndim == 3:
            cell_signal = np.mean(arr, axis=2)
        else:
            cell_signal = arr

        # Normalize for thresholding
        if cell_signal.max() > cell_signal.min():
            cell_signal = (cell_signal - cell_signal.min()) / (cell_signal.max() - cell_signal.min())
        
        # Smooth to reduce noise
        cell_signal = ndimage.gaussian_filter(cell_signal, sigma=2)
        
        # Threshold
        try:
            thresh = threshold_otsu(cell_signal)
            binary_mask = cell_signal > thresh
        except:
            return 0.0
            
        # Morphological cleanup
        binary_mask = closing(binary_mask, square(3))
        
        # Label
        labeled_mask = label(binary_mask)

    # Compute Feature: Mean Extent
    # Extent = Area / (BoundingBox Area)
    regions = regionprops(labeled_mask)
    
    extents = []
    for region in regions:
        # Filter small artifacts (e.g., < 50 pixels)
        if region.area < 50:
            continue
        
        # region.extent is provided by skimage
        # It is defined as: Area / (rows * cols) of the bounding box
        extents.append(region.extent)

    if not extents:
        return 0.0

    result = np.mean(extents)

    return float(result)
