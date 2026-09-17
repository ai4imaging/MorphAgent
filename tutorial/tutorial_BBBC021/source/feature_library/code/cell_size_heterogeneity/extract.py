def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    
    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)
    
    # 1. Determine the Segmentation Mask to use
    # We need an instance segmentation mask where each cell has a unique integer ID.
    # If provided, we use the segmentation masks. If not, we generate a fallback mask from the image.
    
    labels = None
    
    if len(segmentation_masks) > 0:
        # Strategy: Use the provided mask.
        # If multiple masks are provided, we ideally want the "whole cell" mask (cytoplasm/cell body)
        # rather than just nuclei, as cell size heterogeneity is best observed in the whole cell.
        # Without metadata, we assume the mask with the largest total foreground area is the cell mask.
        
        best_mask = None
        max_foreground = -1
        
        for mask in segmentation_masks:
            if mask is None:
                continue
            # Ensure mask is integer for analysis
            curr_mask = np.asarray(mask, dtype=np.int32)
            foreground_area = np.count_nonzero(curr_mask)
            
            if foreground_area > max_foreground:
                max_foreground = foreground_area
                best_mask = curr_mask
        
        if best_mask is not None:
            # Check if it's a binary mask (0 and 1) or instance mask (0, 1, 2, ...)
            if best_mask.max() <= 1:
                # It's binary, we need to label connected components to get instances
                labels, _ = ndimage.label(best_mask)
            else:
                # It's already an instance mask
                labels = best_mask

    # 2. Fallback: Generate mask from raw image if no valid segmentation provided
    if labels is None:
        # Use Channel 0 (Actin/Red) or Channel 1 (Tubulin/Green) as they define cell shape better than DAPI.
        # We'll use the maximum intensity projection of Actin and Tubulin to capture the full cell body.
        if arr.ndim == 3 and arr.shape[2] >= 2:
            # Combine Actin (0) and Tubulin (1)
            cell_signal = np.maximum(arr[..., 0], arr[..., 1])
        elif arr.ndim == 3:
            cell_signal = np.mean(arr, axis=2)
        else:
            cell_signal = arr
            
        # Smooth to reduce noise
        cell_signal = ndimage.gaussian_filter(cell_signal, sigma=2)
        
        # Thresholding
        try:
            thresh = threshold_otsu(cell_signal)
            binary_mask = cell_signal > thresh
        except Exception:
            # Fallback for extremely low contrast or empty images
            binary_mask = cell_signal > np.mean(cell_signal)

        # Morphological cleanup (remove small noise)
        binary_mask = ndimage.binary_opening(binary_mask, structure=np.ones((3,3)))
        
        # Label instances
        labels, _ = ndimage.label(binary_mask)

    # 3. Compute Cell Areas
    # We use bincount to count pixels for each label index.
    # labels.flat flattens the array.
    # We skip index 0 because that is the background.
    if labels.max() == 0:
        return 0.0
        
    areas = np.bincount(labels.ravel())
    
    # Remove background (index 0)
    areas = areas[1:]
    
    # 4. Filter Noise
    # Remove very small artifacts that might skew the standard deviation (e.g., < 50 pixels)
    # 50 pixels is roughly a 7x7 square, reasonable minimum for a cell at 512x512 resolution
    min_area_threshold = 50
    valid_areas = areas[areas > min_area_threshold]
    
    # 5. Compute Feature: Standard Deviation of Areas
    # If we have fewer than 2 cells, standard deviation is not well-defined or is 0.
    if len(valid_areas) < 2:
        return 0.0
        
    # Calculate standard deviation
    # This quantifies the heterogeneity in cell sizes (e.g., mixture of giant senescent cells and normal cells)
    size_heterogeneity = np.std(valid_areas)
    
    return float(size_heterogeneity)
