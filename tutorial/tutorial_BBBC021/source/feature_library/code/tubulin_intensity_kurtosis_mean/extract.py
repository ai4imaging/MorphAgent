def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import stats
    from scipy import ndimage
    from skimage.measure import label
    from skimage.filters import threshold_otsu
    from skimage.morphology import dilation, disk

    # 1. Data Loading and Validation
    # Convert to float64 for precision during statistical moment calculations
    arr = np.asarray(img, dtype=np.float64)

    # Check dimensionality
    # Expected: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # 2. Channel Selection
    # Channel 1 is Tubulin (Green) according to dataset description
    tubulin_channel = arr[..., 1]
    
    # Channel 2 is DAPI (Blue), useful for fallback segmentation
    dapi_channel = arr[..., 2]

    # 3. Segmentation Handling
    # We need a mask that defines individual cells (or at least cytoplasmic regions)
    labeled_mask = None

    if len(segmentation_masks) > 0:
        # If masks are provided, try to find a suitable one.
        # Often masks are passed as (nuclei, cells) or just (nuclei).
        # We prefer the largest mask (likely cells) if multiple are present.
        # If only one mask is present, we use it.
        
        # Simple heuristic: use the last mask provided, assuming it might be the most inclusive (cell body),
        # or check for the one with the largest foreground area.
        best_mask = segmentation_masks[0]
        max_area = np.sum(best_mask > 0)
        
        for mask in segmentation_masks[1:]:
            current_area = np.sum(mask > 0)
            if current_area > max_area:
                max_area = current_area
                best_mask = mask
        
        labeled_mask = best_mask.astype(int)
        
    else:
        # Fallback: Generate a crude segmentation if no masks are provided
        # Strategy: Threshold Tubulin channel to find biological material
        try:
            # Smooth slightly to reduce noise
            smooth_tubulin = ndimage.gaussian_filter(tubulin_channel, sigma=2)
            
            # Calculate threshold (Otsu)
            thresh = threshold_otsu(smooth_tubulin)
            binary_mask = smooth_tubulin > thresh
            
            # Label connected components
            labeled_mask, num_features = ndimage.label(binary_mask)
        except Exception:
            # If Otsu fails (e.g., empty image), return 0.0
            return 0.0

    # 4. Feature Computation: Mean Kurtosis of Tubulin Intensity
    # We iterate over each cell, compute kurtosis, and then average.
    
    # Get unique labels (excluding 0 which is background)
    unique_labels = np.unique(labeled_mask)
    if len(unique_labels) <= 1: # Only background exists
        return 0.0
    
    kurtosis_values = []

    # Optimization: Instead of masking the whole array in a loop (slow),
    # use ndimage.labeled_comprehension or extract slices if possible.
    # Given 512x512, a loop with boolean masking is acceptable but let's be robust.
    
    # We use ndimage.find_objects to get bounding boxes, which speeds up masking significantly
    slices = ndimage.find_objects(labeled_mask)
    
    for i, sl in enumerate(slices):
        if sl is None:
            continue
            
        label_idx = i + 1 # find_objects returns slices for labels 1, 2, ...
        
        # Extract the local region for the specific cell
        mask_slice = labeled_mask[sl]
        img_slice = tubulin_channel[sl]
        
        # Get pixels belonging strictly to this cell
        cell_pixels = img_slice[mask_slice == label_idx]
        
        # Filter out small artifacts
        if len(cell_pixels) < 10:
            continue
            
        # Check variance to avoid division by zero in kurtosis calculation
        # If variance is 0, kurtosis is undefined (or technically -3 depending on def, but practically useless)
        if np.std(cell_pixels) == 0:
            continue
            
        # Calculate Kurtosis
        # fisher=True: Excess Kurtosis (Normal distribution = 0.0)
        # bias=False: Unbiased estimator for sample statistics
        k = stats.kurtosis(cell_pixels, fisher=True, bias=False)
        
        # Check for NaN/Inf which can happen with very small numerical instabilities
        if np.isfinite(k):
            kurtosis_values.append(k)

    # 5. Aggregation
    if not kurtosis_values:
        return 0.0
        
    result = np.mean(kurtosis_values)
    
    return float(result)
