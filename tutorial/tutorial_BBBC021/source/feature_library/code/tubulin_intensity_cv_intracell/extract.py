def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage

    # 1. Input Validation and Preparation
    # Convert image to float32 for precision
    arr = np.asarray(img, dtype=np.float32)

    # Check dimensionality
    # Expected shape: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract Tubulin Channel (Channel 1 - Green)
    # Channel 0: Actin (Red), Channel 1: Tubulin (Green), Channel 2: DAPI (Blue)
    tubulin_channel = arr[..., 1]

    # 2. Handle Segmentation Masks
    # We need at least one mask to define "intracell" regions.
    if not segmentation_masks:
        return float('nan')

    # Select the most appropriate mask.
    # Usually, if multiple masks are provided, one might be nuclei and another cytoplasm/cells.
    # Tubulin is cytoplasmic. We prefer the mask that covers the largest area (likely the whole cell mask).
    selected_mask = None
    max_area = -1

    for mask in segmentation_masks:
        if mask is None:
            continue
        
        # Ensure mask matches image spatial dimensions
        if mask.shape != tubulin_channel.shape:
            continue
            
        # Calculate total area of labeled regions (excluding background 0)
        current_area = np.sum(mask > 0)
        if current_area > max_area:
            max_area = current_area
            selected_mask = mask

    if selected_mask is None:
        return float('nan')

    # 3. Compute Feature: Coefficient of Variation (CV) per cell
    # CV = Standard Deviation / Mean
    
    # Get unique labels (excluding background 0)
    labels = np.unique(selected_mask)
    labels = labels[labels > 0]

    if len(labels) == 0:
        return float('nan')

    # Use scipy.ndimage for efficient calculation over labeled regions
    # mean_intensity per label
    means = ndimage.mean(tubulin_channel, labels=selected_mask, index=labels)
    # standard deviation per label
    stds = ndimage.standard_deviation(tubulin_channel, labels=selected_mask, index=labels)

    # Calculate CV for each cell
    # Handle division by zero if a cell has mean intensity 0 (though unlikely in valid cells)
    cv_values = []
    
    # Ensure means and stds are iterable (if only 1 label, ndimage returns scalar)
    if not isinstance(means, (list, np.ndarray)):
        means = [means]
        stds = [stds]

    for m, s in zip(means, stds):
        if m > 1e-6: # Avoid division by zero or extremely small noise
            cv = s / m
            cv_values.append(cv)
        else:
            # If mean is 0, the region is purely black. Variance is 0. CV is technically undefined or 0.
            # We skip undefined cases to avoid skewing the average with artifacts.
            pass

    # 4. Aggregate Results
    if not cv_values:
        return 0.0

    # The feature is the mean of the CVs across all cells in the image
    result = np.mean(cv_values)

    return float(result)
