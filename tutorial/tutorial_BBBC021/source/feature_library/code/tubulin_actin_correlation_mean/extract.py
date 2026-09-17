def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    
    # Convert to appropriate array type
    # Image is uint8, convert to float32 for statistical calculations
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Expected shape: (512, 512, 3)
    # Channel 0: Actin (Red)
    # Channel 1: Tubulin (Green)
    # Channel 2: DAPI (Blue)
    
    if arr.ndim != 3 or arr.shape[2] < 2:
        # Cannot compute correlation between two channels if they don't exist
        return 0.0

    # Extract relevant channels
    actin = arr[:, :, 0]
    tubulin = arr[:, :, 1]

    # Determine regions of interest (Cells)
    # We need to calculate correlation *within* each cell to be biologically meaningful
    labels = None
    
    # 1. Try to use provided segmentation masks
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        mask_candidate = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions (512, 512)
        if mask_candidate.shape == arr.shape[:2]:
            labels = mask_candidate.astype(np.int32)
    
    # 2. Fallback: Generate labels if no valid mask provided
    # If no mask is available, we must generate one to avoid correlating background noise
    if labels is None:
        # Create a foreground mask based on intensity
        # Combine Actin and Tubulin to find cellular regions
        intensity_sum = actin + tubulin
        
        # Determine a threshold to separate cells from background
        # Use a safe heuristic: if image has content, threshold above background
        if intensity_sum.max() > 0:
            # Simple background exclusion: > 10% of max intensity or fixed low value
            threshold = max(np.percentile(intensity_sum, 90) * 0.1, 10.0)
            binary_mask = intensity_sum > threshold
            
            # Label connected components to treat each blob as a "cell"
            # This allows us to compute local correlations rather than one global correlation
            labels, _ = ndimage.label(binary_mask)
        else:
            return 0.0

    # Get unique object indices (excluding background 0)
    indices = np.unique(labels)
    indices = indices[indices > 0]
    
    if len(indices) == 0:
        return 0.0

    # Compute Pearson Correlation Coefficient per cell
    # Formula: Cov(A, T) / (Std(A) * Std(T))
    # Cov(A, T) = Mean(A*T) - Mean(A)*Mean(T)
    
    # We use scipy.ndimage to vectorize these calculations over all labels
    # This is much faster than looping through each cell in Python
    
    # 1. Mean(Actin) per cell
    mean_a = ndimage.mean(actin, labels=labels, index=indices)
    
    # 2. Mean(Tubulin) per cell
    mean_t = ndimage.mean(tubulin, labels=labels, index=indices)
    
    # 3. Mean(Actin * Tubulin) per cell
    product_at = actin * tubulin
    mean_at = ndimage.mean(product_at, labels=labels, index=indices)
    
    # 4. Standard Deviation of Actin per cell
    std_a = ndimage.standard_deviation(actin, labels=labels, index=indices)
    
    # 5. Standard Deviation of Tubulin per cell
    std_t = ndimage.standard_deviation(tubulin, labels=labels, index=indices)
    
    # Calculate Covariance
    covariance = mean_at - (mean_a * mean_t)
    
    # Calculate Denominator (product of std devs)
    denominator = std_a * std_t
    
    # Avoid division by zero
    # Filter out cells where either channel is flat (std=0)
    valid_mask = denominator > 1e-6
    
    if not np.any(valid_mask):
        return 0.0
        
    # Compute correlations for valid cells
    correlations = np.zeros_like(covariance)
    # Only compute where denominator is valid
    correlations[valid_mask] = covariance[valid_mask] / denominator[valid_mask]
    
    # Extract only the valid computed correlations
    valid_correlations = correlations[valid_mask]
    
    # Clip results to valid Pearson range [-1, 1] to handle potential floating point errors
    valid_correlations = np.clip(valid_correlations, -1.0, 1.0)
    
    # Return the mean correlation across all cells
    result = np.mean(valid_correlations)

    return float(result)
