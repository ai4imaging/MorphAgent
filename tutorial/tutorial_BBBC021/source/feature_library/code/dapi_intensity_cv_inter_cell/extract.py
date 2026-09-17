def extract(img, *segmentation_masks):
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    from skimage.measure import label

    # 1. Input Handling and DAPI Channel Extraction
    # The dataset description specifies Channel 2 (Blue) is DAPI (Nucleus).
    # Image shape is (512, 512, 3).
    img_arr = np.asarray(img)
    
    # Check for valid dimensions
    if img_arr.ndim != 3 or img_arr.shape[2] < 3:
        return 0.0
        
    # Extract DAPI channel and convert to float64 to prevent overflow during summation
    dapi_channel = img_arr[:, :, 2].astype(np.float64)

    # 2. Segmentation Mask Handling
    # We need an instance segmentation mask (labeled regions) to identify individual cells.
    labeled_mask = None
    
    # Check if external masks are provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        mask_input = segmentation_masks[0]
        
        # Ensure mask matches image spatial dimensions
        if mask_input.shape[:2] == dapi_channel.shape:
            # If the mask is binary (max label is 1), label connected components
            if mask_input.max() <= 1:
                labeled_mask = label(mask_input > 0)
            else:
                # Assume it's already an instance mask
                labeled_mask = mask_input.astype(int)
    
    # Fallback: If no valid mask provided, generate one using Otsu thresholding on DAPI
    if labeled_mask is None:
        try:
            # Simple background check to avoid thresholding empty images
            if dapi_channel.max() == dapi_channel.min():
                return 0.0
                
            thresh = threshold_otsu(dapi_channel)
            binary_mask = dapi_channel > thresh
            labeled_mask = label(binary_mask)
        except Exception:
            return 0.0

    # 3. Feature Computation: Integrated Intensity per Cell
    # Get unique labels (excluding 0 which is background)
    unique_labels = np.unique(labeled_mask)
    if len(unique_labels) <= 1: # Only background exists
        return 0.0
    
    # Remove background label (0)
    unique_labels = unique_labels[unique_labels != 0]
    
    # Calculate integrated intensity (sum of pixel values) for each nucleus
    # scipy.ndimage.sum is efficient for this
    integrated_intensities = ndimage.sum(dapi_channel, labeled_mask, unique_labels)
    
    # 4. Statistical Calculation (Coefficient of Variation)
    # We need at least 2 cells to calculate variance meaningfully
    if len(integrated_intensities) < 2:
        return 0.0
        
    mean_intensity = np.mean(integrated_intensities)
    std_intensity = np.std(integrated_intensities, ddof=1) # Use sample standard deviation
    
    # Avoid division by zero
    if mean_intensity <= 0:
        return 0.0
        
    cv = std_intensity / mean_intensity

    return float(cv)
