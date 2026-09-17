def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import regionprops
    from skimage.filters import threshold_otsu

    # 1. Input Validation and Preparation
    # Ensure image is float for calculations, but keep scale relative to original intensity for std dev meaning
    # We work with the original intensity scale (0-255) to keep the standard deviation interpretable
    # (e.g., "variation of 20 gray levels").
    img_arr = np.asarray(img, dtype=np.float32)

    # Check dimensionality
    if img_arr.ndim != 3 or img_arr.shape[2] != 3:
        # If not the expected (H, W, 3), try to handle potential variations or return 0
        if img_arr.ndim == 2:
            # Assume single channel is the relevant one if only 2D provided (unlikely given spec)
            actin_channel = img_arr
        else:
            return 0.0
    else:
        # Extract Channel 0 (Red) -> Actin
        actin_channel = img_arr[:, :, 0]

    # 2. Mask Handling
    labeled_mask = None
    
    # Check if valid segmentation masks are provided
    if segmentation_masks and len(segmentation_masks) > 0:
        # Iterate through masks to find a suitable 2D mask
        for mask in segmentation_masks:
            if mask is None:
                continue
            
            mask_arr = np.asarray(mask)
            
            # Handle potential extra dimensions (e.g., (512, 512, 1))
            if mask_arr.ndim == 3:
                mask_arr = np.squeeze(mask_arr)
            
            # Check if dimensions match image spatial dims
            if mask_arr.ndim == 2 and mask_arr.shape == actin_channel.shape:
                # If mask is already labeled (int > 1), use it
                if np.max(mask_arr) > 1:
                    labeled_mask = mask_arr.astype(int)
                # If mask is binary (0/1 or boolean), label it
                elif np.max(mask_arr) > 0:
                    labeled_mask, _ = ndimage.label(mask_arr > 0)
                
                # If we found a valid mask, stop looking
                if labeled_mask is not None:
                    break
    
    # Fallback: If no valid mask provided, generate a simple foreground mask
    if labeled_mask is None:
        # Simple background exclusion
        # If image is mostly black, Otsu might fail or set threshold too low.
        # We use a safe fallback for very dark images.
        if np.max(actin_channel) == 0:
            return 0.0
            
        try:
            thresh = threshold_otsu(actin_channel)
            binary_mask = actin_channel > thresh
        except:
            # Fallback for extremely low contrast images
            binary_mask = actin_channel > np.mean(actin_channel)
            
        labeled_mask, _ = ndimage.label(binary_mask)

    # 3. Feature Calculation
    # We need to calculate the standard deviation of pixel intensities *per cell*
    # and then take the mean of those standard deviations.
    
    # Get properties for each labeled region
    regions = regionprops(labeled_mask, intensity_image=actin_channel)
    
    std_devs = []
    
    for region in regions:
        # region.image_intensity gives the intensity values within the bounding box
        # region.image gives the binary mask within the bounding box
        # We select only the pixels belonging to the cell
        cell_intensities = region.image_intensity[region.image]
        
        # We need at least 2 pixels to calculate a meaningful standard deviation
        if cell_intensities.size > 1:
            std_val = np.std(cell_intensities)
            std_devs.append(std_val)
        elif cell_intensities.size == 1:
            std_devs.append(0.0)

    # 4. Aggregation
    if not std_devs:
        return 0.0
        
    # Calculate the mean of the standard deviations
    result = np.mean(std_devs)

    return float(result)
