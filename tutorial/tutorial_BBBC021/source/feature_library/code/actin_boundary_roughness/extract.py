def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops, perimeter
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_closing, disk, convex_hull_image

    # 1. Data Loading and Validation
    # Ensure input is a numpy array
    arr = np.asarray(img)
    
    # Check dimensionality and select the Actin channel (Channel 0)
    # Dataset is (512, 512, 3) RGB TIFFs. Channel 0 = Red = Actin.
    if arr.ndim == 3 and arr.shape[2] >= 1:
        actin_channel = arr[..., 0]  # Select Red/Actin channel
    elif arr.ndim == 2:
        # Fallback if single channel provided (unlikely given description but safe)
        actin_channel = arr
    else:
        return 0.0

    # Normalize to float [0, 1] for processing
    if actin_channel.dtype == np.uint8:
        actin_channel = actin_channel.astype(np.float32) / 255.0
    else:
        # Robust normalization for other types
        min_val, max_val = np.min(actin_channel), np.max(actin_channel)
        if max_val > min_val:
            actin_channel = (actin_channel - min_val) / (max_val - min_val)
        else:
            actin_channel = np.zeros_like(actin_channel, dtype=np.float32)

    # 2. Mask Generation
    # Use provided segmentation masks if available, otherwise generate one from Actin
    labeled_mask = None
    
    if segmentation_masks and len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use the first available mask
        mask_input = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions
        if mask_input.shape[:2] == actin_channel.shape[:2]:
            labeled_mask = mask_input.astype(int)
            
    if labeled_mask is None:
        # Generate mask from Actin channel
        # Smooth slightly to reduce pixel noise affecting thresholding
        smoothed = ndimage.gaussian_filter(actin_channel, sigma=1.0)
        
        # Thresholding
        try:
            thresh = threshold_otsu(smoothed)
            binary_mask = smoothed > thresh
        except Exception:
            # Fallback if image is uniform
            return 0.0
            
        # Morphological cleanup
        # Close gaps
        binary_mask = binary_closing(binary_mask, disk(2))
        # Remove small artifacts
        label_img = label(binary_mask)
        regions = regionprops(label_img)
        
        # Filter small objects (< 100 pixels)
        mask_cleaned = np.zeros_like(binary_mask, dtype=bool)
        for prop in regions:
            if prop.area >= 100:
                mask_cleaned[label_img == prop.label] = True
                
        labeled_mask = label(mask_cleaned)

    # 3. Feature Computation: Perimeter / Convex Perimeter
    # We compute this per cell and take the mean
    
    props = regionprops(labeled_mask)
    roughness_values = []

    for prop in props:
        # Skip very small objects or background (label 0 is handled by regionprops skipping)
        if prop.area < 50:
            continue

        # 1. Actual Perimeter
        # regionprops calculates perimeter based on the boundary pixels
        actual_perimeter = prop.perimeter
        
        if actual_perimeter == 0:
            continue

        # 2. Convex Perimeter
        # prop.convex_image gives the binary convex hull of the object (cropped to bbox)
        # We calculate the perimeter of this convex shape
        convex_img = prop.convex_image
        
        # Calculate perimeter of the convex hull
        # We use the same perimeter function from skimage to ensure consistency in measurement units
        conv_perimeter = perimeter(convex_img)
        
        # Avoid division by zero (though convex perimeter of an area>0 object should be >0)
        if conv_perimeter <= 0:
            continue

        # 3. Roughness Ratio
        # Ratio >= 1.0. Higher means more ruffled/irregular.
        roughness = actual_perimeter / conv_perimeter
        roughness_values.append(roughness)

    # 4. Aggregation
    if not roughness_values:
        return 0.0
        
    # Return the mean roughness of the cell population
    result = np.mean(roughness_values)
    
    return float(result)
