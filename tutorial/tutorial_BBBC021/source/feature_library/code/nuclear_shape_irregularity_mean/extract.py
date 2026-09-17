def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.segmentation import clear_border
    from skimage.morphology import binary_opening, disk

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # 1. Data Dimensions & Channel Selection
    # Dataset is (512, 512, 3). Channel 2 is DAPI (Nucleus).
    # If the image is 2D (H, W), assume it's a single channel or pre-processed.
    # If 3D (H, W, C), extract channel 2.
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Extract DAPI channel (Index 2)
        dapi_img = arr[..., 2]
    elif arr.ndim == 2:
        # Fallback: assume the input is already the channel of interest or a grayscale image
        dapi_img = arr
    else:
        # Unexpected format
        return 0.0

    # 2. Segmentation Logic
    # Check if a valid segmentation mask is provided.
    # We prioritize external masks if they exist.
    labeled_mask = None
    
    # Iterate through provided masks to find a suitable one (assuming 2D masks)
    if len(segmentation_masks) > 0:
        for mask in segmentation_masks:
            if mask is not None and mask.ndim == 2 and mask.shape == dapi_img.shape:
                # Assume the mask provided is a label mask or binary mask
                # If it's binary (max value is 1), label it. If it has many labels, use as is.
                if mask.max() <= 1:
                    labeled_mask = label(mask > 0)
                else:
                    labeled_mask = mask.astype(int)
                break
    
    # Fallback: Internal Segmentation if no valid mask provided
    if labeled_mask is None:
        # Normalize for thresholding
        # Robust min/max scaling
        p_min, p_max = np.percentile(dapi_img, (1, 99))
        if p_max > p_min:
            norm_img = (dapi_img - p_min) / (p_max - p_min)
        else:
            norm_img = dapi_img # Should be flat
            
        norm_img = np.clip(norm_img, 0, 1)
        
        # Thresholding (Otsu)
        try:
            thresh = threshold_otsu(norm_img)
            binary_mask = norm_img > thresh
        except ValueError:
            # If image is uniform (e.g. all black), otsu fails
            return 0.0

        # Morphological cleanup
        # Remove small noise
        binary_mask = binary_opening(binary_mask, disk(2))
        
        # Label connected components
        labeled_mask = label(binary_mask)

    # 3. Feature Computation: Shape Irregularity
    # Irregularity = 1 - Circularity
    # Circularity = (4 * pi * Area) / (Perimeter^2)
    
    # Clear objects touching the border (incomplete shapes yield invalid circularity)
    labeled_mask = clear_border(labeled_mask)
    
    props = regionprops(labeled_mask)
    
    irregularity_scores = []
    
    for prop in props:
        # Filter small debris
        if prop.area < 50:
            continue
            
        # Perimeter of a single pixel is 0, avoid division by zero
        if prop.perimeter == 0:
            continue
            
        # Calculate Circularity
        # Note: For discrete pixels, perimeter can be estimated in different ways.
        # regionprops uses an approximation based on the boundary pixels.
        circularity = (4 * np.pi * prop.area) / (prop.perimeter ** 2)
        
        # Clamp circularity to [0, 1] range (discrete geometry can sometimes slightly exceed 1.0 for perfect shapes)
        circularity = min(1.0, max(0.0, circularity))
        
        # Irregularity
        irregularity = 1.0 - circularity
        irregularity_scores.append(irregularity)

    # 4. Aggregation
    if not irregularity_scores:
        return 0.0
        
    result = np.mean(irregularity_scores)

    return float(result)
