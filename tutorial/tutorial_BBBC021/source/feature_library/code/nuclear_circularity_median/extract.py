def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    
    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Expected: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        # If not standard 3-channel image, return 0.0
        return 0.0

    # Identify the Nuclear Mask
    # Strategy: 
    # 1. Check if a segmentation mask is provided via arguments.
    # 2. If yes, use the first mask (assuming it's the primary segmentation).
    # 3. If no, generate a mask from the DAPI channel (Channel 2).

    labeled_mask = None

    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use provided mask
        mask_input = segmentation_masks[0]
        # Ensure it's 2D
        if mask_input.ndim == 3:
            mask_input = mask_input.squeeze()
        
        # If the mask is already labeled (int type with values > 1), use it directly
        # If it's binary (0 and 1/255), label it
        if np.max(mask_input) > 1:
            labeled_mask = mask_input.astype(int)
        else:
            labeled_mask = label(mask_input > 0)
    else:
        # Fallback: Segment Nuclei from DAPI channel (Channel 2)
        dapi_channel = arr[:, :, 2]
        
        # Simple preprocessing
        # Normalize to 0-1 for stability
        dapi_norm = dapi_channel / 255.0
        
        # Gaussian blur to smooth noise before thresholding
        dapi_smooth = ndimage.gaussian_filter(dapi_norm, sigma=2)
        
        # Thresholding
        try:
            thresh = threshold_otsu(dapi_smooth)
            binary_mask = dapi_smooth > thresh
            labeled_mask = label(binary_mask)
        except Exception:
            # Fallback if image is empty or uniform
            return 0.0

    # Calculate Circularity for each nucleus
    # Circularity = 4 * pi * Area / (Perimeter^2)
    
    circularity_values = []
    
    # Get region properties
    regions = regionprops(labeled_mask)
    
    for region in regions:
        area = region.area
        perimeter = region.perimeter
        
        # Filter out small artifacts (e.g., noise dots)
        # A typical nucleus in 512x512 microscopy is usually > 50 pixels
        if area < 50:
            continue
            
        # Avoid division by zero
        if perimeter == 0:
            continue
            
        # Calculate circularity
        # Value range: 0.0 to 1.0 (perfect circle)
        circ = (4 * np.pi * area) / (perimeter ** 2)
        
        # Theoretical max is 1.0, but discrete pixels can sometimes yield slightly > 1.0
        # Clip to valid range for consistency
        circ = min(circ, 1.0)
        
        circularity_values.append(circ)

    # Compute Median
    if not circularity_values:
        result = 0.0
    else:
        result = np.median(circularity_values)

    return float(result)
