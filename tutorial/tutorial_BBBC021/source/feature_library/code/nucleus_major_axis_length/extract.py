def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    
    # 1. Input Validation and Preparation
    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)
    
    # Check dimensionality: Expecting (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        # Fallback for potential 2D inputs or unexpected shapes
        if arr.ndim == 2:
            # If 2D, assume it's a single channel image. 
            # If it's DAPI (nuclei), we proceed. If it's composite, this is ambiguous, 
            # but we treat the whole image as the signal source.
            dapi_channel = arr
        else:
            return 0.0
    else:
        # Extract DAPI channel (Channel 2 based on dataset description)
        # Channel 0: Actin, Channel 1: Tubulin, Channel 2: DAPI (Nucleus)
        dapi_channel = arr[:, :, 2]

    # 2. Segmentation Logic
    labeled_mask = None
    
    # Check if segmentation masks are provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        mask_input = segmentation_masks[0]
        # Ensure mask matches spatial dimensions of the image
        if mask_input.shape[:2] == dapi_channel.shape[:2]:
            # If mask is already labeled (int type with values > 1), use it directly
            if np.issubdtype(mask_input.dtype, np.integer) and np.max(mask_input) > 1:
                labeled_mask = mask_input
            else:
                # If binary mask, label connected components
                labeled_mask = label(mask_input > 0)
    
    # Fallback: Generate segmentation from DAPI channel if no valid mask provided
    if labeled_mask is None:
        # Normalize DAPI channel for thresholding
        dapi_norm = dapi_channel
        vmax = np.percentile(dapi_norm, 99.5) if dapi_norm.size > 0 else 1.0
        if vmax > 0:
            dapi_norm = dapi_norm / vmax
        dapi_norm = np.clip(dapi_norm, 0.0, 1.0)
        
        # Apply Gaussian blur to reduce noise
        dapi_smooth = ndimage.gaussian_filter(dapi_norm, sigma=2.0)
        
        # Determine threshold
        try:
            thresh = threshold_otsu(dapi_smooth)
            binary_mask = dapi_smooth > thresh
        except Exception:
            # Fallback if image is uniform
            binary_mask = dapi_smooth > 0.1
            
        # Label connected components
        labeled_mask = label(binary_mask)

    # 3. Feature Extraction
    # Calculate properties for regions
    regions = regionprops(labeled_mask)
    
    major_axis_lengths = []
    
    for region in regions:
        # Filter small artifacts (e.g., noise dots)
        if region.area < 50:
            continue
            
        # Get major axis length
        # This is the length of the major axis of the ellipse that has the same 
        # normalized second central moments as the region.
        major_axis_lengths.append(region.major_axis_length)
    
    # 4. Aggregation
    if not major_axis_lengths:
        return 0.0
    
    # Return the mean major axis length across all valid nuclei in the image
    result = np.mean(major_axis_lengths)
    
    return float(result)
