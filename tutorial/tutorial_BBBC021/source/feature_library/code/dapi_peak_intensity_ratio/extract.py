def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu

    # 1. Input Validation and Channel Extraction
    # Dataset description: Channel 2 is DAPI (Blue)
    # Image shape is (512, 512, 3)
    if img is None or img.ndim < 3 or img.shape[2] < 3:
        return 0.0

    # Extract DAPI channel (Index 2)
    dapi_channel = img[..., 2]
    
    # Convert to float for calculations to avoid overflow/truncation
    dapi_float = dapi_channel.astype(np.float32)

    # 2. Mask Handling
    # We need a nuclear mask.
    # Strategy: Use provided segmentation if available, otherwise generate one via Otsu.
    mask = None
    
    if len(segmentation_masks) > 0:
        # Check if any mask is provided. 
        # In a typical pipeline, if multiple masks are provided, we need to identify the nuclear one.
        # Without explicit metadata, we assume the first mask or check for a mask that aligns with DAPI.
        # However, usually, the system passes relevant masks. Let's try to use the first one.
        # If the mask is labeled (int), convert to boolean for masking, or use labels directly.
        candidate_mask = segmentation_masks[0]
        if candidate_mask.shape == dapi_channel.shape:
            mask = candidate_mask
    
    # Fallback: Generate mask if none provided or invalid
    if mask is None:
        # Check if image has content (std dev > 0) to avoid Otsu error on empty images
        if np.std(dapi_float) < 1e-6:
            return 0.0
        try:
            thresh = threshold_otsu(dapi_float)
            mask = dapi_float > thresh
        except Exception:
            return 0.0

    # 3. Labeling
    # Ensure mask is integer labeled for regionprops
    if mask.dtype == bool:
        labeled_mask = label(mask)
    else:
        labeled_mask = mask.astype(int)

    # 4. Feature Computation
    # We want the ratio of Max Intensity / Mean Intensity per nucleus.
    # Then we aggregate these ratios (e.g., mean of the ratios) to get a single scalar for the image.
    
    regions = regionprops(labeled_mask, intensity_image=dapi_float)
    
    if not regions:
        return 0.0

    ratios = []
    for region in regions:
        # Filter out very small noise regions
        if region.area < 10:
            continue
            
        mean_val = region.mean_intensity
        max_val = region.max_intensity
        
        # Avoid division by zero
        if mean_val > 0:
            ratio = max_val / mean_val
            ratios.append(ratio)
    
    # 5. Aggregation
    if not ratios:
        return 0.0
        
    # Return the average peak-to-mean ratio across all nuclei in the image
    result = np.mean(ratios)

    return float(result)
