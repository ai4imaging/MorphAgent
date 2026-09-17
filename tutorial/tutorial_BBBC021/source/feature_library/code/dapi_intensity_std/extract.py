def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    
    # 1. Data Loading and Channel Selection
    # Convert to float32 for calculations, but keep original intensity scale (0-255) initially
    # for consistent texture interpretation, or normalize to 0-1. 
    # Standard practice for texture features often uses the original dynamic range or normalized 0-1.
    # Here we normalize to 0-1 to be robust against exposure differences if any, 
    # though the prompt suggests uint8 input.
    arr = np.asarray(img, dtype=np.float32)
    
    # Handle Dimensions: Expecting (512, 512, 3)
    # Channel 2 is DAPI (Blue)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        dapi_channel = arr[:, :, 2]
    elif arr.ndim == 2:
        # If 2D, assume it's already a single channel (though less likely given description)
        dapi_channel = arr
    else:
        return 0.0

    # Normalize intensity to [0, 1] for consistent standard deviation scale
    # This makes the feature comparable across images with slightly different exposure times
    max_val = np.max(dapi_channel)
    if max_val > 0:
        dapi_channel = dapi_channel / max_val
    
    # 2. Mask Handling
    # We need a nuclear mask.
    # If masks are provided, we assume the second one is nuclei (common convention: cell, nuclei)
    # or if only one is provided, we use that.
    # If no masks, we generate one.
    
    mask = None
    
    if len(segmentation_masks) > 0:
        # Heuristic: If multiple masks, often the second one is nuclei in some pipelines,
        # but without strict metadata, we usually check for the one that best overlaps DAPI.
        # However, simpler logic for this constrained task: use the last mask (often nuclei are the finest segmentation)
        # or just the first if only one.
        # Let's try to use the first available mask, assuming it segments the objects of interest.
        # If specific ordering was guaranteed (e.g. cell, nucleus), we would pick by index.
        # Given the prompt's "seg_mask_1, seg_mask_2...", we will use the first one provided.
        candidate_mask = segmentation_masks[0]
        
        # Ensure mask shape matches image (handle potential 3D vs 2D mismatch)
        if candidate_mask.shape == dapi_channel.shape:
            mask = candidate_mask
        elif candidate_mask.ndim == 3 and candidate_mask.shape[:2] == dapi_channel.shape:
             # If mask is 3D (e.g. one-hot), take max projection or first channel
             mask = np.max(candidate_mask, axis=2)
    
    # Fallback: Generate mask if none provided or invalid
    if mask is None:
        # Simple Otsu thresholding on DAPI channel
        # Blur slightly to reduce noise
        blurred = ndimage.gaussian_filter(dapi_channel, sigma=2.0)
        try:
            thresh = threshold_otsu(blurred)
            mask = blurred > thresh
        except Exception:
            # Fallback if image is uniform (e.g. all black)
            return 0.0

    # 3. Feature Computation: Mean of Per-Nucleus Standard Deviation
    # We calculate the std dev of intensities *within* each nucleus, then average across nuclei.
    # This measures the average texture heterogeneity of the nuclei.
    
    # Ensure mask is labeled (integers for distinct objects)
    labeled_mask = label(mask)
    
    # Get properties for each region
    regions = regionprops(labeled_mask, intensity_image=dapi_channel)
    
    if not regions:
        return 0.0
    
    std_devs = []
    for region in regions:
        # region.image_intensity gives the intensity values within the bounding box
        # region.image gives the binary mask within the bounding box
        # We need pixels strictly inside the mask
        intensities = region.image_intensity[region.image]
        
        if intensities.size > 1:
            # ddof=1 for sample standard deviation
            std_val = np.std(intensities, ddof=1)
            std_devs.append(std_val)
        else:
            std_devs.append(0.0)
            
    if not std_devs:
        return 0.0
        
    # Return the mean of the standard deviations
    result = np.mean(std_devs)
    
    return float(result)
