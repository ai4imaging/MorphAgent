def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.feature import graycomatrix
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    
    # 1. Input Validation and Channel Extraction
    # Ensure input is a numpy array
    img = np.asarray(img)
    
    # Check dimensionality and extract DAPI channel (Channel 2)
    # Dataset is (512, 512, 3), Channel 2 is DAPI (Blue)
    if img.ndim == 3 and img.shape[2] >= 3:
        dapi_channel = img[..., 2]
    elif img.ndim == 2:
        # Fallback if only one channel is passed (unlikely based on spec, but safe)
        dapi_channel = img
    else:
        return 0.0

    # Ensure dapi_channel is uint8 for GLCM (0-255)
    # If it's float, normalize and convert. If it's integer but not uint8, cast.
    if dapi_channel.dtype != np.uint8:
        if np.issubdtype(dapi_channel.dtype, np.floating):
            # Normalize float [0,1] to [0,255]
            dapi_channel = np.clip(dapi_channel * 255, 0, 255).astype(np.uint8)
        else:
            # Clip and cast other integer types
            dapi_channel = np.clip(dapi_channel, 0, 255).astype(np.uint8)

    # 2. Segmentation Handling
    # Determine the nuclear mask
    nuclear_mask = None
    
    # Check if segmentation masks are provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Assume the first mask is the relevant one (nuclei) or check logic if multiple
        # Based on typical ordering, nuclei is often primary. We use the first one.
        nuclear_mask = segmentation_masks[0]
        
        # Ensure mask is labeled (integers > 0 for objects)
        # If binary (0 and 1), label it. If already labeled, keep it.
        if nuclear_mask.max() == 1:
            nuclear_mask = label(nuclear_mask)
    else:
        # Fallback: Generate mask using Otsu thresholding on DAPI channel
        try:
            thresh = threshold_otsu(dapi_channel)
            binary_mask = dapi_channel > thresh
            nuclear_mask = label(binary_mask)
        except Exception:
            # If thresholding fails (e.g., constant image), return 0
            return 0.0

    # 3. Feature Computation: Haralick Entropy per Nucleus
    entropy_values = []
    
    # GLCM parameters
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    levels = 256
    
    # Iterate over each segmented nucleus
    props = regionprops(nuclear_mask, intensity_image=dapi_channel)
    
    for prop in props:
        # Skip very small regions to avoid GLCM errors or noise
        if prop.area < 10:
            continue
            
        # Extract the intensity image of the nucleus (bounding box)
        # prop.image is the binary mask within the bbox
        # prop.intensity_image is the intensity within the bbox
        # We need to be careful: the bbox contains background pixels (0).
        # GLCM will count these 0-0 transitions if we aren't careful.
        # However, masking perfectly in GLCM is hard. 
        # Standard approach: Compute GLCM on the rectangular patch.
        # Since we want texture *inside* the nucleus, the background in the bbox 
        # might dilute the result slightly, but it's a consistent approximation.
        
        roi = prop.intensity_image
        
        # Ensure ROI is uint8
        if roi.dtype != np.uint8:
             roi = roi.astype(np.uint8)

        try:
            # Compute GLCM
            # symmetric=True, normed=True
            glcm = graycomatrix(roi, distances=distances, angles=angles, 
                                levels=levels, symmetric=True, normed=True)
            
            # GLCM shape: (levels, levels, num_distances, num_angles)
            # We want to compute entropy.
            # Entropy formula: - sum(p * log2(p))
            
            # Avoid log(0) by adding epsilon
            epsilon = 1e-10
            
            # Calculate entropy for each angle/distance pair
            # We can vectorize this calculation
            p = glcm
            p_norm = p / (np.sum(p, axis=(0, 1), keepdims=True) + epsilon)
            
            # Compute entropy map
            with np.errstate(divide='ignore', invalid='ignore'):
                log_p = np.log2(p_norm + epsilon)
                entropy_per_angle = -np.sum(p_norm * log_p, axis=(0, 1))
            
            # Average entropy across the 4 angles (rotational invariance)
            mean_entropy_nucleus = np.mean(entropy_per_angle)
            
            if np.isfinite(mean_entropy_nucleus):
                entropy_values.append(mean_entropy_nucleus)
                
        except Exception:
            continue

    # 4. Aggregation
    if not entropy_values:
        return 0.0
        
    # Return the mean entropy across all nuclei in the image
    result = np.mean(entropy_values)
    
    return float(result)
