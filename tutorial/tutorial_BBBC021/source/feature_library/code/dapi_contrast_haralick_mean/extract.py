def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.feature import graycomatrix, graycoprops
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    
    # 1. Data Preparation and Validation
    # Ensure input is a numpy array
    arr = np.asarray(img)
    
    # Check dimensionality and extract DAPI channel
    # Dataset spec: (512, 512, 3), Channel 2 is DAPI (Blue)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        dapi_channel = arr[:, :, 2]
    elif arr.ndim == 2:
        # Fallback if only one channel is passed (unlikely based on spec but safe)
        dapi_channel = arr
    else:
        return 0.0

    # Ensure uint8 for GLCM (0-255)
    # If float, normalize and convert. If integer but not uint8, clip and convert.
    if dapi_channel.dtype != np.uint8:
        if np.issubdtype(dapi_channel.dtype, np.floating):
            # Normalize float [0,1] to [0,255]
            dapi_channel = np.clip(dapi_channel * 255, 0, 255).astype(np.uint8)
        else:
            # Clip other integer types to uint8 range
            dapi_channel = np.clip(dapi_channel, 0, 255).astype(np.uint8)

    # 2. Mask Generation (ROI Definition)
    # We need to identify nuclei to compute texture specifically on them
    labeled_mask = None
    
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use provided segmentation mask
        mask_input = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions
        if mask_input.shape[:2] == dapi_channel.shape[:2]:
            # If mask is already labeled (int), use it. If boolean, label it.
            if np.issubdtype(mask_input.dtype, np.integer) and np.max(mask_input) > 1:
                labeled_mask = mask_input
            else:
                labeled_mask = label(mask_input > 0)
    
    # Fallback: Automatic segmentation if no valid mask provided
    if labeled_mask is None:
        try:
            # Check if image is not empty/constant
            if np.min(dapi_channel) == np.max(dapi_channel):
                return 0.0
            thresh = threshold_otsu(dapi_channel)
            binary_mask = dapi_channel > thresh
            labeled_mask = label(binary_mask)
        except Exception:
            return 0.0

    # 3. Feature Extraction
    # We compute Haralick Contrast per nucleus and average it.
    # Computing on the whole image introduces background artifacts.
    
    regions = regionprops(labeled_mask, intensity_image=dapi_channel)
    
    if not regions:
        return 0.0

    contrast_scores = []
    
    # GLCM parameters
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/2] # 0, 45, 90, 135 degrees
    levels = 256
    
    for region in regions:
        # Skip very small regions (noise)
        if region.area < 10:
            continue
            
        # Extract the bounding box intensity image of the nucleus
        # Note: This box includes some background pixels (0) around the nucleus shape.
        # While masked GLCM is ideal, standard skimage GLCM on the bounding box 
        # is a robust and standard approximation for high-content screening features.
        patch = region.intensity_image
        
        # Ensure patch is uint8 (regionprops might return different types depending on input)
        if patch.dtype != np.uint8:
             patch = patch.astype(np.uint8)

        try:
            # Compute GLCM
            glcm = graycomatrix(patch, distances=distances, angles=angles, 
                                levels=levels, symmetric=True, normed=True)
            
            # Compute Contrast
            # Returns array of shape (len(distances), len(angles))
            contrasts = graycoprops(glcm, 'contrast')
            
            # Average across all 4 directions for rotational invariance
            mean_contrast_for_cell = np.mean(contrasts)
            contrast_scores.append(mean_contrast_for_cell)
            
        except Exception:
            continue

    # 4. Aggregation
    if not contrast_scores:
        return 0.0
        
    # Return the mean contrast across all nuclei in the image
    result = np.mean(contrast_scores)
    
    return float(result)
