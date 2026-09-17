def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.feature import graycomatrix, graycoprops
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    
    # 1. Extract the DAPI channel (Channel 2 based on dataset description)
    # The input img is (512, 512, 3)
    if img.ndim == 3 and img.shape[2] >= 3:
        dapi_channel = img[:, :, 2]
    elif img.ndim == 2:
        # Fallback if single channel passed, though unlikely given description
        dapi_channel = img
    else:
        return 0.0

    # Ensure dapi_channel is uint8 for GLCM calculation
    # If it's float, scale to 0-255. If it's already integer-like but not uint8, cast it.
    if np.issubdtype(dapi_channel.dtype, np.floating):
        dapi_channel = (dapi_channel * 255).astype(np.uint8)
    elif dapi_channel.dtype != np.uint8:
        # Handle cases like uint16 by scaling or clipping if necessary, 
        # but dataset says uint8. Safe cast:
        dapi_channel = dapi_channel.astype(np.uint8)

    # 2. Handle Segmentation
    # We need a labeled mask of nuclei to compute texture per nucleus.
    labeled_mask = None
    
    # Check if segmentation masks are provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        mask_input = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions
        if mask_input.shape[:2] == dapi_channel.shape[:2]:
            # If mask is boolean/binary, label it. If integer, assume it's already labeled.
            if mask_input.dtype == bool or np.max(mask_input) == 1:
                labeled_mask = label(mask_input)
            else:
                labeled_mask = mask_input.astype(int)
    
    # Fallback: Generate mask if none provided
    if labeled_mask is None:
        try:
            # Simple Otsu thresholding to find nuclei
            thresh = threshold_otsu(dapi_channel)
            binary_mask = dapi_channel > thresh
            labeled_mask = label(binary_mask)
        except Exception:
            # If thresholding fails (e.g., empty image), return 0
            return 0.0

    # 3. Compute Haralick Contrast per Nucleus
    # We compute GLCM for each nucleus individually to avoid background artifacts
    regions = regionprops(labeled_mask, intensity_image=dapi_channel)
    
    if not regions:
        return 0.0

    contrast_values = []

    for region in regions:
        # Skip very small regions that are likely noise
        if region.area < 10:
            continue

        # Extract the bounding box of the nucleus
        minr, minc, maxr, maxc = region.bbox
        
        # Extract the intensity patch
        intensity_patch = dapi_channel[minr:maxr, minc:maxc]
        
        # Extract the local mask for this specific label
        # region.image is the binary mask of the object within the bbox
        local_mask = region.image
        
        # Apply mask to patch: pixels outside the nucleus in the bbox become 0
        # Note: 0 is usually treated as a valid level in GLCM. 
        # To minimize background influence, we compute GLCM on the masked patch.
        # However, rectangular GLCM always includes the '0' background pixels if the shape is irregular.
        # A common approach is to compute on the rectangular patch where the object is dominant.
        masked_patch = intensity_patch.copy()
        masked_patch[~local_mask] = 0
        
        # Compute GLCM
        # distances=[1] (1 pixel offset)
        # angles=[0, 45, 90, 135] degrees (in radians) for rotational invariance
        # levels=256 (for uint8)
        try:
            glcm = graycomatrix(masked_patch, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                                levels=256, symmetric=True, normed=True)
            
            # Compute Contrast
            # contrast: sum(|i-j|^2 * p(i,j))
            contrast = graycoprops(glcm, 'contrast')
            
            # Average over the 4 directions
            mean_contrast = np.mean(contrast)
            contrast_values.append(mean_contrast)
        except ValueError:
            continue

    # 4. Aggregate Results
    if not contrast_values:
        return 0.0
        
    # Return the mean contrast across all nuclei in the image
    result = np.mean(contrast_values)
    
    return float(result)
