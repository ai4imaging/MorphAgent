def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.feature import graycomatrix, graycoprops
    from skimage.filters import threshold_otsu
    from skimage.measure import label, regionprops
    from skimage.morphology import binary_closing, disk
    from scipy import ndimage

    # 1. Input Validation and Channel Extraction
    # Ensure input is an array
    arr = np.asarray(img)
    
    # Check dimensionality: Expecting (H, W, 3) for BBBC021
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0
    
    # Extract DAPI channel (Channel 2 = Blue)
    # The dataset description specifies: Channel 0=Actin, 1=Tubulin, 2=DAPI
    dapi_channel = arr[:, :, 2]

    # Ensure uint8 for GLCM (required by skimage.feature.graycomatrix for efficiency)
    if dapi_channel.dtype != np.uint8:
        # Normalize to 0-255 if not already uint8
        dapi_float = dapi_channel.astype(np.float32)
        min_val, max_val = dapi_float.min(), dapi_float.max()
        if max_val > min_val:
            dapi_norm = (dapi_float - min_val) / (max_val - min_val)
            dapi_uint8 = (dapi_norm * 255).astype(np.uint8)
        else:
            dapi_uint8 = np.zeros_like(dapi_channel, dtype=np.uint8)
    else:
        dapi_uint8 = dapi_channel

    # 2. Mask Generation / Handling
    # We need to compute texture ONLY within the nuclei to avoid background noise dominating the signal.
    
    labels = None
    
    # Check if segmentation masks are provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Assume the first mask is the relevant one (nuclei or cells)
        mask_input = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions
        if mask_input.shape == dapi_uint8.shape:
            # If it's already labeled (int), use it. If boolean/binary, label it.
            if np.issubdtype(mask_input.dtype, np.integer) and mask_input.max() > 1:
                labels = mask_input
            else:
                labels = label(mask_input > 0)
        else:
            # Fallback if dimensions mismatch
            labels = None

    # Fallback: Generate mask if none provided or invalid
    if labels is None:
        # Simple segmentation pipeline for DAPI
        # 1. Gaussian blur to reduce noise
        blurred = ndimage.gaussian_filter(dapi_uint8, sigma=2)
        
        # 2. Otsu thresholding
        try:
            thresh = threshold_otsu(blurred)
            binary_mask = blurred > thresh
        except ValueError:
            # Handle empty/uniform images
            return 0.0
            
        # 3. Morphological closing to fill holes
        binary_mask = binary_closing(binary_mask, disk(2))
        
        # 4. Label connected components
        labels = label(binary_mask)

    # 3. Feature Computation: Haralick Contrast per Nucleus
    # We compute GLCM for each nucleus individually to capture local texture accurately.
    
    regions = regionprops(labels, intensity_image=dapi_uint8)
    
    if not regions:
        return 0.0

    contrast_values = []
    
    # GLCM parameters
    distances = [1] # Immediate neighbors
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4] # 0, 45, 90, 135 degrees
    levels = 256
    
    for region in regions:
        # Filter small artifacts
        if region.area < 50:
            continue
            
        # Extract the intensity image of the bounding box
        # region.intensity_image contains the pixel values inside the bbox
        # region.image contains the binary mask inside the bbox
        
        # We mask the intensity image so background pixels inside the bbox are 0
        # Note: This introduces a texture edge at the nucleus boundary (0 to value).
        # However, this is consistent across samples.
        # A more robust way is to just use the rectangular crop, but masking is safer for irregular shapes.
        # For standard Haralick on objects, using the masked crop is common.
        roi = region.intensity_image.copy()
        roi[~region.image] = 0
        
        # Compute GLCM
        # We use the masked ROI. The '0' background will contribute to the GLCM
        # but primarily at index (0,0) or (0, x).
        # Since we are looking for high contrast (variance), the internal texture is key.
        try:
            glcm = graycomatrix(roi, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
            
            # Compute Contrast: sum(|i-j|^2 * p(i,j))
            # High contrast = large intensity differences between neighbors
            contrasts = graycoprops(glcm, 'contrast')
            
            # Average across the 4 angles to get rotationally invariant measure for this cell
            mean_contrast = np.mean(contrasts)
            contrast_values.append(mean_contrast)
            
        except (ValueError, IndexError):
            continue

    # 4. Aggregation
    if not contrast_values:
        return 0.0
        
    # Return the mean contrast across all detected nuclei
    # High mean contrast -> Heterogeneous nuclei (e.g., foci, fragmentation)
    return float(np.mean(contrast_values))
