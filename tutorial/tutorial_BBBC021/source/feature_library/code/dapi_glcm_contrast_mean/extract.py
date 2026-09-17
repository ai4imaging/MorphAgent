def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.feature import graycomatrix, graycoprops
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    
    # 1. Data Loading and Validation
    # Convert to appropriate array type, handle potential float inputs by casting to uint8 for GLCM
    # The dataset description says uint8 (0-255). If it's float [0,1], we scale it.
    img_arr = np.asarray(img)
    
    # Handle dimensions: Expecting (512, 512, 3)
    if img_arr.ndim == 3 and img_arr.shape[2] == 3:
        # Channel 2 is DAPI (Blue) based on dataset description
        dapi_channel = img_arr[:, :, 2]
    elif img_arr.ndim == 2:
        # Fallback if single channel passed (unlikely given description but safe)
        dapi_channel = img_arr
    else:
        return 0.0

    # Ensure dapi_channel is uint8 for GLCM quantization
    if dapi_channel.dtype != np.uint8:
        # Normalize to 0-255 if it's float
        if dapi_channel.max() <= 1.0:
            dapi_channel = (dapi_channel * 255).astype(np.uint8)
        else:
            dapi_channel = dapi_channel.astype(np.uint8)

    # 2. Segmentation Mask Handling
    # We need a nuclear mask.
    nuclei_mask = None
    
    # Check provided masks
    if len(segmentation_masks) > 0:
        # Heuristic: usually the nuclear mask is distinct. 
        # If multiple masks, we might need to identify which is which.
        # Given the prompt doesn't specify order, we assume the first one or check for typical properties.
        # However, often the first mask in these datasets is nuclei or the masks are passed in a specific order.
        # Without specific metadata, we check if any mask looks like nuclei (smaller, numerous objects).
        # For simplicity and robustness, if a mask is provided, we use the first one as the primary ROI mask.
        # If it's labeled (instance), convert to binary for masking purposes, though we will relabel later.
        candidate_mask = segmentation_masks[0]
        if candidate_mask.shape == dapi_channel.shape:
             nuclei_mask = candidate_mask > 0
    
    # Fallback: Generate mask if none provided or invalid
    if nuclei_mask is None:
        try:
            thresh = threshold_otsu(dapi_channel)
            nuclei_mask = dapi_channel > thresh
        except Exception:
            return 0.0

    # 3. Preprocessing for GLCM
    # GLCM is computationally expensive on 256 levels. Standard practice is to quantize.
    # We quantize to 64 levels (bins 0-63).
    n_levels = 64
    # Integer division to bin: 256 / 64 = 4
    dapi_quantized = dapi_channel // (256 // n_levels)
    
    # 4. Object-based Feature Extraction
    # We compute GLCM per nucleus to avoid boundary artifacts between nucleus and background.
    labeled_mask = label(nuclei_mask)
    regions = regionprops(labeled_mask, intensity_image=dapi_quantized)
    
    if not regions:
        return 0.0

    contrasts = []
    
    # GLCM Parameters
    # Distances: 1 pixel
    # Angles: 0, 45, 90, 135 degrees (0, np.pi/4, np.pi/2, 3*np.pi/4)
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    for region in regions:
        # Extract the bounding box of the nucleus from the quantized image
        # region.image is the binary mask of the object in the bounding box
        # region.image_intensity is the intensity image in the bounding box
        
        # We need to be careful: region.image_intensity includes background pixels as 0.
        # But 0 is also a valid intensity level in our quantized image (darkest).
        # However, for GLCM contrast inside the nucleus, we ideally only want pixels *inside* the mask.
        # Standard skimage graycomatrix computes on the whole rectangular patch.
        # To mitigate background influence, we can rely on the fact that the bounding box is tight.
        # A more robust way is to replace background pixels with a value outside the range, 
        # but graycomatrix doesn't support "ignore_values" in all versions.
        # Standard approach: Compute on the rectangular patch (intensity_image).
        
        patch = region.image_intensity
        
        # Skip very small regions that might result in empty GLCMs or errors
        if patch.size < 4: 
            continue

        try:
            # Compute GLCM
            # levels=n_levels must match the quantization range
            glcm = graycomatrix(patch, distances=distances, angles=angles, 
                                levels=n_levels, symmetric=True, normed=True)
            
            # Calculate Contrast
            # contrast: sum_{i,j} |i-j|^2 * p(i,j)
            feat = graycoprops(glcm, 'contrast')
            
            # feat is (num_distances, num_angles). We average over all angles.
            mean_contrast_for_cell = np.mean(feat)
            contrasts.append(mean_contrast_for_cell)
            
        except Exception:
            continue

    # 5. Aggregation
    if not contrasts:
        return 0.0
        
    result = np.mean(contrasts)
    
    return float(result)
