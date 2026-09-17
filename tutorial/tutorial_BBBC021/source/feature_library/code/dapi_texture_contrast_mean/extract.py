def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.feature import graycomatrix, graycoprops
    from skimage.measure import regionprops
    
    # 1. Input Validation and Setup
    # Ensure image is the correct shape and type
    img = np.asarray(img)
    if img.ndim != 3 or img.shape[2] != 3:
        return 0.0
    
    # Extract DAPI channel (Channel 2: Blue)
    # Shape becomes (Height, Width)
    dapi_channel = img[:, :, 2]

    # 2. Handle Segmentation Masks
    # We need at least one mask (nuclei) to compute this feature
    if not segmentation_masks:
        return 0.0
    
    # Assume the first mask is the nuclei mask based on standard ordering
    nuclei_mask = segmentation_masks[0]
    
    # Ensure mask matches image spatial dimensions
    if nuclei_mask.shape != dapi_channel.shape:
        return 0.0

    # 3. Preprocessing for GLCM
    # Haralick features are sensitive to the number of gray levels.
    # Computing GLCM on full 0-255 range is sparse and slow.
    # Standard practice is to bin intensity into fewer levels (e.g., 64).
    n_levels = 64
    
    # We need to bin the entire image first to ensure consistent levels across crops,
    # or bin locally. Binning globally is safer for comparison.
    # Binning logic: value // (256 / n_levels)
    # 256 / 64 = 4. So values 0-3 -> 0, 4-7 -> 1, etc.
    dapi_binned = (dapi_channel // (256 // n_levels)).astype(np.uint8)
    
    # Clip to ensure no value equals n_levels (e.g. 255 // 4 = 63, which is fine for 0-63)
    dapi_binned = np.clip(dapi_binned, 0, n_levels - 1)

    # 4. Compute Feature Per Nucleus
    # We iterate over each nucleus to compute texture locally.
    # This avoids computing GLCM on the vast background and handles instance separation.
    
    # Get properties of labeled regions
    # If the mask is semantic (0/1), label it first. If instance (0, 1, 2...), use directly.
    # We assume instance segmentation for "mean per nucleus" calculation.
    # However, if max label is 1, we treat it as a single object or connected components.
    unique_labels = np.unique(nuclei_mask)
    if len(unique_labels) <= 1: # Only background
        return 0.0
        
    # If mask is binary (0 and 1 only), label connected components to treat distinct blobs as nuclei
    if len(unique_labels) == 2 and unique_labels[1] == 1:
        from skimage.measure import label
        labeled_mask = label(nuclei_mask)
        props = regionprops(labeled_mask, intensity_image=dapi_binned)
    else:
        # Already instance labeled
        props = regionprops(nuclei_mask, intensity_image=dapi_binned)

    contrast_values = []

    for prop in props:
        # Extract the bounding box of the nucleus
        # intensity_image in regionprops is already the crop of the passed intensity_image
        # masked by the bounding box. However, it includes background pixels within the bbox.
        # We need to be careful: GLCM on a rectangular crop with zeros (background) 
        # will treat the boundary between nucleus and background as a high contrast edge.
        
        # Strategy:
        # 1. Get the crop of the binned image.
        # 2. Get the crop of the mask.
        # 3. Set background pixels to a neutral value or handle carefully.
        # Actually, skimage graycomatrix doesn't support a mask argument directly to ignore pixels.
        # A common workaround for irregular shapes is to compute on the rectangular bbox
        # but this introduces artifacts.
        # BETTER APPROACH for irregular shapes:
        # Since we can't easily mask out background in standard GLCM without artifacts,
        # we will compute it on the bounding box but try to minimize background impact
        # or accept that shape edges contribute to "texture" in this context.
        # Given the constraints, computing on the masked bbox is the standard approximation.
        
        # Get the crop
        bbox_img = prop.image_intensity # This is dapi_binned cropped to bbox
        
        # Skip tiny nuclei that are smaller than the offset (1 pixel)
        if bbox_img.shape[0] < 2 or bbox_img.shape[1] < 2:
            continue

        # Replace the background (0 in prop.image but valid in data) with a value?
        # regionprops.image_intensity sets pixels outside the region to 0.
        # But 0 is a valid gray level (darkest).
        # To strictly analyze internal texture, we ideally want only internal pixels.
        # However, without a custom GLCM implementation, we use the bbox.
        # We use the bbox provided by regionprops.
        
        # Compute GLCM
        # distances=[1]: pixel adjacency
        # angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]: 0, 45, 90, 135 degrees (isotropic)
        # levels=n_levels: 64
        try:
            glcm = graycomatrix(bbox_img, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                                levels=n_levels, symmetric=True, normed=True)
            
            # Compute Contrast
            # Returns array of shape (len(distances), len(angles))
            contrast = graycoprops(glcm, 'contrast')
            
            # Average over all angles (isotropic texture)
            mean_contrast_nucleus = np.mean(contrast)
            contrast_values.append(mean_contrast_nucleus)
            
        except Exception:
            continue

    # 5. Aggregate Results
    if not contrast_values:
        return 0.0
        
    # Return the mean contrast across all nuclei
    return float(np.mean(contrast_values))
