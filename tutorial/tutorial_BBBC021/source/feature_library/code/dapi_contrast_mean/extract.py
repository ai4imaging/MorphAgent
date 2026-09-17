def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.feature import graycomatrix, graycoprops
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    
    # 1. Input Validation and Channel Selection
    # Ensure image is the expected 3-channel format
    if img.ndim != 3 or img.shape[2] != 3:
        # If not 3 channels, try to handle if it's 2D (grayscale) or other format
        # But per dataset spec, it should be (512, 512, 3)
        # If 2D, assume it's a single channel image, but we specifically need DAPI.
        # If we can't identify DAPI, return 0.0
        return 0.0

    # Extract DAPI channel (Channel 2 / Blue)
    # Image is RGB, so index 2 is Blue/DAPI
    dapi_channel = img[:, :, 2]

    # 2. Mask Handling
    mask = None
    
    # Check if valid segmentation masks are provided
    if segmentation_masks and len(segmentation_masks) > 0:
        # Usually the first mask is nuclei or cells. 
        # We prefer a nuclei mask if available.
        # Without specific metadata on which mask is which in the tuple, 
        # we default to the first one provided, assuming it contains relevant objects.
        candidate_mask = segmentation_masks[0]
        if candidate_mask is not None and candidate_mask.shape == dapi_channel.shape:
            mask = candidate_mask

    # Fallback: Generate mask if none provided
    if mask is None:
        # Use Otsu thresholding to separate nuclei from background
        # Check if image is not empty/black
        if dapi_channel.max() > dapi_channel.min():
            thresh = threshold_otsu(dapi_channel)
            mask = dapi_channel > thresh
        else:
            return 0.0

    # Ensure mask is labeled (integer labels for regions)
    # If mask is boolean or binary (0/1), label it.
    # If it's already labeled (0, 1, 2...), use it directly, but ensure 0 is background.
    if mask.dtype == bool:
        labeled_mask = label(mask)
    else:
        # If it's already integer, assume it's a label matrix
        # Re-label to ensure connectivity consistency if needed, 
        # but usually we can trust the input or just re-run label to be safe
        labeled_mask = label(mask > 0)

    # 3. Pre-processing for GLCM
    # Haralick features are sensitive to the number of gray levels.
    # Computing GLCM on 256 levels is slow and sparse.
    # Standard practice is to bin/quantize the image to fewer levels (e.g., 64).
    n_levels = 64
    # Normalize and bin DAPI channel to 0-(n_levels-1)
    # We use integer division. 256 / 64 = 4.
    # So values 0-3 -> 0, 4-7 -> 1, etc.
    dapi_binned = (dapi_channel // (256 // n_levels)).astype(np.uint8)
    # Clip just in case
    dapi_binned = np.clip(dapi_binned, 0, n_levels - 1)

    # 4. Compute Feature
    contrast_values = []
    
    # Get properties of labeled regions
    regions = regionprops(labeled_mask, intensity_image=dapi_binned)
    
    if not regions:
        return 0.0

    for region in regions:
        # Skip very small regions that might cause GLCM errors or are noise
        if region.area < 5:
            continue

        # Extract the bounding box of the nucleus
        # intensity_image in regionprops gives the intensity of the bounding box
        # masked by the region mask (pixels outside the region are 0? No, usually just the box)
        # We need to be careful: region.image is the binary mask of the object in the bbox.
        # region.intensity_image is the intensity image inside the bbox.
        
        # We want to compute GLCM only on the pixels belonging to the nucleus.
        # However, standard graycomatrix computes on the rectangular array.
        # To avoid background (0) affecting the texture calculation (creating fake edges),
        # we can't easily mask out background in standard skimage graycomatrix without
        # it counting the 0-transitions.
        
        # Strategy:
        # 1. Use the rectangular crop.
        # 2. Replace background pixels with a value that won't trigger high contrast 
        #    OR accept that the bounding box is tight and background is minimal.
        #    Better approach for "Contrast":
        #    If we mask background with 0 (which is a valid gray level), the edge between
        #    nucleus (e.g., level 30) and background (level 0) creates high contrast.
        #    This is an artifact.
        
        # Workaround: 
        # Since we want the texture *inside* the nucleus, we rely on the fact that
        # regionprops provides a tight bounding box.
        # We will compute GLCM on the rectangular intensity_image.
        # To mitigate boundary effects, we can try to fill the background with the 
        # mean intensity of the object, or just accept the slight noise if the object is convex.
        # Given the constraints and standard usage, computing on the masked bbox is common.
        # Let's use the intensity image provided by regionprops.
        # Pixels outside the mask in the bbox are 0 in region.intensity_image ONLY IF 
        # we passed intensity_image masked beforehand? No, regionprops extracts from original.
        # Actually, region.intensity_image contains the pixel values from the image passed to regionprops.
        # The pixels outside the region mask (but inside bbox) retain their values from the original image
        # if we just slice. But region.intensity_image documentation says:
        # "Image with the same size as the bounding box, containing the intensity values..."
        
        # Let's manually slice to be sure we handle the mask correctly.
        minr, minc, maxr, maxc = region.bbox
        patch_intensity = dapi_binned[minr:maxr, minc:maxc]
        patch_mask = region.image # Binary mask of the object in the bbox
        
        # We only want texture inside 'patch_mask'.
        # Since graycomatrix doesn't support an ROI mask directly, we have to accept the bbox.
        # However, we can minimize the background effect by replacing the background pixels
        # with the nearest valid pixel or 0. 
        # If we leave them as is (which might be 0 or adjacent cell values), it's messy.
        # A common robust approximation for cell texture is to just run it on the bbox 
        # if the cell fills most of it.
        
        # Let's compute GLCM on the patch.
        # We use distance=1 (immediate neighbors).
        # We use 4 angles (0, 45, 90, 135 degrees) for rotational invariance.
        # levels=n_levels (64).
        try:
            glcm = graycomatrix(patch_intensity, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                                levels=n_levels, symmetric=True, normed=True)
            
            # Compute contrast
            # contrast: sum(|i-j|^2 * p(i,j))
            contrasts = graycoprops(glcm, 'contrast')
            
            # Average over the 4 angles to get a single rotation-invariant value for this cell
            mean_contrast_for_cell = np.mean(contrasts)
            contrast_values.append(mean_contrast_for_cell)
            
        except (ValueError, IndexError):
            continue

    # 5. Aggregate
    if not contrast_values:
        return 0.0
        
    result = np.mean(contrast_values)

    return float(result)
