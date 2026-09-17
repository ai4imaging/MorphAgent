def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.feature import graycomatrix, graycoprops
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    import warnings

    # Suppress warnings that might arise from empty regions or precision issues
    warnings.filterwarnings("ignore")

    # 1. Data Validation and Preparation
    # Ensure input is a numpy array
    img = np.asarray(img)
    
    # Check dimensionality and extract DAPI channel
    # Dataset spec: (512, 512, 3), Channel 2 is DAPI (Blue)
    if img.ndim == 3 and img.shape[2] >= 3:
        dapi_channel = img[:, :, 2]
    elif img.ndim == 2:
        # Fallback if only one channel is passed (unlikely based on spec but safe)
        dapi_channel = img
    else:
        return 0.0

    # Ensure uint8 for GLCM calculation (required for discrete levels)
    if dapi_channel.dtype != np.uint8:
        # Normalize to 0-255 if not already
        dapi_min, dapi_max = dapi_channel.min(), dapi_channel.max()
        if dapi_max > dapi_min:
            dapi_channel = ((dapi_channel - dapi_min) / (dapi_max - dapi_min) * 255).astype(np.uint8)
        else:
            dapi_channel = dapi_channel.astype(np.uint8)

    # 2. Segmentation Mask Handling
    # We need a labeled mask for nuclei.
    labeled_mask = None
    
    # Check if masks are provided via arguments
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        mask_input = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions
        if mask_input.shape[:2] == dapi_channel.shape[:2]:
            # If mask is boolean or binary (0/1 or 0/255), label it
            if mask_input.max() <= 1 or (mask_input.max() == 255 and np.unique(mask_input).size <= 2):
                labeled_mask = label(mask_input > 0)
            else:
                # Assume it's already an integer instance mask
                labeled_mask = mask_input.astype(int)

    # Fallback: Generate mask if none provided or invalid
    if labeled_mask is None:
        try:
            thresh = threshold_otsu(dapi_channel)
            binary_mask = dapi_channel > thresh
            labeled_mask = label(binary_mask)
        except Exception:
            # If thresholding fails (e.g., constant image), return 0
            return 0.0

    # 3. Feature Computation: Haralick Contrast
    # Parameters for GLCM
    # We reduce bins to 32 or 64 to make the matrix denser and computation faster/more robust
    n_bins = 32
    # Bin the image: 0-255 -> 0-31
    binned_image = (dapi_channel // (256 // n_bins)).astype(np.uint8)
    
    # Distances: 1 pixel
    distances = [1]
    # Angles: 0, 45, 90, 135 degrees (isotropic)
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    regions = regionprops(labeled_mask, intensity_image=binned_image)
    
    if not regions:
        return 0.0

    contrast_values = []

    for region in regions:
        # Skip very small regions that can't support GLCM
        if region.area < 5:
            continue

        # Extract the bounding box of the intensity image
        # region.image is the binary mask of the cell within the bbox
        # region.image_intensity is the intensity image within the bbox
        roi_intensity = region.image_intensity
        roi_mask = region.image

        # We only want texture INSIDE the nucleus.
        # Standard GLCM computes on rectangular arrays.
        # To avoid background (0) affecting texture of the object, we can:
        # 1. Pass the rectangular crop. Background is 0.
        # 2. If we just pass roi_intensity, the 0-background (which is valid bin 0) 
        #    will have high contrast with the object edges.
        # Strategy: We compute GLCM on the rectangular patch. However, we want to minimize
        # the impact of the black background.
        # A common robust approach in cell profiling is to compute on the masked box.
        # Since '0' is a valid intensity bin, we shift the actual data to 1-(n_bins-1) 
        # and leave 0 for background, then ignore row/col 0 in GLCM? 
        # Simpler approach for this pipeline: Compute on the rectangular patch. 
        # The "edge" texture is often considered part of the morphological profile.
        
        # Ensure the ROI is valid
        if roi_intensity.size == 0:
            continue

        try:
            # Compute GLCM
            # levels must be n_bins because max value is n_bins-1
            glcm = graycomatrix(roi_intensity, distances=distances, angles=angles, 
                                levels=n_bins, symmetric=True, normed=True)
            
            # Compute contrast
            # Returns shape (len(distances), len(angles))
            contrasts = graycoprops(glcm, 'contrast')
            
            # Average over the 4 directions to get rotation invariance
            mean_contrast_cell = np.mean(contrasts)
            contrast_values.append(mean_contrast_cell)
            
        except (ValueError, IndexError):
            continue

    # 4. Aggregation
    if not contrast_values:
        return 0.0
        
    # Return the mean contrast across all nuclei
    return float(np.mean(contrast_values))
