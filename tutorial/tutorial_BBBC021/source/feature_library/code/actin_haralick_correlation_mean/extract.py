def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.feature import graycomatrix, graycoprops
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    import warnings

    # Suppress warnings that might arise from GLCM on constant patches (divide by zero)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # 1. Data Preparation
    # Ensure image is uint8 for GLCM (0-255)
    # The dataset description says input is uint8. If it's float, we need to scale it.
    img_arr = np.asarray(img)
    
    # Handle dimensions: (512, 512, 3) -> Extract Channel 0 (Actin)
    if img_arr.ndim == 3 and img_arr.shape[2] >= 1:
        actin_channel = img_arr[:, :, 0]
    elif img_arr.ndim == 2:
        # Fallback if single channel passed
        actin_channel = img_arr
    else:
        return 0.0

    # Ensure uint8 for graycomatrix
    if actin_channel.dtype != np.uint8:
        # Normalize to 0-255 if not already
        if actin_channel.max() <= 1.0:
            actin_channel = (actin_channel * 255).astype(np.uint8)
        else:
            # Clip and cast
            actin_channel = np.clip(actin_channel, 0, 255).astype(np.uint8)

    # 2. Segmentation Handling
    # We need a labeled mask of cells to compute features per cell
    labeled_mask = None

    # Check if segmentation masks are provided
    if len(segmentation_masks) > 0:
        # Iterate through masks to find a suitable one (prefer cell/cytoplasm over nuclei)
        # Heuristic: Larger total area usually implies whole cell vs nucleus
        best_mask = None
        max_area = -1
        
        for mask in segmentation_masks:
            if mask is None:
                continue
            # Ensure mask is 2D
            if mask.ndim == 3:
                mask = np.max(mask, axis=2) # MIP if 3D
            
            current_area = np.sum(mask > 0)
            if current_area > max_area:
                max_area = current_area
                best_mask = mask
        
        if best_mask is not None:
            # If mask is already labeled (int > 1), use it. If binary, label it.
            if best_mask.max() > 1:
                labeled_mask = best_mask.astype(int)
            else:
                labeled_mask = label(best_mask > 0)

    # Fallback: Generate mask from Actin channel if no external mask provided
    if labeled_mask is None:
        # Simple Otsu thresholding on Actin channel to find foreground
        try:
            thresh = threshold_otsu(actin_channel)
            binary_mask = actin_channel > thresh
            labeled_mask = label(binary_mask)
        except Exception:
            # If image is constant or empty
            return 0.0

    # 3. Feature Computation: Haralick Correlation per Cell
    regions = regionprops(labeled_mask, intensity_image=actin_channel)
    
    if not regions:
        return 0.0

    correlations = []

    # GLCM parameters
    # distances=[1]: immediate neighbors
    # angles=[0, 45, 90, 135]: rotational invariance approximation
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    levels = 256

    for region in regions:
        # Skip very small regions that might be noise
        if region.area < 50:
            continue

        # Extract the intensity patch for the single cell
        # region.image is the binary mask of the cell in the bounding box
        # region.intensity_image is the intensity values in the bounding box
        
        # We want to compute texture only on the cell pixels.
        # However, GLCM is rectangular. 
        # Strategy: Use the rectangular bounding box intensity image.
        # Mask out background by setting it to 0 (which is already done in region.intensity_image usually,
        # but region.intensity_image keeps original values inside mask and 0 outside).
        
        patch = region.intensity_image
        
        # Ensure patch is uint8
        if patch.dtype != np.uint8:
            patch = patch.astype(np.uint8)

        # Compute GLCM
        try:
            glcm = graycomatrix(patch, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
            
            # Compute Correlation
            # Returns shape (len(distances), len(angles)) -> (1, 4)
            corr_values = graycoprops(glcm, 'correlation')
            
            # Average over the 4 angles to get a rotation-invariant metric for this cell
            mean_corr_cell = np.mean(corr_values)
            
            # Handle NaN (can happen if patch is perfectly constant)
            if not np.isnan(mean_corr_cell):
                correlations.append(mean_corr_cell)
                
        except Exception:
            continue

    # 4. Aggregation
    if not correlations:
        return 0.0

    # Return the mean correlation across all cells
    result = np.mean(correlations)
    
    return float(result)
