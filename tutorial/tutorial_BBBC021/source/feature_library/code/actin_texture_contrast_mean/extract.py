def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.feature import graycomatrix, graycoprops
    from skimage.filters import threshold_otsu
    from skimage.measure import label, regionprops
    
    # 1. Input Validation and Preparation
    # Convert to appropriate array type if needed, but keep as uint8 for GLCM if possible
    # The dataset description says input is uint8. GLCM works best with discrete integer levels.
    img_arr = np.asarray(img)
    
    # Check dimensionality
    if img_arr.ndim != 3 or img_arr.shape[2] != 3:
        # Fallback or error handling for unexpected shapes
        return 0.0

    # 2. Extract Target Channel (Actin is Channel 0)
    # Shape becomes (Height, Width)
    actin_channel = img_arr[:, :, 0]
    
    # Ensure it is uint8 for graycomatrix (levels=256)
    if actin_channel.dtype != np.uint8:
        # Normalize to 0-255 and cast
        min_val = np.min(actin_channel)
        max_val = np.max(actin_channel)
        if max_val > min_val:
            actin_channel = ((actin_channel - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            actin_channel = np.zeros_like(actin_channel, dtype=np.uint8)

    # 3. Define Region of Interest (ROI) / Segmentation
    # We prefer using provided segmentation masks to isolate cells.
    # If not available, we generate a mask using Otsu thresholding.
    
    mask = None
    if segmentation_masks and len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use the first available mask
        mask = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions
        if mask.shape != actin_channel.shape:
            # If dimensions mismatch (e.g. 3D mask vs 2D image), try to project or resize
            # For this specific dataset, we assume 2D compatibility or return 0
            if mask.ndim == 2 and mask.shape == actin_channel.shape:
                pass # OK
            else:
                mask = None # Fallback to auto-segmentation
    
    if mask is None:
        # Auto-segmentation strategy: Otsu thresholding
        try:
            thresh = threshold_otsu(actin_channel)
            mask = actin_channel > thresh
            # Label the mask to identify individual objects
            mask = label(mask)
        except Exception:
            # If thresholding fails (e.g. uniform image), return 0
            return 0.0
    else:
        # Ensure mask is labeled (integers for distinct objects)
        # If it's a boolean mask, label it. If it's already integer labels, keep it.
        if mask.dtype == bool:
            mask = label(mask)
        # If it's already labeled, we assume 0 is background and 1+ are objects

    # 4. Compute Feature per Object
    # We calculate Haralick Contrast for each cell and average them.
    
    regions = regionprops(mask.astype(int), intensity_image=actin_channel)
    
    if not regions:
        return 0.0

    contrast_values = []
    
    # GLCM Parameters
    distances = [1] # Pixel adjacency
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4] # 0, 45, 90, 135 degrees for rotational invariance
    levels = 256
    
    for region in regions:
        # Skip very small regions that might cause errors or noise
        if region.area < 50:
            continue
            
        # Extract the bounding box of the intensity image
        # This is much faster than processing the whole image
        roi_intensity = region.image_intensity
        
        # region.image_intensity gives the intensity values inside the bounding box,
        # but pixels outside the mask (within bbox) are 0.
        # However, for GLCM, 0 is a valid level.
        # To strictly measure texture *inside* the cell, we ideally want to ignore the background 0s.
        # Standard GLCM implementations are rectangular.
        # A common approach is to compute GLCM on the rectangular ROI.
        # Since the background is 0 (black), the contrast at the boundary will be high.
        # Given the constraints, we compute on the masked ROI provided by regionprops.
        # Ensure the ROI is uint8
        roi_uint8 = roi_intensity.astype(np.uint8)
        
        # Compute GLCM
        # symmetric=True, normed=True are standard for texture features
        try:
            glcm = graycomatrix(roi_uint8, distances=distances, angles=angles, 
                                levels=levels, symmetric=True, normed=True)
            
            # Compute Contrast
            # Returns shape (len(distances), len(angles)) -> (1, 4)
            contrast = graycoprops(glcm, 'contrast')
            
            # Average over the 4 angles to get a rotation-invariant metric for this cell
            mean_contrast_cell = np.mean(contrast)
            contrast_values.append(mean_contrast_cell)
            
        except ValueError:
            continue

    # 5. Aggregate Results
    if not contrast_values:
        return 0.0
        
    # Return the mean contrast across all cells in the image
    result = np.mean(contrast_values)
    
    return float(result)
