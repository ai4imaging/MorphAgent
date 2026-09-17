def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.feature import graycomatrix, graycoprops
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    
    # 1. Input Validation and Channel Selection
    # Ensure input is a numpy array
    arr = np.asarray(img)
    
    # Check dimensionality: Expecting (H, W, C) or (H, W)
    # Dataset description says (512, 512, 3)
    if arr.ndim == 3 and arr.shape[2] >= 2:
        # Channel 1 is Tubulin (Green)
        tubulin_channel = arr[..., 1]
    elif arr.ndim == 2:
        # Fallback if single channel passed (unlikely based on desc, but safe)
        tubulin_channel = arr
    else:
        return 0.0

    # 2. Data Type Handling for GLCM
    # GLCM requires integer types. The dataset is uint8 (0-255).
    # If float, we must scale and cast. If uint8, use directly.
    if np.issubdtype(tubulin_channel.dtype, np.floating):
        # Normalize to 0-255 if it's float [0,1]
        tubulin_channel = (np.clip(tubulin_channel, 0, 1) * 255).astype(np.uint8)
    elif tubulin_channel.dtype != np.uint8:
        # If it's some other int type, clip to 255 and cast
        tubulin_channel = np.clip(tubulin_channel, 0, 255).astype(np.uint8)
        
    # 3. Segmentation Handling
    # We need a mask to define "within cells".
    mask = None
    
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use the first available segmentation mask
        # Assuming mask is same spatial dims as image
        input_mask = segmentation_masks[0]
        # Handle if mask is 3D (e.g. (H, W, 1))
        if input_mask.ndim == 3:
            input_mask = input_mask.squeeze()
        
        # Ensure mask matches image dimensions
        if input_mask.shape == tubulin_channel.shape:
            mask = input_mask
    
    # Fallback: Generate mask if none provided
    if mask is None:
        # Simple background separation
        try:
            thresh = threshold_otsu(tubulin_channel)
            mask = (tubulin_channel > thresh).astype(np.uint8)
        except Exception:
            # If image is uniform, otsu fails
            return 0.0

    # 4. Object Processing
    # Label the mask to identify individual cells
    labeled_mask = label(mask)
    regions = regionprops(labeled_mask, intensity_image=tubulin_channel)
    
    if not regions:
        return 0.0

    contrasts = []
    
    # GLCM Parameters
    # distances=[1]: pixel adjacency
    # angles: 0, 45, 90, 135 degrees for rotational invariance
    # levels=256: for uint8 data
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    levels = 256

    for region in regions:
        # Extract the bounding box of the cell to reduce computation
        minr, minc, maxr, maxc = region.bbox
        
        # Get the sub-image for this cell
        cell_crop = tubulin_channel[minr:maxr, minc:maxc]
        
        # Get the binary mask for this specific cell in the crop
        # region.image is the binary mask of the object within the bbox
        cell_mask = region.image
        
        # Apply mask: We only want texture from the cell, not the background in the bbox.
        # However, GLCM is rectangular. Standard approach:
        # Mask out background pixels (set to 0). Note that this creates a sharp edge 
        # at the boundary (cell intensity -> 0), which contributes to contrast.
        # Given the constraints, calculating on the masked crop is the standard approach.
        masked_crop = cell_crop.copy()
        masked_crop[~cell_mask] = 0
        
        # Skip tiny regions that can't support the offset
        if masked_crop.shape[0] < 2 or masked_crop.shape[1] < 2:
            continue

        # Compute GLCM
        try:
            glcm = graycomatrix(masked_crop, distances=distances, angles=angles, 
                                levels=levels, symmetric=True, normed=True)
            
            # Compute Contrast
            # Contrast measures local variations.
            # High contrast = sharp edges/fibers. Low contrast = smooth/diffuse.
            contrast_vals = graycoprops(glcm, 'contrast')
            
            # Average over the 4 directions
            mean_contrast_for_cell = np.mean(contrast_vals)
            contrasts.append(mean_contrast_for_cell)
            
        except Exception:
            continue

    # 5. Aggregation
    if not contrasts:
        return 0.0
        
    # Return the mean contrast across all cells
    result = np.mean(contrasts)
    
    return float(result)
