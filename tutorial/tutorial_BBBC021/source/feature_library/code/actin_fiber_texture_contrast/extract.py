def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.feature import graycomatrix, graycoprops
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu

    # 1. Input Validation and Preparation
    # Ensure image is numpy array
    img = np.asarray(img)
    
    # Handle dimensionality
    # Dataset spec: (512, 512, 3), Channel 0 = Actin (Red)
    if img.ndim == 3 and img.shape[2] == 3:
        actin_channel = img[:, :, 0]  # Extract Red channel
    elif img.ndim == 2:
        # Fallback if single channel passed (unlikely based on spec but safe)
        actin_channel = img
    else:
        return 0.0

    # Ensure uint8 for GLCM (0-255)
    # If float, normalize and convert. If integer but not uint8, clip and convert.
    if actin_channel.dtype != np.uint8:
        # Normalize to 0-255 if it's float 0-1
        if np.issubdtype(actin_channel.dtype, np.floating):
            if actin_channel.max() <= 1.0:
                actin_channel = (actin_channel * 255).astype(np.uint8)
            else:
                actin_channel = np.clip(actin_channel, 0, 255).astype(np.uint8)
        else:
            # Clip other integer types
            actin_channel = np.clip(actin_channel, 0, 255).astype(np.uint8)

    # 2. Mask Handling
    # We need a mask defining cellular regions to compute texture within cells.
    cell_mask = None
    
    if segmentation_masks and len(segmentation_masks) > 0:
        # Use the first available mask. 
        # Assuming masks are passed as (H, W) or (H, W, 1)
        mask_candidate = segmentation_masks[0]
        if mask_candidate is not None:
            mask_candidate = np.asarray(mask_candidate)
            # Handle potential extra dimensions in mask
            if mask_candidate.ndim == 3:
                mask_candidate = mask_candidate.squeeze()
            
            if mask_candidate.shape == actin_channel.shape:
                cell_mask = mask_candidate

    # Fallback: If no valid mask provided, generate one using Otsu thresholding on Actin
    if cell_mask is None:
        try:
            thresh = threshold_otsu(actin_channel)
            cell_mask = actin_channel > thresh
        except Exception:
            # Fallback for empty/uniform images
            return 0.0

    # 3. Feature Computation: Haralick Contrast
    # Strategy: Compute GLCM per cell to avoid boundary artifacts between cell and background.
    
    # Label the mask to identify individual objects
    labeled_mask = label(cell_mask)
    regions = regionprops(labeled_mask, intensity_image=actin_channel)
    
    if not regions:
        return 0.0

    contrasts = []
    
    # GLCM parameters
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4] # 0, 45, 90, 135 degrees
    levels = 256

    for region in regions:
        # Extract the bounding box of the cell
        # region.image is the binary mask of the cell in the bounding box
        # region.intensity_image is the intensity image in the bounding box
        
        # We only want texture INSIDE the cell.
        # However, GLCM is rectangular.
        # To minimize background influence, we use the masked intensity image.
        # Pixels outside the cell in the bbox are 0.
        # This creates a strong edge at the cell boundary (0 vs cell value).
        # To mitigate this, we can compute GLCM on the masked patch but we must accept
        # that the perimeter will contribute to "contrast". 
        # Given the constraints, this is the standard approach for object-based texture.
        
        patch = region.intensity_image
        # Ensure patch is uint8 (regionprops might return float if input was float, but we converted input to uint8)
        patch = patch.astype(np.uint8)
        
        # Skip very small regions that can't support the offset
        if patch.shape[0] < 2 or patch.shape[1] < 2:
            continue

        # Compute GLCM
        try:
            glcm = graycomatrix(patch, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
            
            # Compute Contrast
            # contrast: element (i,j) is weighted by (i-j)^2
            feat = graycoprops(glcm, 'contrast')
            
            # Average over the 4 angles to get rotation invariance
            avg_contrast = np.mean(feat)
            contrasts.append(avg_contrast)
        except Exception:
            continue

    # 4. Aggregation
    if not contrasts:
        return 0.0
        
    # Return the mean contrast across all cells in the image
    result = np.mean(contrasts)
    
    return float(result)
