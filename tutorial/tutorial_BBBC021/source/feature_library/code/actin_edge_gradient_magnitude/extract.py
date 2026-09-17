def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    
    # 1. Data Loading and Preprocessing
    # Convert to float32 for precision
    arr = np.asarray(img, dtype=np.float32)
    
    # Handle dimensionality and extract Actin channel
    # Dataset is (512, 512, 3), Channel 0 is Actin (Red)
    if arr.ndim == 3 and arr.shape[2] == 3:
        actin_img = arr[:, :, 0]
    elif arr.ndim == 2:
        # Fallback if passed a single channel image, assume it's the relevant one
        actin_img = arr
    else:
        return 0.0

    # Intensity normalization [0, 1]
    # This is crucial for gradient magnitude to be comparable across images
    vmax = np.percentile(actin_img, 99.5) if actin_img.size > 0 else 1.0
    if vmax > 0:
        actin_img = actin_img / vmax
    actin_img = np.clip(actin_img, 0.0, 1.0)

    # 2. Mask Handling
    # We need a mask defining the cell area to find its boundary.
    # If multiple masks are provided, we try to find the "whole cell" mask (usually larger area).
    final_mask = None
    
    if len(segmentation_masks) > 0:
        best_mask = None
        max_area = -1
        
        for mask in segmentation_masks:
            if mask is None:
                continue
            # Ensure mask is 2D and boolean/integer
            m = np.asarray(mask)
            if m.ndim > 2:
                m = np.max(m, axis=-1) # Flatten if 3D
            
            current_area = np.sum(m > 0)
            if current_area > max_area:
                max_area = current_area
                best_mask = m
        
        if best_mask is not None:
            final_mask = best_mask > 0
            
    # Fallback: Generate mask from Actin channel if no external mask provided
    if final_mask is None:
        try:
            # Simple background subtraction/thresholding
            # Smooth slightly before thresholding to reduce noise
            smooth_actin = ndimage.gaussian_filter(actin_img, sigma=2)
            thresh = threshold_otsu(smooth_actin)
            final_mask = smooth_actin > thresh
        except Exception:
            # If thresholding fails (e.g. uniform image), return 0
            return 0.0

    if np.sum(final_mask) == 0:
        return 0.0

    # 3. Boundary Extraction
    # We want the gradient at the edge of the cell.
    # We define the boundary as the pixels just inside the mask.
    # Erode the mask by 1 pixel
    struct = ndimage.generate_binary_structure(2, 1) # 4-connectivity
    eroded_mask = ndimage.binary_erosion(final_mask, structure=struct)
    
    # The boundary is the difference between the original mask and the eroded mask
    boundary_mask = np.logical_xor(final_mask, eroded_mask)
    
    # Exclude image borders from the boundary mask to avoid artifacts at the frame edge
    # (Cells touching the edge of the image shouldn't have that edge counted as a cortex)
    boundary_mask[0, :] = 0
    boundary_mask[-1, :] = 0
    boundary_mask[:, 0] = 0
    boundary_mask[:, -1] = 0

    if np.sum(boundary_mask) == 0:
        return 0.0

    # 4. Gradient Computation
    # Smooth the actin image slightly to reduce high-frequency noise (like individual fibers)
    # We are interested in the structural edge gradient, not texture noise.
    smooth_actin_for_grad = ndimage.gaussian_filter(actin_img, sigma=1.0)
    
    # Compute gradients in X and Y
    sx = ndimage.sobel(smooth_actin_for_grad, axis=0, mode='reflect')
    sy = ndimage.sobel(smooth_actin_for_grad, axis=1, mode='reflect')
    
    # Compute Gradient Magnitude
    grad_mag = np.hypot(sx, sy)

    # 5. Feature Extraction
    # Extract gradient magnitudes only at the boundary pixels
    boundary_gradients = grad_mag[boundary_mask]
    
    # Compute mean
    result = np.mean(boundary_gradients)

    return float(result)
