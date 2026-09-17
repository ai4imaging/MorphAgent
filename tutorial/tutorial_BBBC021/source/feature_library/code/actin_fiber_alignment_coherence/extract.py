def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    
    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3) where Channel 0 is Actin (Red)
    if arr.ndim == 3 and arr.shape[2] >= 1:
        # Extract Actin channel (Channel 0)
        actin_img = arr[:, :, 0]
    elif arr.ndim == 2:
        # Fallback if passed a single channel image (though unlikely given description)
        actin_img = arr
    else:
        return 0.0

    # Intensity normalization
    # Normalize to [0, 1] for gradient calculations
    vmax = np.percentile(actin_img, 99.5) if actin_img.size > 0 else 1.0
    if vmax > 0:
        actin_img = actin_img / vmax
    actin_img = np.clip(actin_img, 0.0, 1.0)

    # Determine Region of Interest (ROI)
    # If segmentation masks are provided, use them. Otherwise, generate a mask via thresholding.
    roi_mask = None
    
    if len(segmentation_masks) > 0:
        # Combine all available masks
        # Masks are typically labeled integers (0=bg, 1..N=cells)
        combined_mask = np.zeros(actin_img.shape, dtype=bool)
        for mask in segmentation_masks:
            if mask is not None:
                # Ensure mask matches image dimensions (handle potential 2D vs 3D mismatch if any)
                if mask.shape == actin_img.shape:
                    combined_mask = combined_mask | (mask > 0)
                elif mask.ndim == 3 and mask.shape[:2] == actin_img.shape:
                     # If mask is 3D (e.g. one-hot or RGB mask), flatten it
                    combined_mask = combined_mask | (np.max(mask, axis=2) > 0)
        
        if np.any(combined_mask):
            roi_mask = combined_mask

    # Fallback: Create mask from Actin channel if no external mask provided or valid
    if roi_mask is None:
        try:
            thresh = threshold_otsu(actin_img)
            roi_mask = actin_img > thresh
        except Exception:
            # If otsu fails (e.g. uniform image), use all pixels
            roi_mask = np.ones(actin_img.shape, dtype=bool)

    # If mask is empty, return 0.0
    if not np.any(roi_mask):
        return 0.0

    # --- Structure Tensor Calculation ---
    # The structure tensor allows us to estimate the local orientation and coherence.
    # It is defined as the smoothed outer product of gradients.
    # J = G_sigma * (grad(I) . grad(I)^T)
    # J = [[ <Ix^2>, <IxIy> ],
    #      [ <IxIy>, <Iy^2> ]]
    
    sigma_grad = 1.0  # Scale for derivative calculation
    sigma_tensor = 3.0 # Scale for integration (window size over which orientation is averaged)

    # 1. Compute Gradients
    # Using Gaussian derivatives is more robust to noise than simple Sobel
    Iy, Ix = np.gradient(actin_img) 
    # Alternatively use ndimage.gaussian_filter with order parameter for analytical derivatives
    # But simple gradient on slightly smoothed image is often sufficient.
    # Let's stick to standard practice:
    
    # 2. Compute Tensor Components
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    # 3. Smooth the components (Integration)
    # This averaging is crucial for the structure tensor to represent a region
    Jxx = ndimage.gaussian_filter(Ixx, sigma_tensor)
    Jyy = ndimage.gaussian_filter(Iyy, sigma_tensor)
    Jxy = ndimage.gaussian_filter(Ixy, sigma_tensor)

    # 4. Compute Eigenvalues of the Structure Tensor
    # The eigenvalues l1, l2 of a 2x2 matrix [[a, b], [b, c]] are:
    # l1,2 = ((a + c) +/- sqrt((a - c)^2 + 4b^2)) / 2
    # Here a=Jxx, b=Jxy, c=Jyy
    
    # Trace = Jxx + Jyy
    # Det = Jxx*Jyy - Jxy*Jxy
    # But we need eigenvalues directly for coherence.
    
    tmp = np.sqrt((Jxx - Jyy)**2 + 4 * Jxy**2)
    l1 = (Jxx + Jyy + tmp) / 2
    l2 = (Jxx + Jyy - tmp) / 2
    
    # 5. Compute Coherence (Anisotropy)
    # Coherence = ((l1 - l2) / (l1 + l2))^2  OR  (l1 - l2) / (l1 + l2)
    # High coherence (-> 1) means l1 >> l2 (strong directionality, fibers)
    # Low coherence (-> 0) means l1 ~ l2 (isotropic, noise or flat)
    
    denominator = l1 + l2
    numerator = l1 - l2
    
    # Avoid division by zero
    # We use a small epsilon or mask out zero regions
    with np.errstate(divide='ignore', invalid='ignore'):
        coherence_map = numerator / (denominator + 1e-7)
        # Often squared to penalize weak orientation, but linear is also common.
        # Let's use the linear version as it's more sensitive to subtle alignment.
        # coherence_map = coherence_map ** 2 
    
    coherence_map = np.nan_to_num(coherence_map)

    # 6. Aggregate over ROI
    # We only care about the coherence within the cell/actin structure
    mean_coherence = np.mean(coherence_map[roi_mask])

    return float(mean_coherence)
