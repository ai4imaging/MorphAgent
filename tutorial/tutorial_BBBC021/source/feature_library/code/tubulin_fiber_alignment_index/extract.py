def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu

    # Convert to float32 for processing
    arr = np.asarray(img, dtype=np.float32)

    # 1. Channel Selection: Target Tubulin (Channel 1)
    # The dataset description specifies 3 channels: Actin (0), Tubulin (1), DAPI (2).
    # We need to handle potential RGBA (4 channels) or single channel inputs robustly.
    target_channel = None
    
    if arr.ndim == 3:
        # Check if we have at least 2 channels to access index 1
        if arr.shape[2] >= 2:
            target_channel = arr[..., 1]  # Green channel is Tubulin
        elif arr.shape[2] == 1:
            target_channel = arr[..., 0]  # Fallback if only 1 channel exists
        else:
            # Fallback for unexpected 3D shape, try to take the middle slice or projection
            target_channel = np.mean(arr, axis=2)
    elif arr.ndim == 2:
        # If already 2D, use as is
        target_channel = arr
    else:
        return 0.0

    # 2. Normalization
    # Normalize to [0, 1] based on robust range
    if target_channel.size == 0:
        return 0.0
        
    p_min, p_max = np.percentile(target_channel, (1, 99))
    if p_max > p_min:
        norm_img = (target_channel - p_min) / (p_max - p_min)
    else:
        if p_max > 0:
            norm_img = target_channel / p_max
        else:
            norm_img = target_channel # All zeros

    norm_img = np.clip(norm_img, 0.0, 1.0)

    # 3. Mask Generation
    # We want to calculate alignment primarily in the foreground (where tubulin exists).
    # If segmentation masks are provided, use them. Otherwise, generate a mask.
    mask = None
    
    # Check provided masks
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use the first mask (often cells or nuclei). If it's nuclei, we might want to dilate, 
        # but usually, the first mask in these datasets is the whole cell or nuclei.
        # Let's assume mask > 0 is the ROI.
        seg = segmentation_masks[0]
        # Ensure mask shape matches image shape (handle 2D vs 3D mismatch if any)
        if seg.shape == norm_img.shape:
            mask = seg > 0
        elif seg.ndim == 3 and norm_img.ndim == 2:
             # If mask is 3D (e.g. HxWx1), squeeze it
             mask = seg.squeeze() > 0
    
    # Fallback: Generate mask from image intensity if no external mask or shape mismatch
    if mask is None:
        try:
            thresh = threshold_otsu(norm_img)
            mask = norm_img > thresh
        except Exception:
            # Fallback if Otsu fails (e.g., uniform image)
            mask = norm_img > 0.1

    # Ensure we have enough pixels to compute statistics
    if np.sum(mask) < 10:
        # If mask is too small, use the whole image
        mask = np.ones_like(norm_img, dtype=bool)

    # 4. Compute Structure Tensor (Gradient Structure Tensor)
    # J = [[Ix^2, IxIy], [IxIy, Iy^2]] smoothed by Gaussian
    # Eigenvalues of J describe the local geometry.
    # Coherence = (lambda1 - lambda2) / (lambda1 + lambda2)^2 or similar metrics.
    # Here we calculate "coherence" or "anisotropy" which indicates alignment.
    
    # Gradients
    sigma_grad = 1.0  # Scale for derivative calculation
    sigma_tensor = 1.5 # Scale for smoothing the tensor (integration scale)
    
    # Compute derivatives
    Iy = ndimage.gaussian_filter(norm_img, sigma=sigma_grad, order=(1, 0))
    Ix = ndimage.gaussian_filter(norm_img, sigma=sigma_grad, order=(0, 1))

    # Compute tensor components
    Ixx = Ix**2
    Iyy = Iy**2
    Ixy = Ix * Iy

    # Smooth tensor components (window integration)
    Jxx = ndimage.gaussian_filter(Ixx, sigma=sigma_tensor)
    Jyy = ndimage.gaussian_filter(Iyy, sigma=sigma_tensor)
    Jxy = ndimage.gaussian_filter(Ixy, sigma=sigma_tensor)

    # 5. Compute Eigenvalues of the Structure Tensor
    # The matrix is [[Jxx, Jxy], [Jxy, Jyy]]
    # Trace = Jxx + Jyy
    # Det = Jxx*Jyy - Jxy^2
    # Lambda1,2 = (Trace +/- sqrt(Trace^2 - 4*Det)) / 2
    # Coherence = ((Lambda1 - Lambda2) / (Lambda1 + Lambda2))^2 
    #           = (sqrt(Trace^2 - 4*Det) / Trace)^2
    #           = (Trace^2 - 4*Det) / Trace^2
    #           = 1 - 4*Det/Trace^2
    # Actually, simpler formula for coherence C:
    # C = sqrt((Jxx - Jyy)**2 + 4*Jxy**2) / (Jxx + Jyy)
    
    # Avoid division by zero
    trace = Jxx + Jyy
    # Add epsilon to denominator
    denominator = trace + 1e-7
    
    numerator = np.sqrt((Jxx - Jyy)**2 + 4 * Jxy**2)
    
    coherence = numerator / denominator

    # 6. Aggregate Feature
    # We want the average alignment (coherence) within the masked region.
    # High coherence = aligned fibers (e.g., bundles).
    # Low coherence = isotropic / disordered.
    
    masked_coherence = coherence[mask]
    
    if masked_coherence.size == 0:
        return 0.0
        
    # Return the mean coherence
    result = np.mean(masked_coherence)
    
    # Sanity check: if result is NaN (shouldn't be due to epsilon), return 0
    if np.isnan(result):
        return 0.0

    return float(result)
