def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.feature import structure_tensor, structure_tensor_eigenvalues
    from skimage.filters import threshold_otsu
    
    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Expected shape: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        # If not the expected shape, try to handle potential variations or return 0
        if arr.ndim == 2:
            # If 2D, assume it's a single channel image, but we need Actin specifically.
            # This is ambiguous, but we'll proceed treating it as the relevant channel.
            actin_channel = arr
        else:
            return 0.0
    else:
        # Extract Channel 0 (Red) for Actin
        actin_channel = arr[..., 0]

    # Intensity normalization
    # Normalize to [0, 1] based on robust range
    vmax = np.percentile(actin_channel, 99.5) if actin_channel.size > 0 else 1.0
    if vmax <= 0:
        vmax = 1.0
    actin_norm = actin_channel / vmax
    actin_norm = np.clip(actin_norm, 0.0, 1.0)

    # Generate Analysis Mask (ROI)
    # We need to calculate alignment only within the cellular regions to avoid background noise.
    roi_mask = None

    # 1. Try to use provided segmentation masks
    if len(segmentation_masks) > 0:
        # Combine all masks (logical OR)
        combined_mask = np.zeros(actin_channel.shape, dtype=bool)
        for mask in segmentation_masks:
            if mask is not None:
                # Ensure mask is 2D and matches image shape
                if mask.ndim == 2 and mask.shape == actin_channel.shape:
                    combined_mask = np.logical_or(combined_mask, mask > 0)
                elif mask.ndim == 3 and mask.shape[:2] == actin_channel.shape:
                    # If mask is 3D (e.g. labeled volume), project or slice
                    combined_mask = np.logical_or(combined_mask, np.max(mask, axis=-1) > 0)
        
        if np.any(combined_mask):
            roi_mask = combined_mask

    # 2. Fallback: Otsu thresholding if no valid masks provided or masks were empty
    if roi_mask is None:
        try:
            thresh = threshold_otsu(actin_norm)
            roi_mask = actin_norm > thresh
            # Fill holes to ensure we capture the internal structure
            roi_mask = ndimage.binary_fill_holes(roi_mask)
        except Exception:
            # Fallback for extremely low contrast images where Otsu fails
            roi_mask = actin_norm > 0.1

    # If mask is still empty (e.g. black image), return 0
    if not np.any(roi_mask):
        return 0.0

    # Compute Structure Tensor
    # Sigma for gradient: small to detect thin fibers (1.0)
    # Sigma for integration: larger to average orientation locally (2.5)
    try:
        # Axx, Axy, Ayy
        st = structure_tensor(actin_norm, sigma=1.0, mode='reflect')
        
        # We need to smooth the structure tensor components to get a robust local orientation
        # structure_tensor function in skimage usually returns smoothed components if sigma is provided,
        # but let's be explicit about the integration scale which is crucial for "texture" vs "noise".
        # The skimage implementation: sigma argument is the scale of the Gaussian derivative.
        # We often need a second smoothing step (window function) for the tensor components.
        # Let's apply additional smoothing to the tensor components to represent the "window".
        sigma_window = 2.0
        st_00 = ndimage.gaussian_filter(st[0], sigma=sigma_window)
        st_01 = ndimage.gaussian_filter(st[1], sigma=sigma_window)
        st_11 = ndimage.gaussian_filter(st[2], sigma=sigma_window)
        
        # Compute Eigenvalues
        # The eigenvalues l1, l2 (l1 >= l2) describe the shape of the local structure.
        # l1 corresponds to the gradient strength in the dominant direction.
        # l2 corresponds to the gradient strength in the perpendicular direction.
        # For a line/fiber: l1 >> l2 (high anisotropy).
        # For a blob/isotropic area: l1 ~ l2.
        
        # Calculate coherence (anisotropy) manually or via eigenvalues
        # Coherence = ((Axx - Ayy)^2 + 4*Axy^2)^0.5 / (Axx + Ayy)
        # This is equivalent to (l1 - l2) / (l1 + l2)
        
        trace = st_00 + st_11
        det = st_00 * st_11 - st_01**2
        
        # Eigenvalues calculation
        # lambda1,2 = trace/2 +/- sqrt((trace/2)^2 - det)
        delta = np.sqrt(np.clip((trace/2)**2 - det, 0, None))
        l1 = trace/2 + delta
        l2 = trace/2 - delta
        
        # Coherence (0 to 1)
        # Add epsilon to prevent division by zero
        epsilon = 1e-7
        coherence = (l1 - l2) / (l1 + l2 + epsilon)
        
        # Weighted Average
        # We weight the coherence by the pixel intensity.
        # Rationale: We care more about the alignment of bright actin bundles than the alignment of dim background noise.
        # We also mask by the ROI.
        
        weights = actin_norm * roi_mask
        weighted_coherence_sum = np.sum(coherence * weights)
        total_weight = np.sum(weights)
        
        if total_weight > 0:
            result = weighted_coherence_sum / total_weight
        else:
            result = 0.0
            
    except Exception:
        return 0.0

    return float(result)
