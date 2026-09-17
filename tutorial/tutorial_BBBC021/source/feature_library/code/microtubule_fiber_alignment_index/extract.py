def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.feature import structure_tensor, structure_tensor_eigenvalues
    from skimage.filters import threshold_otsu
    
    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Check dimensionality
    # Expected: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract Tubulin channel (Channel 1 - Green)
    tubulin_channel = arr[..., 1]

    # Normalize intensity
    # Robust normalization to handle outliers
    p99 = np.percentile(tubulin_channel, 99)
    p1 = np.percentile(tubulin_channel, 1)
    if p99 > p1:
        tubulin_norm = (tubulin_channel - p1) / (p99 - p1)
    else:
        tubulin_norm = tubulin_channel
    tubulin_norm = np.clip(tubulin_norm, 0.0, 1.0)

    # Create a mask for relevant foreground areas
    # If segmentation masks are provided, use them to focus on cellular regions
    # Otherwise, generate a simple foreground mask using Otsu's thresholding
    mask = None
    if len(segmentation_masks) > 0:
        # Combine all available masks to get a general cellular region
        # Masks are typically labeled integers, convert to boolean
        combined_mask = np.zeros(tubulin_channel.shape, dtype=bool)
        for seg in segmentation_masks:
            if seg is not None and seg.shape == tubulin_channel.shape:
                combined_mask = combined_mask | (seg > 0)
        
        if np.any(combined_mask):
            mask = combined_mask

    if mask is None:
        # Fallback: Otsu thresholding on the tubulin channel itself
        try:
            thresh = threshold_otsu(tubulin_norm)
            mask = tubulin_norm > thresh
        except Exception:
            # If thresholding fails (e.g. uniform image), use all pixels
            mask = np.ones(tubulin_channel.shape, dtype=bool)

    # Compute Structure Tensor
    # Sigma for Gaussian weighting (integration scale) and gradient smoothing
    # A small sigma captures fine fiber details
    sigma = 1.0 
    try:
        Axx, Axy, Ayy = structure_tensor(tubulin_norm, sigma=sigma, mode='reflect')
        
        # Compute eigenvalues of the structure tensor
        # l1 is the larger eigenvalue, l2 is the smaller eigenvalue
        # Coherence (anisotropy) = ((l1 - l2) / (l1 + l2))^2 or similar metrics
        # Here we use the coherence metric: C = (l1 - l2) / (l1 + l2 + epsilon)
        
        # FIX: Pass tensor components as a tuple to structure_tensor_eigenvalues
        l1, l2 = structure_tensor_eigenvalues((Axx, Axy, Ayy))
        
        # Calculate local coherence (anisotropy)
        # Values range from 0 (isotropic) to 1 (perfectly anisotropic/aligned)
        epsilon = 1e-7
        coherence = (l1 - l2) / (l1 + l2 + epsilon)
        
        # Apply mask to select only foreground regions
        if mask is not None:
            valid_coherence = coherence[mask]
        else:
            valid_coherence = coherence.flatten()
            
        if valid_coherence.size == 0:
            return 0.0
            
        # The feature is the mean alignment index of the foreground structures
        result = np.mean(valid_coherence)
        
    except Exception:
        return 0.0

    return float(result)
