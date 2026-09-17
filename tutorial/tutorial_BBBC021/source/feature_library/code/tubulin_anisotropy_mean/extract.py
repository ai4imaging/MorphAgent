def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.feature import structure_tensor, structure_tensor_eigenvalues
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_opening, disk

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Check dimensionality
    # Expected: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract Tubulin channel (Channel 1 - Green)
    tubulin = arr[:, :, 1]

    # Normalize Tubulin channel
    # Robust normalization using percentiles to handle outliers
    p1, p99 = np.percentile(tubulin, (1, 99))
    if p99 > p1:
        tubulin_norm = (tubulin - p1) / (p99 - p1)
    else:
        tubulin_norm = tubulin  # Fallback if image is flat
    
    tubulin_norm = np.clip(tubulin_norm, 0.0, 1.0)

    # Determine Region of Interest (ROI)
    # If segmentation masks are provided, use them to mask the cells.
    # Otherwise, create a simple foreground mask based on intensity.
    mask = None
    if len(segmentation_masks) > 0:
        # Combine all available masks into a single boolean mask
        # Assuming masks are labeled (0=bg, >0=cell)
        combined_mask = np.zeros(tubulin.shape, dtype=bool)
        for seg in segmentation_masks:
            if seg is not None and seg.shape == tubulin.shape:
                combined_mask = np.logical_or(combined_mask, seg > 0)
        
        if np.any(combined_mask):
            mask = combined_mask

    if mask is None:
        # Fallback: Otsu thresholding on the tubulin channel to find cellular regions
        try:
            thresh = threshold_otsu(tubulin_norm)
            mask = tubulin_norm > thresh
            # Clean up the mask slightly
            mask = binary_opening(mask, disk(2))
        except Exception:
            # If thresholding fails (e.g. empty image), return 0.0
            return 0.0

    if not np.any(mask):
        return 0.0

    # Compute Structure Tensor
    # Sigma determines the scale of the features (fibers) we are looking for.
    # Tubulin fibers are fine structures, so a small sigma is appropriate.
    sigma = 1.0 
    try:
        # Compute the structure tensor components
        Axx, Axy, Ayy = structure_tensor(tubulin_norm, sigma=sigma, mode='reflect')
        
        # Compute eigenvalues of the structure tensor
        # Note: structure_tensor_eigenvalues signature varies by version.
        # The prompt explicitly warns about passing a list [Axx, Axy, Ayy] vs separate args.
        # Newer skimage versions often expect separate arguments, but the prompt error guidance
        # suggests the installed version might expect a list. However, standard skimage 0.19+ 
        # takes Axx, Axy, Ayy. The prompt error message "takes 1 positional argument but 3 were given"
        # implies the installed function expects a single argument (likely the list/tuple of images).
        # We will follow the explicit guidance provided in the prompt.
        l1, l2 = structure_tensor_eigenvalues([Axx, Axy, Ayy])
        
    except TypeError:
        # Fallback in case the version installed actually wants unpacked arguments
        # (Defensive coding against version mismatch)
        try:
            l1, l2 = structure_tensor_eigenvalues(Axx, Axy, Ayy)
        except Exception:
            return 0.0
    except Exception:
        return 0.0

    # Calculate Anisotropy
    # l1 is the larger eigenvalue, l2 is the smaller eigenvalue.
    # Anisotropy = (l1 - l2) / (l1 + l2)
    # High anisotropy -> l1 >> l2 (linear structure)
    # Low anisotropy -> l1 approx l2 (isotropic structure)
    
    denominator = l1 + l2
    
    # Avoid division by zero
    # We only care about anisotropy where there is actual signal (denominator > epsilon)
    valid_pixels = (denominator > 1e-6) & mask
    
    if not np.any(valid_pixels):
        return 0.0
        
    numerator = l1 - l2
    anisotropy_map = np.zeros_like(l1)
    anisotropy_map[valid_pixels] = numerator[valid_pixels] / denominator[valid_pixels]

    # Compute mean anisotropy within the cellular mask
    mean_anisotropy = np.mean(anisotropy_map[valid_pixels])

    return float(mean_anisotropy)
