def extract(img, *segmentation_masks):
    """
    Computes the 'tubulin_fiber_alignment_strength' feature from the tubulin channel (Channel 1).
    
    This feature measures the anisotropy or alignment strength of the tubulin cytoskeleton using 
    structure tensor analysis. High values indicate parallel bundling of microtubules (a Taxane phenotype), 
    while low values indicate a disordered meshwork.
    
    The implementation uses the Structure Tensor (or Second Moment Matrix) approach:
    1. Compute gradients Ix, Iy.
    2. Compute tensor components Ixx, Ixy, Iyy.
    3. Smooth tensor components to integrate local neighborhood information.
    4. Compute eigenvalues (l1, l2) of the structure tensor.
    5. Calculate coherence/anisotropy: ((l1 - l2) / (l1 + l2))^2.
    6. Aggregate the metric over the foreground (tubulin signal).
    """
    import numpy as np
    from scipy import ndimage
    from skimage.feature import structure_tensor, structure_tensor_eigenvalues
    from skimage.filters import threshold_otsu

    # 1. Input Validation and Preprocessing
    img_arr = np.asarray(img)
    
    # Check for expected dimensions: (H, W, C) = (512, 512, 3)
    if img_arr.ndim != 3 or img_arr.shape[2] != 3:
        return 0.0

    # Extract Tubulin channel (Channel 1 - Green)
    tubulin_channel = img_arr[..., 1].astype(np.float32)

    # Normalize intensity to [0, 1]
    # Robust normalization using percentiles to handle outliers
    p99 = np.percentile(tubulin_channel, 99.5)
    if p99 > 0:
        tubulin_channel = tubulin_channel / p99
    tubulin_channel = np.clip(tubulin_channel, 0.0, 1.0)

    # 2. Define Region of Interest (ROI)
    # We only want to measure alignment where there is actual tubulin signal.
    # If segmentation masks are provided, use them to mask the background.
    # Otherwise, generate a simple foreground mask based on intensity.
    
    mask = None
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Assuming the first mask is a general cell mask or similar
        # Check if mask matches image spatial dimensions
        seg = segmentation_masks[0]
        if seg.shape == tubulin_channel.shape:
            mask = seg > 0
    
    if mask is None:
        # Fallback: Otsu thresholding to identify foreground tubulin
        try:
            thresh = threshold_otsu(tubulin_channel)
            mask = tubulin_channel > thresh
        except Exception:
            # Fallback if image is uniform
            mask = tubulin_channel > 0.1

    # If mask is empty, return 0
    if np.sum(mask) == 0:
        return 0.0

    # 3. Structure Tensor Analysis
    # Sigma for gradient computation (inner scale)
    sigma_g = 1.0 
    # Sigma for tensor smoothing (outer scale - integration scale)
    # A larger outer scale integrates orientation over a larger neighborhood, 
    # which is crucial for detecting "bundling" vs "noise".
    sigma_t = 2.5

    # Compute structure tensor components: Axx, Axy, Ayy
    # structure_tensor returns Axx, Axy, Ayy
    Axx, Axy, Ayy = structure_tensor(tubulin_channel, sigma=sigma_g, mode='reflect')

    # Smooth the structure tensor components to integrate local orientation
    # This is the "window" over which alignment is measured.
    Axx = ndimage.gaussian_filter(Axx, sigma_t)
    Axy = ndimage.gaussian_filter(Axy, sigma_t)
    Ayy = ndimage.gaussian_filter(Ayy, sigma_t)

    # Compute eigenvalues of the structure tensor
    # l1 is the larger eigenvalue (magnitude of gradient in dominant direction)
    # l2 is the smaller eigenvalue (magnitude of gradient in perpendicular direction)
    # Note: structure_tensor_eigenvalues expects a list/tuple of the components
    l1, l2 = structure_tensor_eigenvalues([Axx, Axy, Ayy])

    # 4. Compute Anisotropy / Coherence
    # Coherence measures how strongly the local gradients are aligned.
    # Formula: ((l1 - l2) / (l1 + l2))^2 or similar variants.
    # Here we use (l1 - l2) / (l1 + l2) which ranges from 0 (isotropic) to 1 (perfectly aligned 1D structure).
    # We add a small epsilon to denominator to avoid division by zero.
    epsilon = 1e-7
    coherence = (l1 - l2) / (l1 + l2 + epsilon)

    # 5. Aggregate Feature
    # We calculate the mean coherence only within the foreground mask.
    # We weight the coherence by the signal intensity (or energy l1+l2) to emphasize 
    # alignment in bright fiber bundles over dim background noise.
    
    # Energy (total gradient magnitude squared)
    energy = l1 + l2
    
    # Mask the arrays
    masked_coherence = coherence[mask]
    masked_energy = energy[mask]
    
    if masked_energy.size == 0 or np.sum(masked_energy) == 0:
        return 0.0

    # Weighted mean of coherence:
    # High values mean the bright tubulin structures are highly directional (bundled).
    weighted_alignment = np.sum(masked_coherence * masked_energy) / np.sum(masked_energy)

    return float(weighted_alignment)
