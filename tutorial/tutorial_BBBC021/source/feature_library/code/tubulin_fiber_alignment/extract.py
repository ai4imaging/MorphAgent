def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.feature import structure_tensor, structure_tensor_eigenvalues
    from scipy import ndimage

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Dataset is 512x512x3 (H, W, C)
    # Channel 1 is Tubulin (Green)
    if arr.ndim == 3 and arr.shape[2] == 3:
        tubulin_channel = arr[:, :, 1]
    elif arr.ndim == 2:
        # Fallback if single channel passed, though unlikely given description
        tubulin_channel = arr
    else:
        return 0.0

    # Intensity normalization
    # Normalize to 0-1 range for processing
    v_min, v_max = np.percentile(tubulin_channel, (1, 99))
    if v_max > v_min:
        tubulin_channel = (tubulin_channel - v_min) / (v_max - v_min)
    else:
        if v_max > 0:
            tubulin_channel = tubulin_channel / v_max
    
    tubulin_channel = np.clip(tubulin_channel, 0.0, 1.0)

    # Feature: Tubulin Fiber Alignment
    # Logic: Use Structure Tensor to estimate local orientation and coherence.
    # High alignment (anisotropy) suggests bundling (e.g., Taxol).
    # Low alignment (isotropy) suggests depolymerization or noise.

    # 1. Compute Structure Tensor
    # sigma=1.0 for inner smoothing (gradient), sigma=3.0 for outer smoothing (tensor integration)
    # This scale captures fiber bundles.
    try:
        Axx, Axy, Ayy = structure_tensor(tubulin_channel, sigma=1.0, mode='reflect')
        
        # Smooth the tensor components to integrate over a local neighborhood
        # This is crucial for robust orientation estimation
        Axx = ndimage.gaussian_filter(Axx, 3.0)
        Axy = ndimage.gaussian_filter(Axy, 3.0)
        Ayy = ndimage.gaussian_filter(Ayy, 3.0)

        # 2. Compute Eigenvalues
        # l1 is the larger eigenvalue (gradient magnitude in dominant direction)
        # l2 is the smaller eigenvalue (gradient magnitude in perpendicular direction)
        # FIX: Pass tensor components as a list to structure_tensor_eigenvalues
        l1, l2 = structure_tensor_eigenvalues([Axx, Axy, Ayy])

        # 3. Compute Coherence (Anisotropy)
        # Coherence = ((l1 - l2) / (l1 + l2))^2 or similar metric.
        # Here we use a standard coherence measure: (l1 - l2) / (l1 + l2 + epsilon)
        # Values close to 1 indicate strong directional alignment (fibers).
        # Values close to 0 indicate isotropic structure (noise or uniform regions).
        
        denominator = l1 + l2 + 1e-7
        coherence = (l1 - l2) / denominator
        
        # 4. Masking (Optional but recommended)
        # We only care about alignment where there is actual signal (tubulin).
        # Use a simple threshold to ignore background noise.
        signal_mask = tubulin_channel > 0.1
        
        if np.sum(signal_mask) == 0:
            return 0.0
            
        # 5. Aggregate
        # Return the mean coherence of the signal regions.
        mean_alignment = np.mean(coherence[signal_mask])
        
        return float(mean_alignment)

    except Exception:
        return 0.0
