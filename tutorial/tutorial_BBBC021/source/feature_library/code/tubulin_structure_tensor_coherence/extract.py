def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.feature import structure_tensor, structure_tensor_eigenvalues
    from skimage.filters import threshold_otsu

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3) with channels: 0=Actin, 1=Tubulin, 2=DAPI
    # We need the Tubulin channel (index 1)
    if arr.ndim == 3 and arr.shape[2] == 3:
        tubulin_img = arr[:, :, 1]
    elif arr.ndim == 2:
        # Fallback if passed a single channel image (though unlikely given spec)
        tubulin_img = arr
    else:
        return 0.0

    # Intensity normalization
    # Normalize to [0, 1] for consistent derivative magnitudes
    vmax = np.percentile(tubulin_img, 99.5) if tubulin_img.size > 0 else 1.0
    if vmax > 0:
        tubulin_img = tubulin_img / vmax
    tubulin_img = np.clip(tubulin_img, 0.0, 1.0)

    # Determine the region of interest (ROI)
    # Use segmentation masks if available, otherwise generate a foreground mask
    mask = None
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use the first available mask (assuming it covers cells/cytoplasm)
        # Masks are labeled integers, convert to boolean foreground
        mask = segmentation_masks[0] > 0
        
        # Ensure mask shape matches image shape
        if mask.shape != tubulin_img.shape:
            # Simple resize or crop fallback not implemented, assume matching dimensions
            # If mismatch, fallback to thresholding
            mask = None

    if mask is None:
        # Create a mask based on intensity to ignore background noise
        # Using a low threshold or Otsu if the image has content
        if np.max(tubulin_img) > 0.05:
            try:
                thresh = threshold_otsu(tubulin_img)
                mask = tubulin_img > thresh
            except Exception:
                mask = tubulin_img > 0.05
        else:
            mask = tubulin_img > 0.01

    # If mask is empty, return 0.0
    if np.sum(mask) == 0:
        return 0.0

    # Compute Structure Tensor
    # Sigma determines the scale of the features we are looking at.
    # Microtubules are fine structures, so a small sigma is appropriate.
    sigma = 1.0 
    
    # Axx, Axy, Ayy are the components of the structure tensor matrix
    # S = [[Axx, Axy], [Axy, Ayy]]
    # structure_tensor computes derivatives and smooths them with Gaussian window
    Axx, Axy, Ayy = structure_tensor(tubulin_img, sigma=sigma, mode='reflect')

    # Compute Coherence
    # Coherence measures local anisotropy.
    # Formula based on eigenvalues l1, l2 (l1 >= l2 >= 0):
    # Coherence = ((l1 - l2) / (l1 + l2))^2  (sometimes without square)
    # Here we use the standard definition: C = sqrt((Axx - Ayy)^2 + 4*Axy^2) / (Axx + Ayy)
    # This is equivalent to (l1 - l2) / (l1 + l2)
    
    # Numerator: Difference of eigenvalues
    # This represents the "energy" of the orientation
    numerator = np.sqrt((Axx - Ayy)**2 + 4 * Axy**2)
    
    # Denominator: Sum of eigenvalues (Trace of the tensor)
    # This represents the total gradient energy
    denominator = Axx + Ayy
    
    # Avoid division by zero
    epsilon = 1e-7
    coherence_map = numerator / (denominator + epsilon)

    # We only care about coherence in regions where there is actual structure (high gradient energy).
    # In flat background regions, noise dominates and coherence is meaningless.
    # We use the mask defined earlier, and potentially weight by energy.
    
    # Apply mask
    valid_coherence = coherence_map[mask]
    
    # Optional: Further filter by gradient magnitude to ensure we measure structure, not background noise within the cell
    # Calculate gradient magnitude approximation from the trace
    gradient_energy = denominator[mask]
    
    # Filter out very low energy pixels even within the mask to be safe
    # (e.g., flat regions inside a cell mask)
    energy_thresh = np.percentile(gradient_energy, 10) if gradient_energy.size > 0 else 0.0
    strong_structure_mask = gradient_energy > energy_thresh
    
    if np.sum(strong_structure_mask) == 0:
        return 0.0
        
    final_coherence_values = valid_coherence[strong_structure_mask]

    # Compute the mean coherence
    result = np.mean(final_coherence_values)

    return float(result)
