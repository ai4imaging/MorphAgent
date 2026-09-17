def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import gabor_kernel, threshold_otsu
    
    # 1. Data Loading and Preprocessing
    # Convert to float32 for processing
    arr = np.asarray(img, dtype=np.float32)
    
    # Handle dimensionality and select Tubulin channel (Channel 1 - Green)
    # Dataset is (512, 512, 3) RGB. Channel 1 is Tubulin.
    if arr.ndim == 3 and arr.shape[2] >= 2:
        tubulin_img = arr[:, :, 1]
    elif arr.ndim == 2:
        # Fallback if single channel passed (unlikely based on spec but safe)
        tubulin_img = arr
    else:
        return 0.0

    # Normalize intensity to [0, 1]
    # This is critical for texture features to be comparable across images
    # We use a robust max to avoid outliers skewing the normalization
    vmax = np.percentile(tubulin_img, 99.5)
    if vmax > 0:
        tubulin_img = tubulin_img / vmax
    tubulin_img = np.clip(tubulin_img, 0.0, 1.0)

    # 2. Define Region of Interest (ROI)
    # We want to calculate texture only within cells, not on the background.
    mask = None
    
    # Check if segmentation masks are provided
    if len(segmentation_masks) > 0:
        # Iterate through masks to find a suitable cell mask
        # We prefer a mask that covers the cytoplasm/cell body
        for m in segmentation_masks:
            if m is not None and m.shape == tubulin_img.shape:
                # Assuming labeled mask (0=bg, >0=cells)
                current_mask = m > 0
                if np.sum(current_mask) > 0:
                    # If we find a valid non-empty mask, use it
                    # If multiple masks exist, the logic here takes the first valid one.
                    # In a real scenario, we might prioritize a 'cell' mask over a 'nuclei' mask,
                    # but without explicit naming metadata, taking the first valid one is a reasonable heuristic.
                    mask = current_mask
                    break
    
    # Fallback: Generate mask if none provided or valid
    if mask is None:
        # Use Otsu thresholding on the tubulin channel itself to separate cells from background
        try:
            thresh = threshold_otsu(tubulin_img)
            mask = tubulin_img > thresh
            # Simple morphological closing to fill holes in the cell mask
            mask = ndimage.binary_closing(mask, structure=np.ones((3,3))).astype(bool)
        except Exception:
            # If thresholding fails (e.g. constant image), use full image
            mask = np.ones_like(tubulin_img, dtype=bool)

    # If mask is empty (no cells), return 0.0
    if np.sum(mask) == 0:
        return 0.0

    # 3. Gabor Filter Bank Construction
    # We want to detect linear structures (microtubules) at various orientations.
    # Frequency: Microtubules are fine structures. A wavelength of ~4 pixels is often good for 20x/40x microscopy.
    # Frequency = 1 / wavelength. Let's use 0.25.
    frequency = 0.25
    
    # Orientations: 0, 45, 90, 135 degrees
    thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    # Sigma: Bandwidth of the filter.
    sigma_x = 1.0
    sigma_y = 1.0
    
    # Accumulator for the maximum response across all angles
    max_response = np.zeros_like(tubulin_img)

    # 4. Convolution and Energy Calculation
    for theta in thetas:
        # Create kernel
        # We use the real part for edge detection.
        # Note: gabor_kernel returns complex kernel.
        kernel = gabor_kernel(frequency, theta=theta, sigma_x=sigma_x, sigma_y=sigma_y)
        real_kernel = np.real(kernel)
        
        # Convolve image with kernel
        # mode='reflect' handles boundaries gracefully
        filtered = ndimage.convolve(tubulin_img, real_kernel, mode='reflect')
        
        # We are interested in the energy (magnitude of response)
        # Since we used the real part, we take the absolute value.
        # (Alternatively, one could use magnitude of complex response, but abs(real) is standard for texture strength)
        response = np.abs(filtered)
        
        # We take the maximum response across orientations at each pixel.
        # This makes the feature rotation-invariant: we care if *any* linear structure exists.
        max_response = np.maximum(max_response, response)

    # 5. Aggregation
    # Compute the mean Gabor energy within the cell mask
    masked_response = max_response[mask]
    
    if masked_response.size == 0:
        return 0.0
        
    result = np.mean(masked_response)

    return float(result)
