def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import gabor_kernel, threshold_otsu
    
    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3) - (Height, Width, Channels)
    # Channel 2 is DAPI (Blue)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        dapi_channel = arr[:, :, 2]
    elif arr.ndim == 2:
        # Fallback if only 2D image provided, assume it's the relevant channel
        dapi_channel = arr
    else:
        return 0.0

    # Intensity normalization
    # Normalize to [0, 1] for filter response consistency
    # Check for empty image
    if dapi_channel.max() == 0:
        return 0.0
        
    dapi_norm = dapi_channel / 255.0
    
    # Handle segmentation masks
    mask = None
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use the first provided mask (usually nuclei or cells)
        # Ensure mask matches image dimensions (2D)
        seg = segmentation_masks[0]
        if seg.shape == dapi_norm.shape:
            mask = seg > 0
        elif seg.ndim == 3 and seg.shape[:2] == dapi_norm.shape:
             # If mask is 3D (e.g. one-hot or labeled stack), take max projection or slice
             mask = np.max(seg, axis=2) > 0
    
    # Fallback: Generate mask if none provided or invalid
    if mask is None:
        try:
            thresh = threshold_otsu(dapi_norm)
            mask = dapi_norm > thresh
        except Exception:
            # If thresholding fails (e.g. uniform image), use all pixels
            mask = np.ones_like(dapi_norm, dtype=bool)

    # Check if we have any foreground pixels
    if np.sum(mask) == 0:
        return 0.0

    # Define Gabor filter bank parameters
    # Frequencies: 0.1 (coarse), 0.25 (fine)
    # Orientations: 0, 45, 90, 135 degrees
    frequencies = [0.1, 0.25]
    thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    total_energy_accum = np.zeros_like(dapi_norm, dtype=np.float64)
    
    # Apply filter bank
    for frequency in frequencies:
        for theta in thetas:
            # Create kernel
            # sigma_x and sigma_y control the bandwidth. 
            # Standard choice is often related to frequency, e.g., 1/frequency
            sigma = 1.0 / frequency
            kernel = gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma)
            
            # Separate real and imaginary parts for convolution
            # gabor_kernel returns complex kernel
            real_kernel = np.real(kernel)
            imag_kernel = np.imag(kernel)
            
            # Convolve
            # Using 'reflect' mode to handle boundaries
            filtered_real = ndimage.convolve(dapi_norm, real_kernel, mode='reflect')
            filtered_imag = ndimage.convolve(dapi_norm, imag_kernel, mode='reflect')
            
            # Compute magnitude (energy) for this filter
            magnitude = np.sqrt(filtered_real**2 + filtered_imag**2)
            
            # Accumulate
            total_energy_accum += magnitude

    # Average the accumulated energy over the number of filters
    num_filters = len(frequencies) * len(thetas)
    avg_energy_map = total_energy_accum / num_filters
    
    # Extract mean energy specifically within the nuclear regions
    masked_energy = avg_energy_map[mask]
    
    if masked_energy.size == 0:
        return 0.0
        
    result = np.mean(masked_energy)

    return float(result)
