def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    
    # Convert to appropriate array type
    img_arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Expected shape: (512, 512, 3)
    if img_arr.ndim != 3 or img_arr.shape[2] < 3:
        return 0.0

    # Extract DAPI channel (Channel 2)
    # Normalize to [0, 1] range based on uint8 input
    dapi = img_arr[:, :, 2] / 255.0

    # 1. Define Nuclear Region (Mask)
    # We generate a mask from the DAPI channel itself to ensure we are analyzing 
    # the correct regions. This is robust to variations in provided masks.
    # Smooth slightly to reduce noise for thresholding
    dapi_smooth = ndimage.gaussian_filter(dapi, sigma=2.0)
    
    try:
        # Otsu's method to find a global threshold for nuclei
        thresh = threshold_otsu(dapi_smooth)
        nuclei_mask = dapi_smooth > thresh
    except Exception:
        # Handle cases where image is uniform/empty
        return 0.0

    # 2. Detect Bright Spots (Peaks)
    # We use a maximum filter to find local maxima.
    # Critic feedback: Increase min_distance to >= 10.
    # A filter size of 21 means a pixel must be the max in a 21x21 window 
    # (radius 10), effectively enforcing min_distance=10.
    filter_size = 21
    max_filtered = ndimage.maximum_filter(dapi, size=filter_size, mode='constant')
    
    # Identify pixels that are local maxima
    # Note: dapi and max_filtered are derived from discrete uint8 values, so equality check is stable
    is_peak = (dapi == max_filtered)
    
    # Critic feedback: Lower threshold_abs to 0.3-0.4.
    # We use 0.35 as a robust absolute threshold for "bright" spots in normalized data.
    # This filters out background noise and low-intensity peaks.
    is_bright = dapi > 0.35
    
    # Combine criteria:
    # - Must be a local maximum
    # - Must be bright enough
    # - Must be within the nuclear mask
    detected_peaks = is_peak & is_bright & nuclei_mask
    
    # 3. Count Peaks
    # Use connected components labeling to count peaks.
    # This handles "plateaus" (adjacent pixels with same max value) as a single peak.
    # We use ndimage.label which is available via scipy import
    labeled_peaks, num_peaks = ndimage.label(detected_peaks)
    
    # Return the total count of peaks as requested
    return float(num_peaks)
