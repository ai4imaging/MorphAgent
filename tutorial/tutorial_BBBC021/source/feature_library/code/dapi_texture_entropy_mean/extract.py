def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage import measure, filters
    from scipy import ndimage
    
    # 1. Data Loading & Channel Selection
    # Dataset Description: (512, 512, 3) uint8.
    # Channel 0: Actin (Red), Channel 1: Tubulin (Green), Channel 2: DAPI (Blue).
    # We need the DAPI channel for nuclei analysis.
    
    # Handle dimensionality
    if img.ndim == 3 and img.shape[2] >= 3:
        # Extract DAPI (Channel 2)
        dapi_raw = img[:, :, 2]
    elif img.ndim == 2:
        # Fallback for 2D inputs (assume single channel is DAPI if that's all we have)
        dapi_raw = img
    else:
        # Unexpected format
        return 0.0

    # 2. Preprocessing & Normalization
    # Texture entropy requires a discrete distribution. We normalize the image to use the full 
    # 8-bit dynamic range (0-255) to ensure consistent binning.
    
    # Convert to float for calculation
    dapi_float = dapi_raw.astype(np.float32)
    
    # Robust min/max normalization (percentile based to ignore outliers)
    # This standardizes contrast across images
    val_min = np.percentile(dapi_float, 0.1)
    val_max = np.percentile(dapi_float, 99.9)
    
    if val_max > val_min:
        dapi_norm = (dapi_float - val_min) / (val_max - val_min)
    else:
        dapi_norm = np.zeros_like(dapi_float)
        
    dapi_norm = np.clip(dapi_norm, 0.0, 1.0)
    
    # Convert to uint8 [0, 255] for histogram calculation
    dapi_uint8 = (dapi_norm * 255).astype(np.uint8)

    # 3. Segmentation Logic
    labels = None
    
    # Strategy: Use provided masks if available and valid. 
    # If not, perform Otsu thresholding on the DAPI channel.
    
    if len(segmentation_masks) > 0:
        # Iterate through masks to find a valid candidate
        for mask in segmentation_masks:
            if mask is not None and mask.shape == dapi_uint8.shape:
                # If mask is already integer labels
                if np.issubdtype(mask.dtype, np.integer) and mask.max() > 0:
                    labels = mask
                    break
                # If mask is binary/boolean
                elif (mask.dtype == bool or mask.max() == 1) and mask.max() > 0:
                    labels = measure.label(mask)
                    break
    
    # Fallback: Generate segmentation if no valid mask provided
    if labels is None:
        # Gaussian smooth to reduce noise before thresholding
        blurred = ndimage.gaussian_filter(dapi_uint8.astype(float), sigma=2.0)
        try:
            thresh = filters.threshold_otsu(blurred)
            binary = blurred > thresh
            # Label connected components
            labels = measure.label(binary)
            
            # Basic size filtering to remove noise artifacts (e.g., < 50 pixels)
            props = measure.regionprops(labels)
            mask_clean = np.zeros_like(labels, dtype=bool)
            for p in props:
                if p.area >= 50:
                    # Reconstruct mask for valid objects
                    mask_clean[p.coords[:, 0], p.coords[:, 1]] = True
            labels = measure.label(mask_clean)
        except Exception:
            # If segmentation fails completely (e.g., empty image)
            return 0.0

    # 4. Feature Computation: DAPI Texture Entropy Mean
    # Compute Shannon entropy of pixel intensities within each segmented nucleus
    
    regions = measure.regionprops(labels, intensity_image=dapi_uint8)
    entropies = []
    
    for region in regions:
        # Extract intensity values strictly within the object mask
        # region.image is the binary mask of the object in its bounding box
        # region.intensity_image is the intensity crop
        object_pixels = region.intensity_image[region.image]
        
        if object_pixels.size == 0:
            continue
            
        # Compute histogram
        # We use 257 bin edges to create 256 bins: [0,1), [1,2), ..., [255, 256]
        # This covers the full uint8 range.
        counts, _ = np.histogram(object_pixels, bins=range(257))
        
        # Normalize counts to probabilities
        p = counts / object_pixels.size
        
        # Remove zero probabilities (log(0) is undefined)
        p = p[p > 0]
        
        if p.size > 0:
            # Shannon Entropy formula: - sum(p * log2(p))
            # Using log2 yields units in bits
            entropy = -np.sum(p * np.log2(p))
            entropies.append(entropy)
    
    # 5. Aggregation
    if not entropies:
        return 0.0
        
    # Return the mean entropy across all nuclei
    result = np.mean(entropies)
    
    # Sanity check: Entropy for 8-bit image should be between 0 and 8
    # (log2(256) = 8). If it exceeds this due to float errors, clip it.
    result = np.clip(result, 0.0, 8.0)
    
    return float(result)
