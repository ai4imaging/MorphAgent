def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import stats
    from skimage.filters import threshold_otsu

    # 1. Input Validation and Channel Extraction
    # Ensure input is a numpy array
    img = np.asarray(img)
    
    # Check dimensionality. We expect (H, W, 3) for this dataset.
    # If the image is 2D (H, W), it might be a single channel or projection. 
    # If 3D (H, W, C), we extract the DAPI channel (index 2).
    if img.ndim == 3 and img.shape[2] == 3:
        # Channel 2 is DAPI (Blue) based on dataset description
        dapi_channel = img[:, :, 2]
    elif img.ndim == 2:
        # Fallback: if only 2D provided, assume it's the relevant channel
        dapi_channel = img
    else:
        # Unexpected shape, return 0.0
        return 0.0

    # 2. Define Region of Interest (ROI)
    # We only want to calculate entropy on the nuclear pixels, not the background.
    # Background (0) dominates the histogram and skews entropy.
    
    mask = None
    
    # Check if segmentation masks are provided
    if len(segmentation_masks) > 0:
        # Use the first available mask
        # Masks are typically labeled integers (0=bg, 1..N=cells)
        seg = segmentation_masks[0]
        
        # Ensure mask shape matches image shape (handle potential mismatches)
        if seg.shape == dapi_channel.shape:
            mask = seg > 0
    
    # If no valid mask provided, generate a simple one using Otsu's thresholding
    # This ensures we focus on foreground structure rather than black background
    if mask is None:
        try:
            # Calculate threshold on the DAPI channel
            # Check if image is not empty/constant
            if dapi_channel.max() > dapi_channel.min():
                thresh = threshold_otsu(dapi_channel)
                mask = dapi_channel > thresh
            else:
                # Image is constant, entropy is 0
                return 0.0
        except Exception:
            # Fallback for completely empty/black images
            return 0.0

    # 3. Extract Pixels of Interest
    # Select pixels that fall within the mask
    roi_pixels = dapi_channel[mask]

    # Handle case where mask is empty (no foreground detected)
    if roi_pixels.size == 0:
        return 0.0

    # 4. Compute Histogram and Entropy
    # We compute the histogram of intensity values.
    # Since input is uint8 (0-255), we use 256 bins.
    # If input was float, we would bin it, but dataset spec says uint8.
    # Even if converted to float during loading, we can bin into discrete levels.
    
    # Determine range based on data type or actual values
    if roi_pixels.dtype == np.uint8:
        hist_range = (0, 255)
        bins = 256
    else:
        # If float, bin into 256 bins across the data range
        hist_range = (roi_pixels.min(), roi_pixels.max())
        bins = 256

    # Compute histogram counts
    counts, _ = np.histogram(roi_pixels, bins=bins, range=hist_range)

    # Normalize to get probabilities
    # Filter out zero counts to avoid log(0)
    counts = counts[counts > 0]
    probs = counts / counts.sum()

    # Calculate Shannon Entropy
    # H = -sum(p * log2(p))
    entropy_val = -np.sum(probs * np.log2(probs))

    return float(entropy_val)
