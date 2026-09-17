def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.filters import threshold_otsu

    # 1. Input Validation and Channel Extraction
    # Ensure input is a numpy array
    img = np.asarray(img)
    
    # Check dimensionality. We expect (H, W, 3) for this dataset.
    # If the image is 2D (H, W), it might be a single channel projection, but dataset spec says (512, 512, 3).
    if img.ndim == 3 and img.shape[2] == 3:
        # Extract DAPI channel (Channel 2: Blue)
        dapi_channel = img[:, :, 2]
    elif img.ndim == 2:
        # Fallback if a single channel image is passed (unlikely based on spec, but safe)
        dapi_channel = img
    else:
        # Unexpected format
        return 0.0

    # Ensure dapi_channel is uint8 for standard 256-bin histogram entropy
    # If it's float, normalize to 0-255. If it's another int type, clip/scale.
    if dapi_channel.dtype != np.uint8:
        # Normalize to 0-255 range if not already
        min_val = np.min(dapi_channel)
        max_val = np.max(dapi_channel)
        if max_val > min_val:
            dapi_channel = ((dapi_channel - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            dapi_channel = np.zeros_like(dapi_channel, dtype=np.uint8)

    # 2. Define Region of Interest (ROI)
    # We want to calculate entropy primarily on the nuclear regions to avoid background dominating the statistics.
    
    pixels_of_interest = None

    # Check if segmentation masks are provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Assume the first mask is relevant (often nuclei or cells). 
        # Dataset description mentions 'segmentation/' directory.
        # We combine all available masks to get a broad foreground if multiple are present, 
        # or just use the first one.
        mask = segmentation_masks[0]
        
        # Ensure mask matches image spatial dimensions
        if mask.shape[:2] == dapi_channel.shape[:2]:
            # Create boolean mask where label > 0
            binary_mask = mask > 0
            if np.any(binary_mask):
                pixels_of_interest = dapi_channel[binary_mask]

    # Fallback: If no valid mask pixels found (or no mask provided), use Otsu thresholding
    if pixels_of_interest is None or pixels_of_interest.size == 0:
        try:
            thresh = threshold_otsu(dapi_channel)
            binary_mask = dapi_channel > thresh
            if np.any(binary_mask):
                pixels_of_interest = dapi_channel[binary_mask]
            else:
                # Image might be empty/black
                return 0.0
        except Exception:
            # Fallback for extremely low contrast/empty images where Otsu fails
            return 0.0

    # 3. Compute Entropy
    # Entropy calculation: H = -sum(p * log2(p))
    
    if pixels_of_interest.size == 0:
        return 0.0

    # Compute histogram (counts of each intensity value 0-255)
    counts, _ = np.histogram(pixels_of_interest, bins=256, range=(0, 255))
    
    # Normalize to get probabilities
    total_pixels = pixels_of_interest.size
    p = counts / total_pixels
    
    # Filter out zero probabilities to avoid log(0)
    p = p[p > 0]
    
    # Calculate Shannon entropy
    entropy = -np.sum(p * np.log2(p))

    return float(entropy)
