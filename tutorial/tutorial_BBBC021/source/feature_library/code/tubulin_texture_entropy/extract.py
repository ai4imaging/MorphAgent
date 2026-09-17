def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    
    # 1. Input Validation and Channel Extraction
    # Convert to appropriate array type if needed, but keep as uint8 for histogramming if possible
    # or convert to float for processing then binning.
    # The dataset description says input is uint8.
    
    img_arr = np.asarray(img)
    
    # Handle dimensionality
    # Expected: (512, 512, 3)
    if img_arr.ndim != 3 or img_arr.shape[2] != 3:
        # Fallback for potential unexpected shapes (e.g., if channels are first)
        if img_arr.ndim == 3 and img_arr.shape[0] == 3:
            img_arr = np.transpose(img_arr, (1, 2, 0))
        else:
            return 0.0

    # Extract Tubulin Channel (Channel 1 - Green)
    tubulin_channel = img_arr[:, :, 1]

    # 2. Define Region of Interest (ROI)
    # We want to calculate entropy only on the cellular regions, not the black background.
    # Including the background (huge peak at 0) would distort the entropy measure.
    
    mask = None
    
    # Check if segmentation masks are provided
    if len(segmentation_masks) > 0:
        # Combine all available masks into a single boolean mask
        combined_mask = np.zeros(tubulin_channel.shape, dtype=bool)
        for seg in segmentation_masks:
            if seg is not None:
                # Ensure seg matches image shape (handle potential 2D vs 3D issues)
                if seg.shape == tubulin_channel.shape:
                    combined_mask = combined_mask | (seg > 0)
                elif seg.ndim == 3 and seg.shape[:2] == tubulin_channel.shape:
                     # If mask is 3D (e.g. one-hot), flatten or take max
                    combined_mask = combined_mask | (np.max(seg, axis=2) > 0)
        
        if np.any(combined_mask):
            mask = combined_mask

    # Fallback: If no masks provided or mask is empty, generate a foreground mask
    if mask is None:
        # Simple background subtraction/thresholding
        # Smooth slightly to reduce noise before thresholding
        smoothed = ndimage.gaussian_filter(tubulin_channel.astype(float), sigma=2.0)
        try:
            thresh = threshold_otsu(smoothed)
            mask = smoothed > thresh
        except Exception:
            # Fallback if image is uniform
            mask = np.ones(tubulin_channel.shape, dtype=bool)

    # 3. Extract Pixel Values
    # Get the intensity values only within the mask
    valid_pixels = tubulin_channel[mask]

    # Edge case: No valid pixels
    if valid_pixels.size == 0:
        return 0.0

    # 4. Compute Entropy
    # We treat the pixel intensities as discrete states (0-255 for uint8).
    # If the input was float, we would need to bin it. Since it's uint8, we can use bins 0-256.
    
    # Ensure pixels are uint8 for histogramming
    if valid_pixels.dtype != np.uint8:
        # If float [0,1], scale to 255
        if valid_pixels.max() <= 1.0:
            valid_pixels = (valid_pixels * 255).astype(np.uint8)
        else:
            valid_pixels = valid_pixels.astype(np.uint8)

    # Compute histogram (counts per intensity level)
    counts, _ = np.histogram(valid_pixels, bins=256, range=(0, 255))
    
    # Normalize to get probabilities
    # Avoid division by zero if counts sum is 0 (unlikely given size check above)
    total_count = counts.sum()
    if total_count == 0:
        return 0.0
        
    probs = counts / total_count
    
    # Filter out zero probabilities to avoid log(0)
    probs = probs[probs > 0]
    
    # Shannon Entropy formula: -sum(p * log2(p))
    entropy_val = -np.sum(probs * np.log2(probs))

    return float(entropy_val)
