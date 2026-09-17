def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.filters import threshold_otsu

    # 1. Input Validation and Channel Extraction
    # Ensure input is a numpy array
    img = np.asarray(img)
    
    # Check dimensionality and extract DAPI channel
    # Dataset description: (512, 512, 3), Channel 2 is DAPI (Blue)
    if img.ndim == 3 and img.shape[2] == 3:
        dapi_channel = img[:, :, 2]
    elif img.ndim == 2:
        # Fallback if a single channel image is passed (unlikely based on desc, but safe)
        dapi_channel = img
    else:
        # Unexpected format
        return 0.0

    # Convert to float for statistical calculation to prevent overflow/precision issues
    # We keep the original scale (likely 0-255) to maintain interpretability of "intensity deviation"
    dapi_float = dapi_channel.astype(np.float32)

    # 2. Masking Strategy
    # We need to calculate the standard deviation only within the nuclear regions.
    # Including the large black background would skew the std dev significantly (measuring contrast vs background).
    
    mask = None
    
    # Check if segmentation masks are provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use the first provided mask (assuming it corresponds to nuclei or cells)
        # Ensure mask matches image spatial dimensions
        provided_mask = np.asarray(segmentation_masks[0])
        if provided_mask.shape[:2] == dapi_channel.shape[:2]:
            mask = provided_mask > 0
            
    # Fallback: If no valid mask provided, generate one using Otsu thresholding
    if mask is None:
        # Calculate threshold only on pixels > 0 to avoid background skewing the threshold too low
        # or just use standard Otsu on the whole image
        try:
            thresh = threshold_otsu(dapi_float)
            mask = dapi_float > thresh
        except Exception:
            # Fallback for extremely low contrast/empty images where Otsu might fail
            mask = dapi_float > 10  # Simple noise floor threshold

    # 3. Feature Calculation
    # Select pixels belonging to the mask
    nuclear_pixels = dapi_float[mask]

    # Handle edge case: No foreground pixels found
    if nuclear_pixels.size == 0:
        return 0.0

    # Calculate Standard Deviation of intensity
    # High std dev = high texture/contrast (heterochromatin, apoptotic bodies)
    # Low std dev = smooth/diffuse staining
    std_dev = np.std(nuclear_pixels)

    return float(std_dev)
