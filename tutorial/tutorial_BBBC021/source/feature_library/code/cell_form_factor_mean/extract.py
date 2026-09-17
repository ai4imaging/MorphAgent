def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.segmentation import clear_border
    from skimage.filters import threshold_otsu
    from skimage.morphology import closing, square

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset: (512, 512, 3) uint8, MCF-7 cells
    # If the image is 2D (H, W) or 3D (H, W, C), we need to ensure we have a label mask matching (H, W).
    
    # Check if we have a valid image
    if arr.ndim < 2:
        return 0.0
    
    # Determine image dimensions for mask validation
    if arr.ndim == 3:
        h, w = arr.shape[:2]
    else:
        h, w = arr.shape

    # -------------------------------------------------------------------------
    # 1. Obtain a Labeled Mask
    # -------------------------------------------------------------------------
    labeled_mask = None

    # Strategy: Use provided masks if available, otherwise compute from image
    if len(segmentation_masks) > 0:
        # Prefer the first mask provided. In many pipelines, this might be the 'cell' or 'primary' mask.
        # If multiple masks exist, we assume the first one is the most relevant for "cell" morphology
        # unless specific metadata allows distinguishing nuclei vs cytoplasm (not available here).
        candidate_mask = segmentation_masks[0]
        
        # Ensure mask matches image spatial dimensions
        if candidate_mask.shape[:2] == (h, w):
            # If it's not integer labeled, label it
            if not np.issubdtype(candidate_mask.dtype, np.integer):
                # Assume binary if not integer
                labeled_mask = label(candidate_mask > 0)
            else:
                labeled_mask = candidate_mask
        else:
            # Dimension mismatch, ignore this mask
            pass

    # Fallback: If no valid mask found, generate one from the image
    if labeled_mask is None:
        # Dataset has 3 channels: 0=Actin, 1=Tubulin, 2=DAPI.
        # For "Cell" form factor, Actin (0) and Tubulin (1) define the shape best.
        # We will use the maximum of Actin and Tubulin to capture the full cell body.
        if arr.ndim == 3 and arr.shape[2] >= 2:
            # Normalize channels 0 and 1
            ch0 = arr[..., 0]
            ch1 = arr[..., 1]
            
            # Simple min-max normalization for thresholding
            def norm(c):
                mn, mx = c.min(), c.max()
                return (c - mn) / (mx - mn + 1e-8)
            
            # Combine signals
            signal = np.maximum(norm(ch0), norm(ch1))
        elif arr.ndim == 3:
            # Fallback if fewer channels
            signal = np.mean(arr, axis=2)
        else:
            # Grayscale
            signal = arr

        # Preprocessing
        # Gaussian blur to smooth noise
        signal_smooth = ndimage.gaussian_filter(signal, sigma=2)
        
        # Thresholding
        try:
            thresh = threshold_otsu(signal_smooth)
            binary = signal_smooth > thresh
        except Exception:
            # Fallback for empty/uniform images
            binary = signal_smooth > np.mean(signal_smooth)

        # Morphological cleanup
        # Close gaps
        binary = closing(binary, square(3))
        
        # Label
        labeled_mask = label(binary)

    # -------------------------------------------------------------------------
    # 2. Compute Form Factor
    # -------------------------------------------------------------------------
    
    # Clear objects touching the border to avoid incomplete shapes biasing the form factor
    # (A cut-off circle looks like a D-shape, which has a lower form factor)
    labeled_mask = clear_border(labeled_mask)

    props = regionprops(labeled_mask)
    
    form_factors = []
    
    for prop in props:
        area = prop.area
        perimeter = prop.perimeter
        
        # Filter small noise
        if area < 50:
            continue
            
        # Avoid division by zero
        if perimeter == 0:
            continue
            
        # Calculate Form Factor
        # Formula: (4 * pi * Area) / Perimeter^2
        # Range: 0 to 1 (1 is perfect circle)
        ff = (4.0 * np.pi * area) / (perimeter ** 2)
        
        form_factors.append(ff)

    # -------------------------------------------------------------------------
    # 3. Aggregate Results
    # -------------------------------------------------------------------------
    
    if not form_factors:
        return 0.0
        
    result = np.mean(form_factors)
    
    return float(result)
