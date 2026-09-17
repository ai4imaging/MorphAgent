def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import closing, square
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max
    
    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset is 512x512x3 (H, W, C)
    if arr.ndim != 3 or arr.shape[2] != 3:
        # Fallback for unexpected shapes, e.g., if channels are first or 2D
        if arr.ndim == 2:
            # Treat as single channel grayscale
            actin_channel = arr
            nuclei_channel = arr
        elif arr.ndim == 3 and arr.shape[0] == 3:
            # Channels first
            actin_channel = arr[0]
            nuclei_channel = arr[2]
        else:
            return 0.0
    else:
        # Standard case: (H, W, C) = (512, 512, 3)
        # Channel 0: Actin (Red) - Best for cell body orientation
        # Channel 2: DAPI (Blue) - Best for nuclei seeding
        actin_channel = arr[:, :, 0]
        nuclei_channel = arr[:, :, 2]

    # Normalize channels
    def normalize(ch):
        vmax = np.percentile(ch, 99.5) if ch.size > 0 else 1.0
        if vmax <= 0: vmax = 1.0
        return np.clip(ch / vmax, 0.0, 1.0)

    actin_norm = normalize(actin_channel)
    nuclei_norm = normalize(nuclei_channel)

    # Determine Segmentation Mask
    labeled_mask = None
    
    # 1. Try using provided segmentation masks
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        mask_input = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions
        if mask_input.shape[:2] == actin_channel.shape[:2]:
            if mask_input.ndim == 2:
                labeled_mask = mask_input.astype(int)
            elif mask_input.ndim == 3:
                # If 3D mask, take max projection or first slice
                labeled_mask = np.max(mask_input, axis=-1).astype(int)
    
    # 2. Fallback: Compute segmentation if no valid mask provided
    if labeled_mask is None:
        # Thresholding
        try:
            thresh_nuc = threshold_otsu(nuclei_norm)
            thresh_actin = threshold_otsu(actin_norm)
        except Exception:
            return 0.0 # Image likely empty or uniform

        # Generate markers from nuclei
        nuclei_mask = nuclei_norm > thresh_nuc
        # Distance transform for separating touching nuclei
        distance = ndimage.distance_transform_edt(nuclei_mask)
        # Find peaks
        coords = peak_local_max(distance, min_distance=7, labels=nuclei_mask)
        mask_peaks = np.zeros(distance.shape, dtype=bool)
        mask_peaks[tuple(coords.T)] = True
        markers = label(mask_peaks)
        
        # Cell body mask
        cell_mask = actin_norm > thresh_actin
        cell_mask = closing(cell_mask, square(3))
        
        # Watershed segmentation
        # Use negative actin intensity as "elevation" map so watershed fills bright regions
        labeled_mask = watershed(-actin_norm, markers, mask=cell_mask)

    # Feature Extraction: Orientation Entropy
    # Get properties of labeled regions
    props = regionprops(labeled_mask)
    
    if not props:
        return 0.0

    orientations = []
    
    for prop in props:
        # Filter noise: ignore very small objects
        if prop.area < 50:
            continue
            
        # Filter round objects: Orientation is undefined/unstable for circles.
        # Eccentricity: 0 = circle, 1 = line. 
        # We only care about orientation of elongated cells.
        if prop.eccentricity < 0.2:
            continue
            
        # prop.orientation is in radians [-pi/2, pi/2]
        orientations.append(prop.orientation)

    if not orientations:
        return 0.0

    # Compute Histogram of Orientations
    # Range is -pi/2 to pi/2. 
    # We use 20 bins to capture the distribution shape.
    hist, _ = np.histogram(orientations, bins=20, range=(-np.pi/2, np.pi/2), density=True)
    
    # Normalize histogram to get probabilities (sum = 1)
    # Note: density=True makes integral=1, but for discrete entropy we want sum(p)=1
    # Re-normalize explicitly to be safe with discrete bins
    probabilities = hist / np.sum(hist)
    
    # Remove zeros to avoid log(0)
    probabilities = probabilities[probabilities > 0]
    
    if probabilities.size == 0:
        return 0.0

    # Calculate Shannon Entropy
    # H = -sum(p * log2(p))
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return float(entropy)
