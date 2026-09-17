def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import regionprops, label
    from skimage.filters import threshold_otsu
    
    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Expected: (512, 512, 3)
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Channel 2 is DAPI (Blue), which is critical for nuclear analysis
        dapi_channel = arr[:, :, 2]
    elif arr.ndim == 2:
        # Fallback if passed a single channel image (assume it's the relevant one)
        dapi_channel = arr
    else:
        return 0.0

    # Intensity normalization
    # Normalize to [0, 1] based on uint8 range or data range
    if dapi_channel.max() > 1.0:
        dapi_channel = dapi_channel / 255.0
    dapi_channel = np.clip(dapi_channel, 0.0, 1.0)

    # Handle segmentation masks
    labeled_mask = None
    
    # Check if valid segmentation masks are provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        mask_input = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions
        if mask_input.shape[:2] == dapi_channel.shape[:2]:
            # If mask is not integer labeled, label it
            if mask_input.dtype == bool or np.unique(mask_input).size <= 2:
                labeled_mask = label(mask_input > 0)
            else:
                labeled_mask = mask_input.astype(int)

    # Fallback: Generate segmentation if no mask provided
    if labeled_mask is None:
        # Simple Otsu thresholding on DAPI channel
        # Smooth slightly to reduce noise
        smooth_dapi = ndimage.gaussian_filter(dapi_channel, sigma=2)
        try:
            thresh = threshold_otsu(smooth_dapi)
            binary_mask = smooth_dapi > thresh
            labeled_mask = label(binary_mask)
        except Exception:
            # Fallback for completely empty/black images
            return 0.0

    # Extract properties for each cell
    # We need intensity (from DAPI) and shape (circularity)
    props = regionprops(labeled_mask, intensity_image=dapi_channel)
    
    if not props:
        return 0.0

    # Collect features
    mean_intensities = []
    circularities = []
    areas = []

    for prop in props:
        # Filter small debris
        if prop.area < 50:
            continue
            
        # Calculate circularity: 4 * pi * Area / Perimeter^2
        # Perfect circle = 1.0
        perimeter = prop.perimeter
        if perimeter == 0:
            circ = 0.0
        else:
            circ = (4 * np.pi * prop.area) / (perimeter ** 2)
        
        mean_intensities.append(prop.mean_intensity)
        circularities.append(circ)
        areas.append(prop.area)

    mean_intensities = np.array(mean_intensities)
    circularities = np.array(circularities)
    
    n_cells = len(mean_intensities)
    if n_cells == 0:
        return 0.0

    # Define Mitotic Criteria
    # 1. High Intensity: Mitotic chromatin is condensed and brighter.
    #    We use an adaptive threshold based on population statistics.
    #    Mitotic cells are outliers on the high end.
    if n_cells > 5:
        median_int = np.median(mean_intensities)
        std_int = np.std(mean_intensities)
        # Threshold: Median + 2 std devs (identifies bright outliers)
        intensity_threshold = median_int + (2.0 * std_int)
    else:
        # Fallback for very sparse images: absolute high intensity check
        # Assuming normalized [0,1] data, mitosis is usually very bright
        intensity_threshold = 0.6 

    # 2. High Circularity: Cells round up during mitosis.
    circularity_threshold = 0.80

    # Count mitotic cells
    # Must satisfy BOTH criteria
    is_mitotic = (mean_intensities > intensity_threshold) & (circularities > circularity_threshold)
    n_mitotic = np.sum(is_mitotic)

    # Calculate fraction
    fraction = float(n_mitotic) / float(n_cells)

    return float(fraction)
