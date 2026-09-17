def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu, gaussian
    from skimage.morphology import binary_opening, disk, remove_small_objects
    from skimage.segmentation import watershed, clear_border
    from scipy import ndimage

    # Convert to appropriate array type
    # Image is (512, 512, 3), uint8
    arr = np.asarray(img, dtype=np.float32)

    # Check dimensionality
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract Channels
    # Channel 0: Actin (Cytoskeleton) - Primary target for shape
    # Channel 2: DAPI (Nuclei) - Useful for seeding watershed if no mask provided
    actin_ch = arr[..., 0]
    dapi_ch = arr[..., 2]

    # Normalize channels to [0, 1] for processing
    def normalize(ch):
        vmax = np.percentile(ch, 99.5) if ch.size > 0 else 1.0
        if vmax > 0:
            return np.clip(ch / vmax, 0.0, 1.0)
        return np.zeros_like(ch)

    actin_norm = normalize(actin_ch)
    dapi_norm = normalize(dapi_ch)

    # Determine Segmentation Mask
    labeled_mask = None

    # Strategy 1: Use provided segmentation masks
    if len(segmentation_masks) > 0:
        # Check masks to find a suitable cell mask
        # Often masks are passed as (nuclei, cells) or just (cells)
        # We prefer the one with larger coverage usually, or the last one
        for mask in reversed(segmentation_masks):
            if mask is not None and mask.ndim == 2 and mask.shape == actin_ch.shape:
                # Ensure it's labeled
                if mask.max() > 0:
                    labeled_mask = mask.astype(int)
                    break
    
    # Strategy 2: Self-segmentation (Fallback)
    # MCF-7 cells often clump, so simple thresholding of Actin merges cells.
    # We use a marker-controlled watershed approach.
    if labeled_mask is None:
        try:
            # 1. Create markers from Nuclei (DAPI)
            # Smooth DAPI slightly
            dapi_smooth = gaussian(dapi_norm, sigma=2)
            try:
                thresh_dapi = threshold_otsu(dapi_smooth)
            except ValueError: # Handle empty images
                thresh_dapi = 0.1
            
            nuclei_mask = dapi_smooth > thresh_dapi
            nuclei_mask = binary_opening(nuclei_mask, disk(2))
            nuclei_mask = remove_small_objects(nuclei_mask, min_size=50)
            markers = label(nuclei_mask)

            # 2. Create mask from Actin (Cytoplasm)
            # Smooth Actin to get general shape
            actin_smooth = gaussian(actin_norm, sigma=2)
            try:
                thresh_actin = threshold_otsu(actin_smooth)
            except ValueError:
                thresh_actin = 0.1
            
            # Lower threshold slightly to capture protrusions/blebs which might be dimmer
            cell_mask = actin_smooth > (thresh_actin * 0.8)
            cell_mask = remove_small_objects(cell_mask, min_size=100)

            # 3. Watershed
            # Use negative intensity as elevation map
            elevation_map = -actin_smooth
            labeled_mask = watershed(elevation_map, markers, mask=cell_mask)
            
            # 4. Clear border
            # Cells touching the border have artificial straight edges, skewing perimeter/area ratio
            labeled_mask = clear_border(labeled_mask)

        except Exception:
            # Fallback to simple thresholding if watershed fails
            try:
                thresh = threshold_otsu(actin_norm)
                binary = actin_norm > thresh
                binary = clear_border(binary)
                labeled_mask = label(binary)
            except:
                return 0.0

    if labeled_mask is None or labeled_mask.max() == 0:
        return 0.0

    # Compute Feature: Mean Compactness
    # Compactness = (Perimeter^2) / Area
    # Interpretation: 
    #   Circle: (2*pi*r)^2 / (pi*r^2) = 4*pi^2*r^2 / pi*r^2 = 4*pi ~ 12.57
    #   Square: (4s)^2 / s^2 = 16
    #   Complex shapes: >> 12.57
    
    props = regionprops(labeled_mask)
    compactness_values = []

    for prop in props:
        area = prop.area
        perimeter = prop.perimeter
        
        # Filter tiny artifacts that might have slipped through
        if area < 50:
            continue
            
        # Avoid division by zero
        if area > 0:
            # Calculate compactness
            val = (perimeter ** 2) / area
            compactness_values.append(val)

    if not compactness_values:
        return 0.0

    # Return the mean compactness
    result = np.mean(compactness_values)
    
    return float(result)
