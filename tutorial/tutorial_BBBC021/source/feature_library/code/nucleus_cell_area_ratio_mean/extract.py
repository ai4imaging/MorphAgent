def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu, threshold_li
    from skimage.measure import label
    from skimage.segmentation import watershed
    from skimage.morphology import remove_small_objects, binary_closing, disk

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset: (512, 512, 3), Channels: 0=Actin, 1=Tubulin, 2=DAPI
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract channels
    # Channel 2 is DAPI (Nucleus)
    # Channels 0 and 1 are Actin/Tubulin (Cytoskeleton/Cell Body)
    nuc_channel = arr[..., 2]
    # Combine Actin and Tubulin for a robust cell body signal
    cyto_channel = np.maximum(arr[..., 0], arr[..., 1])

    # Basic normalization for thresholding stability
    def normalize_ch(ch):
        v_min, v_max = np.min(ch), np.max(ch)
        if v_max - v_min > 1e-5:
            return (ch - v_min) / (v_max - v_min)
        return np.zeros_like(ch)

    nuc_norm = normalize_ch(nuc_channel)
    cyto_norm = normalize_ch(cyto_channel)

    # --- 1. Nucleus Segmentation (Seeds) ---
    # Smooth slightly to reduce noise
    nuc_smooth = ndimage.gaussian_filter(nuc_norm, sigma=2)
    
    # Thresholding
    try:
        thresh_nuc = threshold_otsu(nuc_smooth)
    except Exception:
        return 0.0 # Fallback if image is empty/uniform

    mask_nuc = nuc_smooth > thresh_nuc
    
    # Clean up nucleus mask
    mask_nuc = binary_closing(mask_nuc, disk(2))
    mask_nuc = remove_small_objects(mask_nuc, min_size=50)
    
    # Label nuclei
    markers_nuc = label(mask_nuc)
    num_cells = np.max(markers_nuc)

    if num_cells == 0:
        return 0.0

    # --- 2. Cell Body Segmentation (Watershed) ---
    # Smooth cyto channel
    cyto_smooth = ndimage.gaussian_filter(cyto_norm, sigma=2)
    
    # Thresholding for cell body mask
    # Li threshold is often better for cytoplasm which has long tails/faint edges
    try:
        thresh_cyto = threshold_li(cyto_smooth)
    except Exception:
        # Fallback to Otsu if Li fails or just use mean
        thresh_cyto = np.mean(cyto_smooth)

    # The cell mask must include the nucleus mask to ensure containment
    mask_cyto = (cyto_smooth > thresh_cyto) | mask_nuc
    mask_cyto = binary_closing(mask_cyto, disk(3))
    mask_cyto = remove_small_objects(mask_cyto, min_size=100)

    # Watershed segmentation
    # We use the gradient of the image as the "elevation map" or just the inverted intensity
    # Here, inverted intensity works well for bright objects on dark background
    elevation_map = -cyto_smooth
    
    # Run watershed: expand from nucleus markers into the cell mask
    segmentation_cells = watershed(elevation_map, markers_nuc, mask=mask_cyto)

    # --- 3. Feature Calculation ---
    # We need to calculate Area(Nucleus) / Area(Cell) for each cell ID
    
    # Use bincount for fast area summation
    # Labels are integers starting from 1
    # We flatten the arrays to 1D for bincount
    
    # Count pixels for each label in the nucleus mask (should match markers_nuc exactly)
    # Note: We use markers_nuc for nucleus definition and segmentation_cells for cell definition.
    # However, watershed preserves the seed labels. So label 'k' in segmentation_cells corresponds to label 'k' in markers_nuc.
    
    # Get areas from the labeled arrays
    # max_label needs to be at least num_cells to cover all indices
    nuc_areas = np.bincount(markers_nuc.ravel())
    cell_areas = np.bincount(segmentation_cells.ravel())

    # We only care about labels 1 to num_cells
    # Ensure arrays are large enough (bincount size depends on max value in array)
    limit = min(len(nuc_areas), len(cell_areas))
    
    ratios = []
    
    # Iterate through valid cell IDs (skip background 0)
    for i in range(1, limit):
        n_area = nuc_areas[i]
        c_area = cell_areas[i]
        
        # Filter out artifacts
        if n_area < 10 or c_area < 10:
            continue
            
        # Theoretical constraint: Nucleus is inside Cell, so Cell Area >= Nucleus Area
        # If segmentation is perfect, c_area >= n_area.
        # If c_area < n_area (rare segmentation glitch), we clamp to 1.0 or skip.
        if c_area == 0: 
            continue
            
        ratio = n_area / c_area
        
        # Clamp ratio to [0, 1] to be physically meaningful
        if ratio > 1.0:
            ratio = 1.0
            
        ratios.append(ratio)

    if not ratios:
        return 0.0

    result = np.mean(ratios)
    return float(result)
