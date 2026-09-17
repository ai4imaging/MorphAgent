def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    from skimage.measure import label, regionprops
    from skimage.morphology import remove_small_objects, binary_closing, disk
    from skimage.segmentation import watershed

    # Convert to appropriate array type
    # The input is expected to be (H, W, C) = (512, 512, 3)
    arr = np.asarray(img)
    
    # Handle dimensionality and validate input
    if arr.ndim != 3 or arr.shape[2] != 3:
        # If not the expected 3-channel image, return 0.0
        return 0.0

    # Extract relevant channels based on dataset description
    # Channel 0: Actin (Cytoskeleton/Cell Body)
    # Channel 2: DAPI (Nucleus)
    actin_channel = arr[..., 0]
    dapi_channel = arr[..., 2]

    # --- Segmentation Logic ---
    # We need to identify individual cells to compute the ratio per cell.
    # Strategy: Marker-controlled watershed.
    # 1. Identify Nuclei (Seeds) from DAPI
    # 2. Identify Total Cell Area (Mask) from Actin
    # 3. Propagate Nuclei labels to fill Actin mask

    # 1. Nuclei Segmentation
    # Smooth to reduce noise
    dapi_smooth = ndimage.gaussian_filter(dapi_channel, sigma=2)
    
    # Thresholding
    try:
        thresh_nuc = threshold_otsu(dapi_smooth)
        mask_nuc = dapi_smooth > thresh_nuc
    except Exception:
        # Fallback if image is empty or uniform
        return 0.0

    # Clean up nuclei mask
    mask_nuc = remove_small_objects(mask_nuc, min_size=50)
    mask_nuc = binary_closing(mask_nuc, disk(2))
    
    # Label nuclei to create markers
    markers_nuc = label(mask_nuc)
    num_cells = markers_nuc.max()

    if num_cells == 0:
        return 0.0

    # 2. Cell Body Segmentation
    # Smooth actin channel
    actin_smooth = ndimage.gaussian_filter(actin_channel, sigma=2)
    
    # Thresholding for cell body
    # We use a slightly more permissive approach for cell boundaries. 
    # If Otsu fails (e.g. very dim actin), we might fallback to the nuclei mask area,
    # but ideally we want the actin area.
    try:
        thresh_cell = threshold_otsu(actin_smooth)
        mask_cell = actin_smooth > thresh_cell
    except Exception:
        # If actin is empty, assume cell area = nucleus area (ratio 1.0)
        mask_cell = mask_nuc.copy()

    # Ensure cell mask contains nuclei (biological constraint)
    mask_cell = np.logical_or(mask_cell, mask_nuc)
    
    # Clean up cell mask
    mask_cell = remove_small_objects(mask_cell, min_size=100)
    mask_cell = binary_closing(mask_cell, disk(3))

    # 3. Watershed for Instance Segmentation
    # We use the nuclei markers to partition the cell mask
    # We compute the watershed on the negative intensity or distance transform.
    # Here, simply using the mask is often sufficient for morphological separation if cells aren't too clumped,
    # but using the intensity gradient is better. Let's use the inverse of the actin intensity as the "elevation".
    
    # Normalize actin for watershed landscape
    actin_float = actin_smooth.astype(float)
    actin_float /= (actin_float.max() + 1e-6)
    
    # Watershed: expand markers within the mask_cell
    # The 'watershed' function fills the 'mask_cell' starting from 'markers_nuc'
    labeled_cells = watershed(-actin_float, markers_nuc, mask=mask_cell)

    # --- Feature Calculation ---
    # Calculate N/C ratio for each cell
    # Ratio = Area(Nucleus) / Area(Total Cell)
    
    ratios = []
    
    # We iterate through region properties. 
    # Note: labeled_cells contains the full cell extent for each label ID.
    # markers_nuc contains the nucleus extent for each label ID.
    
    # Get properties for total cells
    props_cells = regionprops(labeled_cells)
    
    # Get properties for nuclei (we need to re-measure because markers_nuc might have gaps 
    # if we just used the label indices, but here indices are aligned by watershed)
    props_nuclei = regionprops(markers_nuc)
    
    # Create a lookup for nuclei areas by label
    nuc_areas = {p.label: p.area for p in props_nuclei}
    
    for prop in props_cells:
        cell_label = prop.label
        cell_area = prop.area
        
        # Retrieve corresponding nucleus area
        nuc_area = nuc_areas.get(cell_label, 0)
        
        if cell_area > 0 and nuc_area > 0:
            # Biological constraint: Nucleus cannot be larger than the cell.
            # In our segmentation, cell_mask includes mask_nuc, so cell_area >= nuc_area is guaranteed geometrically
            # unless watershed did something weird with boundaries (unlikely with mask constraint).
            ratio = nuc_area / float(cell_area)
            
            # Clamp for safety
            ratio = min(max(ratio, 0.0), 1.0)
            ratios.append(ratio)

    # --- Aggregation ---
    if not ratios:
        return 0.0
        
    result = np.mean(ratios)

    return float(result)
