def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    from skimage.morphology import closing, square, remove_small_objects
    from skimage.measure import label, regionprops
    from skimage.segmentation import watershed

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Expected: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        # If not standard 3-channel image, return 0.0 as fallback
        return 0.0

    # Channel Mapping based on dataset description:
    # Ch0: Actin (Cytoskeleton)
    # Ch1: Tubulin (Microtubules)
    # Ch2: DAPI (Nucleus)
    
    # 1. Define Nucleus Signal (Channel 2)
    nuc_channel = arr[..., 2]
    
    # 2. Define Cell Body Signal
    # Combine Actin (Ch0) and Tubulin (Ch1) to get a robust cell body representation.
    # Some drugs affect one but not the other, so max projection is safer.
    cell_channel = np.maximum(arr[..., 0], arr[..., 1])

    # --- Segmentation Logic ---
    
    # Check if valid segmentation masks are provided
    # We expect masks to be labeled integer arrays if provided.
    # However, standard pipeline often requires re-segmentation to ensure consistency 
    # between nucleus and cytoplasm definitions for this specific calculation.
    # If masks are provided, we try to use them. If not, we compute them.
    
    nuclei_labels = None
    cell_labels = None

    if len(segmentation_masks) >= 2:
        # Assuming order: Cell Mask, Nuclei Mask (common convention, but checking overlap is safer)
        # Let's check sizes to guess. Nuclei are usually smaller.
        m1 = segmentation_masks[0]
        m2 = segmentation_masks[1]
        
        if np.sum(m1 > 0) > np.sum(m2 > 0):
            cell_labels = m1
            nuclei_labels = m2
        else:
            cell_labels = m2
            nuclei_labels = m1
            
    elif len(segmentation_masks) == 1:
        # If only one mask, assume it's the cell mask (whole cell)
        cell_labels = segmentation_masks[0]
        # We still need nuclei to compute cytoplasm. We will segment nuclei from the image.
    
    # --- On-the-fly Segmentation (if needed) ---
    
    # 1. Segment Nuclei if missing
    if nuclei_labels is None:
        # Smooth
        nuc_smooth = ndimage.gaussian_filter(nuc_channel, sigma=2)
        
        # Threshold
        try:
            thresh_nuc = threshold_otsu(nuc_smooth)
        except ValueError: # Handle empty images
            thresh_nuc = 0
            
        nuc_binary = nuc_smooth > thresh_nuc
        
        # Clean up
        nuc_binary = closing(nuc_binary, square(3))
        nuc_binary = remove_small_objects(nuc_binary, min_size=50)
        
        # Label
        nuclei_labels = label(nuc_binary)

    # 2. Segment Cells (Watershed) if missing
    if cell_labels is None:
        # Smooth cell signal
        cell_smooth = ndimage.gaussian_filter(cell_channel, sigma=2)
        
        # Threshold
        try:
            thresh_cell = threshold_otsu(cell_smooth)
        except ValueError:
            thresh_cell = 0
            
        # Create binary mask
        # Note: The cell mask must contain the nucleus. 
        # We enforce this by union with nuc_binary derived from nuclei_labels
        cell_binary = cell_smooth > thresh_cell
        nuc_mask_from_labels = nuclei_labels > 0
        cell_binary = np.logical_or(cell_binary, nuc_mask_from_labels)
        
        # Clean up
        cell_binary = closing(cell_binary, square(3))
        cell_binary = remove_small_objects(cell_binary, min_size=100)
        
        # Watershed
        # Use nuclei as markers to split touching cells
        # Use inverted intensity as topographic map
        distance = -cell_smooth
        cell_labels = watershed(distance, markers=nuclei_labels, mask=cell_binary)

    # --- Feature Computation: Cytoplasm Area Mean ---
    
    # Get properties
    props_nuc = regionprops(nuclei_labels)
    props_cell = regionprops(cell_labels)
    
    # Create a lookup for nucleus areas: {label_id: area}
    nuc_areas = {p.label: p.area for p in props_nuc}
    
    cytoplasm_areas = []
    
    # Iterate through cells
    for p_cell in props_cell:
        label_id = p_cell.label
        
        # Only consider cells that have a corresponding nucleus
        if label_id in nuc_areas:
            area_cell = p_cell.area
            area_nuc = nuc_areas[label_id]
            
            # Cytoplasm Area = Cell Area - Nucleus Area
            area_cyto = area_cell - area_nuc
            
            # Physical constraint: Area cannot be negative
            if area_cyto < 0:
                area_cyto = 0.0
                
            cytoplasm_areas.append(area_cyto)

    # Compute Mean
    if not cytoplasm_areas:
        return 0.0
        
    result = np.mean(cytoplasm_areas)

    return float(result)
