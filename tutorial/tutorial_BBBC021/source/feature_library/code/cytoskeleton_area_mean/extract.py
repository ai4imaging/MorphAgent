def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.segmentation import watershed
    from skimage.morphology import binary_closing, disk

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset: (512, 512, 3), Channels: 0=Actin, 1=Tubulin, 2=DAPI
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract channels
    # Ch0: Actin (Cytoskeleton)
    # Ch1: Tubulin (Cytoskeleton)
    # Ch2: DAPI (Nuclei)
    actin = arr[..., 0]
    tubulin = arr[..., 1]
    dapi = arr[..., 2]

    # Initialize masks
    nuclei_labels = None
    cell_labels = None

    # Check if segmentation masks are provided
    # We expect potentially two masks: one for nuclei, one for cells/cytoplasm
    # If provided, we assume they are instance masks (labeled 1..N) or binary masks
    if len(segmentation_masks) >= 2:
        # Assuming order: cell_mask, nuclei_mask based on typical derived data structures,
        # but we need to be robust. Often segmentation dirs have 'nuclei.tif' and 'cells.tif'.
        # Without explicit naming in the *args, we might have to infer or just compute our own
        # to be safe and consistent with the "unbiased feature" goal.
        # However, the prompt implies we should use them if available.
        # Let's try to use them, but if they are just binary, we label them.
        
        # Heuristic: usually the smaller total area is nuclei, larger is cell
        mask1 = np.array(segmentation_masks[0], dtype=np.int32)
        mask2 = np.array(segmentation_masks[1], dtype=np.int32)
        
        area1 = np.count_nonzero(mask1)
        area2 = np.count_nonzero(mask2)
        
        if area1 < area2:
            nuclei_mask_input = mask1
            cell_mask_input = mask2
        else:
            nuclei_mask_input = mask2
            cell_mask_input = mask1
            
        # Ensure they are labeled (instance segmentation)
        if nuclei_mask_input.max() <= 1:
            nuclei_labels = label(nuclei_mask_input)
        else:
            nuclei_labels = nuclei_mask_input
            
        if cell_mask_input.max() <= 1:
            cell_labels = label(cell_mask_input)
        else:
            cell_labels = cell_mask_input

    # Fallback: Compute segmentation if masks are missing or insufficient
    if nuclei_labels is None or cell_labels is None:
        # 1. Nuclei Segmentation (Channel 2)
        # Smooth and threshold
        dapi_smooth = ndimage.gaussian_filter(dapi, sigma=2)
        try:
            thresh_nuc = threshold_otsu(dapi_smooth)
        except ValueError: # Handle empty/uniform images
            thresh_nuc = 0
            
        nuclei_binary = dapi_smooth > thresh_nuc
        nuclei_binary = ndimage.binary_fill_holes(nuclei_binary)
        # Label nuclei
        nuclei_labels = label(nuclei_binary)
        
        # 2. Cell Body Segmentation (Channels 0 & 1)
        # Combine Actin and Tubulin for a robust cell body signal
        cyto_signal = np.maximum(actin, tubulin)
        cyto_smooth = ndimage.gaussian_filter(cyto_signal, sigma=2)
        try:
            thresh_cyto = threshold_otsu(cyto_smooth)
            # Lower threshold slightly to capture faint edges of spreading cells
            thresh_cyto = thresh_cyto * 0.8 
        except ValueError:
            thresh_cyto = 0
            
        cell_binary = cyto_smooth > thresh_cyto
        cell_binary = binary_closing(cell_binary, disk(3))
        cell_binary = ndimage.binary_fill_holes(cell_binary)
        
        # 3. Instance Segmentation via Watershed
        # Use nuclei as markers to split the cell mask into individual cells
        # This ensures 1:1 mapping between nucleus and cell body
        if np.max(nuclei_labels) > 0:
            # Distance transform is often used for watershed, but here we have markers
            # We use the inverse intensity as the "basin"
            image_basin = -cyto_smooth
            cell_labels = watershed(image_basin, nuclei_labels, mask=cell_binary)
        else:
            cell_labels = np.zeros_like(cell_binary, dtype=np.int32)

    # Calculate Cytoplasm Area Mean
    # Logic: For each cell, Area_Cytoplasm = Area_Cell - Area_Nucleus
    
    # Get properties
    props_nuclei = regionprops(nuclei_labels)
    props_cells = regionprops(cell_labels)
    
    # Map areas by label ID
    # Note: Watershed propagates the label ID from marker (nucleus) to the mask (cell)
    # So label 'k' in nuclei_labels corresponds to label 'k' in cell_labels
    
    nuclei_areas = {p.label: p.area for p in props_nuclei}
    cell_areas = {p.label: p.area for p in props_cells}
    
    cytoplasm_areas = []
    
    # Iterate through cell labels (keys)
    for label_id, c_area in cell_areas.items():
        if label_id in nuclei_areas:
            n_area = nuclei_areas[label_id]
            # Cytoplasm is the part of the cell not covered by the nucleus
            # In a perfect segmentation, Cell Mask includes Nucleus.
            # cyto_area = cell_area - nucleus_area
            cyto_area = max(0.0, float(c_area) - float(n_area))
            
            # Filter small artifacts
            if cyto_area > 10: # Minimum valid cytoplasm size in pixels
                cytoplasm_areas.append(cyto_area)
    
    if not cytoplasm_areas:
        return 0.0
        
    result = np.mean(cytoplasm_areas)

    return float(result)
