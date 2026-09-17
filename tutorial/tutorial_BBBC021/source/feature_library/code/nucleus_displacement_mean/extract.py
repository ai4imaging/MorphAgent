def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.segmentation import watershed, clear_border
    from skimage.morphology import binary_opening, disk

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality
    # Expected shape: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Normalize intensity to [0, 1] for processing
    # We process channels individually later, but good to have a normalized copy
    arr_norm = np.clip(arr / 255.0, 0.0, 1.0)

    # Define masks for Nuclei and Cells
    nuclei_mask = None
    cells_mask = None

    # Strategy:
    # 1. Use provided segmentation masks if available and valid.
    # 2. If not, generate them on the fly using channel information.
    #    - Ch2 (Blue) = Nuclei
    #    - Ch0 (Red) + Ch1 (Green) = Cell Body

    if len(segmentation_masks) >= 2:
        # Assuming standard order: first mask often cells, second nuclei, or vice versa.
        # However, without strict metadata, we often have to guess or rely on file naming which isn't available here.
        # A robust heuristic: Nuclei are usually smaller and contained within cells.
        # Let's try to identify which is which based on area or containment, 
        # but for simplicity in this constrained environment, we will check if we can map them.
        # If the system provides specific masks, we usually assume:
        # mask 0: cell/cytoplasm, mask 1: nuclei (common convention)
        # OR we check the labels.
        
        # Let's try to use the provided masks directly.
        # We need to determine which is which.
        m1 = segmentation_masks[0]
        m2 = segmentation_masks[1]
        
        # Heuristic: The mask with more total area is likely the cell mask.
        if np.sum(m1 > 0) > np.sum(m2 > 0):
            cells_mask = m1
            nuclei_mask = m2
        else:
            cells_mask = m2
            nuclei_mask = m1
            
    elif len(segmentation_masks) == 1:
        # Only one mask provided. If it's cells, we need nuclei. If it's nuclei, we need cells.
        # We will generate the missing one.
        provided_mask = segmentation_masks[0]
        
        # Generate nuclei from Ch2
        nuc_ch = arr_norm[..., 2]
        try:
            thresh_n = threshold_otsu(nuc_ch)
        except ValueError:
            thresh_n = 0.1
        generated_nuclei_mask = label(nuc_ch > thresh_n)
        
        # Compare provided mask with generated nuclei
        # If provided mask is significantly larger than generated nuclei, assume provided is cells.
        if np.sum(provided_mask > 0) > 1.5 * np.sum(generated_nuclei_mask > 0):
            cells_mask = provided_mask
            nuclei_mask = generated_nuclei_mask
        else:
            # Provided is likely nuclei
            nuclei_mask = provided_mask
            # Generate cell mask from Ch0 + Ch1
            cell_ch = arr_norm[..., 0] + arr_norm[..., 1]
            try:
                thresh_c = threshold_otsu(cell_ch)
            except ValueError:
                thresh_c = 0.1
            # Use watershed to split cells based on nuclei markers
            binary_cells = cell_ch > thresh_c
            cells_mask = watershed(-cell_ch, nuclei_mask, mask=binary_cells)

    else:
        # No masks provided. Generate both.
        # 1. Nuclei Segmentation (Channel 2 - Blue)
        nuc_ch = arr_norm[..., 2]
        # Smooth slightly to reduce noise
        nuc_ch = ndimage.gaussian_filter(nuc_ch, sigma=1)
        try:
            thresh_n = threshold_otsu(nuc_ch)
        except ValueError:
            thresh_n = 0.0
        
        binary_nuc = nuc_ch > thresh_n
        binary_nuc = binary_opening(binary_nuc, footprint=disk(2))
        nuclei_mask = label(binary_nuc)
        
        # 2. Cell Segmentation (Channel 0 - Red + Channel 1 - Green)
        # Actin and Tubulin define the cell body
        cell_ch = arr_norm[..., 0] + arr_norm[..., 1]
        cell_ch = ndimage.gaussian_filter(cell_ch, sigma=2)
        try:
            thresh_c = threshold_otsu(cell_ch)
        except ValueError:
            thresh_c = 0.0
            
        binary_cells = cell_ch > thresh_c
        
        # 3. Watershed to define cell boundaries based on nuclei
        # This ensures 1-to-1 mapping or at least that cells contain nuclei
        cells_mask = watershed(-cell_ch, nuclei_mask, mask=binary_cells)

    # Validation: Ensure we have valid masks
    if nuclei_mask is None or cells_mask is None:
        return 0.0
    
    if nuclei_mask.max() == 0 or cells_mask.max() == 0:
        return 0.0

    # Clear border objects to avoid bias from cut-off cells
    # We clear cells touching the border, and their corresponding nuclei
    cells_mask = clear_border(cells_mask)
    # Filter nuclei: keep only those that are inside the remaining valid cells
    # Fast way: mask nuclei with the binary version of the cleared cell mask
    nuclei_mask = np.where(cells_mask > 0, nuclei_mask, 0)

    # Compute properties
    # We need centroids.
    # regionprops returns a list of properties.
    nuc_props = regionprops(nuclei_mask)
    cell_props = regionprops(cells_mask)

    # Create a lookup for cell centroids by label
    # Note: regionprops labels match the mask values
    cell_centroids = {prop.label: np.array(prop.centroid) for prop in cell_props}

    displacements = []

    # Iterate through nuclei to find their parent cells
    for n_prop in nuc_props:
        n_label = n_prop.label
        n_centroid = np.array(n_prop.centroid) # (row, col)
        
        # Find the corresponding cell label at the nucleus centroid location
        # We cast centroid to int to index the array
        r, c = int(n_centroid[0]), int(n_centroid[1])
        
        # Boundary check just in case
        if 0 <= r < cells_mask.shape[0] and 0 <= c < cells_mask.shape[1]:
            parent_cell_label = cells_mask[r, c]
            
            # If the nucleus centroid falls on a valid cell (not background 0)
            if parent_cell_label in cell_centroids:
                c_centroid = cell_centroids[parent_cell_label]
                
                # Calculate Euclidean distance
                # dist = sqrt((x1-x2)^2 + (y1-y2)^2)
                dist = np.linalg.norm(n_centroid - c_centroid)
                displacements.append(dist)
            else:
                # Fallback: If nucleus centroid is not directly on a cell label (e.g. slight gap),
                # we could try to find which cell label overlaps most with this nucleus.
                # For efficiency, we skip this edge case in this implementation.
                pass

    if not displacements:
        return 0.0

    result = np.mean(displacements)
    return float(result)
