def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.segmentation import clear_border
    from skimage.morphology import remove_small_objects

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Expected shape: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract relevant channels
    # Channel 0: Actin (Cell Body context)
    # Channel 2: DAPI (Nucleus context)
    actin_channel = arr[..., 0]
    dapi_channel = arr[..., 2]

    # Define masks for Nuclei and Cell Bodies
    nuclei_mask = None
    cell_mask = None

    # Strategy: Use provided masks if available, otherwise generate them
    if len(segmentation_masks) >= 2:
        # Assuming typical order: cell/cytoplasm mask, then nuclei mask, or vice versa.
        # We need to determine which is which. Usually, nuclei are smaller and contained within cells.
        # However, without metadata, we often assume the order or check properties.
        # Let's try to infer or use a heuristic: Nuclei masks usually have smaller total area coverage than cell masks.
        
        mask1 = segmentation_masks[0]
        mask2 = segmentation_masks[1]
        
        # Simple heuristic: The mask with greater total area is likely the cell body (Actin/Cyto), 
        # the smaller one is Nuclei.
        area1 = np.sum(mask1 > 0)
        area2 = np.sum(mask2 > 0)
        
        if area1 > area2:
            cell_mask = mask1
            nuclei_mask = mask2
        else:
            cell_mask = mask2
            nuclei_mask = mask1

    elif len(segmentation_masks) == 1:
        # If only one mask is provided, it's ambiguous. It might be a cell mask.
        # We will generate the nuclei mask from the DAPI channel and use the provided mask as cell body.
        cell_mask = segmentation_masks[0]
        
    # Fallback: Generate masks if they are still None
    if nuclei_mask is None:
        # Generate Nuclei Mask from DAPI (Channel 2)
        try:
            thresh_n = threshold_otsu(dapi_channel)
            nuclei_bool = dapi_channel > thresh_n
            nuclei_bool = remove_small_objects(nuclei_bool, min_size=50)
            nuclei_mask = label(nuclei_bool)
        except Exception:
            return 0.0

    if cell_mask is None:
        # Generate Cell Mask from Actin (Channel 0)
        try:
            thresh_c = threshold_otsu(actin_channel)
            cell_bool = actin_channel > thresh_c
            # Fill holes to ensure centroid is robust
            cell_bool = ndimage.binary_fill_holes(cell_bool)
            cell_bool = remove_small_objects(cell_bool, min_size=200)
            # Clear border cells to avoid bias in centroid calculation due to cropping
            cell_bool = clear_border(cell_bool)
            cell_mask = label(cell_bool)
        except Exception:
            return 0.0

    # Ensure masks are labeled integers
    if nuclei_mask.dtype == bool:
        nuclei_mask = label(nuclei_mask)
    if cell_mask.dtype == bool:
        cell_mask = label(cell_mask)

    # Calculate properties
    nuclei_props = regionprops(nuclei_mask)
    cell_props = regionprops(cell_mask)

    if not nuclei_props or not cell_props:
        return 0.0

    # Create a mapping of Cell Label -> Cell Centroid
    # We use a dictionary for fast lookup
    cell_centroids = {prop.label: prop.centroid for prop in cell_props}

    displacements = []

    # Iterate through nuclei to find their parent cells
    # We assume a nucleus belongs to a cell if the nucleus centroid falls within the cell mask
    for n_prop in nuclei_props:
        yc, xc = n_prop.centroid
        
        # Check bounds
        yi, xi = int(yc), int(xc)
        if 0 <= yi < cell_mask.shape[0] and 0 <= xi < cell_mask.shape[1]:
            parent_cell_label = cell_mask[yi, xi]
            
            # If the nucleus centroid lands on a valid cell label
            if parent_cell_label > 0 and parent_cell_label in cell_centroids:
                # Get centroids
                n_centroid = np.array(n_prop.centroid)
                c_centroid = np.array(cell_centroids[parent_cell_label])
                
                # Calculate Euclidean distance
                dist = np.linalg.norm(n_centroid - c_centroid)
                displacements.append(dist)

    # If no valid nucleus-cell pairs were found
    if not displacements:
        return 0.0

    # Return the mean displacement index
    result = np.mean(displacements)
    
    return float(result)
