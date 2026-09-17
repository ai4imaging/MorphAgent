def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu

    # 1. Data Loading and Validation
    # Ensure input is a numpy array
    arr = np.asarray(img)
    
    # Check dimensions: Expected (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # 2. Channel Extraction
    # Channel 1: Tubulin (Green) -> Cell Body proxy
    # Channel 2: DAPI (Blue) -> Nucleus
    tubulin_ch = arr[..., 1].astype(np.float32)
    dapi_ch = arr[..., 2].astype(np.float32)

    # 3. Segmentation Handling
    # We need to identify individual cells to calculate per-cell displacement.
    # If segmentation masks are provided, use them. Otherwise, generate a crude mask.
    
    labels = None
    
    if len(segmentation_masks) > 0:
        # Check for a valid mask. Usually, the first mask is nuclei or cells.
        # We prefer a cell mask if available, or nuclei mask to seed cells.
        # Let's assume the first mask provided is usable.
        mask_candidate = segmentation_masks[0]
        if mask_candidate is not None and mask_candidate.shape == dapi_ch.shape:
            labels = mask_candidate.astype(int)
    
    # Fallback: Generate segmentation if no valid mask provided
    if labels is None:
        # Simple nuclei segmentation using Otsu
        try:
            thresh = threshold_otsu(dapi_ch)
            binary_nuc = dapi_ch > thresh
            # Clean up noise
            binary_nuc = ndimage.binary_opening(binary_nuc, structure=np.ones((3,3)))
            # Label connected components
            labels = label(binary_nuc)
        except Exception:
            return 0.0

    # 4. Feature Computation: Center Displacement
    # We calculate the center of mass for DAPI (nucleus) and Tubulin (cell body)
    # for each labeled object.
    
    # Get properties for the labels based on the DAPI channel (to get bounding boxes/indices)
    props = regionprops(labels, intensity_image=dapi_ch)
    
    displacements = []
    
    for prop in props:
        # Filter small artifacts
        if prop.area < 50:
            continue
            
        # Get the bounding box slice
        min_row, min_col, max_row, max_col = prop.bbox
        
        # Extract local crops for the specific cell
        # We use the mask to isolate the single cell within the crop
        cell_mask = (labels[min_row:max_row, min_col:max_col] == prop.label)
        
        local_dapi = dapi_ch[min_row:max_row, min_col:max_col]
        local_tubulin = tubulin_ch[min_row:max_row, min_col:max_col]
        
        # Calculate Center of Mass (weighted by intensity)
        # We mask the intensity images so we only calculate CoM for the specific cell pixels
        
        # DAPI Center of Mass
        # Fix: Pass input as a single argument to avoid TypeError
        masked_dapi = local_dapi * cell_mask
        if np.sum(masked_dapi) == 0:
            continue
        com_dapi = ndimage.center_of_mass(masked_dapi)
        
        # Tubulin Center of Mass
        # Note: Tubulin signal might extend slightly beyond the nuclear mask if we only have nuclear labels.
        # However, without a cytoplasm mask, restricting to the nuclear mask + dilation or just the nuclear mask 
        # is a standard approximation, or we use the nuclear mask to define the "cell" object.
        # Ideally, we'd measure the shift of the bulk tubulin signal relative to the nucleus.
        # If we only have nuclear labels, the tubulin signal *inside* the nucleus isn't very informative for polarization.
        # A better heuristic without a cyto mask: Dilate the mask slightly to capture perinuclear tubulin.
        
        dilated_mask = ndimage.binary_dilation(cell_mask, iterations=5)
        masked_tubulin = local_tubulin * dilated_mask
        
        if np.sum(masked_tubulin) == 0:
            continue
            
        com_tubulin = ndimage.center_of_mass(masked_tubulin)
        
        # Calculate Euclidean distance between the two centers
        # com_dapi and com_tubulin are (row, col) tuples
        dist = np.sqrt((com_dapi[0] - com_tubulin[0])**2 + (com_dapi[1] - com_tubulin[1])**2)
        displacements.append(dist)

    # 5. Aggregation
    if not displacements:
        return 0.0
        
    # Return the mean displacement across all cells in the image
    return float(np.mean(displacements))
