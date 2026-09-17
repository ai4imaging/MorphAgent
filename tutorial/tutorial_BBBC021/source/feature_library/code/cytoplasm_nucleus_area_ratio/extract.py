def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu, threshold_triangle
    from skimage.morphology import binary_closing, disk
    
    # Convert to appropriate array type
    # Image is (512, 512, 3), uint8
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality
    if arr.ndim != 3 or arr.shape[2] != 3:
        # If not the expected 3-channel image, return 0.0
        return 0.0

    # Define channel indices based on dataset description
    CH_ACTIN = 0   # Red
    CH_TUBULIN = 1 # Green
    CH_DAPI = 2    # Blue

    # Initialize masks
    mask_nuclei = None
    mask_cells = None

    # Strategy:
    # 1. Try to use provided segmentation masks.
    # 2. If not provided, generate them from the image channels.
    
    # Check provided segmentation masks
    # The dataset description suggests masks might be available.
    # If 2 masks are provided, assume [cells, nuclei] or [nuclei, cells].
    # Usually, nuclei are smaller and contained within cells.
    
    if len(segmentation_masks) >= 2:
        # Heuristic to distinguish nuclei from cell masks if not labeled:
        # Nuclei usually have smaller total area than whole cells.
        m1 = np.asarray(segmentation_masks[0]) > 0
        m2 = np.asarray(segmentation_masks[1]) > 0
        
        area1 = np.sum(m1)
        area2 = np.sum(m2)
        
        if area1 < area2:
            mask_nuclei = m1
            mask_cells = m2
        else:
            mask_nuclei = m2
            mask_cells = m1
            
    elif len(segmentation_masks) == 1:
        # If only one mask is provided, it's ambiguous. 
        # However, in HCS, the primary segmentation is often nuclei.
        # We will treat it as nuclei and generate cell body from image.
        mask_nuclei = np.asarray(segmentation_masks[0]) > 0
    
    # Fallback / Generation Logic
    
    # 1. Generate Nuclear Mask if missing
    if mask_nuclei is None:
        dapi_channel = arr[..., CH_DAPI]
        # Normalize DAPI
        dapi_max = np.max(dapi_channel)
        if dapi_max > 0:
            # Simple Otsu thresholding for nuclei
            try:
                thresh_val = threshold_otsu(dapi_channel)
                mask_nuclei = dapi_channel > thresh_val
            except Exception:
                # Fallback for very low signal
                mask_nuclei = dapi_channel > (dapi_max * 0.2)
        else:
            mask_nuclei = np.zeros(dapi_channel.shape, dtype=bool)

    # 2. Generate Cell Body Mask if missing
    if mask_cells is None:
        # Combine Actin and Tubulin for cell body
        # These markers stain the cytoskeleton
        cyto_signal = np.maximum(arr[..., CH_ACTIN], arr[..., CH_TUBULIN])
        
        # Smooth to connect fragmented cytoskeleton
        cyto_signal = ndimage.gaussian_filter(cyto_signal, sigma=2)
        
        cyto_max = np.max(cyto_signal)
        if cyto_max > 0:
            try:
                # Triangle threshold is often better for cytoplasm which has a long tail in histogram
                thresh_val = threshold_triangle(cyto_signal)
                mask_cells = cyto_signal > thresh_val
            except Exception:
                mask_cells = cyto_signal > (cyto_max * 0.1)
                
            # Morphological closing to fill holes
            mask_cells = binary_closing(mask_cells, disk(3))
        else:
            mask_cells = np.zeros(cyto_signal.shape, dtype=bool)

    # Enforce biological constraint: Nucleus must be inside the cell
    # The cell mask should be the union of the detected cytoplasm and the nucleus
    mask_cells = np.logical_or(mask_cells, mask_nuclei)

    # Calculate Areas
    # We calculate the global ratio for the image (Total Cytoplasm / Total Nuclei)
    # This is robust to segmentation errors where individual cells touch.
    
    area_nuclei = np.sum(mask_nuclei)
    area_total_cell = np.sum(mask_cells)
    
    # Cytoplasm area = Total Cell Area - Nuclei Area
    # (Note: Since we enforced mask_cells = mask_cells | mask_nuclei, 
    # area_total_cell is guaranteed to be >= area_nuclei)
    area_cytoplasm = area_total_cell - area_nuclei

    # Compute Ratio
    # Feature description: "Calculates the ratio of cytoplasmic area... to Nuclear Area"
    if area_nuclei == 0:
        return 0.0
    
    ratio = float(area_cytoplasm) / float(area_nuclei)

    return float(ratio)
