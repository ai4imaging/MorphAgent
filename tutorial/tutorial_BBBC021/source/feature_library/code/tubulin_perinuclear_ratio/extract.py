def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu, gaussian
    from skimage.morphology import dilation, disk
    
    # Convert to appropriate array type and normalize
    # Image is (512, 512, 3), uint8.
    # Channel 0: Actin (R), Channel 1: Tubulin (G), Channel 2: DAPI (B)
    arr = np.asarray(img, dtype=np.float32)
    
    # Check dimensions
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Normalize intensity to [0, 1]
    # We use a robust max to avoid hot pixel artifacts
    vmax = np.percentile(arr, 99.9) if arr.size > 0 else 255.0
    if vmax <= 0: vmax = 1.0
    arr = arr / vmax
    arr = np.clip(arr, 0.0, 1.0)

    # Extract specific channels
    # Channel 1 is Tubulin (Target for intensity measurement)
    tubulin_ch = arr[:, :, 1]
    # Channel 2 is DAPI (Target for Nucleus segmentation)
    dapi_ch = arr[:, :, 2]
    # Channel 0 is Actin (Target for Cell Body segmentation help)
    actin_ch = arr[:, :, 0]

    # --- Segmentation Logic ---
    # We need masks for: Nuclei, Cell Body.
    
    mask_nuclei = None
    mask_cells = None

    # Check if segmentation masks were provided via arguments
    # The prompt implies masks might be passed, but we must be robust if they aren't or are empty
    if len(segmentation_masks) > 0:
        # Heuristic: Try to identify which mask is which based on overlap with channels or order
        # Common convention: often nuclei is first or second. 
        # Without metadata, we fallback to computing our own to ensure consistency with the specific definition of "perinuclear ring"
        # However, if provided, we can use them. Let's stick to a robust internal computation 
        # because "perinuclear" requires a specific geometric relationship that general masks might not capture perfectly
        # (e.g. we need the exact nucleus boundary to dilate from).
        # But to respect the interface, if we had labeled masks, we would use them. 
        # Given the "optional" nature and the specific geometric requirement, calculating fresh binary masks 
        # on the provided image is often safer for this specific morphological feature.
        pass

    # 1. Generate Nuclei Mask (from DAPI)
    # Smooth DAPI to reduce noise
    dapi_smooth = gaussian(dapi_ch, sigma=2)
    try:
        thresh_nuc = threshold_otsu(dapi_smooth)
        mask_nuclei = dapi_smooth > thresh_nuc
    except Exception:
        # Fallback if image is blank
        return 0.0

    # 2. Generate Cell Body Mask (Union of Tubulin + Actin + Nuclei)
    # We combine channels to get the full cellular extent
    cell_signal = np.maximum(tubulin_ch, actin_ch)
    cell_signal = np.maximum(cell_signal, dapi_ch) # Ensure nucleus is inside
    cell_smooth = gaussian(cell_signal, sigma=2)
    try:
        thresh_cell = threshold_otsu(cell_smooth)
        # Lower threshold slightly to capture faint edges
        mask_cells = cell_smooth > (thresh_cell * 0.7)
    except Exception:
        return 0.0
    
    # Ensure nucleus is part of the cell
    mask_cells = np.logical_or(mask_cells, mask_nuclei)

    # --- ROI Definition ---
    
    # Define Perinuclear Ring
    # We want a ring *immediately* surrounding the nucleus.
    # Radius of dilation: ~5-10 pixels is typical for 512x512 microscopy images of cells
    ring_radius = 6
    selem = disk(ring_radius)
    
    # Dilate nuclei
    dilated_nuclei = dilation(mask_nuclei, selem)
    
    # The ring is the dilated area MINUS the original nucleus area
    # AND it must be within the cell body
    mask_ring = np.logical_and(dilated_nuclei, np.logical_not(mask_nuclei))
    mask_ring = np.logical_and(mask_ring, mask_cells)

    # Define "Rest of Cytoplasm"
    # This is the Cell Body MINUS the Nucleus MINUS the Ring
    mask_rest = np.logical_and(mask_cells, np.logical_not(mask_nuclei))
    mask_rest = np.logical_and(mask_rest, np.logical_not(mask_ring))

    # --- Feature Calculation ---

    # We calculate the mean intensity of Tubulin in these two regions.
    # We do this globally (aggregating all cells) to be robust against segmentation errors splitting touching cells.
    
    pixels_ring = tubulin_ch[mask_ring]
    pixels_rest = tubulin_ch[mask_rest]

    if pixels_ring.size == 0:
        return 0.0
    
    mean_ring = np.mean(pixels_ring)
    
    if pixels_rest.size == 0:
        # If there is no "rest of cytoplasm" (cells are just nuclei + rings), 
        # the ratio is technically infinite or undefined. 
        # We return a high value or just the ring intensity. 
        # Returning a high constant implies "very concentrated".
        return float(mean_ring * 10.0) 

    mean_rest = np.mean(pixels_rest)

    # Avoid division by zero if background is perfectly black
    if mean_rest <= 1e-9:
        return float(mean_ring * 100.0) # Cap at a high ratio

    ratio = mean_ring / mean_rest

    return float(ratio)
