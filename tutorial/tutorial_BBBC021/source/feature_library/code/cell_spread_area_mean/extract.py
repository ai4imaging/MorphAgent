def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    from skimage.measure import label
    from skimage.morphology import binary_closing, remove_small_objects, disk

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Expected: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        # If not the expected shape, try to handle potential variations or return 0
        if arr.ndim == 2:
            # If 2D, assume it's a single channel projection, treat as grayscale
            # This is a fallback and likely incorrect for this specific multi-channel feature request
            # but prevents crashing.
            return 0.0
        return 0.0

    # Channel Mapping based on dataset description
    # Channel 0: Actin (Cytoskeleton) - Best for cell spread
    # Channel 1: Tubulin (Microtubules) - Also good for cell body
    # Channel 2: DAPI (Nucleus) - For counting cells
    actin_ch = arr[..., 0]
    tubulin_ch = arr[..., 1]
    dapi_ch = arr[..., 2]

    # Normalize channels to [0, 1]
    def normalize(ch):
        vmax = np.percentile(ch, 99.5) if ch.size > 0 else 1.0
        if vmax <= 0: vmax = 1.0
        return np.clip(ch / vmax, 0.0, 1.0)

    actin_norm = normalize(actin_ch)
    tubulin_norm = normalize(tubulin_ch)
    dapi_norm = normalize(dapi_ch)

    # --- Step 1: Determine Cell Count (N) ---
    num_cells = 0
    
    # Check if segmentation masks are available for nuclei
    # Typically mask[0] is nuclei if available, but we can't be 100% sure of order without metadata.
    # However, if masks are provided, we trust the first one for object counting if it looks like nuclei.
    # If no masks, we segment DAPI.
    
    nuclei_mask = None
    
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use provided mask
        nuclei_mask = segmentation_masks[0]
        # Ensure it's labeled
        if nuclei_mask.max() <= 1:
            nuclei_mask = label(nuclei_mask > 0)
        num_cells = nuclei_mask.max()
    else:
        # Auto-segment nuclei from DAPI
        # Smooth DAPI
        dapi_smooth = ndimage.gaussian_filter(dapi_norm, sigma=2.0)
        
        # Threshold
        try:
            thresh_val = threshold_otsu(dapi_smooth)
        except ValueError:
            thresh_val = 0.0
            
        binary_nuclei = dapi_smooth > thresh_val
        
        # Clean up
        binary_nuclei = remove_small_objects(binary_nuclei, min_size=50)
        binary_nuclei = binary_closing(binary_nuclei, disk(2))
        
        # Label
        nuclei_labels = label(binary_nuclei)
        num_cells = nuclei_labels.max()

    if num_cells == 0:
        return 0.0

    # --- Step 2: Determine Total Cellular Spread Area ---
    # We need the area of the entire cell body (cytoplasm + nucleus).
    # Actin and Tubulin are the best markers for this.
    
    total_cell_area = 0.0
    
    # If a second mask is provided, it might be the cell mask
    if len(segmentation_masks) > 1 and segmentation_masks[1] is not None:
        cell_mask = segmentation_masks[1]
        total_cell_area = np.sum(cell_mask > 0)
    else:
        # Auto-segment cell body
        # Combine Actin and Tubulin signals. Using max captures the union of structures.
        # Senescent cells often have spread out actin stress fibers.
        cell_signal = np.maximum(actin_norm, tubulin_norm)
        
        # Smooth significantly to connect cytoskeletal fibers into a blob
        cell_smooth = ndimage.gaussian_filter(cell_signal, sigma=2.5)
        
        # Threshold
        # Since background is usually dark and cells are bright, Otsu works reasonably well.
        # However, for spread cells, the edges might be dim.
        try:
            thresh_cell = threshold_otsu(cell_smooth)
        except ValueError:
            thresh_cell = 0.0
            
        # Create binary mask
        binary_cell = cell_smooth > thresh_cell
        
        # Morphological cleanup
        # Fill holes inside the cell
        binary_cell = ndimage.binary_fill_holes(binary_cell)
        # Remove small noise (debris)
        binary_cell = remove_small_objects(binary_cell, min_size=100)
        
        total_cell_area = np.sum(binary_cell)

    # --- Step 3: Compute Mean Area ---
    # Feature: cell_spread_area_mean
    # Result is average pixels per cell
    
    mean_area = total_cell_area / float(num_cells)
    
    return float(mean_area)
