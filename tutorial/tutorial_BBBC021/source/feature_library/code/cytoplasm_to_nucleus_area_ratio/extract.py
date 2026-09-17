def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.filters import threshold_otsu
    
    # Convert to float32 for processing
    img_arr = np.asarray(img, dtype=np.float32)
    
    # Handle dimensionality according to dataset format
    # The dataset is (512, 512, 3).
    # Case 1: 2D Image (H, W) -> Treat as single channel, duplicate to 3 for consistency
    if img_arr.ndim == 2:
        img_arr = np.stack([img_arr, img_arr, img_arr], axis=-1)
    
    # Case 2: 3D Image (H, W, C)
    # Ensure we have at least 3 channels for the indexing below.
    # If fewer than 3, pad or duplicate.
    if img_arr.ndim == 3:
        if img_arr.shape[2] == 1:
            img_arr = np.concatenate([img_arr, img_arr, img_arr], axis=-1)
        elif img_arr.shape[2] < 3:
            # Pad with zeros if we have e.g. 2 channels (unlikely given dataset desc, but safe)
            pad_width = ((0,0), (0,0), (0, 3-img_arr.shape[2]))
            img_arr = np.pad(img_arr, pad_width, mode='constant')
        # If > 3 channels, we just use the first 3 indices.

    # Intensity normalization (0-1 range)
    if img_arr.max() > 0:
        img_arr = img_arr / img_arr.max()
        
    # Channel Mapping based on Dataset Description
    # Ch0: Actin (Red) -> Cytoskeleton
    # Ch1: Tubulin (Green) -> Cytoskeleton/Microtubules
    # Ch2: DAPI (Blue) -> Nucleus
    
    # Extract signals
    # Nucleus signal is primarily Blue channel (Index 2)
    nuc_signal = img_arr[..., 2]
    
    # Cytoplasm/Cell signal is primarily Red (0) + Green (1).
    # We create a "Total Cell Body" signal by taking the maximum projection of all channels.
    # This ensures we capture the extent of the cell including the nucleus.
    cell_signal = np.max(img_arr, axis=2)
    
    # --- Segmentation Logic ---
    # We prioritize channel-based segmentation over passed masks because the channel 
    # definitions (DAPI vs Actin) are explicit and reliable for this dataset.
    
    # 1. Nucleus Mask
    # Use Otsu's method to find a threshold for the nucleus
    if np.std(nuc_signal) > 1e-5:
        try:
            thresh_nuc = threshold_otsu(nuc_signal)
            mask_nuc = nuc_signal > thresh_nuc
        except Exception:
            # Fallback: simple mean threshold if Otsu fails
            mask_nuc = nuc_signal > np.mean(nuc_signal)
    else:
        # Flat image fallback
        mask_nuc = nuc_signal > 0.1

    # 2. Total Cell Mask
    # Use Otsu's method on the maximum projection
    if np.std(cell_signal) > 1e-5:
        try:
            thresh_cell = threshold_otsu(cell_signal)
            mask_cell = cell_signal > thresh_cell
        except Exception:
            mask_cell = cell_signal > np.mean(cell_signal)
    else:
        mask_cell = cell_signal > 0.05

    # 3. Refinement
    # Biologically, the nucleus must be inside the cell.
    # Enforce: Cell Mask = Union(Cell Mask, Nucleus Mask)
    # This prevents negative cytoplasm areas.
    mask_cell = np.logical_or(mask_cell, mask_nuc)
    
    # --- Feature Calculation ---
    
    # Calculate areas (number of pixels)
    area_nucleus = np.sum(mask_nuc)
    area_total_cell = np.sum(mask_cell)
    
    # Cytoplasm Area = Total Cell Area - Nucleus Area
    area_cytoplasm = area_total_cell - area_nucleus
    
    # Compute Ratio: Cytoplasm / Nucleus
    # Handle division by zero
    if area_nucleus == 0:
        return 0.0
        
    ratio = area_cytoplasm / area_nucleus
    
    return float(ratio)
