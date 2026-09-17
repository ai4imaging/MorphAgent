def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_opening, disk

    # Convert to appropriate array type and normalize
    # The dataset is uint8 (0-255). We normalize to [0, 1] for intensity features.
    arr = np.asarray(img, dtype=np.float32)
    
    # Handle dimensionality
    # Expected shape: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        # Fallback for potential unexpected shapes, though dataset spec says (512, 512, 3)
        return 0.0

    # Extract Tubulin channel (Channel 1 = Green)
    # Channel 0: Actin (Red), Channel 1: Tubulin (Green), Channel 2: DAPI (Blue)
    tubulin_channel = arr[:, :, 1]
    
    # Normalize intensity to [0, 1] range
    # This is critical for consistent feature extraction across different exposure times or bit depths
    tubulin_normalized = tubulin_channel / 255.0
    tubulin_normalized = np.clip(tubulin_normalized, 0.0, 1.0)

    # Determine the Cell Mask
    cell_mask = None

    # Strategy 1: Use provided segmentation masks
    if len(segmentation_masks) > 0:
        # We need to identify which mask corresponds to the whole cell (cytoplasm/cell body).
        # Typically, in a cell/nuclei pair, the cell mask covers a larger area.
        # If only one mask is provided, we assume it's the relevant one.
        
        best_mask = None
        max_area = -1
        
        for mask in segmentation_masks:
            # Ensure mask is boolean or labeled
            if mask is None:
                continue
            
            # Convert labeled mask to binary (foreground vs background)
            binary_mask = mask > 0
            current_area = np.sum(binary_mask)
            
            # We assume the largest mask covering the image is the cell mask (vs nuclei)
            if current_area > max_area:
                max_area = current_area
                best_mask = binary_mask
        
        cell_mask = best_mask

    # Strategy 2: Fallback if no masks provided or masks were empty
    if cell_mask is None:
        # Create a mask based on the Tubulin channel itself.
        # Tubulin is usually a good proxy for cell shape in this dataset.
        # We use Otsu's thresholding to separate foreground (cells) from background.
        
        # Handle case where image is completely black/uniform
        if np.min(tubulin_normalized) == np.max(tubulin_normalized):
            return 0.0
            
        try:
            thresh = threshold_otsu(tubulin_normalized)
            cell_mask = tubulin_normalized > thresh
            
            # Clean up the mask: remove small noise
            # Using a small disk for opening to remove salt noise
            cell_mask = binary_opening(cell_mask, footprint=disk(2))
        except Exception:
            # Fallback for extremely low contrast images where otsu might fail
            return 0.0

    # Compute Feature: Mean Intensity within the Cell Mask
    # We only consider pixels that are part of a cell.
    
    # Check if mask is empty (no cells detected)
    if np.sum(cell_mask) == 0:
        return 0.0

    # Select pixels within the mask
    masked_pixels = tubulin_normalized[cell_mask]
    
    # Calculate mean intensity
    mean_intensity = np.mean(masked_pixels)

    return float(mean_intensity)
