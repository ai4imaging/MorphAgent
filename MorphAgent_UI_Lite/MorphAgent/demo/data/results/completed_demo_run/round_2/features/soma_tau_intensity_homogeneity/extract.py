def extract(img, seg):
    # IMPORT REQUIRED LIBRARIES
    import numpy as np

    # Convert image to float32 for processing
    arr = np.asarray(img, dtype=np.float32)
    
    # Validate input image dimensions and segmentation
    if arr.ndim != 2 or arr.shape != (2048, 2048):
        return 0.0
    cell_mask = seg.get("mask_cell")
    if cell_mask is None or cell_mask.shape != arr.shape or np.sum(cell_mask) == 0:
        return 0.0

    # Apply mask to extract Tau intensities within soma
    masked_pixels = arr[cell_mask == 1]
    if masked_pixels.size == 0:
        return 0.0

    # Compute mean and standard deviation of Tau intensity in soma
    mean_intensity = np.mean(masked_pixels)
    std_intensity = np.std(masked_pixels)

    # Handle edge case where mean intensity is too small to avoid division by zero
    if mean_intensity < 1e-6:
        return 0.0

    # Calculate coefficient of variation (CV)
    cv = std_intensity / mean_intensity

    return float(cv)
