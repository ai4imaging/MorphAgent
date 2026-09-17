def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.filters import threshold_otsu
    from scipy import ndimage

    # Convert to appropriate array type
    # Input is expected to be (512, 512, 3) uint8
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality and validate input
    if arr.ndim != 3 or arr.shape[2] != 3:
        # If the input format is unexpected (e.g., not 3 channels), return 0.0
        return 0.0

    # Extract relevant channels for confluence: Actin (Ch0) and Tubulin (Ch1)
    # These markers define the cell body/cytoplasm, which is the correct measure for confluence.
    # DAPI (Ch2) is ignored as nuclei are a subset of the cell area.
    actin = arr[:, :, 0]
    tubulin = arr[:, :, 1]

    # Combine channels to get a "total cellular material" intensity map
    # We use the maximum projection to capture the union of structures
    combined_intensity = np.maximum(actin, tubulin)

    # Check for empty/noise-only images
    # If the dynamic range is extremely low (e.g., < 5 intensity levels out of 255),
    # it's likely an empty background image or failed acquisition.
    # Otsu's threshold would force a split in the noise, creating false positives.
    data_range = np.max(combined_intensity) - np.min(combined_intensity)
    if data_range < 5.0:
        return 0.0

    # Preprocessing: Gaussian blur to smooth texture and noise
    # This helps create a more contiguous mask rather than fragmented pixel clusters
    smoothed = ndimage.gaussian_filter(combined_intensity, sigma=2.0)

    # Segmentation: Otsu's Thresholding
    # Calculate a global threshold to separate foreground (cells) from background
    try:
        thresh = threshold_otsu(smoothed)
    except ValueError:
        # Handle cases where the image is uniform (e.g., all zeros)
        return 0.0

    # Create binary mask
    binary_mask = smoothed > thresh

    # Morphological Post-Processing
    # 1. Fill holes: Cells often have dimmer internal regions that shouldn't be counted as background
    binary_mask = ndimage.binary_fill_holes(binary_mask)
    
    # 2. Opening: Remove small noise specks (dust, hot pixels) that are too small to be cells
    # Structure size 2 corresponds to a small 3x3 or 5x5 kernel depending on implementation details,
    # sufficient to remove single-pixel noise.
    binary_mask = ndimage.binary_opening(binary_mask, structure=np.ones((3,3)))

    # Calculate Confluence
    # Fraction of total pixels that are labeled as foreground
    total_pixels = binary_mask.size
    foreground_pixels = np.sum(binary_mask)
    
    if total_pixels == 0:
        return 0.0

    confluence = foreground_pixels / total_pixels

    return float(confluence)
