def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_opening, disk, binary_dilation

    # Convert to appropriate array type
    # The input is expected to be (512, 512, 3) uint8
    arr = np.asarray(img)
    
    # Validation of input dimensions
    if arr.ndim != 3 or arr.shape[2] != 3:
        # If not the expected 3-channel image, return 0.0
        return 0.0

    # Channel 0 is Actin (Red), Channel 2 is DAPI (Blue)
    actin_channel = arr[:, :, 0]
    dapi_channel = arr[:, :, 2]

    # Define masks for Nuclei and Cytoplasm
    nuclei_mask = None
    cell_mask = None

    # Check if segmentation masks are provided
    # We expect potentially two masks: one for nuclei, one for cells (or cytoplasm)
    # If provided, we try to use them.
    if len(segmentation_masks) >= 2:
        # Heuristic: usually masks are passed in order or distinguishable by size
        # Often mask 0 is nuclei, mask 1 is cells, but let's check coverage
        m1 = segmentation_masks[0] > 0
        m2 = segmentation_masks[1] > 0
        
        sum1 = np.sum(m1)
        sum2 = np.sum(m2)
        
        # Usually cell mask covers more area than nuclei mask
        if sum1 < sum2:
            nuclei_mask = m1
            cell_mask = m2
        else:
            nuclei_mask = m2
            cell_mask = m1
            
    elif len(segmentation_masks) == 1:
        # If only one mask, assume it's the cell mask (or nuclei), hard to distinguish perfectly
        # without more info. Let's assume it's a cell mask and we need to segment nuclei internally.
        cell_mask = segmentation_masks[0] > 0
        # We will generate nuclei mask below
    
    # Fallback Segmentation Logic if masks are missing
    if nuclei_mask is None:
        # Segment Nuclei from DAPI (Channel 2)
        # Simple Otsu thresholding + morphological cleanup
        try:
            thresh_nuc = threshold_otsu(dapi_channel)
            nuclei_mask = dapi_channel > thresh_nuc
            nuclei_mask = binary_opening(nuclei_mask, footprint=disk(2))
        except Exception:
            # Fallback if image is empty/uniform
            nuclei_mask = np.zeros(dapi_channel.shape, dtype=bool)

    if cell_mask is None:
        # Segment Cells from Actin (Channel 0)
        # Actin usually defines the cell body well
        try:
            # Smooth slightly to connect fibers
            actin_smooth = ndimage.gaussian_filter(actin_channel, sigma=2)
            thresh_cell = threshold_otsu(actin_smooth)
            cell_mask = actin_smooth > thresh_cell
            # Dilate to ensure we capture the full extent
            cell_mask = binary_dilation(cell_mask, footprint=disk(3))
        except Exception:
            cell_mask = np.zeros(actin_channel.shape, dtype=bool)

    # Define Cytoplasm: Cell body MINUS Nucleus
    # Ensure masks are boolean and same shape
    if nuclei_mask.shape != actin_channel.shape:
        nuclei_mask = np.zeros(actin_channel.shape, dtype=bool)
    if cell_mask.shape != actin_channel.shape:
        cell_mask = np.zeros(actin_channel.shape, dtype=bool)

    cytoplasm_mask = np.logical_and(cell_mask, np.logical_not(nuclei_mask))

    # Extract pixels from the Actin channel within the cytoplasm
    cytoplasm_pixels = actin_channel[cytoplasm_mask]

    # If no cytoplasm detected, return 0.0
    if cytoplasm_pixels.size == 0:
        return 0.0

    # Compute Shannon Entropy
    # 1. Compute histogram of pixel intensities (0-255 for uint8)
    # We use 256 bins because the data is uint8
    hist, _ = np.histogram(cytoplasm_pixels, bins=256, range=(0, 256))

    # 2. Normalize to get probability distribution
    p_i = hist / cytoplasm_pixels.size

    # 3. Filter out zero probabilities to avoid log(0)
    p_i = p_i[p_i > 0]

    # 4. Calculate Entropy: -sum(p * log2(p))
    entropy_val = -np.sum(p_i * np.log2(p_i))

    return float(entropy_val)
