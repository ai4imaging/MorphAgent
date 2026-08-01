def extract(img, seg):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops

    # Validate input image dimensions
    if img.ndim != 2 or img.shape not in [(1536, 1536), (2048, 2048)]:
        return 0.0

    # Convert the input image to a float array for processing
    arr = np.asarray(img, dtype=np.float32)

    # Normalize image data to range [0, 1] using the maximum intensity in the original range
    arr = (arr - arr.min()) / (arr.max() - arr.min())

    # Validate segmentation presence
    if not seg:
        return 0.0

    # Extract required segmentation masks
    cell_mask = seg.get("mask_cell")
    filament_mask = seg.get("mask_filament")

    # Ensure masks are valid and match image dimensions
    if cell_mask is None or filament_mask is None:
        return 0.0
    if cell_mask.shape != arr.shape or filament_mask.shape != arr.shape:
        return 0.0

    # Apply the cell mask to the image
    cell_area = arr * cell_mask

    # Apply the filament mask to extract fluorescence signal within Tau filaments
    filament_area = cell_area * filament_mask

    # Thresholding to segment bead-like inclusions within the filament mask
    threshold_value = np.percentile(filament_area[filament_area > 0], 95)  # Use 95th percentile as intensity threshold
    bead_mask = filament_area > threshold_value

    # Label connected components within the bead mask
    labeled_beads = label(bead_mask)
    bead_regions = regionprops(labeled_beads)

    # Count bead-like inclusions within filaments
    bead_count = len(bead_regions)

    # Normalize bead count to the filament mask area for density calculation
    filament_area_size = np.sum(filament_mask)
    if filament_area_size == 0:  # Avoid division by zero
        return float(0.0)

    bead_density = bead_count / filament_area_size

    return float(bead_density)
