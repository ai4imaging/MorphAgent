def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import remove_small_objects
    
    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Expected: (512, 512, 3)
    if arr.ndim != 3:
        # If 2D (512, 512), it might be a single channel or composite grayscale. 
        # Without channel info, we can't reliably extract Tubulin. Return 0.0.
        return 0.0
    
    # Check channel ordering. Dataset says (Height, Width, Channels).
    # If channels are first (3, 512, 512), transpose.
    if arr.shape[0] == 3 and arr.shape[2] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    
    # Extract Tubulin Channel (Channel 1 - Green)
    # Channel 0: Actin, Channel 1: Tubulin, Channel 2: DAPI
    if arr.shape[2] >= 2:
        tubulin_channel = arr[:, :, 1]
    else:
        # Fallback if less than 2 channels (unexpected for this dataset)
        return 0.0

    # Determine the Cell Mask
    # We need a mask that defines the cell body to measure tubulin intensity.
    cell_mask = None

    # Check if segmentation masks are provided
    if len(segmentation_masks) > 0:
        # Heuristic: If multiple masks, usually they are [cells, nuclei] or similar.
        # We prefer the mask that covers a larger area (likely the cell body/cytoplasm).
        # If only one mask, we use it.
        
        best_mask = None
        max_area = -1
        
        for mask in segmentation_masks:
            if mask is None:
                continue
            
            # Ensure mask is 2D
            current_mask = np.asarray(mask)
            if current_mask.ndim > 2:
                current_mask = np.max(current_mask, axis=-1) # Project if 3D/multichannel
            
            # Check if it's a valid label mask
            if current_mask.shape == tubulin_channel.shape:
                # Calculate total foreground area to guess if it's nuclei or cells
                # Cells usually occupy more space than nuclei
                foreground_area = np.count_nonzero(current_mask)
                if foreground_area > max_area:
                    max_area = foreground_area
                    best_mask = current_mask
        
        cell_mask = best_mask

    # Fallback: If no valid mask provided, generate one from the image
    if cell_mask is None:
        # Create a foreground mask using Tubulin (Ch1) and Actin (Ch0)
        # Tubulin and Actin define the cell body structure.
        # We use the maximum projection of these structural channels.
        if arr.shape[2] >= 1:
            structure_img = np.maximum(arr[:, :, 0], arr[:, :, 1])
        else:
            structure_img = arr[:, :, 1]
            
        # Smooth slightly to reduce noise
        from scipy.ndimage import gaussian_filter
        structure_smooth = gaussian_filter(structure_img, sigma=2)
        
        # Threshold
        try:
            thresh = threshold_otsu(structure_smooth)
            binary_mask = structure_smooth > thresh
            # Remove small artifacts (debris)
            binary_mask = remove_small_objects(binary_mask, min_size=50)
            # Label connected components
            cell_mask = label(binary_mask)
        except Exception:
            # If thresholding fails (e.g., uniform image), return 0
            return 0.0

    # Ensure cell_mask is integer labeled
    if cell_mask.dtype == bool:
        cell_mask = label(cell_mask)
    else:
        cell_mask = cell_mask.astype(int)

    # Compute Mean Intensity Per Cell
    # We use regionprops to get the mean intensity of the tubulin channel for each labeled region
    props = regionprops(cell_mask, intensity_image=tubulin_channel)
    
    if not props:
        return 0.0
    
    # Extract mean intensity for each cell
    mean_intensities = [prop.mean_intensity for prop in props]
    
    # Compute the average of these means across the population
    # This represents the "average cell's tubulin intensity"
    if len(mean_intensities) > 0:
        result = np.mean(mean_intensities)
    else:
        result = 0.0

    return float(result)
