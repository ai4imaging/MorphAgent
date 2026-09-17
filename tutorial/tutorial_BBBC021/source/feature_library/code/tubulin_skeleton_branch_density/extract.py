def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.morphology import skeletonize
    from skimage.filters import threshold_otsu
    
    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3) - Channel 1 is Tubulin (Green)
    if arr.ndim == 3 and arr.shape[2] == 3:
        tubulin_channel = arr[:, :, 1]
    elif arr.ndim == 2:
        # Fallback if single channel passed, though dataset spec says 3-channel
        tubulin_channel = arr
    else:
        return 0.0

    # Intensity normalization
    # Normalize to [0, 1] range for processing
    vmax = np.percentile(tubulin_channel, 99.5) if tubulin_channel.size > 0 else 1.0
    if vmax > 0:
        tubulin_norm = tubulin_channel / vmax
    else:
        tubulin_norm = tubulin_channel
    tubulin_norm = np.clip(tubulin_norm, 0.0, 1.0)

    # --- Step 1: Define the Region of Interest (Cell Area) ---
    # We need a mask to define the cellular area for normalization (density denominator)
    # and to constrain the skeletonization to relevant regions.
    
    cell_mask = None
    
    # Check if valid segmentation masks are provided
    if len(segmentation_masks) > 0:
        # Heuristic: Use the first mask provided. 
        # If multiple masks exist, usually the largest covering mask (cell body) is useful.
        # We assume the mask is labeled (0=bg, >0=cells).
        candidate_mask = segmentation_masks[0]
        if candidate_mask.shape == tubulin_channel.shape:
            cell_mask = candidate_mask > 0
    
    # Fallback: If no mask provided, generate one from the image itself
    if cell_mask is None:
        # Use a low threshold to capture the cell body area
        try:
            thresh = threshold_otsu(tubulin_norm)
            # Create a binary mask
            binary_mask = tubulin_norm > thresh
            # Fill holes and smooth to get a solid area
            cell_mask = ndimage.binary_fill_holes(binary_mask)
            # Optional: morphological opening to remove noise
            cell_mask = ndimage.binary_opening(cell_mask, structure=np.ones((3,3)))
        except Exception:
            # If thresholding fails (e.g. empty image), return 0
            return 0.0

    # Calculate Cell Area (Denominator)
    cell_area = np.sum(cell_mask)
    if cell_area < 10.0: # Avoid division by zero or noise
        return 0.0

    # --- Step 2: Extract Tubulin Structure (Filaments) ---
    # We need a binary representation of the microtubule filaments
    
    # Gaussian blur to smooth noise before thresholding for structure
    smoothed = ndimage.gaussian_filter(tubulin_norm, sigma=1.0)
    
    # Adaptive thresholding often works better for filaments than global Otsu
    # However, without cv2.adaptiveThreshold, we can simulate it or use a slightly higher global threshold
    # relative to the local background.
    # Here, we use a simple logic: pixels significantly brighter than the local mean.
    
    local_mean = ndimage.uniform_filter(smoothed, size=15)
    # Filaments are bright structures on top of background
    filament_mask = smoothed > (local_mean + 0.05) # 0.05 is a small offset in [0,1] space
    
    # Constrain filaments to the cell mask area
    filament_mask = np.logical_and(filament_mask, cell_mask)

    # --- Step 3: Skeletonization ---
    # Reduce filaments to 1-pixel wide lines
    skeleton = skeletonize(filament_mask)

    # --- Step 4: Detect Branch Points ---
    # A branch point in a skeleton is a pixel with > 2 neighbors.
    # We use a convolution kernel to count neighbors for every pixel.
    
    # Kernel for 8-connectivity neighbor counting
    # The center is weighted 10 to distinguish it from neighbors easily
    # [[1, 1, 1],
    #  [1, 10, 1],
    #  [1, 1, 1]]
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]], dtype=np.uint8)
    
    # Convolve. The skeleton is boolean, convert to uint8 (0 or 1)
    skel_uint8 = skeleton.astype(np.uint8)
    filtered = ndimage.convolve(skel_uint8, kernel, mode='constant', cval=0)
    
    # Logic:
    # Center pixel = 1 -> value += 10
    # Neighbors = 1 -> value += 1 per neighbor
    # A pixel is part of the skeleton if value >= 10.
    # It is an endpoint if value == 11 (10 + 1 neighbor).
    # It is a line segment if value == 12 (10 + 2 neighbors).
    # It is a branch point if value > 12 (10 + 3 or more neighbors).
    
    # Note: This simple logic works for standard skeletons. 
    # Sometimes 'crossings' might have 4 neighbors.
    
    branch_points = filtered > 12
    num_branches = np.sum(branch_points)

    # --- Step 5: Compute Density ---
    # Feature: Branch points per unit area
    branch_density = float(num_branches) / float(cell_area)

    return float(branch_density)
