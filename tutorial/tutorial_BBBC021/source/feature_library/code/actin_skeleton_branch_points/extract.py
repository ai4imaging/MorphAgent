def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.morphology import skeletonize
    from skimage.filters import threshold_otsu, gaussian
    from skimage.measure import label

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset: (512, 512, 3), Channel 0 = Actin, Channel 2 = DAPI
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract channels
    # Channel 0: Actin (Red) - Target for skeleton analysis
    # Channel 2: DAPI (Blue) - Used for cell counting if no segmentation is provided
    actin_channel = arr[:, :, 0]
    dapi_channel = arr[:, :, 2]

    # --- Step 1: Preprocessing Actin Channel ---
    # Normalize actin channel
    if np.max(actin_channel) > 0:
        actin_norm = actin_channel / np.max(actin_channel)
    else:
        return 0.0

    # Apply Gaussian blur to smooth the actin fibers before thresholding
    # This is critical for skeletonization to avoid creating tiny spurs from noise
    actin_smooth = gaussian(actin_norm, sigma=1.0)

    # Thresholding to create binary mask
    try:
        thresh = threshold_otsu(actin_smooth)
        actin_binary = actin_smooth > thresh
    except Exception:
        # Fallback if image is uniform
        return 0.0

    # --- Step 2: Skeletonization ---
    # Skeletonize the binary actin mask
    # This reduces fibers to 1-pixel wide lines
    skeleton = skeletonize(actin_binary)

    # --- Step 3: Branch Point Detection Logic ---
    # A branch point in a skeleton is a pixel with > 2 neighbors.
    # We use a convolution kernel to count neighbors for every pixel.
    # Kernel:
    # [[1, 1, 1],
    #  [1, 10, 1],
    #  [1, 1, 1]]
    #
    # If the center pixel is 1 (part of skeleton), it contributes 10.
    # Neighbors contribute 1 each.
    # - Value 11: Endpoint (1 neighbor)
    # - Value 12: Line segment (2 neighbors)
    # - Value >= 13: Branch point (3 or more neighbors)
    
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]], dtype=np.uint8)
    
    # Convert skeleton to uint8 (0 and 1) for convolution
    skeleton_int = skeleton.astype(np.uint8)
    
    # Convolve
    filtered = ndimage.convolve(skeleton_int, kernel, mode='constant', cval=0)
    
    # Identify branch points (pixels >= 13)
    # Note: We only care about pixels that were part of the skeleton (>= 10)
    branch_points_mask = filtered >= 13

    # --- Step 4: Aggregation (Per Cell) ---
    
    # Check if segmentation masks are available
    has_segmentation = len(segmentation_masks) > 0 and segmentation_masks[0] is not None
    
    if has_segmentation:
        # Use the first provided mask (usually cell or nuclei segmentation)
        seg_mask = segmentation_masks[0]
        
        # Ensure mask shape matches image (handle potential 2D vs 3D issues if any)
        if seg_mask.shape != actin_channel.shape:
            # If shapes don't match, fallback to global calculation
            # This handles cases where mask might be squeezed or different
            pass 
        else:
            # Calculate branch points per labeled region
            unique_labels = np.unique(seg_mask)
            unique_labels = unique_labels[unique_labels > 0] # Exclude background
            
            if len(unique_labels) == 0:
                return 0.0
            
            branch_counts = []
            for lab in unique_labels:
                # Create a mask for the current cell
                cell_mask = (seg_mask == lab)
                
                # Count branch points within this cell
                # We use logical AND to find branch points that fall inside the cell mask
                count = np.sum(branch_points_mask & cell_mask)
                branch_counts.append(count)
            
            return float(np.mean(branch_counts))

    # --- Fallback: Global Calculation Normalized by Estimated Cell Count ---
    # If no segmentation mask is provided, we estimate cell count from DAPI
    
    # 1. Count total branch points in the image
    total_branch_points = np.sum(branch_points_mask)
    
    # 2. Estimate number of cells using DAPI
    try:
        dapi_smooth = gaussian(dapi_channel, sigma=2.0)
        dapi_thresh = threshold_otsu(dapi_smooth)
        dapi_binary = dapi_smooth > dapi_thresh
        
        # Label nuclei
        labeled_nuclei, num_cells = label(dapi_binary, return_num=True)
    except Exception:
        num_cells = 1 # Fallback to 1 if thresholding fails
        
    if num_cells < 1:
        num_cells = 1.0
        
    # Return average branch points per cell
    result = total_branch_points / float(num_cells)

    return float(result)
