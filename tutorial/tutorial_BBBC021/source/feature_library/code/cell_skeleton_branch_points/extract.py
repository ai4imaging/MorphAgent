def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.morphology import skeletonize
    from skimage.filters import threshold_otsu, threshold_local

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset: (512, 512, 3) - Channel 0 is Actin (Red)
    if arr.ndim == 3 and arr.shape[2] == 3:
        actin_channel = arr[..., 0]
    elif arr.ndim == 2:
        # Fallback if passed a single channel image
        actin_channel = arr
    else:
        return 0.0

    # Intensity normalization
    # Normalize to [0, 1] based on dtype range for uint8 (0-255)
    # or robust min/max if float
    if np.max(actin_channel) > 1.0:
        actin_channel = actin_channel / 255.0
    actin_channel = np.clip(actin_channel, 0.0, 1.0)

    # Pre-processing: Gaussian blur to reduce noise before skeletonization
    # A small sigma helps prevent single-pixel noise from becoming spurious branches
    actin_smooth = ndimage.gaussian_filter(actin_channel, sigma=1.0)

    # Create a binary mask of the actin structure
    # We use adaptive thresholding to capture local filaments rather than just global bright spots
    # Block size must be odd
    block_size = 35
    local_thresh = threshold_local(actin_smooth, block_size, offset=0.02)
    binary_structure = actin_smooth > local_thresh

    # Handle segmentation masks (ROI filtering)
    # If segmentation masks are provided, we restrict the analysis to the cellular regions
    # to avoid skeletonizing background noise.
    roi_mask = None
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Combine all masks if multiple are provided, or just use the first valid one
        # Assuming masks are labeled (0=bg, >0=cell)
        combined_mask = np.zeros_like(segmentation_masks[0], dtype=bool)
        for mask in segmentation_masks:
            if mask is not None:
                # Ensure mask shape matches image shape (handle potential 3D vs 2D mismatch)
                if mask.shape == actin_channel.shape:
                    combined_mask = combined_mask | (mask > 0)
        
        if np.any(combined_mask):
            roi_mask = combined_mask

    # If no valid segmentation mask provided, create a rough foreground mask using global Otsu
    if roi_mask is None:
        try:
            global_thresh = threshold_otsu(actin_smooth)
            roi_mask = actin_smooth > global_thresh
        except Exception:
            # Fallback for completely uniform images
            roi_mask = np.ones_like(actin_smooth, dtype=bool)

    # Apply ROI mask to the binary structure
    binary_structure = binary_structure & roi_mask

    # Skeletonize the binary structure
    # This reduces filaments to 1-pixel wide lines
    skeleton = skeletonize(binary_structure)

    # Count branch points
    # A branch point in a skeleton is a pixel with > 2 neighbors in its 8-neighborhood.
    # We can detect this using convolution.
    # Kernel:
    # 1 1 1
    # 1 10 1  <-- Center pixel weighted 10
    # 1 1 1
    #
    # If the center is part of the skeleton (value 1), the convolution result will be 10 + sum(neighbors).
    # - Endpoint: 1 neighbor -> 11
    # - Line segment: 2 neighbors -> 12
    # - Branch point: 3 neighbors -> 13
    # - Crossing: 4 neighbors -> 14
    
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]], dtype=np.uint8)

    # Convolve. Note: skeleton is boolean, convert to uint8 for convolution
    filtered = ndimage.convolve(skeleton.astype(np.uint8), kernel, mode='constant', cval=0)

    # Branch points are pixels where the value is >= 13 (Center=1 + Neighbors>=3)
    # We specifically look for pixels that were part of the skeleton (>=10) and have >2 neighbors.
    # 10 (isolated) is rare in skeletons but possible. 11 is endpoint. 12 is line. >=13 is branch.
    branch_points = (filtered >= 13)

    result = np.sum(branch_points)

    return float(result)
