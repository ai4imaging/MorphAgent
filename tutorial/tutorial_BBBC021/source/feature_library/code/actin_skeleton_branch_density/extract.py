def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.morphology import skeletonize
    from skimage.filters import threshold_otsu

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3) where Channel 0 is Actin
    if arr.ndim == 3 and arr.shape[2] >= 1:
        actin_channel = arr[..., 0]
    elif arr.ndim == 2:
        # Fallback if only 2D image provided, assume it's the relevant channel
        actin_channel = arr
    else:
        return 0.0

    # Intensity normalization
    # Normalize to [0, 1] for processing
    vmax = np.percentile(actin_channel, 99.5) if actin_channel.size > 0 else 1.0
    if vmax > 0:
        actin_channel = actin_channel / vmax
    actin_channel = np.clip(actin_channel, 0.0, 1.0)

    # Preprocessing: Mild Gaussian blur to reduce noise before skeletonization
    # This prevents pixel noise from creating false "hairy" branches
    actin_smooth = ndimage.gaussian_filter(actin_channel, sigma=1.0)

    # Determine Binary Mask
    # Use segmentation mask if available (prefer the first one as a cell/foreground mask)
    # Otherwise, generate one using Otsu thresholding on the smoothed actin channel
    binary_mask = None
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Assuming mask is labeled (0=bg, >0=cells), convert to boolean
        mask_input = segmentation_masks[0]
        # Handle potential dimension mismatch if mask is 3D or different shape
        if mask_input.shape == actin_channel.shape:
            binary_mask = mask_input > 0
        elif mask_input.ndim == 3 and mask_input.shape[:2] == actin_channel.shape:
             # If mask is (H, W, 1) or similar
             binary_mask = mask_input[..., 0] > 0
    
    if binary_mask is None:
        # Fallback: Otsu thresholding
        try:
            thresh = threshold_otsu(actin_smooth)
            binary_mask = actin_smooth > thresh
        except Exception:
            # If image is constant or empty
            return 0.0

    # Ensure binary mask is boolean
    binary_mask = binary_mask.astype(bool)
    
    # Calculate Foreground Area (denominator)
    foreground_area = np.sum(binary_mask)
    
    # If no foreground, return 0
    if foreground_area < 10: # Minimum area check to avoid noise
        return 0.0

    # Skeletonize the binary mask
    # This reduces actin fibers to 1-pixel wide lines
    skeleton = skeletonize(binary_mask)
    
    # Branch Point Detection
    # A branch point in a skeleton is a pixel with > 2 neighbors in its 8-neighborhood
    
    # Define 3x3 kernel for neighbor counting
    # [[1, 1, 1],
    #  [1, 0, 1],
    #  [1, 1, 1]]
    # However, a simpler way is to sum all 9 neighbors and subtract the center pixel value
    # if the center is 1.
    
    # Convert skeleton to integer (0, 1)
    skeleton_int = skeleton.astype(int)
    
    # Convolve with a kernel of all ones to count neighbors (including self)
    kernel = np.ones((3, 3), dtype=int)
    neighbor_count = ndimage.convolve(skeleton_int, kernel, mode='constant', cval=0)
    
    # A pixel is part of the skeleton if skeleton_int == 1
    # The number of actual neighbors is (neighbor_count - 1) because the kernel includes the center
    # We are looking for pixels where skeleton is present AND neighbors > 2
    # So neighbor_count > 3 (since self is 1, plus >2 neighbors = >3 total sum)
    
    branch_points = (skeleton_int == 1) & (neighbor_count > 3)
    num_branch_points = np.sum(branch_points)

    # Compute Density
    # Density = Number of Branch Points / Total Actin Area
    # This normalizes for cell size/confluence
    branch_density = num_branch_points / float(foreground_area)

    return float(branch_density)
