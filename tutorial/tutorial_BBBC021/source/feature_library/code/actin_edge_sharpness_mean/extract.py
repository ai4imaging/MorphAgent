def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_erosion, binary_dilation, disk

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Expected shape: (512, 512, 3)
    # Channel 0 = Actin (Red)
    actin_channel = None
    
    if arr.ndim == 3 and arr.shape[2] == 3:
        actin_channel = arr[:, :, 0]
    elif arr.ndim == 2:
        # Fallback if passed a single channel image, assume it's the relevant one
        actin_channel = arr
    else:
        return 0.0

    # Intensity normalization [0, 1]
    # This ensures gradient magnitudes are comparable across images
    vmax = np.percentile(actin_channel, 99.5) if actin_channel.size > 0 else 1.0
    if vmax <= 0:
        vmax = 1.0
    actin_norm = actin_channel / vmax
    actin_norm = np.clip(actin_norm, 0.0, 1.0)

    # Define the mask for cells
    # If segmentation masks are provided, combine them to get a total cellular area
    # If not, generate a mask using Otsu thresholding on the actin channel
    combined_mask = np.zeros(actin_norm.shape, dtype=bool)
    
    if len(segmentation_masks) > 0:
        for mask in segmentation_masks:
            if mask is not None:
                # Ensure mask matches image dimensions (handle potential 2D vs 3D mismatch)
                if mask.shape == actin_norm.shape:
                    combined_mask = np.logical_or(combined_mask, mask > 0)
                elif mask.ndim == 3 and mask.shape[:2] == actin_norm.shape:
                     # If mask is 3D (e.g. one-hot encoded or just extra dim), flatten it
                    combined_mask = np.logical_or(combined_mask, np.any(mask > 0, axis=-1))
    
    # If no valid masks were found or provided, generate one
    if not np.any(combined_mask):
        # Blur slightly to reduce noise before thresholding
        blurred = ndimage.gaussian_filter(actin_norm, sigma=2.0)
        try:
            thresh = threshold_otsu(blurred)
            combined_mask = blurred > thresh
        except Exception:
            # Fallback for empty/uniform images
            return 0.0

    # Clean the mask: remove small holes and noise
    # Binary closing (dilation then erosion) to fill holes
    # Binary opening (erosion then dilation) to remove small noise
    struct = disk(2)
    combined_mask = binary_dilation(combined_mask, struct)
    combined_mask = binary_erosion(combined_mask, struct)
    
    # If mask is empty after cleaning, return 0
    if not np.any(combined_mask):
        return 0.0

    # Define the boundary region
    # The "edge sharpness" is best measured at the transition from cell to background.
    # We define the boundary as the region consisting of the outer edge of the mask.
    # boundary = mask XOR eroded_mask
    eroded_mask = binary_erosion(combined_mask, disk(1))
    boundary_mask = np.logical_xor(combined_mask, eroded_mask)
    
    # Exclude image borders from the boundary mask to avoid artifacts
    # (e.g., if a cell is cut off by the image edge, the gradient there is artificial)
    border_margin = 2
    boundary_mask[:border_margin, :] = False
    boundary_mask[-border_margin:, :] = False
    boundary_mask[:, :border_margin] = False
    boundary_mask[:, -border_margin:] = False

    if not np.any(boundary_mask):
        return 0.0

    # Compute Gradient Magnitude on the normalized Actin channel
    # Using Sobel operator
    sx = ndimage.sobel(actin_norm, axis=0, mode='reflect')
    sy = ndimage.sobel(actin_norm, axis=1, mode='reflect')
    gradient_magnitude = np.hypot(sx, sy)

    # Extract gradient values specifically at the boundaries
    boundary_gradients = gradient_magnitude[boundary_mask]

    # Compute the mean sharpness
    result = np.mean(boundary_gradients)

    return float(result)
