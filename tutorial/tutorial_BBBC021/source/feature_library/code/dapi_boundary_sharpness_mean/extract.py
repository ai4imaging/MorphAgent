def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu, sobel
    from skimage.morphology import binary_dilation, binary_erosion, disk

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3) - Channel 2 is DAPI (Blue)
    # If the input is 3D with 3 channels, we extract the DAPI channel.
    # If it's already 2D, we assume it's the relevant channel or a projection.
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Extract Channel 2 (Blue/DAPI)
        dapi_img = arr[:, :, 2]
    elif arr.ndim == 2:
        dapi_img = arr
    else:
        # Unexpected format, return 0.0
        return 0.0

    # Intensity normalization
    # Normalize to [0, 1] for consistent gradient magnitude calculation
    # Using simple min-max or division by 255 if uint8-like range
    if dapi_img.max() > 1.0:
        dapi_img = dapi_img / 255.0
    dapi_img = np.clip(dapi_img, 0.0, 1.0)

    # Preprocessing: Mild Gaussian blur to suppress pixel noise
    # We want structural sharpness, not noise sharpness
    dapi_smooth = ndimage.gaussian_filter(dapi_img, sigma=1.0)

    # Determine Segmentation Mask
    # We need a binary mask of the nuclei to define the boundary
    mask = None
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use provided segmentation mask
        # Assuming mask is labeled (0=bg, >0=cells)
        mask = segmentation_masks[0] > 0
        
        # Ensure mask shape matches image shape (handle potential mismatches)
        if mask.shape != dapi_img.shape:
            # If mask is different size, fallback to internal generation
            mask = None
    
    # Fallback: Generate mask if none provided or invalid
    if mask is None:
        try:
            thresh = threshold_otsu(dapi_smooth)
            mask = dapi_smooth > thresh
        except Exception:
            # If thresholding fails (e.g., uniform image), return 0.0
            return 0.0

    # Ensure mask is boolean
    mask = mask.astype(bool)

    # If no objects detected, return 0.0
    if not np.any(mask):
        return 0.0

    # Define the Boundary Region
    # We want a narrow band around the edge of the nuclei.
    # Dilate - Erode gives us the morphological gradient (the edge).
    # Using a disk of radius 1 creates a boundary approx 2-3 pixels wide.
    selem = disk(1)
    dilated = binary_dilation(mask, footprint=selem)
    eroded = binary_erosion(mask, footprint=selem)
    boundary_mask = dilated & (~eroded)

    # If boundary mask is empty (e.g., image is all foreground or all background), return 0.0
    if not np.any(boundary_mask):
        return 0.0

    # Compute Gradient Magnitude
    # Sobel filter computes approximation of the gradient
    # We use the smoothed image to avoid noise
    grad_mag = sobel(dapi_smooth)

    # Extract gradients at the boundary
    boundary_gradients = grad_mag[boundary_mask]

    # Compute Mean Sharpness
    # Higher value = sharper boundary (intact nucleus)
    # Lower value = diffuse boundary (NEBD/fuzzy)
    result = np.mean(boundary_gradients)

    return float(result)
