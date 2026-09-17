def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_dilation, binary_erosion, disk

    # 1. Data Loading and Preprocessing
    # Convert to float32 for precision
    arr = np.asarray(img, dtype=np.float32)

    # Check dimensionality and extract DAPI channel
    # Dataset info: (512, 512, 3), Channel 2 = Blue = DAPI
    if arr.ndim == 3 and arr.shape[2] == 3:
        dapi_channel = arr[:, :, 2]
    elif arr.ndim == 2:
        # Fallback if only one channel is passed (unlikely based on spec but safe)
        dapi_channel = arr
    else:
        return 0.0

    # Normalize Intensity to [0, 1]
    # This ensures gradient magnitudes are comparable regardless of absolute intensity scaling
    # We use a robust max to avoid outliers skewing the normalization too much, 
    # but for uint8 data, dividing by 255 is standard. 
    # Given the prompt suggests normalization, we'll use min-max based on the image content 
    # or simply divide by 255 if it looks like uint8.
    # Let's use robust min-max to be safe across potential bit-depths.
    min_val = np.min(dapi_channel)
    max_val = np.max(dapi_channel)
    if max_val > min_val:
        dapi_norm = (dapi_channel - min_val) / (max_val - min_val)
    else:
        return 0.0

    # 2. Mask Acquisition
    # We need a binary mask of the nuclei to define the boundary.
    mask = None
    
    # Check if segmentation masks are provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Assume the first mask is the nuclear mask (standard convention)
        # Masks are labeled integers. Convert to binary.
        mask = segmentation_masks[0] > 0
        
        # Ensure mask shape matches image shape (handle potential 2D/3D mismatch)
        if mask.shape != dapi_channel.shape:
            # If shapes don't match, try to use the fallback
            mask = None

    # Fallback: Generate mask if none provided or invalid
    if mask is None:
        try:
            thresh = threshold_otsu(dapi_norm)
            mask = dapi_norm > thresh
        except Exception:
            # If image is constant or empty
            return 0.0

    # If mask is empty (no cells), return 0
    if not np.any(mask):
        return 0.0

    # 3. Define Boundary Region of Interest (ROI)
    # We want the gradient *at* the edge. A simple edge mask can be found by
    # taking the difference between a dilation and an erosion of the nuclear mask.
    # This creates a "rim" around the nucleus.
    
    # Use a small structural element (radius 1 or 2)
    selem = disk(1)
    
    dilated_mask = binary_dilation(mask, footprint=selem)
    eroded_mask = binary_erosion(mask, footprint=selem)
    
    # The boundary is the region added by dilation AND the region removed by erosion
    # This covers the transition zone from inside to outside.
    boundary_mask = dilated_mask & (~eroded_mask)

    if not np.any(boundary_mask):
        return 0.0

    # 4. Compute Gradient Magnitude
    # We compute the gradient on the normalized DAPI image.
    # Sobel filter is a standard way to approximate the gradient.
    
    # Gradient in X
    gx = ndimage.sobel(dapi_norm, axis=0)
    # Gradient in Y
    gy = ndimage.sobel(dapi_norm, axis=1)
    
    # Gradient Magnitude
    gradient_magnitude = np.sqrt(gx**2 + gy**2)

    # 5. Extract Feature
    # Calculate the mean gradient magnitude specifically within the boundary ROI
    boundary_gradients = gradient_magnitude[boundary_mask]
    
    if boundary_gradients.size == 0:
        return 0.0
        
    mean_boundary_gradient = np.mean(boundary_gradients)

    return float(mean_boundary_gradient)
