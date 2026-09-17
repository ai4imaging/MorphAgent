def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    
    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3) where Channel 1 is Tubulin (Green)
    # If the image is 3D with channels in the last dimension
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Extract Tubulin channel (Index 1)
        tubulin_img = arr[:, :, 1]
    elif arr.ndim == 2:
        # If already 2D, assume it's the relevant channel or a projection
        tubulin_img = arr
    else:
        # Unexpected format, return 0.0
        return 0.0

    # Intensity normalization
    # Normalize to [0, 1] range based on uint8 assumption (0-255)
    # This ensures gradient magnitude is scale-invariant relative to bit-depth
    tubulin_img = tubulin_img / 255.0
    tubulin_img = np.clip(tubulin_img, 0.0, 1.0)

    # Handle segmentation masks
    roi_mask = None
    
    # Check if any masks were provided
    if len(segmentation_masks) > 0:
        # Combine all available masks into a single boolean mask
        # Masks are typically labeled integers (0=bg, 1..N=cells)
        combined_mask = np.zeros(tubulin_img.shape, dtype=bool)
        
        for mask in segmentation_masks:
            if mask is not None:
                # Ensure mask matches image spatial dimensions
                if mask.shape == tubulin_img.shape:
                    combined_mask = np.logical_or(combined_mask, mask > 0)
                elif mask.ndim == 3 and mask.shape[:2] == tubulin_img.shape:
                    # Handle case where mask might be 3D (e.g. one-hot or RGB mask)
                    # Flatten to 2D by taking max projection or checking any channel
                    mask_2d = np.any(mask > 0, axis=-1)
                    combined_mask = np.logical_or(combined_mask, mask_2d)
        
        if np.any(combined_mask):
            roi_mask = combined_mask

    # Fallback: If no valid mask provided or mask is empty, generate one using Otsu thresholding
    if roi_mask is None:
        # Apply slight smoothing before thresholding to reduce noise
        smooth_img = ndimage.gaussian_filter(tubulin_img, sigma=2.0)
        
        # Calculate Otsu threshold
        # Handle edge case where image is uniform (min==max)
        if np.min(smooth_img) == np.max(smooth_img):
            thresh = np.min(smooth_img)
        else:
            thresh = threshold_otsu(smooth_img)
            
        roi_mask = smooth_img > thresh

    # Check if mask is empty after fallback
    if not np.any(roi_mask):
        return 0.0

    # Feature Computation: Gradient Magnitude
    # 1. Apply Gaussian smoothing to reduce high-frequency noise that amplifies gradients
    #    Sigma=1.0 is a standard choice for texture analysis at this resolution
    smoothed_tubulin = ndimage.gaussian_filter(tubulin_img, sigma=1.0)

    # 2. Compute gradients in X and Y directions using Sobel operator
    #    Sobel is preferred for edge detection/gradient magnitude
    grad_x = ndimage.sobel(smoothed_tubulin, axis=0)
    grad_y = ndimage.sobel(smoothed_tubulin, axis=1)

    # 3. Compute Euclidean magnitude of the gradient
    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # 4. Extract values within the ROI (Region of Interest)
    masked_magnitudes = magnitude[roi_mask]

    # 5. Calculate the mean gradient magnitude
    #    This quantifies the average "sharpness" or texture complexity of the tubulin structure
    result = np.mean(masked_magnitudes)

    return float(result)
