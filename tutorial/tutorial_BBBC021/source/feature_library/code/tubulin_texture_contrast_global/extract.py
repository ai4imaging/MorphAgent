def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.feature import graycomatrix, graycoprops
    
    # 1. Input Validation and Preparation
    # Ensure input is a numpy array
    arr = np.asarray(img)
    
    # Check dimensionality and extract Tubulin channel (Channel 1 - Green)
    # Dataset format: (Height, Width, Channels) = (512, 512, 3)
    if arr.ndim == 3 and arr.shape[2] >= 2:
        # Extract Channel 1 (Green/Tubulin)
        tubulin_channel = arr[:, :, 1]
    elif arr.ndim == 2:
        # Fallback if passed a single channel image (unlikely based on spec, but safe)
        tubulin_channel = arr
    else:
        # Unexpected format
        return 0.0

    # 2. Data Type Handling
    # GLCM requires integer types. The dataset is uint8 (0-255).
    # If the input was normalized to float, we need to rescale to integers for GLCM.
    if np.issubdtype(tubulin_channel.dtype, np.floating):
        # Normalize to 0-255 and cast to uint8
        min_val = tubulin_channel.min()
        max_val = tubulin_channel.max()
        if max_val > min_val:
            tubulin_channel = ((tubulin_channel - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            tubulin_channel = np.zeros_like(tubulin_channel, dtype=np.uint8)
    elif tubulin_channel.dtype != np.uint8:
        # Cast other integer types to uint8 (clipping if necessary)
        tubulin_channel = np.clip(tubulin_channel, 0, 255).astype(np.uint8)

    # 3. Masking (Optional but Recommended)
    # If segmentation masks are provided, use them to focus on cellular regions.
    # This reduces the impact of background noise on texture calculations.
    if segmentation_masks and len(segmentation_masks) > 0:
        # Combine all available masks into a single boolean foreground mask
        # Masks are typically labeled integers (0=bg, 1..N=cells)
        combined_mask = np.zeros(tubulin_channel.shape, dtype=bool)
        for mask in segmentation_masks:
            if mask is not None:
                # Ensure mask shape matches image shape (handle potential 3D vs 2D mismatch)
                if mask.shape == tubulin_channel.shape:
                    combined_mask = np.logical_or(combined_mask, mask > 0)
        
        # Apply mask: Set background pixels to 0
        # Note: In GLCM, 0 is a valid level. Setting background to 0 treats it as a large
        # uniform region. This is standard for texture analysis when ROI is irregular.
        # Alternatively, one could compute GLCM only on ROI pixels, but standard implementation
        # usually involves masking the array.
        if np.any(combined_mask):
            tubulin_channel = tubulin_channel * combined_mask
        else:
            # If masks exist but are empty, return 0.0 or process whole image?
            # Safer to return 0.0 as there is no biological content.
            return 0.0

    # 4. GLCM Computation
    # Parameters:
    # - distances=[1]: Pixel adjacency for fine texture.
    # - angles=[0, 45, 90, 135]: 4 directions for rotational invariance.
    # - levels=256: Standard for uint8.
    # - symmetric=True, normed=True: Standard GLCM properties.
    try:
        glcm = graycomatrix(
            tubulin_channel, 
            distances=[1], 
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
            levels=256, 
            symmetric=True, 
            normed=True
        )
    except ValueError:
        # Handle cases where image might be empty or invalid
        return 0.0

    # 5. Feature Extraction: Contrast
    # Contrast measures the local variations in the gray-level co-occurrence matrix.
    # High contrast = sharp transitions (e.g., microtubule bundles).
    # Low contrast = smooth/diffuse regions (e.g., depolymerized tubulin).
    contrast_values = graycoprops(glcm, 'contrast')
    
    # Average across all 4 directions to get a rotation-invariant global scalar
    global_contrast = np.mean(contrast_values)

    return float(global_contrast)
