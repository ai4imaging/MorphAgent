def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.feature import graycomatrix, graycoprops
    
    # Convert to appropriate array type
    # The input image is expected to be (512, 512, 3) uint8 based on dataset info.
    arr = np.asarray(img)
    
    # Handle dimensionality and Channel Selection
    # Dataset Info:
    # Channel 0: Red (Actin)
    # Channel 1: Green (Tubulin)
    # Channel 2: Blue (DAPI/Nucleus)
    #
    # The feature request asks for "Actin texture contrast".
    # Previous feedback noted confusion about channel indices. 
    # Based on the provided table: Channel 0 is Actin.
    
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Select Channel 0 for Actin
        actin_channel = arr[:, :, 0]
    elif arr.ndim == 2:
        # Fallback if single channel is passed (unlikely given description, but safe)
        actin_channel = arr
    else:
        return 0.0

    # Normalization for GLCM
    # Haralick features require integer inputs (levels).
    # Standard practice is to quantize the image into a smaller number of levels (e.g., 256 or 64)
    # to reduce computation time and noise sensitivity.
    # The input is uint8 (0-255). We will use this directly or bin it.
    # Using 256 levels is standard for 8-bit images.
    
    # Ensure it's uint8
    if actin_channel.dtype != np.uint8:
        # Normalize to 0-255 if not already
        min_val = np.min(actin_channel)
        max_val = np.max(actin_channel)
        if max_val > min_val:
            actin_channel = ((actin_channel - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            actin_channel = np.zeros_like(actin_channel, dtype=np.uint8)

    # Compute Gray Level Co-occurrence Matrix (GLCM)
    # Distances: 1 pixel (capture local texture)
    # Angles: 0, 45, 90, 135 degrees (average over directions for rotation invariance)
    # Levels: 256 (since we have uint8 data)
    try:
        glcm = graycomatrix(actin_channel, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                            levels=256, symmetric=True, normed=True)
    except ValueError:
        # Fallback for empty or invalid images
        return 0.0

    # Compute Contrast
    # Contrast measures the intensity contrast between a pixel and its neighbor over the whole image.
    # Range: [0, (levels-1)^2]. For 256 levels, max is 255^2 = 65025.
    contrast = graycoprops(glcm, 'contrast')
    
    # Average contrast across all 4 angles
    mean_contrast = np.mean(contrast)
    
    # Normalize the result to [0, 1] range as requested by previous feedback.
    # The maximum theoretical contrast for a GLCM with N levels is (N-1)^2.
    # Here N=256, so max_contrast = 255^2 = 65025.
    # However, real biological images rarely hit the theoretical max (checkerboard of 0 and 255).
    # Normalizing by the theoretical max ensures the output is strictly in [0, 1].
    
    max_theoretical_contrast = 255.0 ** 2
    normalized_contrast = mean_contrast / max_theoretical_contrast

    return float(normalized_contrast)
