def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage

    # 1. Input Validation and Preparation
    # Ensure image is a numpy array
    img = np.asarray(img)
    
    # Check dimensionality. We expect (H, W, C) = (512, 512, 3) or similar 2D multichannel
    if img.ndim != 3 or img.shape[2] < 2:
        # If dimensions are unexpected (e.g., 2D grayscale), return 0.0
        return 0.0

    # 2. Extract and Normalize the Tubulin Channel
    # According to dataset info: Channel 1 = Green = Tubulin
    # Convert to float64 for precision during summation and normalize to [0, 1]
    tubulin_channel = img[:, :, 1].astype(np.float64) / 255.0

    # 3. Determine the Segmentation Mask
    # The feature requires intensity "within segmented cell boundaries".
    # We need a mask that covers the cytoplasm/whole cell, not just the nucleus.
    
    final_mask = None

    if len(segmentation_masks) > 0:
        # Heuristic: If multiple masks are provided (e.g., nuclei and cells), 
        # the whole-cell mask will have the largest total area (sum of non-zero pixels).
        # We iterate through provided masks to find the one with the largest coverage.
        max_area = -1
        for mask in segmentation_masks:
            # Ensure mask is 2D matching the image spatial dims
            if mask.shape == img.shape[:2]:
                current_area = np.count_nonzero(mask)
                if current_area > max_area:
                    max_area = current_area
                    final_mask = mask
        
        # If we found a valid mask, convert it to boolean
        if final_mask is not None:
            final_mask = final_mask > 0
    
    # Fallback: If no valid segmentation masks provided
    if final_mask is None:
        # Create a simple background threshold mask to exclude empty space noise.
        # A low threshold (e.g., 10/255 approx 0.04) removes pure black background.
        final_mask = tubulin_channel > 0.04

    # 4. Compute Total Integrated Intensity
    # Apply mask to the normalized tubulin channel
    masked_pixels = tubulin_channel[final_mask]

    # If mask is empty (no cells), return 0.0
    if masked_pixels.size == 0:
        return 0.0

    # Sum the intensity values
    total_intensity = np.sum(masked_pixels)

    return float(total_intensity)
