def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from scipy import ndimage

    # Convert to appropriate array type
    # We use float32 to prevent overflow during summation, but we keep the original intensity scale (0-255)
    # because "Mass" (Integrated Intensity) is usually calculated on raw values to preserve absolute abundance info.
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract Tubulin Channel (Channel 1 - Green)
    tubulin_channel = arr[:, :, 1]

    # Determine Segmentation Mask
    labeled_mask = None

    # 1. Try using provided segmentation masks
    if len(segmentation_masks) > 0:
        # Use the first available mask. 
        # We assume the first mask passed is the most relevant (e.g., whole cell or cytoplasm).
        mask_input = segmentation_masks[0]
        
        # Ensure mask matches image spatial dimensions
        if mask_input.shape == tubulin_channel.shape:
            # If it's already labeled (integer mask), use it directly
            if np.issubdtype(mask_input.dtype, np.integer) and mask_input.max() > 1:
                labeled_mask = mask_input
            # If it's a binary mask (bool or 0/1), label it
            else:
                labeled_mask = label(mask_input > 0)

    # 2. Fallback: Generate segmentation if no valid mask provided
    if labeled_mask is None:
        # Create a proxy for cell body using Actin (Ch0) + Tubulin (Ch1)
        # Actin often defines cell boundaries well.
        actin_channel = arr[:, :, 0]
        combined_signal = actin_channel + tubulin_channel
        
        # Smooth to reduce noise
        smoothed = ndimage.gaussian_filter(combined_signal, sigma=2)
        
        # Threshold
        try:
            thresh = threshold_otsu(smoothed)
            binary_mask = smoothed > thresh
        except ValueError:
            # Handle case where image is uniform (e.g., all black)
            return 0.0
            
        # Fill holes and label
        binary_mask = ndimage.binary_fill_holes(binary_mask)
        labeled_mask = label(binary_mask)

    # Compute Feature: Mean Total Integrated Intensity (Mass) per Cell
    # We use the labeled mask to define regions, and the tubulin_channel for intensity
    regions = regionprops(labeled_mask, intensity_image=tubulin_channel)

    if not regions:
        return 0.0

    masses = []
    for props in regions:
        # Filter out small artifacts (e.g., debris < 50 pixels)
        if props.area < 50:
            continue
            
        # Integrated Intensity = Mean Intensity * Area
        # This is equivalent to summing all pixel values within the region
        # regionprops calculates mean_intensity efficiently
        mass = props.mean_intensity * props.area
        masses.append(mass)

    if not masses:
        return 0.0

    # Calculate the mean of the masses across all valid cells
    result = np.mean(masses)

    return float(result)
