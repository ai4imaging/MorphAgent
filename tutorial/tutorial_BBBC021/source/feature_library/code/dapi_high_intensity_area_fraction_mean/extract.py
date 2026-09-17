def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    from skimage.measure import label

    # Convert to appropriate array type
    # The input is expected to be (512, 512, 3) uint8 based on dataset description
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality and channel selection
    # Dataset: (H, W, C) = (512, 512, 3)
    # Channel 2 is DAPI (Blue), which targets the Nucleus
    if arr.ndim == 3 and arr.shape[2] == 3:
        dapi_channel = arr[:, :, 2]
    elif arr.ndim == 2:
        # Fallback if only one channel is passed (unlikely given description, but safe)
        dapi_channel = arr
    else:
        return 0.0

    # Determine Segmentation Mask
    # We prioritize the provided segmentation masks.
    # If masks are provided, we assume the first one corresponds to nuclei/cells.
    # If no masks are provided, we generate one using Otsu thresholding on the DAPI channel.
    labels = None
    
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use the first provided mask
        labels = segmentation_masks[0]
        # Ensure labels are integers
        if labels.dtype != int and labels.dtype != np.int32 and labels.dtype != np.int64:
            labels = labels.astype(np.int32)
        
        # Check if mask dimensions match image dimensions (ignoring channels)
        if labels.shape != dapi_channel.shape:
            # If shapes don't match, we can't use the mask reliably. Fallback to computing one.
            labels = None

    if labels is None:
        # Fallback: Compute segmentation mask
        # 1. Smooth slightly to reduce noise
        smooth = ndimage.gaussian_filter(dapi_channel, sigma=2)
        # 2. Threshold
        try:
            thresh = threshold_otsu(smooth)
            binary_mask = smooth > thresh
        except ValueError:
            # Handle case where image is uniform (e.g., all black)
            return 0.0
        # 3. Label connected components
        labels = label(binary_mask)

    # Check if we have any objects
    unique_labels = np.unique(labels)
    # Remove background (0)
    unique_labels = unique_labels[unique_labels != 0]
    
    if len(unique_labels) == 0:
        return 0.0

    # Determine High Intensity Threshold
    # The feature is "fraction of nuclear area that exceeds a high intensity threshold".
    # A robust way to define "high intensity" is relative to the population of nuclear pixels.
    # We take the 90th percentile of all pixels belonging to nuclei (foreground).
    
    # Extract all pixel values that fall within any nucleus
    foreground_mask = labels > 0
    foreground_pixels = dapi_channel[foreground_mask]
    
    if foreground_pixels.size == 0:
        return 0.0
        
    # Calculate the global high-intensity threshold (e.g., 90th percentile of nuclear pixels)
    # This defines "bright condensed chromatin" relative to the general nuclear staining.
    high_intensity_threshold = np.percentile(foreground_pixels, 90)

    # Create a binary mask of high intensity pixels
    high_intensity_mask = (dapi_channel > high_intensity_threshold) & foreground_mask

    # Compute statistics per object
    # We need: (Area of high intensity within object) / (Total area of object)
    
    # 1. Total area per label
    # ndimage.sum with a mask of ones gives the count of pixels per label
    ones_image = np.ones_like(dapi_channel)
    areas = ndimage.sum(ones_image, labels, index=unique_labels)
    
    # 2. High intensity area per label
    # ndimage.sum of the high_intensity_mask over the labels
    high_int_areas = ndimage.sum(high_intensity_mask, labels, index=unique_labels)
    
    # Calculate fractions
    # Avoid division by zero (though areas should be > 0 since they come from unique_labels)
    fractions = np.divide(high_int_areas, areas, out=np.zeros_like(areas), where=areas!=0)
    
    # Compute the mean of these fractions across all nuclei
    result = np.mean(fractions)

    return float(result)
