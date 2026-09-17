def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.morphology import skeletonize
    from skimage.filters import threshold_otsu
    
    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset: (512, 512, 3), Channel 1 is Tubulin
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Extract Green channel (Tubulin)
        tubulin_channel = arr[:, :, 1]
    elif arr.ndim == 2:
        # Fallback if passed a single channel image (though unlikely given description)
        tubulin_channel = arr
    else:
        return 0.0

    # Intensity normalization
    # Normalize to [0, 1] range for consistent thresholding
    vmax = np.percentile(tubulin_channel, 99.5) if tubulin_channel.size > 0 else 1.0
    if vmax > 0:
        tubulin_channel = tubulin_channel / vmax
    tubulin_channel = np.clip(tubulin_channel, 0.0, 1.0)

    # Determine Region of Interest (ROI)
    # If segmentation masks are provided, use them to mask the analysis area.
    # If not, generate a foreground mask based on intensity.
    roi_mask = None
    if len(segmentation_masks) > 0:
        # Combine all available masks into a single binary mask
        combined_mask = np.zeros(tubulin_channel.shape, dtype=bool)
        for mask in segmentation_masks:
            if mask is not None:
                # Ensure mask is 2D and matches image shape
                if mask.ndim == 2 and mask.shape == tubulin_channel.shape:
                    combined_mask = combined_mask | (mask > 0)
                elif mask.ndim == 3 and mask.shape[:2] == tubulin_channel.shape:
                     # Handle potential 3D mask (e.g. if passed as (H, W, 1))
                    combined_mask = combined_mask | (mask[:, :, 0] > 0)
        
        if np.any(combined_mask):
            roi_mask = combined_mask

    # If no valid segmentation mask was created, create a simple intensity-based one
    if roi_mask is None:
        # Simple background subtraction/masking
        # Use a low threshold to just exclude empty background
        try:
            thresh_bg = threshold_otsu(tubulin_channel)
            roi_mask = tubulin_channel > thresh_bg
        except Exception:
            # Fallback for extremely low contrast or empty images
            roi_mask = tubulin_channel > 0.1

    # Preprocessing for Skeletonization
    # 1. Smooth the image to reduce noise that causes false branches
    smoothed_tubulin = ndimage.gaussian_filter(tubulin_channel, sigma=1.0)
    
    # 2. Binarize the tubulin structures
    # We need a binary representation of the "tubes" to skeletonize them.
    # We apply the threshold only within the ROI.
    try:
        # Calculate threshold on pixels within the ROI to be adaptive
        roi_pixels = smoothed_tubulin[roi_mask]
        if roi_pixels.size == 0:
            return 0.0
        
        # Otsu on the ROI pixels usually gives a good separation of structure vs diffuse cyto
        thresh_val = threshold_otsu(roi_pixels)
        binary_structure = (smoothed_tubulin > thresh_val) & roi_mask
    except Exception:
        return 0.0

    # Skeletonization
    # This reduces the binary tubes to 1-pixel wide lines
    skeleton = skeletonize(binary_structure)

    # Quantification
    # Count the number of pixels in the skeleton.
    # This is a direct proxy for the total length of the network.
    # Note: A more precise length would weight diagonal neighbors by sqrt(2),
    # but pixel count is a standard robust morphological feature for screening.
    skeleton_length = np.sum(skeleton)

    return float(skeleton_length)
