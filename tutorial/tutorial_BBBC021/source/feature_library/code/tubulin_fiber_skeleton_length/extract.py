def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage import filters, morphology, measure
    
    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3) - Channel 1 is Tubulin
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Extract Tubulin channel (Index 1)
        tubulin = arr[:, :, 1]
    elif arr.ndim == 2:
        # Fallback if only 2D image is passed (unlikely given description, but safe)
        tubulin = arr
    else:
        return 0.0

    # Intensity normalization
    # Normalize to [0, 1] range for filter stability
    vmax = np.percentile(tubulin, 99.5) if tubulin.size > 0 else 1.0
    if vmax > 0:
        tubulin = tubulin / vmax
    tubulin = np.clip(tubulin, 0.0, 1.0)

    # Preprocessing: Mild Gaussian blur to reduce pixel noise before ridge detection
    tubulin_smooth = ndimage.gaussian_filter(tubulin, sigma=1.0)

    # Feature Extraction Logic: Ridge Detection -> Skeletonization
    # We use the Meijering filter (neuriteness) to detect continuous fibers
    # This filter is sensitive to tubular structures and suppresses diffuse background
    try:
        # Meijering filter returns a response map where ridges are high intensity
        # black_ridges=False because fluorescence is bright on dark background
        ridges = filters.meijering(tubulin_smooth, sigmas=range(1, 4), black_ridges=False)
    except Exception:
        # Fallback if filters fail (e.g. image too small)
        return 0.0

    # Thresholding the ridge map
    # Otsu's method is generally robust for separating the enhanced ridges from the suppressed background
    try:
        thresh = filters.threshold_otsu(ridges)
        binary_fibers = ridges > thresh
    except ValueError:
        # Can happen if image is uniform
        return 0.0

    # Handle Segmentation Masks
    # If a segmentation mask is provided, we restrict the analysis to the cellular regions
    # to avoid skeletonizing background noise artifacts.
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        mask = segmentation_masks[0]
        # Ensure mask matches image dimensions (handle potential 3D/2D mismatch)
        if mask.shape == binary_fibers.shape:
            binary_fibers = binary_fibers & (mask > 0)
    else:
        # Fallback: Create a rough foreground mask from the raw intensity
        # to avoid analyzing empty background space
        background_thresh = np.percentile(tubulin, 20)
        foreground_mask = tubulin > background_thresh
        binary_fibers = binary_fibers & foreground_mask

    # Cleanup: Remove small disconnected specks that are likely noise, not fibers
    # min_size=10 pixels is a conservative threshold for a "fiber segment"
    binary_fibers = morphology.remove_small_objects(binary_fibers, min_size=10)

    # Skeletonization
    # Reduces the binary fibers to 1-pixel wide lines
    skeleton = morphology.skeletonize(binary_fibers)

    # Quantification
    # Sum the pixels in the skeleton.
    # While a simple sum counts pixels, it is a standard proxy for length in HCS.
    # A more precise length would weight diagonal pixels by sqrt(2), but raw pixel count
    # is robust and sufficient for relative comparison.
    skeleton_length = np.sum(skeleton)

    return float(skeleton_length)
