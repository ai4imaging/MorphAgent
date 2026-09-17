def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.morphology import skeletonize, disk
    from skimage.filters import threshold_otsu, gaussian
    from skimage.util import img_as_bool

    # 1. Data Loading and Validation
    # Convert to float32 for processing
    arr = np.asarray(img, dtype=np.float32)

    # Check dimensionality
    # Expected shape is (512, 512, 3) for BBBC021
    if arr.ndim != 3 or arr.shape[2] != 3:
        # If not 3 channels, we can't reliably identify Tubulin (Ch1) vs Actin (Ch0)
        return 0.0

    # Normalize to [0, 1]
    if arr.max() > 1.0:
        arr /= 255.0
    arr = np.clip(arr, 0.0, 1.0)

    # 2. Channel Extraction
    # Channel 0: Actin (Red) - Cytoskeleton
    # Channel 1: Green (G) - Tubulin (Target for skeletonization)
    # Channel 2: Blue (B) - DAPI (Nucleus)
    actin_ch = arr[..., 0]
    tubulin_ch = arr[..., 1]

    # 3. Define Foreground Mask (Denominator)
    # The foreground is the total cellular area.
    # If a segmentation mask is provided, use it. Otherwise, generate one from Actin + Tubulin.
    foreground_mask = None

    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use the first provided mask (assumed to be cell mask)
        seg = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions
        if seg.shape == arr.shape[:2]:
            foreground_mask = seg > 0
    
    if foreground_mask is None:
        # Fallback: Create a mask from the image content
        # Combine Actin and Tubulin to get the full cell body shape
        combined_signal = np.maximum(actin_ch, tubulin_ch)
        
        # Smooth to reduce noise and merge internal gaps
        smoothed_signal = gaussian(combined_signal, sigma=2.0)
        
        # Threshold
        try:
            thresh_val = threshold_otsu(smoothed_signal)
            foreground_mask = smoothed_signal > thresh_val
        except Exception:
            # Fallback if image is empty or uniform
            return 0.0
            
        # Fill holes to make the cell area solid
        foreground_mask = ndimage.binary_fill_holes(foreground_mask)

    # Calculate Denominator (Foreground Area)
    foreground_area = np.sum(foreground_mask)
    
    if foreground_area == 0:
        return 0.0

    # 4. Tubulin Segmentation and Skeletonization (Numerator)
    # We need a binary representation of the tubulin fibers to skeletonize them.
    
    # Smooth tubulin channel slightly to avoid noise creating spurious branches
    tubulin_smooth = gaussian(tubulin_ch, sigma=1.0)
    
    # Mask the tubulin channel with the foreground to ignore background noise
    # We only care about tubulin inside the identified cell regions
    masked_tubulin = tubulin_smooth * foreground_mask
    
    # Threshold the tubulin signal
    # We compute threshold only on pixels within the foreground to be more adaptive
    tubulin_pixels = masked_tubulin[foreground_mask]
    
    if tubulin_pixels.size == 0:
        return 0.0
        
    try:
        # Use Otsu on the foreground pixels
        t_thresh = threshold_otsu(tubulin_pixels)
        binary_tubulin = masked_tubulin > t_thresh
    except Exception:
        return 0.0

    # Clean up the binary tubulin mask
    # Remove small specks that aren't filaments
    # Binary opening with a small structure
    # Note: ndimage.binary_opening is faster than cv2 or skimage for simple structuring elements
    struct = ndimage.generate_binary_structure(2, 1) # 4-connectivity
    binary_tubulin = ndimage.binary_opening(binary_tubulin, structure=struct, iterations=1)

    # Skeletonize
    # skeletonize expects boolean input
    skeleton = skeletonize(binary_tubulin)

    # Calculate Numerator (Skeleton Length)
    # In a skeletonized image, the sum of pixels approximates the length
    skeleton_length = np.sum(skeleton)

    # 5. Compute Feature
    # Density = Total Skeleton Length / Total Foreground Area
    density = skeleton_length / foreground_area

    return float(density)
