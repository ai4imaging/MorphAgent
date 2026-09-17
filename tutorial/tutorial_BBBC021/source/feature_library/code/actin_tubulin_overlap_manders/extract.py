def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage

    # Convert to appropriate array type (float64 for precision in accumulation)
    arr = np.asarray(img, dtype=np.float64)

    # Handle dimensionality according to dataset format
    # Expected shape: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] < 2:
        return 0.0

    # Extract relevant channels
    # Channel 0: Actin (Red)
    # Channel 1: Tubulin (Green)
    actin = arr[:, :, 0]
    tubulin = arr[:, :, 1]

    # Determine Region of Interest (ROI)
    # If segmentation masks are provided, use the first one (assuming it defines cell bodies)
    # If not, generate a simple threshold-based mask to exclude background
    roi_mask = None
    
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use the first provided mask
        mask_input = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions
        if mask_input.shape == actin.shape:
            # Treat all non-zero labels as the cellular region
            roi_mask = mask_input > 0
    
    # Fallback if no valid mask provided
    if roi_mask is None:
        # Create a background mask based on intensity
        # Simple threshold: pixel is valid if either channel has significant signal
        # Using a low threshold (e.g., 10/255 for uint8 data) to capture cell bodies
        threshold = 10.0
        roi_mask = (actin > threshold) | (tubulin > threshold)

    # Check if mask is empty
    if not np.any(roi_mask):
        return 0.0

    # Flatten arrays based on the mask to get pixel vectors for the ROI
    a_flat = actin[roi_mask]
    b_flat = tubulin[roi_mask]

    # Calculate Manders' Overlap Coefficient (MOC)
    # Formula: sum(Ai * Bi) / sqrt(sum(Ai^2) * sum(Bi^2))
    
    # Numerator: Sum of products
    numerator = np.sum(a_flat * b_flat)

    # Denominator: Sqrt of product of sums of squares
    sum_sq_a = np.sum(a_flat ** 2)
    sum_sq_b = np.sum(b_flat ** 2)
    
    denominator = np.sqrt(sum_sq_a * sum_sq_b)

    # Handle division by zero (if one channel is completely silent in the mask)
    if denominator == 0:
        return 0.0

    result = numerator / denominator

    # Ensure result is within [0, 1] (though mathematically it should be)
    return float(np.clip(result, 0.0, 1.0))
