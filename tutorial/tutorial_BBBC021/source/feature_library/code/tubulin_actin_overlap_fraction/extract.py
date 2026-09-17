def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    
    # Convert to appropriate array type (float32) to prevent overflow during calculations
    arr = np.asarray(img, dtype=np.float32)
    
    # Handle dimensionality and validate input
    # Expected shape: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract relevant channels
    # Channel 0: Actin (Red) - The structural reference for overlap
    # Channel 1: Tubulin (Green) - The signal we want to quantify the overlap OF
    actin_ch = arr[:, :, 0]
    tubulin_ch = arr[:, :, 1]
    
    # Preprocessing: Mild Gaussian blur to reduce high-frequency noise
    # This prevents single-pixel noise from affecting the overlap calculation
    sigma = 1.0
    actin_smooth = ndimage.gaussian_filter(actin_ch, sigma)
    tubulin_smooth = ndimage.gaussian_filter(tubulin_ch, sigma)
    
    # Define Region of Interest (ROI) - The "Cell Mask"
    # We only want to calculate statistics within the cells, ignoring background
    cell_mask = None
    
    if len(segmentation_masks) > 0:
        # If segmentation masks are provided, combine them to form the cell mask
        # Masks are typically labeled integers (0=bg, 1..N=cells)
        combined_mask = np.zeros(arr.shape[:2], dtype=bool)
        for mask in segmentation_masks:
            if mask is not None and mask.shape == arr.shape[:2]:
                combined_mask = np.logical_or(combined_mask, mask > 0)
        
        if np.any(combined_mask):
            cell_mask = combined_mask

    # Fallback if no valid segmentation masks provided or masks were empty
    if cell_mask is None:
        # Create a rough tissue mask based on total intensity
        # Use the sum of Actin and Tubulin to find biological material
        total_intensity = actin_smooth + tubulin_smooth
        # Check if image is not empty
        if np.max(total_intensity) > 0:
            try:
                thresh_tissue = threshold_otsu(total_intensity)
                cell_mask = total_intensity > thresh_tissue
            except Exception:
                # Fallback for very low contrast images
                cell_mask = total_intensity > np.mean(total_intensity)
        else:
            return 0.0

    # Apply ROI mask to the smoothed channels
    # We set background pixels to 0 so they don't contribute to signal or thresholding
    actin_roi = np.where(cell_mask, actin_smooth, 0)
    tubulin_roi = np.where(cell_mask, tubulin_smooth, 0)
    
    # Determine signal thresholds for Manders' Coefficient
    # We need to distinguish "real signal" from "cellular background/autofluorescence"
    # We calculate thresholds based only on the pixels within the ROI to be more accurate
    roi_pixels_actin = actin_roi[cell_mask]
    roi_pixels_tubulin = tubulin_roi[cell_mask]
    
    if roi_pixels_actin.size == 0 or roi_pixels_tubulin.size == 0:
        return 0.0
        
    # Calculate thresholds (Otsu is standard for bimodal distributions common in fluorescence)
    # If the signal is too uniform (e.g., all background or all saturated), Otsu might fail or return min/max
    try:
        t_actin = threshold_otsu(roi_pixels_actin) if np.max(roi_pixels_actin) > np.min(roi_pixels_actin) else 0
        t_tubulin = threshold_otsu(roi_pixels_tubulin) if np.max(roi_pixels_tubulin) > np.min(roi_pixels_tubulin) else 0
    except Exception:
        t_actin = np.mean(roi_pixels_actin)
        t_tubulin = np.mean(roi_pixels_tubulin)

    # Calculate Manders' Overlap Coefficient M1 (Fraction of Tubulin overlapping Actin)
    # M1 = Sum(Tubulin_i_coloc) / Sum(Tubulin_i_total)
    # Where Tubulin_i_coloc = Tubulin_i if Actin_i > T_actin else 0
    
    # 1. Identify pixels where Actin is considered "present" (above threshold)
    actin_binary = actin_roi > t_actin
    
    # 2. Identify pixels where Tubulin is considered "present" (above threshold)
    # This is the denominator: Total valid Tubulin signal
    tubulin_mask = tubulin_roi > t_tubulin
    
    # 3. Identify pixels where Tubulin is present AND Actin is present
    # This is the numerator: Tubulin signal that overlaps with Actin
    overlap_mask = np.logical_and(tubulin_mask, actin_binary)
    
    # Sum intensities
    # Note: Manders' typically sums the raw intensities, not just the count of pixels.
    # We sum the Tubulin intensity at the overlapping locations.
    numerator = np.sum(tubulin_roi[overlap_mask])
    denominator = np.sum(tubulin_roi[tubulin_mask])
    
    # Avoid division by zero
    if denominator == 0:
        return 0.0
        
    result = numerator / denominator
    
    return float(result)
