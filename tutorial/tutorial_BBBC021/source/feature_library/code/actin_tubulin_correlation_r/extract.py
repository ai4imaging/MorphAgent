def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.filters import threshold_otsu
    
    # Convert to appropriate array type
    # The dataset is uint8, convert to float32 for calculations to avoid overflow
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Expected shape: (512, 512, 3)
    # If the image is not 3D or doesn't have at least 2 channels, we cannot compute correlation
    if arr.ndim != 3 or arr.shape[2] < 2:
        return 0.0

    # Extract channels based on dataset description
    # Channel 0: Actin (Red)
    # Channel 1: Tubulin (Green)
    actin = arr[:, :, 0]
    tubulin = arr[:, :, 1]

    # Determine the regions of interest (ROI)
    # We want to compute correlation only within cellular regions to avoid background noise
    # artificially inflating the correlation (since background is low in both channels).
    
    correlations = []

    # Check if segmentation masks are provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use the first available mask (assuming it's a cell or nuclei mask)
        mask = segmentation_masks[0]
        
        # Ensure mask shape matches image spatial dimensions
        if mask.shape != actin.shape:
            # If mask doesn't match, fall back to global foreground
            pass 
        else:
            # Get unique labels (excluding 0 which is background)
            labels = np.unique(mask)
            labels = labels[labels > 0]

            # If we have labeled objects, compute correlation per object
            if len(labels) > 0:
                for label_id in labels:
                    # Boolean mask for the current cell
                    cell_mask = (mask == label_id)
                    
                    # Extract pixel values for this cell
                    act_vals = actin[cell_mask]
                    tub_vals = tubulin[cell_mask]
                    
                    # Skip if too few pixels to compute meaningful correlation
                    if len(act_vals) < 10:
                        continue
                        
                    # Check for zero variance (constant intensity) which causes NaN in correlation
                    if np.std(act_vals) == 0 or np.std(tub_vals) == 0:
                        continue
                        
                    # Compute Pearson correlation
                    # np.corrcoef returns a matrix [[1, r], [r, 1]], we want [0, 1]
                    r = np.corrcoef(act_vals, tub_vals)[0, 1]
                    
                    if not np.isnan(r):
                        correlations.append(r)

    # Fallback: If no valid correlations computed (no masks, empty masks, or mask mismatch)
    # Compute correlation on the global foreground
    if not correlations:
        # Create a foreground mask using Otsu thresholding on the sum of channels
        # This separates cells from empty background
        sum_img = actin + tubulin
        
        # Handle completely black images
        if np.max(sum_img) == 0:
            return 0.0
            
        try:
            thresh = threshold_otsu(sum_img)
            fg_mask = sum_img > thresh
        except Exception:
            # Fallback for extremely low contrast or uniform images
            fg_mask = sum_img > np.mean(sum_img)

        # Extract pixels in foreground
        act_vals = actin[fg_mask]
        tub_vals = tubulin[fg_mask]

        if len(act_vals) >= 10 and np.std(act_vals) > 0 and np.std(tub_vals) > 0:
            r = np.corrcoef(act_vals, tub_vals)[0, 1]
            if not np.isnan(r):
                correlations.append(r)

    # Return the mean correlation across all cells (or the single global correlation)
    if not correlations:
        return 0.0
        
    result = np.mean(correlations)
    
    # Ensure result is within valid range [-1, 1] (floating point errors can cause slight deviations)
    result = np.clip(result, -1.0, 1.0)

    return float(result)
