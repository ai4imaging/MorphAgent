def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.filters import threshold_otsu

    # Convert to appropriate array type (float32 for calculation precision)
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3) where:
    # Ch 0: Actin (Red) - Cytoskeleton
    # Ch 1: Tubulin (Green) - Microtubules
    # Ch 2: DAPI (Blue) - Nucleus
    
    if arr.ndim != 3 or arr.shape[2] != 3:
        # Fallback for unexpected shapes, though dataset spec says (512, 512, 3)
        return 0.0

    # Extract relevant channels
    # We want correlation between Nucleus (Ch 2) and Tubulin (Ch 1)
    tubulin_ch = arr[:, :, 1]
    dapi_ch = arr[:, :, 2]

    # Helper function to calculate Pearson Correlation Coefficient
    def calculate_pcc(x, y):
        # Flatten arrays
        x_flat = x.flatten()
        y_flat = y.flatten()
        
        # Check for sufficient data points
        if x_flat.size < 2:
            return 0.0
            
        # Check for zero variance (constant signal) to avoid division by zero
        x_std = np.std(x_flat)
        y_std = np.std(y_flat)
        
        if x_std < 1e-6 or y_std < 1e-6:
            return 0.0
            
        # Compute correlation
        # Pearson r = Cov(X, Y) / (Std(X) * Std(Y))
        # np.corrcoef returns the correlation matrix [[1, r], [r, 1]]
        return np.corrcoef(x_flat, y_flat)[0, 1]

    # Strategy:
    # 1. If segmentation masks are provided, calculate PCC per cell and average.
    #    This is biologically more accurate as it avoids correlating background noise.
    # 2. If no masks, generate a foreground mask (Otsu) to exclude empty background,
    #    then calculate global PCC on the foreground.

    correlations = []

    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use the first available mask (assuming it's a cell or nuclei mask)
        mask = segmentation_masks[0]
        
        # Ensure mask matches image spatial dimensions
        if mask.shape[:2] == arr.shape[:2]:
            # Get unique labels (excluding background 0)
            labels = np.unique(mask)
            labels = labels[labels > 0]
            
            if len(labels) > 0:
                for label_id in labels:
                    # Extract pixels for this specific cell
                    cell_indices = (mask == label_id)
                    
                    cell_tubulin = tubulin_ch[cell_indices]
                    cell_dapi = dapi_ch[cell_indices]
                    
                    pcc = calculate_pcc(cell_tubulin, cell_dapi)
                    correlations.append(pcc)
                
                # Return the mean correlation across all cells
                if correlations:
                    return float(np.mean(correlations))
                else:
                    return 0.0
    
    # Fallback: No valid segmentation masks provided or mask processing failed
    # Create a "biological foreground" mask to avoid correlating the large black background
    try:
        # Calculate thresholds. Handle completely black images safely.
        if np.max(dapi_ch) > 0:
            thresh_dapi = threshold_otsu(dapi_ch)
        else:
            thresh_dapi = 0
            
        if np.max(tubulin_ch) > 0:
            thresh_tubulin = threshold_otsu(tubulin_ch)
        else:
            thresh_tubulin = 0
            
        # Foreground is where EITHER signal is significant
        foreground_mask = (dapi_ch > thresh_dapi) | (tubulin_ch > thresh_tubulin)
        
        if np.sum(foreground_mask) < 2:
            return 0.0
            
        # Calculate global correlation on foreground pixels
        fg_tubulin = tubulin_ch[foreground_mask]
        fg_dapi = dapi_ch[foreground_mask]
        
        result = calculate_pcc(fg_tubulin, fg_dapi)
        return float(result)
        
    except Exception:
        # In case of Otsu failure or other numerical errors
        return 0.0
