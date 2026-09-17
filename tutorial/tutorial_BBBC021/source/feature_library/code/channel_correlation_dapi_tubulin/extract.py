def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    
    # 1. Data Preparation and Validation
    # Convert to float32 for calculation to avoid overflow/underflow
    arr = np.asarray(img, dtype=np.float32)
    
    # Check dimensionality: Expecting (H, W, C) where C >= 3
    # Dataset spec: (512, 512, 3) -> Ch1=Tubulin, Ch2=DAPI
    if arr.ndim != 3 or arr.shape[2] < 3:
        return 0.0
        
    # Extract relevant channels based on dataset description
    # Channel 1: Tubulin (Green)
    # Channel 2: DAPI (Nucleus/Blue)
    tubulin = arr[:, :, 1]
    dapi = arr[:, :, 2]
    
    # 2. Define Region of Interest (ROI)
    # We want to calculate correlation either per-cell (if mask exists) or globally on foreground
    
    correlations = []
    
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # --- Strategy A: Segmentation Mask Available ---
        # Use the first mask provided (assuming it's a cell or nuclei label mask)
        mask = segmentation_masks[0]
        
        # Ensure mask shape matches image spatial dimensions
        if mask.shape != tubulin.shape:
            # If mask is 3D or mismatched, try to use it if spatial dims match, else fallback
            if mask.ndim == 3 and mask.shape[:2] == tubulin.shape:
                mask = mask[:, :, 0] # Take first slice if 3D
            elif mask.shape != tubulin.shape:
                # Fallback to global if shapes are totally incompatible
                mask = None
        
        if mask is not None:
            # Get unique object labels (excluding background 0)
            labels = np.unique(mask)
            labels = labels[labels > 0]
            
            # If we have labels, compute correlation per object
            if len(labels) > 0:
                # Pre-calculate object slices for performance (optional but good practice)
                # Here we iterate labels directly as it's robust
                for label_id in labels:
                    # Boolean mask for current cell
                    cell_mask = (mask == label_id)
                    
                    # Extract pixel values
                    t_vals = tubulin[cell_mask]
                    d_vals = dapi[cell_mask]
                    
                    # Statistical validity checks:
                    # 1. Need at least 2 pixels to compute correlation
                    # 2. Standard deviation must be > 0 (cannot correlate constant signals)
                    if len(t_vals) > 2:
                        t_std = np.std(t_vals)
                        d_std = np.std(d_vals)
                        
                        if t_std > 1e-6 and d_std > 1e-6:
                            # Compute Pearson correlation
                            # np.corrcoef returns [[1, r], [r, 1]]
                            r = np.corrcoef(t_vals, d_vals)[0, 1]
                            if not np.isnan(r):
                                correlations.append(r)
    
    # --- Strategy B: Fallback (No Mask or No Valid Objects) ---
    if not correlations:
        # If no segmentation or no valid cells found, compute on global foreground.
        # Create a simple intensity threshold mask to ignore empty background
        # (Background 0,0 correlation is meaningless)
        
        # Simple thresholding (assuming uint8 input range 0-255 roughly)
        # Using a low threshold to catch cell areas
        bg_threshold = 10.0
        foreground_mask = (tubulin > bg_threshold) | (dapi > bg_threshold)
        
        if np.sum(foreground_mask) > 10: # Need some foreground pixels
            t_vals = tubulin[foreground_mask]
            d_vals = dapi[foreground_mask]
            
            t_std = np.std(t_vals)
            d_std = np.std(d_vals)
            
            if t_std > 1e-6 and d_std > 1e-6:
                r = np.corrcoef(t_vals, d_vals)[0, 1]
                if not np.isnan(r):
                    correlations.append(r)
            else:
                return 0.0
        else:
            return 0.0

    # 3. Aggregation
    # Return the mean correlation across all cells (or the single global correlation)
    if correlations:
        result = np.mean(correlations)
        # Clamp result to valid range [-1, 1] just in case of float errors
        return float(np.clip(result, -1.0, 1.0))
    else:
        return 0.0
