def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    from skimage.measure import label
    
    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset: (512, 512, 3), uint8.
    # Channel 0: Actin (Red)
    # Channel 1: Tubulin (Green)
    # Channel 2: DAPI (Blue)
    
    # Check for valid shape
    if arr.ndim != 3 or arr.shape[2] != 3:
        # If not the expected 3-channel image, return 0.0
        return 0.0

    # Extract relevant channels
    actin_channel = arr[:, :, 0]
    dapi_channel = arr[:, :, 2]

    # --- Step 1: Calculate Total Actin Area ---
    
    # Normalize Actin channel for thresholding
    # Robust max to handle outliers
    vmax_actin = np.percentile(actin_channel, 99.5) if actin_channel.size > 0 else 1.0
    if vmax_actin <= 0: vmax_actin = 1.0
    actin_norm = actin_channel / vmax_actin
    actin_norm = np.clip(actin_norm, 0.0, 1.0)

    # Apply Gaussian blur to smooth actin texture and reduce noise
    actin_smooth = ndimage.gaussian_filter(actin_norm, sigma=2.0)

    # Determine threshold for Actin
    # If image is completely black, otsu might fail or return 0. Handle empty image case.
    if np.max(actin_smooth) == 0:
        total_actin_area = 0.0
    else:
        try:
            thresh_actin = threshold_otsu(actin_smooth)
            actin_mask = actin_smooth > thresh_actin
            total_actin_area = np.sum(actin_mask)
        except Exception:
            # Fallback if otsu fails (e.g. uniform image)
            total_actin_area = 0.0

    # --- Step 2: Calculate Number of Nuclei ---
    
    num_nuclei = 0
    
    # Strategy A: Use Segmentation Masks if available
    # We assume the first mask provided is likely the nuclei mask or a cell mask containing nuclei info
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        mask = segmentation_masks[0]
        # Ensure mask is 2D
        if mask.ndim == 3:
            mask = np.max(mask, axis=2) # Project if 3D
            
        # If mask is labeled (integers > 1), max label is count
        # If mask is binary (0/1 or 0/255), we need to label it
        if np.max(mask) > 1:
            # Check if it looks like a label matrix (many unique values) or just a binary mask with value 255
            unique_vals = np.unique(mask)
            if len(unique_vals) > 2:
                # Likely a label matrix
                num_nuclei = np.max(mask)
            else:
                # Likely binary mask (e.g. 0 and 255)
                labeled_mask, num_features = label(mask > 0, return_num=True)
                num_nuclei = num_features
        else:
            # Binary 0/1
            labeled_mask, num_features = label(mask > 0, return_num=True)
            num_nuclei = num_features
            
    # Strategy B: Fallback to DAPI channel detection
    else:
        # Normalize DAPI
        vmax_dapi = np.percentile(dapi_channel, 99.5) if dapi_channel.size > 0 else 1.0
        if vmax_dapi <= 0: vmax_dapi = 1.0
        dapi_norm = dapi_channel / vmax_dapi
        dapi_norm = np.clip(dapi_norm, 0.0, 1.0)
        
        # Blur DAPI slightly
        dapi_smooth = ndimage.gaussian_filter(dapi_norm, sigma=1.0)
        
        if np.max(dapi_smooth) == 0:
            num_nuclei = 0
        else:
            try:
                thresh_dapi = threshold_otsu(dapi_smooth)
                dapi_mask = dapi_smooth > thresh_dapi
                
                # Fill holes to make nuclei solid
                dapi_mask = ndimage.binary_fill_holes(dapi_mask)
                
                # Label connected components
                labeled_nuclei, count = label(dapi_mask, return_num=True)
                
                # Filter small noise (optional but good for robustness)
                # Assume a nucleus must be at least some pixels (e.g., 20)
                sizes = ndimage.sum(dapi_mask, labeled_nuclei, range(count + 1))
                # sizes[0] is background
                valid_nuclei = np.sum(sizes[1:] > 20)
                num_nuclei = valid_nuclei
                
            except Exception:
                num_nuclei = 0

    # --- Step 3: Compute Ratio ---
    
    if num_nuclei == 0:
        return 0.0
    
    ratio = total_actin_area / float(num_nuclei)

    return float(ratio)
