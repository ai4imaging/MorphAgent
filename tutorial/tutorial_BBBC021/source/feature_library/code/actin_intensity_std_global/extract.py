def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    
    # 1. Handle Image Format and Channel Selection
    # Dataset Description: 
    # - Dimensions: (512, 512, 3)
    # - Channel 0: Red (Actin) - THIS IS THE TARGET CHANNEL
    # - Channel 1: Green (Tubulin)
    # - Channel 2: Blue (DAPI)
    
    # Ensure input is an array
    img = np.asarray(img)
    
    # Check dimensionality to avoid errors with unexpected inputs
    if img.ndim != 3 or img.shape[2] < 1:
        return 0.0
        
    # Extract the Actin channel (Channel 0)
    # Convert to float64 for accurate statistical computation
    actin_channel = img[:, :, 0].astype(np.float64)

    # 2. Define Region of Interest (ROI)
    # Ideally, we calculate statistics only on the cellular regions to avoid 
    # the large black background skewing the standard deviation.
    
    pixels_to_measure = None
    
    if segmentation_masks and len(segmentation_masks) > 0:
        # If segmentation masks are provided, create a combined boolean mask
        # Masks are typically labeled integers (0=bg, 1..N=cells)
        combined_mask = np.zeros(actin_channel.shape, dtype=bool)
        
        has_valid_mask = False
        for mask in segmentation_masks:
            if mask is not None:
                # Ensure mask shape matches image shape (handle potential broadcasting issues if any)
                if mask.shape == actin_channel.shape:
                    combined_mask = np.logical_or(combined_mask, mask > 0)
                    has_valid_mask = True
        
        if has_valid_mask and np.any(combined_mask):
            pixels_to_measure = actin_channel[combined_mask]
        else:
            # Fallback if masks are empty or invalid shapes
            pixels_to_measure = actin_channel.flatten()
    else:
        # No segmentation masks provided: use the whole image
        # Note: This includes background pixels, which is acceptable for a "global" 
        # feature if no segmentation is available, though less specific.
        pixels_to_measure = actin_channel.flatten()

    # 3. Compute Feature: Standard Deviation
    # Handle edge case where the pixel array might be empty
    if pixels_to_measure.size == 0:
        return 0.0
        
    # Calculate standard deviation
    # High std -> Heterogeneous cytoskeleton (stress fibers)
    # Low std -> Diffuse staining
    actin_std = np.std(pixels_to_measure)
    
    return float(actin_std)
