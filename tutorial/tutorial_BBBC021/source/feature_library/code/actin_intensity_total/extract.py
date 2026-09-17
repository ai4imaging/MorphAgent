def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    
    # Convert to appropriate array type
    # The input is expected to be (512, 512, 3) uint8
    arr = np.asarray(img)
    
    # Handle dimensionality according to dataset format
    # Dataset description: (Height, Width, Channels) = (512, 512, 3)
    # Channel 0 is Actin (Red)
    
    # Check if the image has the expected structure
    if arr.ndim != 3 or arr.shape[2] < 1:
        # If dimensions are unexpected (e.g., 2D grayscale), return 0.0
        return 0.0

    # Extract the Actin channel (Channel 0)
    # We convert to float64 immediately to prevent overflow during summation
    # (uint8 max is 255, summing 512*512 pixels will overflow uint32)
    actin_channel = arr[..., 0].astype(np.float64)

    # Handle segmentation masks if available
    # If masks are provided, we use them to restrict the sum to cellular regions.
    # If no masks are provided, we sum the entire field of view (standard for total intensity).
    
    mask_to_use = None
    
    if len(segmentation_masks) > 0:
        # Combine all available masks into a single binary mask
        # Masks are typically labeled integers (0=bg, 1..N=cells)
        combined_mask = np.zeros(actin_channel.shape, dtype=bool)
        
        for mask in segmentation_masks:
            if mask is not None:
                # Ensure mask shape matches image shape (handle potential 2D vs 3D mismatch)
                # Masks are usually 2D (H, W)
                if mask.shape == actin_channel.shape:
                    combined_mask = np.logical_or(combined_mask, mask > 0)
                elif mask.ndim == 2 and mask.shape == actin_channel.shape[:2]:
                    combined_mask = np.logical_or(combined_mask, mask > 0)
        
        # If we successfully created a mask, apply it
        if np.any(combined_mask):
            mask_to_use = combined_mask

    # Compute the feature: Total Intensity
    if mask_to_use is not None:
        # Sum only pixels within the segmented regions
        total_intensity = np.sum(actin_channel[mask_to_use])
    else:
        # Sum the entire image
        total_intensity = np.sum(actin_channel)

    return float(total_intensity)
