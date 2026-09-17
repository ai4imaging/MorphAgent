def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    
    # Define epsilon to prevent division by zero
    epsilon = 1e-7

    # Convert image to float64 to prevent overflow during summation
    # The input is uint8, so summing 512*512 pixels can easily exceed 2^32
    arr = np.asarray(img, dtype=np.float64)

    # Handle dimensionality according to dataset format
    # Expected shape: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        # If the shape is unexpected (e.g., grayscale or wrong channels), return 0.0
        return 0.0

    # Extract specific channels based on dataset description
    # Channel 0: Actin (Cytoskeleton)
    # Channel 2: DAPI (Nucleus)
    actin_channel = arr[:, :, 0]
    dapi_channel = arr[:, :, 2]

    # Handle segmentation masks if available
    # If a mask is provided, we use it to restrict the calculation to cellular regions
    # This reduces background noise contribution to the ratio
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        mask = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions
        if mask.shape == actin_channel.shape:
            # Create a binary mask (foreground > 0)
            binary_mask = mask > 0
            
            # Apply mask to channels
            actin_channel = actin_channel * binary_mask
            dapi_channel = dapi_channel * binary_mask

    # Compute Total Integrated Intensity for each channel
    total_actin_intensity = np.sum(actin_channel)
    total_dapi_intensity = np.sum(dapi_channel)

    # Calculate Ratio: Total Actin / Total DAPI
    # This metric indicates the relative abundance of cytoskeletal structure vs DNA content
    # High values suggest hypertrophy or cell spreading; low values suggest atrophy or rounding
    ratio = total_actin_intensity / (total_dapi_intensity + epsilon)

    return float(ratio)
