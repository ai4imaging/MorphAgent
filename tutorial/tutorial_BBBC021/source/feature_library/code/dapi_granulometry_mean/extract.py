def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.morphology import opening, disk
    from skimage.filters import threshold_otsu
    
    # Convert to appropriate array type
    # The image is uint8, but for summation operations we need higher precision
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3)
    # Channel 2 is DAPI (Blue), which is the target for chromatin analysis
    if arr.ndim == 3 and arr.shape[2] >= 3:
        dapi_channel = arr[:, :, 2]
    elif arr.ndim == 2:
        # Fallback if passed a single channel image, assume it's the relevant one
        dapi_channel = arr
    else:
        return 0.0

    # Determine Region of Interest (ROI) - The Nuclei
    # Granulometry should only be performed on the nuclei to avoid background noise
    mask = None
    
    # Check if segmentation masks are provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Assume the first mask corresponds to nuclei or cells
        # Masks are typically labeled integers (0=bg, 1..N=cells)
        input_mask = segmentation_masks[0]
        
        # Ensure mask matches image dimensions (handle potential 2D vs 3D mismatch)
        if input_mask.shape == dapi_channel.shape:
            mask = input_mask > 0
        elif input_mask.ndim == 3 and input_mask.shape[:2] == dapi_channel.shape:
             # If mask is 3D (e.g. one-hot or RGB), flatten it
            mask = np.max(input_mask, axis=2) > 0
            
    # Fallback: Generate mask if none provided or dimensions mismatch
    if mask is None:
        # Simple background exclusion using Otsu
        try:
            # Check if image is not empty/constant
            if np.min(dapi_channel) == np.max(dapi_channel):
                return 0.0
            thresh = threshold_otsu(dapi_channel)
            mask = dapi_channel > thresh
        except Exception:
            # Fallback for extremely low contrast or empty images
            return 0.0

    # Apply mask to DAPI channel
    # We zero out the background so it doesn't contribute to the volume sums
    masked_dapi = dapi_channel * mask

    # Granulometry Parameters
    # We are looking for chromatin speckles, which are fine textures.
    # We use a small range of radii for the structuring element.
    # Radius 1 to 5 covers diameters 3 to 11 pixels.
    radii = range(1, 6)
    
    # Initial Volume (Sum of intensities)
    # This is the volume of the "topographic surface" of the image
    prev_volume = np.sum(masked_dapi)
    
    if prev_volume == 0:
        return 0.0

    weighted_sum_radii = 0.0
    total_volume_loss = 0.0

    # Iterative Opening (Sieving)
    for r in radii:
        # Create structuring element (disk)
        selem = disk(r)
        
        # Perform morphological opening
        # Opening removes bright features smaller than the structuring element
        opened_img = opening(masked_dapi, selem)
        
        # Calculate new volume
        current_volume = np.sum(opened_img)
        
        # Calculate volume loss (the "mass" of features of size 'r')
        # Loss = (Volume at r-1) - (Volume at r)
        loss = prev_volume - current_volume
        
        # Accumulate statistics
        # We weight the radius by the amount of intensity lost at that step
        if loss > 0:
            weighted_sum_radii += loss * r
            total_volume_loss += loss
        
        # Update previous volume for next iteration
        prev_volume = current_volume

    # Compute Mean Granulometric Size
    if total_volume_loss > 0:
        result = weighted_sum_radii / total_volume_loss
    else:
        # If no texture was removed (image is very smooth or flat), return 0
        result = 0.0

    return float(result)
