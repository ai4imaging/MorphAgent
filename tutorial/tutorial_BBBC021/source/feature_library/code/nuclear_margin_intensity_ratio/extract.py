def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_erosion, disk

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        # If not the expected 3-channel image, return 0.0
        return 0.0

    # Extract DAPI channel (Channel 2 based on dataset description)
    # Channel 0: Actin (R), Channel 1: Tubulin (G), Channel 2: DAPI (B)
    dapi_channel = arr[:, :, 2]

    # Normalize intensity to prevent overflow/underflow issues during calculation, 
    # though ratio is scale-invariant, working in float is safer.
    # We don't strictly need 0-1 normalization for a ratio, but it's good practice.
    
    # Determine Segmentation Mask
    # We need a nuclear mask.
    # Strategy: 
    # 1. Check if segmentation_masks are provided.
    # 2. If provided, try to find the one corresponding to nuclei. 
    #    The prompt mentions masks like 'cyto.tif', 'nuclei.tif'. 
    #    Without explicit metadata mapping index to type, we usually assume the one 
    #    that overlaps best with DAPI intensity or simply the second one if standard.
    #    However, a robust fallback is to generate one if the provided ones are ambiguous or missing.
    
    nuclei_mask = None
    
    # Heuristic: If masks are provided, check if any looks like a nuclear mask.
    # Often nuclear masks are smaller and more numerous than cell masks.
    # For this specific implementation, to ensure we measure the *nuclear* margin specifically,
    # if a mask is provided, we use it. If multiple, we might default to the one with the most objects 
    # (often nuclei) or just the first one if only one exists.
    # Given the ambiguity, a robust approach for this specific feature (which relies heavily on accurate 
    # nuclear boundaries) is to calculate a fresh mask from the DAPI channel if the input masks are not clearly labeled.
    # However, the instructions say "Masks are automatically loaded...". 
    # Let's try to use the provided mask if available.
    
    if len(segmentation_masks) > 0:
        # Use the first available mask as the primary object mask
        # In many pipelines, the first mask is often the primary object or nuclei.
        # We assume the provided mask labels the objects of interest.
        input_mask = segmentation_masks[0]
        if input_mask.shape == dapi_channel.shape:
             nuclei_mask = input_mask > 0
    
    # Fallback: Generate mask if none provided or shape mismatch
    if nuclei_mask is None:
        # Gaussian blur to reduce noise
        blurred = ndimage.gaussian_filter(dapi_channel, sigma=2)
        try:
            thresh = threshold_otsu(blurred)
            nuclei_mask = blurred > thresh
        except Exception:
            # Fallback for very low contrast images
            return 0.0

    # Label the nuclei
    labeled_nuclei = label(nuclei_mask)
    regions = regionprops(labeled_nuclei, intensity_image=dapi_channel)

    if not regions:
        return 0.0

    ratios = []
    
    # Define erosion radius for margin calculation
    # For a 512x512 image of cells, a 3-pixel radius is a reasonable approximation for the nuclear envelope/rim.
    erosion_radius = 3
    selem = disk(erosion_radius)

    for region in regions:
        # Extract the bounding box image for efficiency
        # region.image is the binary mask of the object in the bounding box
        # region.intensity_image is the intensity image in the bounding box
        
        mask_full = region.image
        intensity_full = region.intensity_image
        
        # Skip very small nuclei where erosion would consume everything
        if np.sum(mask_full) < selem.sum():
            continue

        # Define Center (Inner Core) via erosion
        # We need to pad the mask slightly to handle boundary erosion correctly within the bbox context,
        # but region.image is tightly cropped. Binary erosion treats borders as 0 usually.
        mask_center = binary_erosion(mask_full, footprint=selem)
        
        # Define Margin (Rim)
        mask_margin = np.logical_and(mask_full, ~mask_center)
        
        # Check if we have valid pixels for both
        if np.sum(mask_center) == 0 or np.sum(mask_margin) == 0:
            continue
            
        # Calculate Mean Intensities
        mean_center = np.mean(intensity_full[mask_center])
        mean_margin = np.mean(intensity_full[mask_margin])
        
        # Calculate Ratio: Margin / Center
        # Add epsilon to denominator to avoid division by zero
        epsilon = 1e-6
        ratio = mean_margin / (mean_center + epsilon)
        
        ratios.append(ratio)

    if not ratios:
        return 0.0

    # Return the mean ratio across all valid nuclei in the image
    result = np.mean(ratios)
    
    return float(result)
