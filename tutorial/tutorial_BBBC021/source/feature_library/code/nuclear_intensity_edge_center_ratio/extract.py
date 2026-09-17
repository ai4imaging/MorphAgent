def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.morphology import binary_erosion, disk
    from skimage.filters import threshold_otsu

    # Convert to appropriate array type
    # The input is (512, 512, 3) uint8. We need float for calculations.
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality and channel selection
    # Dataset info: (Height, Width, Channels) = (512, 512, 3)
    # Channel 2 is DAPI (Nucleus), which is the target for this feature.
    if arr.ndim == 3 and arr.shape[2] >= 3:
        dapi_channel = arr[:, :, 2]
    elif arr.ndim == 2:
        # Fallback if a single channel image is passed (unlikely based on spec but safe)
        dapi_channel = arr
    else:
        return 0.0

    # Normalize intensity to avoid overflow/underflow issues, though ratio is scale-invariant
    # Simple min-max normalization to 0-1 range
    dapi_min, dapi_max = dapi_channel.min(), dapi_channel.max()
    if dapi_max > dapi_min:
        dapi_channel = (dapi_channel - dapi_min) / (dapi_max - dapi_min)
    else:
        return 0.0 # Flat image, no features

    # Handle segmentation masks
    labeled_mask = None
    
    # Check if valid segmentation masks are provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        mask_input = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions
        if mask_input.shape[:2] == dapi_channel.shape[:2]:
            # If mask is not integer labeled (e.g. binary), label it
            if mask_input.dtype == bool or len(np.unique(mask_input)) <= 2:
                labeled_mask = label(mask_input)
            else:
                labeled_mask = mask_input.astype(int)
    
    # Fallback: Generate segmentation if none provided
    if labeled_mask is None:
        # Simple Otsu thresholding for nuclei
        try:
            thresh = threshold_otsu(dapi_channel)
            binary_mask = dapi_channel > thresh
            # Remove small noise
            binary_mask = ndimage.binary_opening(binary_mask, structure=np.ones((3,3)))
            labeled_mask = label(binary_mask)
        except Exception:
            return 0.0

    # Feature Calculation: Edge vs Center Intensity Ratio
    # We define "Edge" as a rim of pixels at the boundary, and "Center" as the interior.
    # Erosion radius determines the thickness of the edge.
    erosion_radius = 3
    selem = disk(erosion_radius)
    
    ratios = []
    
    # Iterate over each nucleus
    props = regionprops(labeled_mask, intensity_image=dapi_channel)
    
    for prop in props:
        # Extract the binary mask for this specific nucleus within its bounding box
        # prop.image is the binary mask of the object in the bounding box
        nucleus_mask = prop.image
        
        # Skip very small nuclei where erosion would remove everything
        if np.sum(nucleus_mask) < selem.sum(): 
            continue
            
        # Create the "Center" mask by eroding the full nucleus mask
        # This shrinks the mask. The remaining part is the center.
        # We pad the mask slightly to handle boundary erosion correctly within the bbox context if needed,
        # but regionprops bbox is tight. Binary erosion treats outside as 0.
        center_mask = binary_erosion(nucleus_mask, footprint=selem)
        
        # Create the "Edge" mask: pixels in original but NOT in center
        edge_mask = np.logical_and(nucleus_mask, ~center_mask)
        
        # Check if we have valid pixels for both regions
        if np.sum(center_mask) == 0 or np.sum(edge_mask) == 0:
            continue
            
        # Extract intensities
        # prop.intensity_image is the intensity image within the bounding box
        # We mask it with our local masks
        center_pixels = prop.intensity_image[center_mask]
        edge_pixels = prop.intensity_image[edge_mask]
        
        mean_center = np.mean(center_pixels)
        mean_edge = np.mean(edge_pixels)
        
        # Calculate ratio
        # Add epsilon to denominator to prevent division by zero
        if mean_center > 1e-6:
            ratio = mean_edge / mean_center
            ratios.append(ratio)
            
    # Aggregate results
    if not ratios:
        return 0.0
        
    # Return the mean ratio across all valid nuclei in the image
    result = np.mean(ratios)

    return float(result)
