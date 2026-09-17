def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.morphology import binary_erosion, disk
    from skimage.filters import threshold_otsu

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Expected shape: (512, 512, 3)
    # Channel 2 is DAPI (Nucleus)
    dapi_channel = None
    
    if arr.ndim == 3 and arr.shape[2] == 3:
        dapi_channel = arr[:, :, 2]
    elif arr.ndim == 2:
        # Fallback if single channel passed, assume it's the relevant one
        dapi_channel = arr
    else:
        return 0.0

    # Intensity normalization
    # Normalize to 0-1 range for consistent calculation, though ratio is scale-invariant
    vmax = np.percentile(dapi_channel, 99.5) if dapi_channel.size > 0 else 1.0
    if vmax > 0:
        dapi_channel = dapi_channel / vmax
    dapi_channel = np.clip(dapi_channel, 0.0, 1.0)

    # Handle segmentation masks
    labeled_mask = None
    
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use provided mask
        mask_input = segmentation_masks[0]
        # Ensure it matches image spatial dimensions
        if mask_input.shape[:2] == dapi_channel.shape:
            # If mask is boolean or binary, label it. If integer, assume it's instance labels.
            if mask_input.dtype == bool or len(np.unique(mask_input)) <= 2:
                labeled_mask = label(mask_input > 0)
            else:
                labeled_mask = mask_input.astype(int)
    
    # Fallback: Generate mask if none provided or invalid
    if labeled_mask is None:
        # Simple Otsu thresholding
        try:
            thresh = threshold_otsu(dapi_channel)
            binary_mask = dapi_channel > thresh
            # Remove small noise
            binary_mask = ndimage.binary_opening(binary_mask, structure=np.ones((3,3)))
            labeled_mask = label(binary_mask)
        except Exception:
            return 0.0

    # Feature Computation: Boundary vs Center Intensity Ratio
    # We define the boundary as the outer ring of the nucleus and the center as the eroded core.
    
    # Parameters
    erosion_radius = 4  # Pixels to erode to define the "core" vs "ring"
    min_area = 50       # Minimum area to consider a valid nucleus
    epsilon = 1e-6      # Avoid division by zero

    ratios = []
    
    # Iterate over each nucleus
    regions = regionprops(labeled_mask, intensity_image=dapi_channel)
    
    for region in regions:
        if region.area < min_area:
            continue
            
        # Get the binary mask of the single cell within its bounding box
        cell_mask = region.image
        
        # Get the intensity image within the bounding box
        cell_intensity = region.intensity_image
        
        # Define the structuring element for erosion
        selem = disk(erosion_radius)
        
        # If the cell is smaller than the structuring element, erosion will wipe it out.
        # We need to handle this gracefully.
        if cell_mask.shape[0] <= selem.shape[0] or cell_mask.shape[1] <= selem.shape[1]:
            continue

        # Create the "Center" mask by eroding the full cell mask
        # We pad the mask slightly to handle edge cases with binary_erosion if tight
        center_mask = binary_erosion(cell_mask, selem)
        
        # Create the "Boundary" (Ring) mask by subtracting center from full
        boundary_mask = np.logical_and(cell_mask, ~center_mask)
        
        # Check if we have valid pixels for both regions
        if np.sum(center_mask) == 0 or np.sum(boundary_mask) == 0:
            continue
            
        # Calculate mean intensities
        mean_center = np.mean(cell_intensity[center_mask])
        mean_boundary = np.mean(cell_intensity[boundary_mask])
        
        # Compute ratio: Boundary / Center
        # Ratio > 1 implies marginalization (nuclear cap)
        # Ratio < 1 implies central condensation
        ratio = mean_boundary / (mean_center + epsilon)
        ratios.append(ratio)

    # Aggregate results
    if not ratios:
        return 0.0
        
    # Return the mean ratio across all valid nuclei in the image
    result = np.mean(ratios)

    return float(result)
