def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    from skimage.measure import label, regionprops
    from skimage.morphology import binary_erosion, disk

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset: (512, 512, 3), Channel 0 = Actin, Channel 1 = Tubulin, Channel 2 = DAPI
    # We need Channel 0 (Actin) for this feature
    if arr.ndim == 3 and arr.shape[2] == 3:
        actin_img = arr[..., 0]
    elif arr.ndim == 2:
        # Fallback if single channel passed (unlikely based on desc, but safe)
        actin_img = arr
    else:
        return 0.0

    # Intensity normalization
    # Normalize to 0-1 range for consistent thresholding if needed, though ratios are scale-invariant
    vmax = np.percentile(actin_img, 99.5) if actin_img.size > 0 else 1.0
    if vmax > 0:
        actin_img = actin_img / vmax
    actin_img = np.clip(actin_img, 0.0, 1.0)

    # Handle segmentation masks
    # We need a labeled cell mask.
    labeled_mask = None
    
    # Check if valid segmentation masks are provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        mask_candidate = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions
        if mask_candidate.shape[:2] == actin_img.shape[:2]:
            # If mask is not integer labeled (e.g. binary), label it
            if mask_candidate.max() <= 1 and mask_candidate.dtype != bool:
                 # It might be a binary mask 0/1 or 0/255
                 labeled_mask = label(mask_candidate > 0)
            else:
                 # Assume it's already labeled
                 labeled_mask = mask_candidate.astype(int)

    # Fallback: Generate segmentation from Actin channel if no mask provided
    if labeled_mask is None:
        try:
            # Simple background separation
            thresh = threshold_otsu(actin_img)
            binary_mask = actin_img > thresh
            # Clean up noise
            binary_mask = ndimage.binary_opening(binary_mask, structure=np.ones((3,3)))
            binary_mask = ndimage.binary_fill_holes(binary_mask)
            labeled_mask = label(binary_mask)
        except Exception:
            # If thresholding fails (e.g. empty image), return 0
            return 0.0

    # Feature Calculation: Actin Cortical Ratio
    # Ratio = Mean Intensity of Cortex / Mean Intensity of Center
    
    # Parameters
    erosion_radius = 4  # Pixels to erode to define "center" vs "cortex"
    min_area = 50       # Minimum area to consider a valid cell
    epsilon = 1e-6      # Avoid division by zero

    ratios = []
    
    # Get properties for all regions
    regions = regionprops(labeled_mask, intensity_image=actin_img)
    
    for props in regions:
        if props.area < min_area:
            continue
            
        # Extract the crop for the current cell to speed up processing
        # props.image is the binary mask of the cell in the bounding box
        # props.intensity_image is the intensity image in the bounding box
        cell_mask_crop = props.image
        cell_intensity_crop = props.intensity_image
        
        # Define Center: Erode the cell mask
        # We use a disk structuring element for isotropic erosion
        selem = disk(erosion_radius)
        
        # If the cell is smaller than the structuring element, erosion might make it empty
        # We check dimensions to avoid errors in binary_erosion if image is smaller than selem
        if cell_mask_crop.shape[0] <= selem.shape[0] or cell_mask_crop.shape[1] <= selem.shape[1]:
            # Cell is too small/thin to have a distinct center -> treat as all cortex or skip
            # Biologically, very thin cells are mostly cortex-like structures
            continue

        center_mask = binary_erosion(cell_mask_crop, footprint=selem)
        
        # Define Cortex: Cell Mask - Center Mask
        cortex_mask = np.logical_and(cell_mask_crop, ~center_mask)
        
        # Check if we have valid pixels for both regions
        if np.sum(center_mask) == 0 or np.sum(cortex_mask) == 0:
            continue
            
        # Calculate Mean Intensities
        mean_center = np.mean(cell_intensity_crop[center_mask])
        mean_cortex = np.mean(cell_intensity_crop[cortex_mask])
        
        # Compute Ratio
        ratio = mean_cortex / (mean_center + epsilon)
        ratios.append(ratio)

    # Aggregate results
    if not ratios:
        return 0.0
        
    result = np.mean(ratios)
    
    return float(result)
