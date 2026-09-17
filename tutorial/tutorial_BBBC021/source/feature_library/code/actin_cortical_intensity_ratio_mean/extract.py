def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.morphology import binary_erosion, disk
    from skimage.filters import threshold_otsu

    # Convert to appropriate array type and normalize
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality and extract Actin channel (Channel 0)
    # Dataset is (512, 512, 3), Channel 0 is Actin
    if arr.ndim == 3 and arr.shape[2] >= 1:
        actin_img = arr[:, :, 0]
    elif arr.ndim == 2:
        # Fallback if passed a single channel 2D image
        actin_img = arr
    else:
        return 0.0

    # Intensity normalization [0, 1]
    vmax = np.percentile(actin_img, 99.5) if actin_img.size > 0 else 1.0
    if vmax > 0:
        actin_img = actin_img / vmax
    actin_img = np.clip(actin_img, 0.0, 1.0)

    # Determine Segmentation Mask
    # We need a cell mask (cytoplasm/whole cell) to define the boundary.
    # If masks are provided, we try to find a suitable one.
    # If not, we generate one from the actin channel.
    
    cell_labels = None
    
    if len(segmentation_masks) > 0:
        # Heuristic: If multiple masks, pick the one with larger total area, 
        # assuming cells are larger than nuclei.
        best_mask = None
        max_area = -1
        
        for mask in segmentation_masks:
            if mask is None: continue
            # Ensure mask is 2D
            if mask.ndim == 3:
                # If 3D mask, take max projection or slice matching image
                mask_2d = np.max(mask, axis=0) if mask.shape[0] < mask.shape[1] else mask[:,:,0]
            else:
                mask_2d = mask
                
            # Check if it's a label mask or binary
            current_area = np.sum(mask_2d > 0)
            if current_area > max_area:
                max_area = current_area
                best_mask = mask_2d
        
        if best_mask is not None:
            # Ensure it's labeled
            if best_mask.max() <= 1:
                cell_labels = label(best_mask)
            else:
                cell_labels = best_mask.astype(int)

    # Fallback: Generate mask from Actin channel if no external mask provided
    if cell_labels is None:
        try:
            # Simple segmentation pipeline
            # 1. Smooth
            smooth = ndimage.gaussian_filter(actin_img, sigma=2)
            # 2. Threshold
            thresh = threshold_otsu(smooth)
            binary = smooth > thresh
            # 3. Clean up (fill holes, remove small objects)
            binary = ndimage.binary_fill_holes(binary)
            # 4. Label
            cell_labels = label(binary)
        except Exception:
            return 0.0

    # Feature Calculation: Cortical vs Internal Intensity Ratio
    # Cortical region: The outer boundary of the cell.
    # Internal region: The rest of the cell.
    
    # Parameters
    cortical_thickness = 4  # pixels
    min_cell_area = 100     # pixels
    
    ratios = []
    
    # Get properties for each cell
    props = regionprops(cell_labels, intensity_image=actin_img)
    
    for prop in props:
        if prop.area < min_cell_area:
            continue
            
        # Extract the cell's bounding box image and mask to speed up processing
        # (Working on the whole 512x512 array for every cell is slow)
        bbox_img = prop.image  # Binary mask of the cell within bbox
        bbox_intensity = prop.intensity_image # Intensity within bbox
        
        # Define Cortical Mask
        # Erode the cell mask to define the "inner" part
        # The cortex is the difference between the original mask and the eroded mask
        selem = disk(cortical_thickness)
        
        # Pad bbox slightly to handle edge effects of erosion if necessary, 
        # but regionprops bbox usually is tight. 
        # We perform erosion on the binary mask.
        eroded_mask = binary_erosion(bbox_img, footprint=selem)
        
        cortical_mask = np.logical_xor(bbox_img, eroded_mask)
        internal_mask = eroded_mask
        
        # Check if internal mask exists (cell might be too thin)
        if np.sum(internal_mask) == 0:
            continue
            
        # Calculate Mean Intensities
        # We use the intensity image provided by regionprops
        mean_cortical = np.mean(bbox_intensity[cortical_mask])
        mean_internal = np.mean(bbox_intensity[internal_mask])
        
        # Avoid division by zero
        if mean_internal == 0:
            ratio = mean_cortical / 1e-6 # Arbitrary large number
        else:
            ratio = mean_cortical / mean_internal
            
        ratios.append(ratio)

    # Aggregate results
    if not ratios:
        return 0.0
        
    result = np.mean(ratios)
    
    return float(result)
