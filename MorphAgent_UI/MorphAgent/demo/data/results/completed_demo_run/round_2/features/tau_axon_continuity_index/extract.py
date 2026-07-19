def extract(img, seg=None):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import regionprops, label
    
    # Ensure `img` is a 2D grayscale image of proper type and range
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim != 2 or arr.dtype != np.float32:
        return 0.0
    
    # Normalize the intensity values to [0, 1]
    arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    
    # If segmentation masks are provided, calculate continuity index for axons
    if seg:
        axon_mask = seg.get("mask_bundle")  # Use the 'mask_bundle' key for axonal structures
        if axon_mask is None or axon_mask.ndim != 2 or axon_mask.shape != arr.shape:
            return 0.0
        
        axon_mask = axon_mask.astype(bool)  # Convert to boolean mask
        
        # Skeletonize the axon mask for continuity analysis
        from skimage.morphology import skeletonize
        axon_skeleton = skeletonize(axon_mask)
        
        # Detect gaps in the axonal regions (holes)
        complemented_mask = np.logical_not(axon_mask)  # Invert the axon mask
        gaps = np.logical_and(complemented_mask, axon_skeleton)
        
        # Calculate the number and size of gaps
        labeled_gaps = label(gaps)
        gap_properties = regionprops(labeled_gaps)
        total_gap_area = sum(region.area for region in gap_properties)
        num_gaps = len(gap_properties)
        
        # Calculate axon area
        axon_area = np.sum(axon_mask)
        
        # Handle case if the axonal area is zero
        if axon_area == 0:
            return 0.0
        
        # Continuity index: (axon total area - total gap area) / axon total area
        tau_axon_continuity_index = (axon_area - total_gap_area) / axon_area
        return float(tau_axon_continuity_index)
    
    # If no segmentation masks are provided, return 0.0 as the computation requires segmentation
    return 0.0
