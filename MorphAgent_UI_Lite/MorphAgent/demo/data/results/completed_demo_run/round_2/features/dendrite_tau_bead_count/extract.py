def extract(img, seg):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    
    # Convert image to float32 for processing
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim != 2 or arr.shape not in [(1536, 1536), (2048, 2048)]:
        return 0.0  # Unexpected format
    
    # Validate segmentation dictionary
    if not seg:
        return 0.0
    dendrite_mask = seg.get("mask_filament")
    if dendrite_mask is None:
        return 0.0  # No dendrite mask available
    
    # Ensure dendrite mask is binary with valid dimensions
    dendrite_mask = np.asarray(dendrite_mask, dtype=np.uint8)
    if dendrite_mask.shape != arr.shape or dendrite_mask.max() > 1:
        return 0.0  # Invalid mask
    
    # Isolate regions in dendritic structures
    labeled_regions = label(dendrite_mask)  # Label connected components
    tau_positive_bead_count = 0
    
    for region in regionprops(labeled_regions, intensity_image=arr):
        # Extract region intensity mask
        region_mask = region.mean_intensity  # Mean intensity within the region
        
        # Threshold Tau-positive beads within dendritic regions
        region_intensity_threshold = threshold_otsu(arr)  # Otsu threshold across the image
        bead_count = np.sum(region.intensity_image > region_intensity_threshold)  # Count bright spots/beads
        
        # Accumulate count
        tau_positive_bead_count += bead_count
    
    return float(tau_positive_bead_count)
