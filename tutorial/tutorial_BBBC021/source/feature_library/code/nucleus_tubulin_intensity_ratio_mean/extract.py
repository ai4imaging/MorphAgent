def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.morphology import binary_dilation, disk
    
    # Convert to appropriate array type
    # Image is expected to be (512, 512, 3) uint8 based on dataset description
    arr = np.asarray(img)
    
    # Handle dimensionality
    # Return 0.0 if dimensions are not as expected (H, W, C)
    if arr.ndim != 3:
        return 0.0
    # Need at least up to Channel 1 (Tubulin)
    if arr.shape[2] < 2: 
        return 0.0
        
    # Extract Tubulin channel (Channel 1: Green)
    # Normalize to float for calculation to prevent overflow and allow division
    tubulin_img = arr[:, :, 1].astype(np.float32)
    
    # Handle segmentation masks
    # If no masks are provided, we cannot compute this spatial feature reliably
    if not segmentation_masks:
        return 0.0
        
    # Assume first mask is nuclei (standard convention)
    nuclei_mask = segmentation_masks[0]
    
    # Check if a second mask (cells/cytoplasm) is available
    cell_mask = None
    if len(segmentation_masks) > 1:
        cell_mask = segmentation_masks[1]
        
    # Ensure masks are integer labels
    nuclei_mask = nuclei_mask.astype(np.int32)
    if cell_mask is not None:
        cell_mask = cell_mask.astype(np.int32)
        
    # Get unique cell labels (excluding background 0)
    unique_labels = np.unique(nuclei_mask)
    unique_labels = unique_labels[unique_labels > 0]
    
    if len(unique_labels) == 0:
        return 0.0
        
    ratios = []
    
    # Define structuring element for fallback dilation (approx 5px radius ring)
    selem = disk(5)
    
    for label in unique_labels:
        # 1. Define Nucleus Mask for this cell
        n_mask = (nuclei_mask == label)
        
        # 2. Define Cytoplasm Mask for this cell
        c_mask = None
        
        if cell_mask is not None:
            # Try to find corresponding cell in cell_mask
            # We look for the most frequent label in cell_mask that overlaps with n_mask
            overlap_vals = cell_mask[n_mask]
            overlap_vals = overlap_vals[overlap_vals > 0] # Exclude background
            
            if len(overlap_vals) > 0:
                # Find mode (most common label)
                vals, counts = np.unique(overlap_vals, return_counts=True)
                matched_cell_label = vals[np.argmax(counts)]
                
                # Cytoplasm = (Cell == matched_label) AND NOT (Nucleus == label)
                # We use the specific nucleus mask to exclude the nucleus area
                c_mask = (cell_mask == matched_cell_label) & (~n_mask)
        
        # Fallback: If no cell mask or matching failed, use dilation
        if c_mask is None or np.sum(c_mask) == 0:
            # Dilate nucleus to approximate cytoplasm (perinuclear region)
            dilated_n = binary_dilation(n_mask, footprint=selem)
            c_mask = dilated_n & (~n_mask)
            
        # 3. Compute Intensities
        # Ensure we have valid pixels in both regions
        if np.sum(n_mask) == 0 or np.sum(c_mask) == 0:
            continue
            
        mean_nuc = np.mean(tubulin_img[n_mask])
        mean_cyto = np.mean(tubulin_img[c_mask])
        
        # 4. Compute Ratio
        # Add epsilon to denominator to avoid division by zero
        # If mean_cyto is 0, the ratio is effectively determined by the numerator
        ratio = mean_nuc / (mean_cyto + 1e-6)
        ratios.append(ratio)
        
    # Aggregate: Compute mean of ratios across all valid cells
    if not ratios:
        return 0.0
        
    result = np.mean(ratios)
    
    return float(result)
