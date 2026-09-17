def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.filters import threshold_otsu
    
    # Convert to appropriate array type and normalize to [0, 1]
    # Dataset is uint8 (0-255), shape (512, 512, 3)
    # Channel 0: Actin (Red)
    # Channel 1: Tubulin (Green)
    # Channel 2: DAPI (Blue)
    arr = np.asarray(img, dtype=np.float32)
    arr = arr / 255.0
    
    # Extract the Actin channel (Channel 0)
    if arr.ndim == 3 and arr.shape[2] >= 1:
        actin_channel = arr[..., 0]
    else:
        # Fallback for unexpected shapes, though dataset spec says (512, 512, 3)
        return 0.0

    # Define the Cytoplasm Mask
    cytoplasm_mask = None

    # Strategy:
    # 1. If segmentation masks are provided, use them.
    #    We assume standard ordering if multiple are present: Cell (0) and Nucleus (1).
    #    If only one is present, we might assume it's the cell and try to subtract a computed nucleus.
    # 2. If no masks are provided, compute them on the fly using Otsu thresholding on Actin and DAPI channels.

    if len(segmentation_masks) >= 2:
        # Best case: We have at least two masks.
        # Usually, masks are passed as (cell_mask, nucleus_mask, ...) or similar.
        # We assume the first is the whole cell and the second is the nucleus.
        # Masks are typically labeled integers. We convert to boolean (foreground > 0).
        cell_mask = segmentation_masks[0] > 0
        nucleus_mask = segmentation_masks[1] > 0
        
        # Ensure shapes match (handle potential broadcasting or mismatch issues)
        if cell_mask.shape == actin_channel.shape and nucleus_mask.shape == actin_channel.shape:
            cytoplasm_mask = np.logical_and(cell_mask, np.logical_not(nucleus_mask))
    
    # Fallback or if masks were insufficient/mismatched
    if cytoplasm_mask is None:
        # Generate masks on the fly
        
        # 1. Generate Nucleus Mask from DAPI (Channel 2)
        if arr.shape[2] > 2:
            dapi_channel = arr[..., 2]
            try:
                thresh_nuc = threshold_otsu(dapi_channel)
                # Add a small epsilon or check if image is not empty to avoid errors
                nucleus_mask = dapi_channel > thresh_nuc
            except Exception:
                # If image is uniform (e.g. all black), Otsu fails
                nucleus_mask = np.zeros_like(dapi_channel, dtype=bool)
        else:
            nucleus_mask = np.zeros_like(actin_channel, dtype=bool)

        # 2. Generate Cell Mask from Actin (Channel 0)
        # Actin usually provides a good outline of the cell body
        try:
            thresh_cell = threshold_otsu(actin_channel)
            cell_mask = actin_channel > thresh_cell
        except Exception:
            cell_mask = np.zeros_like(actin_channel, dtype=bool)
            
        # 3. Define Cytoplasm
        cytoplasm_mask = np.logical_and(cell_mask, np.logical_not(nucleus_mask))

    # Compute Feature: Standard Deviation of Actin Intensity in Cytoplasm
    
    # Check if we have any cytoplasmic pixels
    if np.sum(cytoplasm_mask) == 0:
        return 0.0
    
    # Extract pixels belonging to the cytoplasm
    cytoplasmic_pixels = actin_channel[cytoplasm_mask]
    
    # Calculate standard deviation
    # ddof=1 is often used for sample standard deviation, but default (ddof=0) is fine for image descriptors
    intensity_std = np.std(cytoplasmic_pixels)
    
    return float(intensity_std)
