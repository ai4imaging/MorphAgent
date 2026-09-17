def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu, threshold_triangle
    from skimage.segmentation import watershed, clear_border
    from skimage.morphology import binary_opening, binary_closing, disk

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset: (512, 512, 3), uint8
    # Channel 0: Actin (Red) - Target structure (Cytoskeleton)
    # Channel 1: Tubulin (Green)
    # Channel 2: DAPI (Blue) - Nuclei
    
    if arr.ndim != 3 or arr.shape[2] != 3:
        # Fallback for unexpected shapes, though dataset spec says (512, 512, 3)
        return 0.0

    # Extract relevant channels
    # Channel 0 is Actin (Cytoskeleton)
    actin_channel = arr[..., 0]
    # Channel 2 is DAPI (Nuclei) - needed for seeding individual cells
    nuclei_channel = arr[..., 2]

    # Intensity normalization [0, 1]
    # Robust max to handle potential hot pixels
    vmax_actin = np.percentile(actin_channel, 99.5) if actin_channel.size > 0 else 1.0
    if vmax_actin > 0:
        actin_channel = actin_channel / vmax_actin
    actin_channel = np.clip(actin_channel, 0.0, 1.0)

    vmax_nuc = np.percentile(nuclei_channel, 99.5) if nuclei_channel.size > 0 else 1.0
    if vmax_nuc > 0:
        nuclei_channel = nuclei_channel / vmax_nuc
    nuclei_channel = np.clip(nuclei_channel, 0.0, 1.0)

    # --- Segmentation Logic ---
    # To calculate the MEAN area per cell, we need to identify individual cells.
    # We use a marker-controlled watershed approach:
    # 1. Define the total cellular foreground using Actin.
    # 2. Define cell centers (markers) using Nuclei.
    # 3. Partition the Actin foreground using the markers.

    # 1. Define Actin Foreground (Cytoplasm Mask)
    # Smooth slightly to reduce noise in filaments
    actin_smooth = ndimage.gaussian_filter(actin_channel, sigma=2)
    
    # Use Triangle thresholding for Actin (often better for fluorescence tails)
    try:
        thresh_actin = threshold_triangle(actin_smooth)
    except Exception:
        thresh_actin = 0.1 # Fallback
        
    actin_mask = actin_smooth > thresh_actin
    
    # Clean up mask: remove small noise, fill holes
    actin_mask = binary_opening(actin_mask, footprint=disk(2))
    actin_mask = binary_closing(actin_mask, footprint=disk(3))

    # 2. Define Markers (Nuclei)
    # Check if segmentation masks are provided and usable
    labeled_cells = None
    
    # If we have external masks, we might use them, but the prompt asks for 
    # "actin_area_mean" specifically defined by the Actin channel. 
    # Often external masks are just nuclei or general cell masks. 
    # To be robust and self-contained for this specific feature definition, 
    # we will perform the segmentation internally unless a very specific mask is passed.
    # However, if a labeled mask is passed that looks like cell labels, we can use it 
    # to mask the actin channel directly.
    
    use_internal_segmentation = True
    if len(segmentation_masks) > 0:
        mask_candidate = segmentation_masks[0]
        # Check if it's a label mask (integer type) and matches shape
        if mask_candidate.shape == actin_channel.shape and np.issubdtype(mask_candidate.dtype, np.integer):
            # If provided mask is good, we intersect it with our actin threshold
            # to measure the actin area specifically within those cells
            labeled_cells = mask_candidate
            use_internal_segmentation = False

    if use_internal_segmentation:
        # Internal Watershed Segmentation
        
        # Threshold Nuclei for markers
        nuc_smooth = ndimage.gaussian_filter(nuclei_channel, sigma=2)
        try:
            thresh_nuc = threshold_otsu(nuc_smooth)
        except Exception:
            thresh_nuc = 0.1
            
        nuc_mask = nuc_smooth > thresh_nuc
        nuc_mask = binary_opening(nuc_mask, footprint=disk(2))
        
        # Label markers
        markers = label(nuc_mask)
        
        # Watershed
        # We use the inverse of actin intensity as the "elevation map" so watershed fills bright areas
        elevation_map = -actin_smooth
        labeled_cells = watershed(elevation_map, markers, mask=actin_mask)

    # 3. Feature Computation
    if labeled_cells is None or labeled_cells.max() == 0:
        return 0.0

    # Remove objects touching the border to avoid partial cells skewing the mean area
    labeled_cells_cleared = clear_border(labeled_cells)
    
    # If clear_border removes everything (e.g. zoomed in image), revert to original
    if labeled_cells_cleared.max() == 0 and labeled_cells.max() > 0:
        labeled_cells_cleared = labeled_cells

    # Calculate properties
    props = regionprops(labeled_cells_cleared)
    
    if not props:
        return 0.0

    # Extract Area
    # Area is the number of pixels in the region
    areas = [p.area for p in props]
    
    if not areas:
        return 0.0
        
    result = np.mean(areas)

    return float(result)
