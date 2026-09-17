def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_dilation, disk

    # 1. Input Validation and Conversion
    # Convert to float32 for calculations
    arr = np.asarray(img, dtype=np.float32)

    # Check dimensions: Expected (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # 2. Channel Mapping
    # Channel 0: Red (Actin) - The signal to measure
    # Channel 1: Green (Tubulin) - Useful for cytoplasmic segmentation (as per feedback)
    # Channel 2: Blue (DAPI) - Nucleus
    actin_channel = arr[..., 0]
    tubulin_channel = arr[..., 1]
    dapi_channel = arr[..., 2]

    # 3. Normalization (per channel)
    # Normalize actin channel for intensity measurements
    p99_actin = np.percentile(actin_channel, 99.5)
    if p99_actin > 0:
        actin_norm = actin_channel / p99_actin
    else:
        actin_norm = actin_channel

    # Normalize structural channels for segmentation
    p99_dapi = np.percentile(dapi_channel, 99.5)
    dapi_norm = dapi_channel / p99_dapi if p99_dapi > 0 else dapi_channel
    
    p99_tubulin = np.percentile(tubulin_channel, 99.5)
    tubulin_norm = tubulin_channel / p99_tubulin if p99_tubulin > 0 else tubulin_channel

    # 4. Segmentation Logic
    # We need a Nuclear Mask and a Cytoplasmic Mask.
    
    # Strategy:
    # A. Use provided masks if available.
    # B. If not, generate masks using Otsu thresholding.
    #    - Nuclei from DAPI (Channel 2)
    #    - Cell body from Tubulin (Channel 1) to avoid circular dependency with Actin (Channel 0)
    
    nuclear_mask = None
    cytoplasmic_mask = None

    if len(segmentation_masks) >= 2:
        # Assuming standard order often seen: cell_mask, nuclei_mask or vice versa.
        # Without explicit metadata on mask order, we often have to guess or rely on size.
        # Usually, nuclei are smaller and contained within cells.
        mask1 = segmentation_masks[0]
        mask2 = segmentation_masks[1]
        
        # Simple heuristic: The mask with smaller total area is likely the nucleus
        area1 = np.sum(mask1 > 0)
        area2 = np.sum(mask2 > 0)
        
        if area1 < area2:
            nuclear_mask = (mask1 > 0)
            cell_mask = (mask2 > 0)
        else:
            nuclear_mask = (mask2 > 0)
            cell_mask = (mask1 > 0)
            
        # Define cytoplasm as Cell - Nucleus
        cytoplasmic_mask = np.logical_and(cell_mask, ~nuclear_mask)

    elif len(segmentation_masks) == 1:
        # If only one mask, assume it's nuclei if it's small/fragmented, or cell if large.
        # Fallback to computing the other.
        # For robustness, let's just re-compute from scratch to ensure consistency between N and C.
        pass

    # Fallback / Primary Segmentation (if masks not provided or insufficient)
    if nuclear_mask is None or cytoplasmic_mask is None:
        # 4.1 Segment Nuclei (DAPI)
        try:
            thresh_dapi = threshold_otsu(dapi_norm)
            nuclear_mask = dapi_norm > thresh_dapi
        except Exception:
            # Fallback if image is blank
            return 0.0

        # Fill holes in nuclei
        nuclear_mask = ndimage.binary_fill_holes(nuclear_mask)

        # 4.2 Segment Cell Body (Tubulin)
        # Using Tubulin (Channel 1) as suggested by feedback to avoid circularity with Actin
        try:
            thresh_tubulin = threshold_otsu(tubulin_norm)
            # Tubulin is often faint; sometimes a lower threshold or log scale helps, 
            # but standard Otsu is a safe baseline.
            # Combine with DAPI because cells contain nuclei
            cell_mask = (tubulin_norm > thresh_tubulin) | nuclear_mask
        except Exception:
            # If tubulin is empty, try to dilate nuclei to approximate cytoplasm
            cell_mask = binary_dilation(nuclear_mask, disk(15))

        # Fill holes in cell mask
        cell_mask = ndimage.binary_fill_holes(cell_mask)

        # 4.3 Define Cytoplasm
        # Cytoplasm = Cell Mask AND NOT Nuclear Mask
        # To be safe, we can erode the nuclear mask slightly before subtraction to ensure 
        # we don't catch the nuclear rim in the cytoplasm, and dilate the nuclear mask 
        # slightly for the "Nuclear Region" to ensure we are truly inside.
        # However, for "Nuclear Actin Exclusion", we want to compare the *bulk* nucleus vs *bulk* cytoplasm.
        
        # Strict Nuclear Mask (erode slightly to be safe inside nucleus)
        # nuclear_mask_strict = ndimage.binary_erosion(nuclear_mask, structure=np.ones((3,3)))
        
        # Strict Cytoplasmic Mask (Cell - Dilated Nucleus)
        # This ensures we don't sample the transition zone (nuclear envelope) as cytoplasm
        nuclear_dilated = binary_dilation(nuclear_mask, disk(3))
        cytoplasmic_mask = np.logical_and(cell_mask, ~nuclear_dilated)

    # 5. Feature Calculation
    # Measure Actin (Channel 0) intensity in both regions
    
    # Sanity check for empty masks
    if np.sum(nuclear_mask) == 0 or np.sum(cytoplasmic_mask) == 0:
        return 0.0

    mean_nuclear_actin = np.mean(actin_norm[nuclear_mask])
    mean_cytoplasmic_actin = np.mean(actin_norm[cytoplasmic_mask])

    # Avoid division by zero
    if mean_cytoplasmic_actin == 0:
        return 0.0

    # Ratio: Nucleus / Cytoplasm
    # Expected behavior: Low value (<< 1.0) for healthy cells (exclusion).
    # High value (> 1.0) suggests rupture or error.
    ratio = mean_nuclear_actin / mean_cytoplasmic_actin

    return float(ratio)
