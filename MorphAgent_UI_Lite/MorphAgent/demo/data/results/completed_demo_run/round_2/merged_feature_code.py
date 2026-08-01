import numpy as np
from scipy import ndimage
from skimage.measure import regionprops, label
from skimage.morphology import skeletonize
from skimage.filters import threshold_otsu

def extract_all(img, seg):
    results = {}

    # Feature 1: tau_axon_continuity_index
    try:
        # Ensure `img` is a 2D grayscale image of proper type and range
        arr = np.asarray(img, dtype=np.float32)
        if arr.ndim != 2 or arr.dtype != np.float32:
            raise ValueError("Invalid image format for tau_axon_continuity_index")

        # Normalize the intensity values to [0, 1]
        arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

        # Process axon mask from segmentation dictionary
        axon_mask = seg.get("mask_bundle")
        if axon_mask is None or axon_mask.ndim != 2 or axon_mask.shape != arr.shape:
            raise ValueError("Invalid mask for tau_axon_continuity_index")

        axon_mask = axon_mask.astype(bool)
        axon_skeleton = skeletonize(axon_mask)

        # Detect gaps in axonal regions
        complemented_mask = np.logical_not(axon_mask)
        gaps = np.logical_and(complemented_mask, axon_skeleton)
        labeled_gaps = label(gaps)
        gap_properties = regionprops(labeled_gaps)
        total_gap_area = sum(region.area for region in gap_properties)
        num_gaps = len(gap_properties)

        # Calculate axon area
        axon_area = np.sum(axon_mask)

        if axon_area == 0:
            raise ValueError("Axonal area is zero for tau_axon_continuity_index")

        # Calculate continuity index
        tau_axon_continuity_index = (axon_area - total_gap_area) / axon_area
        results["tau_axon_continuity_index"] = float(tau_axon_continuity_index)
    except Exception as e:
        results["tau_axon_continuity_index"] = 0.0

    # Feature 2: soma_tau_intensity_homogeneity
    try:
        # Convert image to float32 for processing
        arr = np.asarray(img, dtype=np.float32)
        if arr.ndim != 2 or arr.shape != (2048, 2048):
            raise ValueError("Invalid image format for soma_tau_intensity_homogeneity")

        # Process cell mask from segmentation dictionary
        cell_mask = seg.get("mask_cell")
        if cell_mask is None or cell_mask.shape != arr.shape or np.sum(cell_mask) == 0:
            raise ValueError("Invalid mask for soma_tau_intensity_homogeneity")

        # Apply mask to extract Tau intensities within soma
        masked_pixels = arr[cell_mask == 1]
        if masked_pixels.size == 0:
            raise ValueError("No masked pixels for soma_tau_intensity_homogeneity")

        # Compute mean and standard deviation of Tau intensity in soma
        mean_intensity = np.mean(masked_pixels)
        std_intensity = np.std(masked_pixels)

        if mean_intensity < 1e-6:
            raise ValueError("Mean intensity too small in soma_tau_intensity_homogeneity")

        # Calculate coefficient of variation (CV)
        cv = std_intensity / mean_intensity
        results["soma_tau_intensity_homogeneity"] = float(cv)
    except Exception as e:
        results["soma_tau_intensity_homogeneity"] = 0.0

    # Feature 3: dendrite_tau_bead_count
    try:
        # Convert image to float32 for processing
        arr = np.asarray(img, dtype=np.float32)
        if arr.ndim != 2 or arr.shape not in [(1536, 1536), (2048, 2048)]:
            raise ValueError("Invalid image format for dendrite_tau_bead_count")

        # Process dendrite mask from segmentation dictionary
        dendrite_mask = seg.get("mask_filament")
        if dendrite_mask is None:
            raise ValueError("No dendrite mask available for dendrite_tau_bead_count")

        dendrite_mask = np.asarray(dendrite_mask, dtype=np.uint8)
        if dendrite_mask.shape != arr.shape or dendrite_mask.max() > 1:
            raise ValueError("Invalid mask for dendrite_tau_bead_count")

        # Isolate regions in dendritic structures
        labeled_regions = label(dendrite_mask)
        tau_positive_bead_count = 0

        for region in regionprops(labeled_regions, intensity_image=arr):
            # Extract intensity mask
            region_mask = region.mean_intensity
            region_intensity_threshold = threshold_otsu(arr)
            bead_count = np.sum(region.intensity_image > region_intensity_threshold)
            tau_positive_bead_count += bead_count

        results["dendrite_tau_bead_count"] = float(tau_positive_bead_count)
    except Exception as e:
        results["dendrite_tau_bead_count"] = 0.0

    # Return all results
    return results