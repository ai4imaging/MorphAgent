import numpy as np
from scipy import ndimage
from skimage.measure import label, regionprops

def extract_all(img, seg):
    results = {}

    # Feature 1: Axon to Soma Tau Intensity Ratio
    try:
        arr = np.asarray(img, dtype=np.float32)
        arr /= arr.max()

        cell_mask = seg.get("mask_cell")
        bundle_mask = seg.get("mask_bundle")

        if cell_mask is None or bundle_mask is None:
            raise ValueError("Missing cell_mask or bundle_mask")

        cell_mask = cell_mask.astype(bool)
        bundle_mask = bundle_mask.astype(bool)

        soma_intensity_mean = np.mean(arr[cell_mask]) if np.any(cell_mask) else 0.0
        axon_intensity_mean = np.mean(arr[bundle_mask]) if np.any(bundle_mask) else 0.0

        if soma_intensity_mean == 0.0:
            axon_to_soma_ratio = 0.0
        else:
            axon_to_soma_ratio = axon_intensity_mean / soma_intensity_mean

        results["axon_to_soma_tau_intensity_ratio"] = float(axon_to_soma_ratio)
    except Exception as e:
        results["axon_to_soma_tau_intensity_ratio"] = float('nan')

    # Feature 2: Tau Neurite Bead Density
    try:
        arr = np.asarray(img, dtype=np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min())

        cell_mask = seg.get("mask_cell")
        filament_mask = seg.get("mask_filament")

        if cell_mask is None or filament_mask is None:
            raise ValueError("Missing cell_mask or filament_mask")
        if cell_mask.shape != arr.shape or filament_mask.shape != arr.shape:
            raise ValueError("Mask dimensions do not match image dimensions")

        cell_area = arr * cell_mask
        filament_area = cell_area * filament_mask

        threshold_value = np.percentile(filament_area[filament_area > 0], 95)
        bead_mask = filament_area > threshold_value

        labeled_beads = label(bead_mask)
        bead_regions = regionprops(labeled_beads)

        bead_count = len(bead_regions)
        filament_area_size = np.sum(filament_mask)

        if filament_area_size == 0:
            bead_density = 0.0
        else:
            bead_density = bead_count / filament_area_size

        results["tau_neurite_bead_density"] = float(bead_density)
    except Exception as e:
        results["tau_neurite_bead_density"] = float('nan')

    # Feature 3: Distal to Proximal Axon Tau Intensity Ratio
    try:
        img = np.asarray(img, dtype=np.float32)
        img = img / 9042.0

        proximal_mask = seg.get("mask_bundle")
        distal_mask = seg.get("mask_filament")

        if proximal_mask is None or distal_mask is None:
            raise ValueError("Missing proximal_mask or distal_mask")
        if proximal_mask.shape != img.shape or distal_mask.shape != img.shape:
            raise ValueError("Mask dimensions do not match image dimensions")

        proximal_intensity = np.mean(img[proximal_mask > 0])
        distal_intensity = np.mean(img[distal_mask > 0])

        if proximal_intensity == 0:
            distal_to_proximal_ratio = 0.0
        else:
            distal_to_proximal_ratio = distal_intensity / proximal_intensity

        results["distal_to_proximal_axon_tau_intensity_ratio"] = float(distal_to_proximal_ratio)
    except Exception as e:
        results["distal_to_proximal_axon_tau_intensity_ratio"] = float('nan')

    return results