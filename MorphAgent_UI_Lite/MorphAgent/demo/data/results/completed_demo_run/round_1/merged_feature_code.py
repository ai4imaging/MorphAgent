import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize, remove_small_objects
from skimage.filters import threshold_otsu
from skimage.feature import peak_local_max


def extract_all(img, seg):
    results = {}

    try:
        arr = np.asarray(img)
        arr = np.squeeze(arr)

        if arr.ndim == 2:
            img2d = arr.astype(np.float64, copy=False)
        elif arr.ndim == 3:
            if arr.shape[-1] in (3, 4):
                img2d = np.mean(arr[..., :3].astype(np.float64, copy=False), axis=-1)
            elif arr.shape[0] in (3, 4):
                img2d = np.mean(arr[:3, ...].astype(np.float64, copy=False), axis=0)
            else:
                img2d = np.max(arr.astype(np.float64, copy=False), axis=0)
        else:
            img2d = None

        feature_value = 0.0

        if img2d is not None and img2d.ndim == 2 and img2d.size != 0 and seg and isinstance(seg, dict):
            def _mask_bool(mask):
                if mask is None:
                    return None
                m = np.asarray(mask)
                m = np.squeeze(m)
                if m.ndim == 3:
                    if m.shape[-1] in (3, 4):
                        m = np.max(m[..., :3], axis=-1)
                    elif m.shape[0] in (3, 4):
                        m = np.max(m[:3, ...], axis=0)
                    else:
                        m = np.max(m, axis=0)
                if m.shape != img2d.shape:
                    return None
                return np.isfinite(m) & (m > 0)

            cell = _mask_bool(seg.get("mask_cell"))
            if cell is not None and int(np.count_nonzero(cell)) != 0:
                nucleus = _mask_bool(seg.get("mask_nucleus"))
                bundle = _mask_bool(seg.get("mask_bundle"))
                filament = _mask_bool(seg.get("mask_filament"))

                if nucleus is None:
                    nucleus = np.zeros_like(cell, dtype=bool)

                axon_raw = np.zeros_like(cell, dtype=bool)
                if bundle is not None:
                    axon_raw |= bundle
                if filament is not None:
                    axon_raw |= filament

                axon_raw &= ~nucleus
                min_pixels = 25

                valid_regions = True
                if np.count_nonzero(axon_raw) >= min_pixels:
                    axon_in_cell = axon_raw & cell
                    if np.count_nonzero(axon_in_cell) >= min_pixels:
                        axon_mask = axon_in_cell
                    else:
                        axon_mask = axon_raw
                    soma_mask = cell & (~nucleus) & (~axon_mask)
                    if np.count_nonzero(soma_mask) < min_pixels:
                        soma_mask = cell & (~nucleus)
                else:
                    cytoplasm = cell & (~nucleus)
                    if np.count_nonzero(cytoplasm) < 2 * min_pixels or np.count_nonzero(nucleus) == 0:
                        valid_regions = False
                        axon_mask = np.zeros_like(cell, dtype=bool)
                        soma_mask = np.zeros_like(cell, dtype=bool)
                    else:
                        dist_to_nucleus = ndimage.distance_transform_edt(~nucleus)
                        vals = dist_to_nucleus[cytoplasm]
                        vals = vals[np.isfinite(vals)]
                        if vals.size < 2 * min_pixels:
                            valid_regions = False
                            axon_mask = np.zeros_like(cell, dtype=bool)
                            soma_mask = np.zeros_like(cell, dtype=bool)
                        else:
                            thresh = np.percentile(vals, 35.0)
                            soma_mask = cytoplasm & (dist_to_nucleus <= thresh)
                            axon_mask = cytoplasm & (dist_to_nucleus > thresh)

                if valid_regions and np.count_nonzero(axon_mask) >= min_pixels and np.count_nonzero(soma_mask) >= min_pixels:
                    finite_img = np.isfinite(img2d)
                    if np.any(finite_img):
                        outside_cell = (~cell) & finite_img
                        if np.count_nonzero(outside_cell) >= min_pixels:
                            background = np.percentile(img2d[outside_cell], 10.0)
                        else:
                            background = np.percentile(img2d[finite_img], 1.0)

                        corrected = img2d - background
                        corrected[~finite_img] = np.nan
                        corrected = np.maximum(corrected, 0.0)

                        axon_vals = corrected[axon_mask & finite_img]
                        soma_vals = corrected[soma_mask & finite_img]

                        if axon_vals.size >= min_pixels and soma_vals.size >= min_pixels:
                            axon_mean = np.nanmean(axon_vals)
                            soma_mean = np.nanmean(soma_vals)

                            if np.isfinite(axon_mean) and np.isfinite(soma_mean) and soma_mean > 0:
                                result = axon_mean / soma_mean
                                if np.isfinite(result):
                                    feature_value = float(result)

        results["axon_to_soma_tau_intensity_ratio"] = float(feature_value)
    except Exception:
        results["axon_to_soma_tau_intensity_ratio"] = 0.0

    try:
        arr = np.asarray(img, dtype=np.float32)
        arr = np.squeeze(arr)
        feature_value = float(0.0)

        if arr.ndim == 2 and arr.size != 0:
            finite = np.isfinite(arr)
            if np.any(finite):
                max_finite = float(np.max(arr[finite]))
                arr = np.nan_to_num(arr, nan=0.0, posinf=max_finite, neginf=0.0).astype(np.float32, copy=False)

                shape = arr.shape

                def _as_bool_mask(mask):
                    if mask is None:
                        return None
                    m = np.asarray(mask)
                    m = np.squeeze(m)
                    if m.shape != shape:
                        return None
                    return m > 0

                seg_dict = seg if isinstance(seg, dict) else {}

                cell_mask = _as_bool_mask(seg_dict.get("mask_cell"))
                if cell_mask is None or not np.any(cell_mask):
                    cell_mask = np.ones(shape, dtype=bool)

                nucleus_mask = _as_bool_mask(seg_dict.get("mask_nucleus"))
                if nucleus_mask is not None and np.any(nucleus_mask):
                    cytoplasm_mask = cell_mask & (~nucleus_mask)
                else:
                    cytoplasm_mask = cell_mask.copy()

                if np.any(cytoplasm_mask):
                    norm_values = arr[cytoplasm_mask]
                    if norm_values.size >= 2:
                        p_low = float(np.percentile(norm_values, 1.0))
                        p_high = float(np.percentile(norm_values, 99.5))
                        if np.isfinite(p_low) and np.isfinite(p_high) and p_high > p_low:
                            img_norm = np.clip((arr - p_low) / (p_high - p_low), 0.0, 1.0).astype(np.float32, copy=False)
                            img_smooth = ndimage.gaussian_filter(img_norm, sigma=1.0)

                            bundle_mask = _as_bool_mask(seg_dict.get("mask_bundle"))
                            filament_mask = _as_bool_mask(seg_dict.get("mask_filament"))

                            neurite_mask = np.zeros(shape, dtype=bool)
                            if bundle_mask is not None:
                                neurite_mask |= bundle_mask
                            if filament_mask is not None:
                                neurite_mask |= filament_mask

                            neurite_mask &= cytoplasm_mask

                            if np.any(neurite_mask):
                                neurite_mask = ndimage.binary_closing(neurite_mask, structure=np.ones((3, 3), dtype=bool))
                                neurite_mask = remove_small_objects(neurite_mask.astype(bool), min_size=20)

                            valid_neurite = True
                            if np.count_nonzero(neurite_mask) < 20:
                                vals = img_smooth[cytoplasm_mask]
                                if vals.size < 10:
                                    valid_neurite = False
                                else:
                                    p75 = float(np.percentile(vals, 75.0))
                                    try:
                                        otsu_thr = float(threshold_otsu(vals))
                                    except Exception:
                                        otsu_thr = p75
                                    fallback_thr = max(otsu_thr, p75)
                                    neurite_mask = (img_smooth > fallback_thr) & cytoplasm_mask
                                    neurite_mask = remove_small_objects(neurite_mask.astype(bool), min_size=30)
                                    if np.count_nonzero(neurite_mask) < 20:
                                        valid_neurite = False

                            if valid_neurite:
                                skeleton = skeletonize(neurite_mask.astype(bool))
                                skeleton_length = int(np.count_nonzero(skeleton))
                                if skeleton_length >= 10:
                                    distance_to_skeleton = ndimage.distance_transform_edt(~skeleton)
                                    alignment_radius = 4.0
                                    near_skeleton = distance_to_skeleton <= alignment_radius
                                    peak_search_mask = cytoplasm_mask & near_skeleton

                                    if np.any(peak_search_mask):
                                        vals = img_smooth[cytoplasm_mask]
                                        med = float(np.median(vals))
                                        mad_sigma = float(1.4826 * np.median(np.abs(vals - med)))
                                        p90 = float(np.percentile(vals, 90.0))
                                        threshold_abs = max(p90, med + 2.0 * mad_sigma)

                                        if np.isfinite(threshold_abs):
                                            coords = peak_local_max(
                                                img_smooth,
                                                min_distance=3,
                                                threshold_abs=threshold_abs,
                                                exclude_border=False,
                                                labels=peak_search_mask.astype(np.uint8),
                                            )

                                            if coords is not None and len(coords) != 0:
                                                aligned_peak_count = int(np.count_nonzero(distance_to_skeleton[coords[:, 0], coords[:, 1]] <= alignment_radius))
                                                density = 100.0 * float(aligned_peak_count) / float(skeleton_length)

                                                if np.isfinite(density):
                                                    feature_value = float(density)

        results["tau_neurite_bead_density"] = float(feature_value)
    except Exception:
        results["tau_neurite_bead_density"] = 0.0

    try:
        arr = np.asarray(img)
        arr = np.squeeze(arr)
        feature_value = float(0.0)

        if arr.ndim == 2:
            arr = arr.astype(np.float32, copy=False)
            finite = np.isfinite(arr)
            if np.any(finite):
                finite_median = float(np.median(arr[finite]))
                arr = np.where(finite, arr, finite_median).astype(np.float32, copy=False)

                if seg and isinstance(seg, dict):
                    def _mask_from_seg(key):
                        m = seg.get(key)
                        if m is None:
                            return None
                        m = np.squeeze(np.asarray(m))
                        if m.shape != arr.shape:
                            return None
                        return m > 0

                    cell = _mask_from_seg("mask_cell")
                    if cell is not None and int(np.count_nonzero(cell)) >= 100:
                        nucleus = _mask_from_seg("mask_nucleus")
                        if nucleus is not None:
                            nucleus = nucleus & cell
                            roi = cell & (~nucleus)
                        else:
                            roi = cell.copy()

                        if int(np.count_nonzero(roi)) >= 100:
                            outside_cell = (~cell) & np.isfinite(arr)
                            if np.count_nonzero(outside_cell) >= 100:
                                background = float(np.median(arr[outside_cell]))
                            else:
                                background = float(np.percentile(arr[np.isfinite(arr)], 1.0))

                            img_corr = arr - background
                            img_corr[img_corr < 0] = 0.0

                            if nucleus is not None and np.count_nonzero(nucleus) > 0:
                                dist = ndimage.distance_transform_edt(~nucleus)
                            else:
                                ys, xs = np.nonzero(cell)
                                if ys.size > 0:
                                    cy = float(np.mean(ys))
                                    cx = float(np.mean(xs))
                                    yy, xx = np.indices(arr.shape)
                                    dist = np.sqrt((yy.astype(np.float32) - cy) ** 2 + (xx.astype(np.float32) - cx) ** 2)
                                else:
                                    dist = None

                            if dist is not None:
                                roi_dist = dist[roi]
                                if roi_dist.size >= 100 and np.all(np.isfinite(roi_dist)):
                                    q33, q67 = np.percentile(roi_dist, [33.333333, 66.666667])
                                    if np.isfinite(q33) and np.isfinite(q67) and q67 > q33:
                                        proximal = roi & (dist <= q33)
                                        distal = roi & (dist >= q67)

                                        min_pixels = max(25, int(0.01 * np.count_nonzero(roi)))
                                        if np.count_nonzero(proximal) >= min_pixels and np.count_nonzero(distal) >= min_pixels:
                                            proximal_values = img_corr[proximal]
                                            distal_values = img_corr[distal]

                                            if proximal_values.size != 0 and distal_values.size != 0:
                                                proximal_mean = float(np.mean(proximal_values))
                                                distal_mean = float(np.mean(distal_values))

                                                eps = 1e-6
                                                if np.isfinite(proximal_mean) and np.isfinite(distal_mean) and proximal_mean > eps:
                                                    result = distal_mean / proximal_mean
                                                    if np.isfinite(result):
                                                        feature_value = float(result)

        results["distal_to_proximal_axon_tau_intensity_ratio"] = float(feature_value)
    except Exception:
        results["distal_to_proximal_axon_tau_intensity_ratio"] = 0.0

    return results