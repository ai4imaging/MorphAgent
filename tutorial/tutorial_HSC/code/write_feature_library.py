"""Write per-feature extract.py files, the catalog CSV, and the 25-name list."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIB = ROOT / "source" / "feature_library"
CODE = LIB / "code"

FEATURES = [
    ("mitochondria_background_signal_heterogeneity", "code", "intensity",
     "Measures the standard deviation of intensities in the background region (pixels not covered by any mitochondrial label) after masking out the cell area if available, yielding a scalar that reflects residual background noise and possible diffuse mitochondrial signal."),
    ("mito_perimeter_intensity_gradient_mean", "code", "intensity",
     "For each mitochondrial instance, samples intensities just inside and just outside its perimeter and computes the average inward-minus-outward intensity difference, then averages across objects weighted by perimeter length."),
    ("background_mito_signal_to_noise", "code", "intensity",
     "Estimates a signal-to-noise ratio by computing the mean mitochondrial intensity inside the cell mask and dividing by the standard deviation of intensities in a surrounding background ring."),
    ("mitochondria_neighbor_intensity_autocorrelation", "code", "spatial",
     "Computes a local Moran's I spatial autocorrelation statistic on mitochondrial intensities within the cell mask using 8-connected pixel neighborhoods."),
    ("mitochondria_texture_anisotropy_glcm", "code", "texture",
     "Captures directional organization of mitochondrial texture by computing GLCM contrast at 0/45/90/135 degrees and returning the coefficient of variation across angles."),
    ("mitochondria_long_tubule_fraction", "code", "morphology",
     "Calculates the fraction of mitochondrial objects whose major-axis length exceeds a high-length threshold, yielding the proportion of the population forming long tubules rather than small fragments."),
    ("mitochondrial_area_fraction", "code", "morphology",
     "Measures the fraction of the image area occupied by mitochondrial signal (nonzero labels)."),
    ("mitochondria_local_autocorrelation_length", "code", "texture",
     "Computes the 2D spatial autocorrelation function of mitochondrial intensity within the cell and estimates the correlation length as the radius at which autocorrelation decays to 1/e of its maximum."),
    ("mito_intensity_autocorrelation_length", "code", "texture",
     "Computes the 2D spatial autocorrelation of intensity restricted to mitochondrial pixels and estimates the characteristic correlation length (decay to 1/e)."),
    ("mitochondria_to_cell_intensity_dynamic_range_ratio", "code", "intensity",
     "Computes the dynamic range (95th minus 5th percentile) of pixel intensities within mitochondrial objects and divides it by the dynamic range in the entire cell mask."),
    ("background_noise_robust_snr", "code", "intensity",
     "Estimates a robust SNR as median mitochondrial intensity divided by the MAD of background pixels."),
    ("mito_intensity_geodesic_variation", "code", "spatial",
     "Computes the slope of intensity coefficient of variation versus geodesic distance from the brightest mitochondrial pixel along the network."),
    ("mitochondria_local_entropy_mean", "code", "texture",
     "Applies a local entropy filter to the mitochondrial intensity image within the cell mask, then averages the entropy values over all cell pixels."),
    ("mitochondrial_radial_anisotropy_index", "code", "spatial",
     "Constructs radial mitochondrial intensity profiles along angular sectors around the cell centroid and returns the coefficient of variation across sectors."),
    ("mitochondria_network_continuity_score", "code", "morphology",
     "Quantifies how network-like versus fragmented the mitochondrial morphology is (skeleton length relative to object size). High values indicate long continuous tubules."),
    ("mitochondria_length_width_skewness", "code", "morphology",
     "For each mitochondria instance, computes the major/minor axis ratio and then the skewness of that per-object distribution over the whole cell."),
    ("mitochondria_intensity_entropy", "code", "texture",
     "Computes the Shannon entropy of the mitochondrial intensity histogram within the cell mask."),
    ("mito_clustered_vs_isolated_object_fraction", "code", "spatial",
     "Identifies mitochondrial instances that have a neighbour within a small distance versus those that remain isolated; returns the fraction of total mitochondrial area contributed by clustered objects."),
    ("mitochondria_intensity_gini_inside_cell", "code", "intensity",
     "Computes the Gini coefficient of mitochondrial signal intensities for all pixels within the cell mask."),
    ("mitochondria_area_fraction_in_perinuclear_band", "code", "spatial",
     "Measures the fraction of total mitochondrial area that resides in a perinuclear band around the nucleus."),
    ("mitochondria_intensity_entropy_within_cell", "code", "texture",
     "Computes the Shannon entropy of the intensity histogram for pixels inside the mitochondrial region, using 64 bins."),
    ("mitochondrial_clusteredness_by_nearest_neighbor", "code", "spatial",
     "Quantifies how spatially clustered mitochondrial objects are by comparing mean nearest-neighbour distances to a uniform spatial null. Higher values indicate stronger clusteredness."),
    ("mitochondrial_network_fragmentation_score", "code", "morphology",
     "Summarizes mitochondrial fragmentation on a continuum from fused/network-like (low) to punctate (high)."),
    ("mitochondrial_network_compactness", "vlm", "spatial",
     "Visually scores how compact versus dispersed the mitochondrial network appears within the cell. Higher values indicate a more compact, centrally concentrated organization."),
    ("mitochondrial_neighbor_intensity_variogram_slope", "code", "texture",
     "Estimates the slope of the empirical semi-variogram of mitochondrial intensity between neighbouring pixels at increasing spatial lags."),
]

EXTRACT_PY = '''def extract(img, *segmentation_masks):
    """Locked HSC feature: {name}."""
    from pathlib import Path
    import sys

    _shared = Path(__file__).resolve().parents[2] / "_shared"
    if str(_shared) not in sys.path:
        sys.path.insert(0, str(_shared))
    import mito_features as F
    return F.{name}(img, *segmentation_masks)
'''


def main() -> None:
    import csv

    names_path = ROOT / "source" / "morphagent_hsc_25_feature_names.csv"
    catalog_path = ROOT / "source" / "feature_catalog.csv"
    manifest_path = LIB / "manifest.csv"

    names_path.write_text("feature_name\n" + "\n".join(n for n, *_ in FEATURES) + "\n")

    with catalog_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "feature_name", "method", "category", "description"])
        for i, (name, method, cat, desc) in enumerate(FEATURES, 1):
            w.writerow([i, name, method, cat, desc])

    with manifest_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature_name", "method", "category", "library_path", "description"])
        for name, method, cat, desc in FEATURES:
            if method == "code":
                rel = f"code/{name}/extract.py"
            else:
                rel = f"vlm/{name}/feature.json"
            w.writerow([name, method, cat, rel, desc])

    for name, method, cat, desc in FEATURES:
        if method != "code":
            continue
        d = CODE / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "extract.py").write_text(EXTRACT_PY.format(name=name))

    print("Wrote", names_path)
    print("Wrote", catalog_path)
    print("Wrote", manifest_path)
    print("Wrote", sum(1 for *_, m, _, _ in [(a, b, c, d) for a, b, c, d in FEATURES if b == "code"]), "extract.py files")


if __name__ == "__main__":
    main()
