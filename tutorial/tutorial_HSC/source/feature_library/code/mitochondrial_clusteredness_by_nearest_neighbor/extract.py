def extract(img, *segmentation_masks):
    """Locked HSC feature: mitochondrial_clusteredness_by_nearest_neighbor."""
    from pathlib import Path
    import sys

    _shared = Path(__file__).resolve().parents[2] / "_shared"
    if str(_shared) not in sys.path:
        sys.path.insert(0, str(_shared))
    import mito_features as F
    return F.mitochondrial_clusteredness_by_nearest_neighbor(img, *segmentation_masks)
