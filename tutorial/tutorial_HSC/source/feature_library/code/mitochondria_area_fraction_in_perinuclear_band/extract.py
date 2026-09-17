def extract(img, *segmentation_masks):
    """Locked HSC feature: mitochondria_area_fraction_in_perinuclear_band."""
    from pathlib import Path
    import sys

    _shared = Path(__file__).resolve().parents[2] / "_shared"
    if str(_shared) not in sys.path:
        sys.path.insert(0, str(_shared))
    import mito_features as F
    return F.mitochondria_area_fraction_in_perinuclear_band(img, *segmentation_masks)
