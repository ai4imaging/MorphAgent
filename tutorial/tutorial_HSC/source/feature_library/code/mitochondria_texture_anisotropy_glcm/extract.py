def extract(img, *segmentation_masks):
    """Locked HSC feature: mitochondria_texture_anisotropy_glcm."""
    from pathlib import Path
    import sys

    _shared = Path(__file__).resolve().parents[2] / "_shared"
    if str(_shared) not in sys.path:
        sys.path.insert(0, str(_shared))
    import mito_features as F
    return F.mitochondria_texture_anisotropy_glcm(img, *segmentation_masks)
