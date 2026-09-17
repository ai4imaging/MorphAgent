# MorphAgent HSC 25-feature library

One folder per named feature from Supplementary Feature List 2.

| Kind | Count | Path |
|------|-------|------|
| Code (`extract.py`) | 24 | `code/<feature_name>/extract.py` |
| VLM (planner record) | 1 | `vlm/mitochondrial_network_compactness/feature.json` |

`manifest.csv` lists every name, its method (`code` / `vlm`), and a short description.

Code extractors expose `extract(img, *segmentation_masks) → float`. Implementations live in `_shared/mito_features.py` so the 24 extractors stay in lock-step with the catalog.

The VLM feature scores mitochondrial network compactness on a 0–100 scale. This tutorial does not ship an API key.
