# MorphAgent 467-feature library

One folder per named feature from Supplementary Feature List 1. Nothing here depends on another MorphAgent checkout.

| Kind | Count | Path |
|------|-------|------|
| Code (`extract.py`) | 438 | `code/<feature_name>/extract.py` |
| VLM (planner record) | 29 | `vlm/<feature_name>/feature.json` |

`manifest.csv` lists every name, its method (`code` / `vlm`), category, and a path relative to this folder.

Code extractors expose `extract(img, seg)` (or `extract(img)`). `img` is the RGB `image.tif`; `seg` is the dict of `segmentation/*.tif` masks.

VLM replay uses `source/vlm_scoring.json` plus **your** OpenAI-compatible vision endpoint. This folder does not include a base URL or API key.
