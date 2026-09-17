# MorphAgent BBBC021 tutorial

Self-contained walkthrough of the MorphAgent BBBC021 results. The **467 feature extractors**, MoA map, L1000 tables, and cached figure metrics all live in `source/`. The only external payload is the image dataset (Zenodo).

Recorded walkthrough: [`tutorial_BBBC021.mp4`](tutorial_BBBC021.mp4) (4½ min) runs all three notebooks and shows where the desktop app fits — designing the feature set from the images, and reusing it on a new dataset.

## Layout

```
tutorial_BBBC021/
  notebook/
    01_setup_environment_and_data.ipynb
    02_extract_morphagent_features.ipynb
    03_reproduce_main_figures.ipynb
  code/
    paths.py
    download_dataset.py
    extract_morphagent_features.py
    reproduce_main_figures.py
  source/
    feature_library/          439 code extractors + 28 VLM feature specs
    cached_results/           small JSON/CSV/PNG used to replot the paper figures
    morphagent_467_feature_names.csv
    drug_moa_source.csv
    l1000_per_method.csv
    l1000_pairwise.csv
    vlm_scoring.json
  data/
    dataset/                  gitignored; Zenodo images (or a local copy)
    outputs/                  gitignored; files written by the notebooks
```

## Setup

```bash
python -m pip install -r requirements.txt
```

Open the notebooks from `notebook/` (Jupyter / VS Code / Cursor). Run them in order:

1. **01_setup_environment_and_data** — install packages and download images into `data/dataset/`.
2. **02_extract_morphagent_features** — smoke-test one code feature and (optionally) one VLM feature.
3. **03_reproduce_main_figures** — replay the manuscript panels from `source/cached_results/`.

Images are on Zenodo: [record 22763120](https://zenodo.org/records/22763120) (DOI [10.5281/zenodo.22763120](https://doi.org/10.5281/zenodo.22763120)). Notebook 01 downloads the split `dataset.zip.part_*` files, concatenates them, and extracts into `data/dataset/`. If that folder already exists, the download is skipped. CLI:

```bash
python code/download_dataset.py
```

## Images

Each sample directory looks like:

```
data/dataset/<sample_id>/
  image.tif              RGB uint8, 512×512×3 (R=actin, G=tubulin, B=DAPI)
  slices/*.png           per-channel PNGs used by VLM features
  segmentation/*.tif     cyto / cytoplasm / nuclei masks used by code features
```

## VLM features

28 of the 467 names start with `vlm_`. They call an OpenAI-compatible vision endpoint (`POST /v1/chat/completions` with `image_url` parts). **This tutorial does not ship a base URL or API key.** In notebook 02, paste your own values into:

```python
VLM_API_BASE_URL = ""
VLM_API_KEY = ""
VLM_MODEL = "gpt-4o"
```

Leave them empty to skip the live VLM cell. CLI equivalent:

```bash
python code/extract_morphagent_features.py --smoke \
  --api-base https://YOUR_HOST/v1 \
  --api-key YOUR_KEY \
  --api-model YOUR_MODEL
```

## What is reproduced (notebook 03)

| Panel | Metric | CellProfiler | DeepProfiler | DP retrained | MorphAgent |
|-------|--------|--------------|--------------|--------------|------------|
| Non-redundant subset (\|r\| > 0.9) | # features | 1,284 | 440 | — | **291** |
| Perturbation detection | mAP | 0.154 | 0.190 | 0.042 | **0.228** |
| MoA prediction | kNN accuracy | 0.472 | 0.579 | 0.166 | **0.621** |
| Same-MoA matching | mAP | 0.524 | 0.543 | 0.462 | **0.602** |
| L1000 regression | R² | 0.096 | 0.159 | −0.005 | **0.177** |

Default path: replot from `source/cached_results/` (seconds). Optional full recompute needs large evaluation CSVs under `data/evaluation/` (not in this repository).

## CLI

```bash
python code/extract_morphagent_features.py --smoke
python code/reproduce_main_figures.py
```

## Expected runtime

- Notebook 03 (cached replay): seconds.
- Code-feature smoke (5 images): < 1 s.
- VLM-feature smoke (1–2 images): tens of seconds, API-dependent.
- Full 3,552 × 467 extraction is **not** launched by the notebooks. Serial wall time is on the order of a day for code features and much longer for naive per-feature VLM calls.
