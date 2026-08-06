# MorphAgent UI Lite

MorphAgent UI Lite provides the desktop interface, bundled Tau demo, feature extraction pipeline, historical code reuse, and result exploration in one Conda environment.

## Included

- Qt desktop UI
- Bundled 10-sample Tau demo
- Code-based feature extraction
- Online multimodal feature scoring
- Historical merged-code reuse without model calls
- Ask MorphAgent result questions
- Prepared expert, research, and literature summaries
- Existing segmentation-mask reuse
- PNG, JPEG, TIFF/OME-TIFF, GIF, WebP, MRC, MAP, and REC input support

Automatic segmentation and live literature downloading are not included. Missing optional knowledge or masks do not block a normal Lite run.

## Install

Prerequisites:

- Miniconda or Anaconda
- Network access during the first installation

### macOS / Linux

From this directory:

```bash
bash scripts/setup.sh
bash scripts/start_ui.sh
```

### Windows

Open `scripts\` and double-click:

1. `setup_windows.bat`
2. `start_ui_windows.bat`

Or run:

```powershell
.\scripts\setup_windows.bat
.\scripts\start_ui_windows.bat
```

Setup creates the single `morphagent_lite` environment and installs the application from `MorphAgent/`.

## Run the demo

1. Open the UI.
2. Click **Load demo dataset**.
3. Configure an OpenAI-compatible text and multimodal API.
4. Choose the number of rounds and features.
5. Click **Run MorphAgent**.
6. Review retained features and evidence when the run finishes.

A completed demo run can be opened without an API key.

## Use your own images

Choose the directory containing the sample folders:

```text
dataset/
├── sample_1/
│   ├── image.tif
│   └── segmentation/
│       └── mask_cell.tif
└── sample_2/
    └── image.png
```

Non-2D images are expanded into grayscale slices. Generated slice metadata is recorded in `slice_manifest.json`.

Existing masks are reused. If no mask is present, Lite continues without automatic segmentation unless a historical reuse job explicitly requires that mask.

## Reuse historical features

From Home, open **Reuse history features** and select:

1. A completed MorphAgent results directory.
2. A new dataset.
3. An output directory.

Lite applies each round's merged code to the new samples without LLM or VLM calls. Required segmentation masks are checked before execution.

## Prepared knowledge

The demo summaries are stored under:

```text
MorphAgent/demo/precomputed/
```

When enabled, Lite uses prepared summaries first. If one is unavailable, it can synthesize a short summary with the configured LLM and cache it under `.knowledge_precomputed/`. A failed optional knowledge call is skipped instead of stopping the run.

## Verify

```bash
conda run -n morphagent_lite python scripts/verify_install.py
```

To recreate the environment:

```bash
MORPHAGENT_RECREATE_ENVS=1 bash scripts/setup.sh
```
