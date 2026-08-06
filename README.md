# MorphAgent UI Lite

MorphAgent UI Lite is a lightweight desktop application for designing, extracting, reviewing, and reusing microscopy image features. It runs in one Conda environment and supports OpenAI-compatible LLM and multimodal APIs.

## UI demo

The bundled Tau dataset provides 10 ready-to-run samples and a completed result for browsing the interface without preparing data first.

Workflow:

**Home → Configure → Run → Features → Evidence**

The UI also includes:

- **Reuse history features** — apply merged code from an earlier run to a new dataset without calling an LLM or VLM.
- **Ask MorphAgent** — ask questions about completed feature results.
- **Prepared knowledge** — use bundled expert, research, and literature summaries for the demo.

Demo video:

https://github.com/user-attachments/assets/efa9fb0b-0f2d-48f2-8899-7abb2b74b6f5

## Install

First install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda.

```bash
git clone https://github.com/ai4imaging/MorphAgent.git
cd MorphAgent/MorphAgent_UI_Lite
```

### macOS / Linux

```bash
bash scripts/setup.sh
bash scripts/start_ui.sh
```

### Windows

Open `MorphAgent_UI_Lite\scripts\` and double-click:

1. `setup_windows.bat`
2. `start_ui_windows.bat`

Or run the same files from Anaconda Prompt or PowerShell:

```powershell
.\scripts\setup_windows.bat
.\scripts\start_ui_windows.bat
```

The installer creates one environment named `morphagent_lite`.

## Use your own dataset

Select a directory containing one subdirectory per sample:

```text
dataset/
├── sample_1/
│   ├── image.tif
│   └── segmentation/       # optional existing masks
└── sample_2/
    └── image.png
```

PNG, JPEG, TIFF/OME-TIFF, GIF, WebP, MRC, MAP, and REC images are supported. Multichannel, Z-stack, and time-series inputs are expanded into grayscale slices while preserving their axes in `slice_manifest.json`.

Lite reuses segmentation masks when present and continues without segmentation when masks are absent.

## Verify

From `MorphAgent_UI_Lite`:

```bash
conda run -n morphagent_lite python scripts/verify_install.py
```

More details are available in [`MorphAgent_UI_Lite/README_LITE.md`](MorphAgent_UI_Lite/README_LITE.md).
