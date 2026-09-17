# Agentic cell profiling across microscopy modalities via biologically grounded feature design

Enze Ye, Xiaoxuan Wu, Rui Peng, Wenjia Hu, Xiangyou Li, Xuefei Zhang, Mengxiao Niu, Yaorong Guo, Jinzhuo Wang, Liangyi Chen, He Sun

MorphAgent is an LLM/VLM agent that designs and extracts quantitative morphological features from microscopy images. This repository's default path is the **desktop UI**.

Supplementary video:

https://github.com/user-attachments/assets/0bf0fd55-e039-4551-a5be-eddc6035a114

## Layout

| Path | What it is |
|------|------------|
| [`MorphAgent_UI/`](MorphAgent_UI/) | Desktop UI (`scripts/` + nested `MorphAgent/` app) |
| [`MorphAgent_CLI/`](MorphAgent_CLI/) | Headless `python main.py` pipeline, with its own [README](MorphAgent_CLI/README.md) |

UI flow: **Home → Configure → Run → Features → Evidence**. The UI includes the bundled Tau demo, feature extraction, historical code reuse, Ask MorphAgent, and support for multidimensional PNG, TIFF/OME-TIFF, GIF, WebP, and MRC-family images.

## Install the desktop UI

```bash
git clone https://github.com/ai4imaging/MorphAgent.git
cd MorphAgent/MorphAgent_UI
bash scripts/setup.sh
bash scripts/start_ui.sh
```

On Windows, open `MorphAgent_UI\scripts\` and run `setup_windows.bat`, then `start_ui_windows.bat`.

Detailed UI notes: [`MorphAgent_UI/README.md`](MorphAgent_UI/README.md).

Setup creates the `morphagent_lite` conda env, pip-installs the science stack, and editable-installs `MorphAgent_UI/MorphAgent`. API keys are configured in the UI (or in `MorphAgent_UI/MorphAgent/.env`).

## Dataset layout (for your own images)

In the UI, select the parent folder that contains `dataset/<sample>/`:

```
INPUT/
├── dataset/                   # one subdirectory per sample
│   ├── WT_1/
│   │   ├── image.tif          # primary image (Code features)
│   │   ├── slices/            # optional 2D slices (VLM prefers these)
│   │   └── segmentation/      # optional masks, e.g. mask_cell.tif
│   └── MU_1/
│       └── image.tif
├── expert_knowledge/          # optional
├── deep_research/             # optional
└── RAG/                       # optional
```

- Sample ID = subdirectory name. Recommend ≥5 samples.
- Primary files sit directly in the sample folder. `segmentation/` masks are keyed by filename stem (`seg["mask_cell"]`).
- If a sample has no masks, this UI skips auto-segmentation and continues.
- A short `dataset_index.txt` (or README) under `dataset/` describing channels and dimensions helps planning.

## Command-line pipeline

The original CLI is packaged separately:

```bash
cd MorphAgent/MorphAgent_CLI
```

See [`MorphAgent_CLI/README.md`](MorphAgent_CLI/README.md) and [`MorphAgent_CLI/installation_skill.md`](MorphAgent_CLI/installation_skill.md) for the `morphagent` conda env, `python main.py …`, Cellpose-SAM / Allen, and all CLI flags.

## For coding agents

| Goal | Read | Working directory |
|------|------|-------------------|
| Desktop UI | [`MorphAgent_UI/README.md`](MorphAgent_UI/README.md) | **`MorphAgent_UI/`** |
| CLI pipeline | [`MorphAgent_CLI/installation_skill.md`](MorphAgent_CLI/installation_skill.md) | **`MorphAgent_CLI/`** |
