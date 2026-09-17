# Agentic cell profiling across microscopy modalities via biologically grounded feature design

Enze Ye, Xiaoxuan Wu, Rui Peng, Wenjia Hu, Xiangyou Li, Xuefei Zhang, Mengxiao Niu, Yaorong Guo, Jinzhuo Wang, Liangyi Chen, He Sun

MorphAgent is an LLM/VLM agent that designs and extracts quantitative morphological features from microscopy images. This repository's default path is the **desktop UI**.

Supplementary video:

https://github.com/user-attachments/assets/0bf0fd55-e039-4551-a5be-eddc6035a114

## Layout

| Path | What it is |
|------|------------|
| [`MorphAgent_UI/`](MorphAgent_UI/) | Self-contained desktop UI: pipeline, UI package, frontend, installer, and bundled Tau demo |
| [`MorphAgent_CLI/`](MorphAgent_CLI/) | Headless `python main.py` pipeline, with its own [README](MorphAgent_CLI/README.md) |
| [`tutorial/`](tutorial/) | Reproduction of the paper's main results |

The UI is a native window (Qt6 + WebEngine, no browser required) with four pages in the left sidebar:

| Page | What you do there |
|------|-------------------|
| **Design** | Attach a dataset and optional knowledge files, ask a biological question, set the feature count, review the configuration, then watch progress and live output on the same page |
| **Compute** | Apply the feature scripts saved by a previous run to a new dataset — no model calls, no feature redesign |
| **Visualize** | Load a run's `feature_value.csv` and browse retained features with All / Code / VLM filters and distribution histograms |
| **Help** | Ask MorphAgent about the paper, figures, methods, or implementation, answered from bundled manuscript and code excerpts |

Completed runs export a timestamped folder and ZIP automatically. Input images may be multidimensional PNG, TIFF/OME-TIFF, GIF, WebP, or MRC-family files.

## Paper result reproduction

[`tutorial/`](tutorial/) reproduces the main results in the paper. Each subfolder is self-contained (notebooks, code, and cached tables):

| Tutorial | What it reproduces |
|----------|--------------------|
| [`tutorial/tutorial_BBBC021/`](tutorial/tutorial_BBBC021/) | BBBC021 MoA benchmarks and main figures (467 MorphAgent features) |
| [`tutorial/tutorial_Tau/`](tutorial/tutorial_Tau/) | Tau genotype separation, classification, and transcriptome prediction |
| [`tutorial/tutorial_HSC/`](tutorial/tutorial_HSC/) | Young/Old HSC Figure 3 panels |

Start from the README in each folder and run the notebooks there. Large image payloads (BBBC021) are downloaded separately; they are not in the git tree.

## Install the desktop UI

```bash
git clone https://github.com/ai4imaging/MorphAgent.git
cd MorphAgent/MorphAgent_UI
bash scripts/setup.sh
bash scripts/start_ui.sh
```

On Windows, open `MorphAgent_UI\scripts\` and run `setup_windows.bat`, then `start_ui_windows.bat`.

Setup creates the `morphagent_lite` conda environment, pip-installs the science stack together with PySide6 / Qt WebEngine, editable-installs `MorphAgent_UI`, and finishes with `scripts/verify_install.py`. Rerun the same command to upgrade an existing installation; it reuses the environment and keeps your history and results.

Open **Settings** in the UI and enter your own OpenAI-compatible Base URL, key, and model. Credentials stay in memory for the current session only and are never written to disk.

Already installed? Launch without a browser:

```bash
cd MorphAgent_UI
conda activate morphagent_lite
python launch_desktop_ui.py
```

A browser workspace with the same history and results is available via `python launch_web_ui.py` (then open http://127.0.0.1:8766). The older Qt5 interface remains at `python launch_ui.py`.

Detailed UI notes: [`MorphAgent_UI/README.md`](MorphAgent_UI/README.md) (desktop), [`MorphAgent_UI/README_WEB.md`](MorphAgent_UI/README_WEB.md) (browser workflow and data formats).

## Dataset layout (for your own images)

In **Design → Add data**, select the parent folder that contains `dataset/<sample>/`:

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
| Paper result reproduction | [`tutorial/`](tutorial/) | the matching `tutorial/tutorial_*` folder |
