# MorphAgent

MorphAgent is an **automatic microscopy image feature extraction agent** built on large language models (LLMs) and multimodal vision-language models (VLMs). Given a batch of microscopy images and a one-sentence natural-language task description, it automatically:

1. **Understands the dataset** (dimensions, channels, markers, naming conventions);
2. **Automatically segments** cells / nuclei / cytoplasm and other structures (Cellpose-SAM, or Allen aicssegmentation);
3. **Plans features** (morphology, intensity, texture, distribution, spatial, and other categories);
4. Extracts scalar features via two complementary paths:
   - **Code features**: the LLM generates an `extract()` Python function that runs in an isolated sandbox environment, self-debugs, and executes in batch;
   - **VLM features**: a multimodal model scores images feature by feature (a continuous score of 0–100);
5. (Optional) injects external knowledge: **expert_knowledge** (expert materials), **auto_deep_research** (deep research reports), **auto_literature_retrieval / RAG** (literature corpus);
6. **Deterministically validates** and filters features, and outputs a feature table CSV.

> This repository is the **public, general-purpose build**: it accesses models only through an **OpenAI-compatible API** (**no local model deployment whatsoever**), and it contains no specific datasets, experiments, or paper-analysis code. You only need to prepare your own dataset and configure an API to run it on any microscopy dataset.

---

## Table of Contents

- [Graphical User Interface (UI)](#graphical-user-interface-morphagent-ui)
- [Key Features](#key-features)
- [Directory Structure](#directory-structure)
- [Environment & Installation](#environment--installation)
- [Configuration (API & Models)](#configuration-api--models)
- [Input Data Format (Important)](#input-data-format-important)
- [Quick Start](#quick-start)
- [Output](#output)
- [Auto Segmentation (auto_segmentation)](#auto-segmentation-auto_segmentation)
- [Auto Deep Research & Literature Retrieval](#auto-deep-research--literature-retrieval)
- [Common Command-Line Arguments](#common-command-line-arguments)
- [FAQ](#faq)

---

## Graphical User Interface (MorphAgent UI)

Prefer not to drive everything from the command line? This repo also ships a desktop Qt UI under [`MorphAgent_UI/`](MorphAgent_UI/) that wraps the same `main.py` pipeline with a guided workflow: **Home → Configure → Run → Features → Evidence**.

Watch the ~4m40s demo video first to see the full workflow end-to-end:

https://github.com/user-attachments/assets/efa9fb0b-0f2d-48f2-8899-7abb2b74b6f5

> English narration with burned-in subtitles; a matching `.srt` file is included alongside it: [`MorphAgent_UI/demo_video/MorphAgent_demo_english.srt`](MorphAgent_UI/demo_video/MorphAgent_demo_english.srt).

### Try it

```bash
cd MorphAgent_UI
bash scripts/setup_macos_linux.sh   # one-click environment setup (Windows: scripts/setup_windows.ps1)
bash scripts/start_ui.sh            # launch the desktop app (Windows: scripts/start_ui.ps1)
```

Highlights:

- **Bundled demo dataset** — click **Use bundled Tau demo** to run the full pipeline on 5 sample images without preparing your own data first;
- **Load a previous run** — instantly browse a completed run's `Features` and `Evidence` pages, no API key required;
- **In-app API configuration** — fill in Base URL / API key / model on the Configure page; it writes straight to `MorphAgent/.env`, no manual file editing needed.

Full install/usage instructions, system requirements, and troubleshooting live in [`MorphAgent_UI/README_UI.md`](MorphAgent_UI/README_UI.md).

---

## Key Features

| Capability | Description | Dependencies |
|------|------|------|
| Code feature extraction | The LLM generates/repairs an `extract(img, seg)` function and runs it in batch | LLM API + sandbox conda environment |
| VLM feature scoring | A multimodal model scores images feature by feature on a continuous scale | Multimodal API (e.g. GPT-4o) |
| auto_segmentation | Generates masks with Cellpose-SAM (default) or Allen aicssegmentation | GPU (Cellpose) / CPU (Allen) |
| auto_deep_research | Reads research reports under `deep_research/` (.md/.txt/.pdf) → LLM digests → injects into planning | LLM API (PDF requires PaddleX) |
| auto_literature_retrieval (RAG) | Reads literature under `RAG/` (.xml/.pdf) → LLM digests → injects into planning | LLM API (PDF requires PaddleX) |
| expert_knowledge | Reads expert materials under `expert_knowledge/` → LLM digests | LLM API |
| Deterministic validation | Unsupervised / supervised (metadata) feature filtering with multi-round deduplication | None |

---

## Directory Structure

```
MorphAgent/
├── main.py                  # Main entry point (processes the whole dataset in batch)
├── config.py                # Global configuration (USER CONFIGURATION block at the top)
├── graph.py / state.py      # LangGraph pipeline and state
├── utils_helpers.py         # Dataset indexing / image path lookup
├── nodes/                   # researcher → prompt_gen → execution nodes
├── tools/                   # Code generation/execution/repair, VLM client, segmentation calls
├── knowledge/               # Dataset understanding, expert/deep_research/RAG, prompts/
│   └── prompts/             # 6 general-purpose prompt templates (JSON)
├── utils/ utils_modules/    # Cell context, data preprocessing, channel parsing, reproducibility
├── validation/              # Deterministic feature validation and registry
├── segmentation_allen/      # Allen aicssegmentation backend (vendored + CLI entry point)
├── envs/                    # Environment yaml files (see "Environment & Installation")
├── .env.example             # Configuration template (copy to .env)
└── README.md
```

---

## Environment & Installation

This project has **no local large model**: all LLM/VLM calls go through an OpenAI-compatible API. The environment only needs to cover
agent orchestration, sandbox scientific computing, and Cellpose-SAM segmentation. Everything is merged into **one** unified environment.

- **`morphagent` (unified environment, Python 3.10)**: runs the main program (LangChain/LangGraph/OpenAI client), the sandbox that executes generated code (numpy/scipy/scikit-image/opencv/tifffile/mahotas, etc.), and **Cellpose-SAM** segmentation (cellpose ≥ 4.0 + PyTorch). For convenience, the agent, sandbox, and segmentation are merged into the same environment.
- **`morphagent_allen` (optional/legacy, Python 3.6)**: Allen `aicssegmentation` has older dependencies (scikit-image 0.15, numpy 1.19) that **cannot be merged with a modern environment**, so it must be created separately. The default Cellpose-SAM path does not use it at all.

Installation:

```bash
# 1) Unified main environment (recommended)
conda env create -f envs/environment.yml      # creates morphagent
conda activate morphagent

# Or: manage your own Python 3.10 environment (venv/pyenv/uv)
#   pip install -r envs/requirements.txt

# 2) Post-install self-check
python - <<'PY'
import langchain, langgraph, openai, cellpose, torch
import numpy, pandas, skimage, cv2, tifffile, mahotas, bs4
print("torch CUDA available:", torch.cuda.is_available())
print("cellpose:", cellpose.version); print("OK")
PY

# 3) (Optional) extra features such as PDF parsing / local VLM
#   pip install -r envs/requirements-optional.txt

# 4) (Optional/legacy) Allen segmentation environment
conda env create -f envs/environment_allen.yml # creates morphagent_allen
conda activate morphagent_allen
pip install -e segmentation_allen              # installs vendored aicssegmentation
python segmentation_allen/check_installation.py
export SEGMENTATION_CONDA_ENV=morphagent_allen # make the main program use the Allen segmentation backend
```

> See `envs/README.md` for details.

---

## Configuration (API & Models)

There are two ways (you **must explicitly specify the model name to use**):

**Option A: Environment variables (recommended, keeps keys out of the repo)**

```bash
cp .env.example .env      # edit .env and fill in your values
source .env               # or export manually

export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_API_KEY="sk-..."
export LLM_MODEL="gpt-4o"          # text model used for planning/writing code/review

export VLM_BASE_URL="https://api.openai.com/v1"
export VLM_API_KEY="sk-..."
export VLM_MODEL="gpt-4o"          # multimodal (vision) model used for scoring
```

**Option B: Edit `config.py` directly**

Open the `USER CONFIGURATION` block at the top of `config.py` and change the default values of `DEFAULT_LLM_*` / `DEFAULT_VLM_*` to your own.

> Any service that implements the OpenAI Chat Completions protocol will work: OpenAI, Azure OpenAI, DeepSeek, Together, OpenRouter, or a self-hosted gateway (vLLM / Ollama / LM Studio). If a gateway requires custom HTTP headers, set them in `DEFAULT_LLM_HEADERS` / `DEFAULT_VLM_HEADERS` in `config.py`.

To temporarily switch endpoints, register a named preset in `API_PROVIDER_PRESETS` in `config.py` and select it at runtime with `--api-provider <name>`.

---

## Input Data Format (Important)

MorphAgent makes **very general** assumptions about the data layout: **one dataset = one directory, in which each subdirectory is one sample**. Details below.

### 1. Top-level layout

The directory you point `--data-root` at (denoted `INPUT`) can be one of two forms, and the program will recognize them automatically:

```
# Form 1: INPUT is directly the dataset root
INPUT/                         # == data_root
├── dataset_index.txt          # dataset description file (see below)
├── sample_0001/               # one sample = one subdirectory
├── sample_0002/
└── ...

# Form 2: INPUT is the project root, containing a dataset/ subdirectory (recommended, allows knowledge folders)
INPUT/                         # == project_root
├── dataset/                   # == data_root (auto-detected)
│   ├── dataset_index.txt
│   ├── sample_0001/
│   └── ...
├── expert_knowledge/          # optional
├── deep_research/             # optional
└── RAG/                       # optional
```

- **Sample ID = subdirectory name**: `read_dataset_index()` directly scans all non-hidden subdirectories under `data_root` as the sample list, and **does not rely on any manifest inside an index file**. The directory name is the sample ID (processed in sorted order).
- If a `dataset/` directory exists under `INPUT`, then `data_root=INPUT/dataset` and `project_root=INPUT`; otherwise `data_root=INPUT` and `project_root` is the parent directory of `INPUT`. Knowledge folders (`expert_knowledge/`, `deep_research/`, `RAG/`) always live under **project_root**.

### 2. Inside each sample directory

```
sample_0001/
├── image.tif                  # (1) primary file (raw data), used by Code features
├── ...(there may be multiple primary image files)
├── slices/                    # (2) secondary directory (derived data), preferred by VLM features
│   ├── slice_0000_channel_0.png
│   └── ...
└── segmentation/              # (3) segmentation masks (auto-generated by this tool, or bring your own)
    ├── mask_cell.tif
    ├── mask_nucleus.tif
    └── ...
```

- **(1) Primary files (primary)**: image files **directly** under the sample directory (excluding subdirectories). These are the input for **Code features** (the `img` in `extract(img, ...)`). Supports `.tif/.tiff/.png/.jpg/.jpeg/.bmp/.gif`. The generated code chooses the reading method by extension (tifffile for TIFF, PIL for PNG/JPG).
- **(2) Secondary directories (secondary)**: image files inside subdirectories of the sample directory (e.g. `slices/*.png`). **VLM features** prefer these 2D slices that are easy to view visually; if none exist, they fall back to the primary files. These slices can be generated automatically by this tool's preprocessing stage (normalized slices for multi-channel/z-stack data), or you can bring your own.
- **(3) `segmentation/`**: the segmentation mask directory (see the next section).

### 3. Segmentation masks and the `seg` dictionary (key point)

The generated Code feature function signature is:

```python
def extract(img, seg):   # when segmentation is available
def extract(img):        # when there is no segmentation
```

Here **`seg` is a dictionary** whose keys are the **file name stems** of the mask files under the `segmentation/` directory:

| File | Key in `seg` | Typical meaning |
|------|-------------|----------|
| `segmentation/mask_cell.tif` | `seg["mask_cell"]` | whole cell |
| `segmentation/mask_nucleus.tif` | `seg["mask_nucleus"]` | nucleus |
| `segmentation/cyto.tif` | `seg["cyto"]` | Cellpose-SAM whole-cell instance labels |
| `segmentation/nuclei.tif` | `seg["nuclei"]` | Cellpose-SAM nucleus instance labels |

- The code **always accesses masks by key name** (`seg.get("mask_cell")`), never by positional index — different datasets have different mask names/counts.
- A mask can be binary (0/background) or an instance label map (multiple integer labels); the generated code processes objects using `regionprops` and similar tools.
- **You can bring your own segmentation**: just place masks into each sample's `segmentation/` directory, and by default (`--segmentation-skip-if-present`) the program will skip auto-segmentation and use your masks directly.

### 4. Dataset description file (description)

Placed under `data_root` and located by priority: `dataset_index.txt` → `README.md` → `README.txt` → `dataset_description.json` → `description.txt` (you can also specify the path explicitly with `--description`).

It is **free text** handed to the LLM to understand, telling the agent about: the data dimensions (2D/3D/multi-channel/z-stack), what each channel captures (markers/colors), file naming conventions, etc. The clearer it is written, the more accurate the feature planning. Example:

```
Dataset: my_microscopy
Dimension: 2D multi-channel (C, H, W), 3 channels
Channels:
  Channel 0 | Red   | Actin (F-actin)
  Channel 1 | Green | Tubulin
  Channel 2 | Blue  | DAPI (nucleus)
File layout: each sample folder contains one multi-channel TIFF `image.tif`.
Notes: single cell per image; pixel size ~0.65 um.
```

> The description file **only affects understanding and planning**; it does not determine the sample list (the sample list comes from scanning subdirectories).

### 5. Optional knowledge folders (placed under project_root)

| Folder | What to put in | How it is used |
|--------|--------|-----------|
| `expert_knowledge/` | Expert notes, example images, review PDFs (`.txt/.md/.csv/.json/.pdf/images`) | The LLM digests them into expert knowledge blocks and injects them into planning |
| `deep_research/` | Deep research reports (**`.md/.txt` recommended**; `.pdf` also supported) | The LLM digests them into a research summary and injects it into planning |
| `RAG/` | Literature corpus (PMC `.xml`; `.pdf` also supported), placed flat at the top level of this directory | The LLM digests them in batch into literature knowledge and injects it into planning (with hash caching) |

- These are all **optional** and can be turned off with `--disable-expert-knowledge` / `--disable-deep-research` / `--disable-rag`.
- `.pdf` parsing requires PaddleX (heavy, GPU); if you use `.md/.txt/.xml`, **PaddleX is not needed**.

---

## Quick Start

```bash
conda activate morphagent
source .env    # API/models already configured

python main.py "Generate unbiased morphological features for these microscopy images" \
  --data-root /path/to/INPUT \
  --method both \
  --features-per-iteration 10 \
  --target-feature-count 200
```

- The positional argument is the **natural-language task description** (it feeds into the feature planning prompt).
- `--method both`: use both code and vlm; you can also use `code` or `vlm`.
- The first run automatically: understands the dataset → (optionally) segments → plans features → generates/executes code + VLM scoring → validates → writes CSV.

---

## Output

By default, results are written to `<project_root>/results/run_<timestamp>/` (can be set with `--results-dir`):

| File | Meaning |
|------|------|
| `features.csv` | All extracted features (rows = samples, columns = features) |
| `retained_features.csv` | Features retained after deterministic validation |
| `feature_registry.json` | Feature registry (with metadata/decisions for each feature) |
| `feature_plan.json` | The list of features planned in this round |
| `dataset_description.txt` | The dataset understanding generated by the LLM |
| `expert_knowledge_summary.txt` / `deep_research_summary.txt` / `rag_knowledge_summary.txt` | Summaries of each knowledge source (if enabled) |
| `segmentation_summary.json` | Segmentation statistics (if enabled) |

---

## Auto Segmentation (auto_segmentation)

### Cellpose-SAM (default, integrated into the main pipeline)

Step 2.4 of the main program automatically segments all samples (requires a GPU). You can also call it separately:

```bash
# Single image
conda activate morphagent
python tools/segment_tif_with_cpsam.py input.tif -o sample_dir/segmentation/ -c 0 2

# Batch (Python API)
python -c "from tools.segmentation import segment_all_samples; ..."
```

This outputs `cyto.tif` / `nuclei.tif` / `cytoplasm.tif` to each sample's `segmentation/`. Use `--disable-segmentation` to turn it off.

### Allen aicssegmentation (optional, separate environment)

Good for classic nucleus/cytoplasm segmentation and fiber/mitochondria-like punctate structures. Requires the `morphagent_allen` environment (see `segmentation_allen/README_SEGMENTATION.md`):

```bash
conda activate morphagent_allen
# Nucleus + cytoplasm (TIFF)
python segmentation_allen/run_segment_image_tif.py input.tif -o out_dir/ -c 2 0 1
# Mitochondria / punctate structures
python segmentation_allen/run_segment_mitochondria.py input.tif -o out_dir/
# PNG batch
python segmentation_allen/segmentation_pipeline.py -d png_dir/ -o out_dir/
```

Place the generated masks into each sample's `segmentation/`, and the main pipeline will use them directly (skipping Cellpose by default).

---

## Auto Deep Research & Literature Retrieval

At **runtime**, these two capabilities manifest as "read a local knowledge folder → LLM digests → inject into feature planning":

- **auto_deep_research**: save your research reports (produced by Gemini/other deep research tools) as `.md`/`.txt` into `project_root/deep_research/`, and they are digested automatically at runtime.
- **auto_literature_retrieval (RAG)**: place PMC literature `.xml` (or `.pdf`) flat into `project_root/RAG/`, and they are digested and cached automatically at runtime.

> The offline "online deep research agent" and "PubMed batch download" are heavy standalone subsystems and are not included in this general-purpose build; this build focuses on **digesting knowledge files you have already prepared**.

---

## Common Command-Line Arguments

| Argument | Default | Description |
|------|------|------|
| `user_query` (positional) | required | Natural-language task description |
| `--data-root` | config | Dataset root / project root (auto-detects `dataset/`) |
| `--description` | none | Explicitly specify the description file path |
| `--results-dir` | auto | Output directory |
| `--method` | `both` | `code` / `vlm` / `both` |
| `--features-per-iteration` | 10 | Number of features planned per round |
| `--target-feature-count` | 1000 | Target total number of features (used in the planning prompt) |
| `--num-rounds` | 1 | Multi-round iteration |
| `--api-provider` | `default` | LLM endpoint preset name (see config) |
| `--vlm-api-provider` | `online` | `online`/`api` (API, recommended) or `qwen` (local, advanced) |
| `--llm-model` / `--vlm-online-model` | none | Override the model name |
| `--enable-segmentation` / `--disable-segmentation` | enabled | Auto-segmentation toggle |
| `--segmentation-skip-if-present` | enabled | Skip segmentation if masks already exist (your own masks take priority) |
| `--enable-* / --disable-*` (expert-knowledge / deep-research / rag) | enabled | Toggles for each knowledge source |
| `--reproduce` | off | Deterministic mode (temperature=0 + VLM caching) |
| `--code-parallel-workers` | 1 | Number of parallel processes for running merged code across all samples |
| `--vlm-online-concurrency` | 1 | Number of concurrent threads for the online VLM API |

(See `python main.py -h` for the full list.)

---

## FAQ

- **Do I really need a GPU?** LLM/VLM go through the API and need no local GPU; but **Cellpose-SAM segmentation requires a GPU**. Without a GPU, you can disable segmentation (`--disable-segmentation`) or use your own masks / Allen (CPU) instead.
- **Code execution reports missing packages?** Generated code runs in `CONDA_ENV` (default `morphagent`) and will try to `pip/conda install` automatically. Pre-installing common scientific-computing libraries into that environment is more reliable.
- **Don't want to install PaddleX?** Use `.md/.txt` for deep_research and `.xml` for RAG, and PaddleX is not needed.
- **VLM scoring is very slow / times out?** Increase `--vlm-online-concurrency`, or tune environment variables such as `VLM_ONLINE_REQUEST_TIMEOUT` (see `config.py`).
