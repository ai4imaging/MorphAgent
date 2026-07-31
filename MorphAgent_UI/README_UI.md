# MorphAgent UI Handoff Notes

This directory is a reviewable handoff package for the MorphAgent graphical interface. It includes the current UI source, backend pipeline, dependency manifests, install/launch/self-check scripts, Tau neuron demo data provided by the instructor, one completed run result, and the final English demo video.

## 0. Docker one-click (recommended)

No conda install needed. Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/) (macOS / Windows) or Docker Engine (Linux). Double-click the starter; the MorphAgent Qt UI opens in the browser via noVNC.

| Platform | Download |
|----------|----------|
| macOS | [MorphAgent-UI-Docker-macOS.zip](https://github.com/ai4imaging/MorphAgent/releases/download/ui-docker-20260731/MorphAgent-UI-Docker-macOS.zip) |
| Windows | [MorphAgent-UI-Docker-Windows.zip](https://github.com/ai4imaging/MorphAgent/releases/download/ui-docker-20260731/MorphAgent-UI-Docker-Windows.zip) |
| Linux | [MorphAgent-UI-Docker-Linux.zip](https://github.com/ai4imaging/MorphAgent/releases/download/ui-docker-20260731/MorphAgent-UI-Docker-Linux.zip) |

After download: unzip → double-click `MorphAgent-UI.command` (Mac) / `MorphAgent-UI.bat` (Windows) / run `MorphAgent-UI.sh` (Linux) → browser opens at `http://localhost:6080/vnc.html?autoconnect=true&resize=remote`.

First launch builds the image (several minutes). Later starts are fast. For a quick review without an API key: **Home → Load a previous run →** `completed_demo_run`.

From this repo (developers):

```bash
# macOS
open docker/mac/MorphAgent-UI.command
# Linux
bash docker/linux/MorphAgent-UI.sh
# Windows
docker\win\MorphAgent-UI.bat

# Rebuild the three downloadable zip packages locally (does not push):
bash scripts/build_docker_packages.sh
# Optional: embed a pre-built image tarball into each zip (much larger):
bash scripts/build_docker_packages.sh --with-image
```

Packages are written to `docker/dist/` for manual upload to the GitHub Release above.

## 1. Package Contents

```text
MorphAgent_handoff_20260719/
├── MorphAgent/                         # Current runnable source
│   ├── launch_ui.py                    # UI entry point
│   ├── main.py                         # Full backend pipeline entry point
│   ├── morphagent_ui/                  # Home/Configure/Run/Features/Evidence
│   ├── envs/                           # Original repo dependency files
│   ├── demo/data/                      # 10 Tau demo samples and knowledge materials
│   └── demo/data/results/
│       └── completed_demo_run/         # Completed results, ready to load and inspect
├── dependencies/                       # Dependency manifests organized by purpose
├── scripts/                            # One-click install, launch, and self-check
├── demo_video/
│   ├── MorphAgent_demo_english.mp4     # ~4 min 40 sec English demo (compressed)
│   └── MorphAgent_demo_english.srt     # Matching subtitle file
├── MANIFEST.md                         # Package contents and run strategy
└── SHA256SUMS.txt                      # File integrity checksums
```

The package does not include `.git`, Python caches, historical debug outputs, or a real `.env`. API keys are never written into source, commands, manifests, logs, or the delivery archive.

## 2. Recommended Review Order

For a first review, the recipient should follow this order:

1. Play `demo_video/MorphAgent_demo_english.mp4` first to see the full workflow.
2. Install the “UI + bundled demo” dependencies.
3. Launch the UI, choose **Load a previous run**, load the completed results, and inspect Features and Evidence.
4. On the Configure page, enter your own API credentials, then run the full pipeline with **Load demo dataset**.
5. `bash scripts/setup.sh` also creates **`morphagent_allen`** (UI default auto-mask backend for custom data). Set `MORPHAGENT_INSTALL_ALLEN=0` to skip.
This lets you verify the UI and result views first, then external APIs, and only then the longer full computation path.

## 3. System Requirements

- macOS, Linux, or Windows 10/11.
- Miniforge, Mambaforge, Miniconda, or Anaconda recommended.
- Python 3.10; the install scripts create a dedicated environment named `morphagent`.
- UI + bundled demo: at least 8 GB RAM and 5 GB free disk recommended; GPU not required.
- Custom data without masks: `setup.sh` installs `morphagent_allen` by default; if missing, segmentation is skipped gracefully and VLM still scores images without masks.
- A full new experiment needs a reachable OpenAI Chat Completions–compatible API.
- Models used for the Code + VLM route must support both text and image input; if the model does not support images, use Code only or configure a separate VLM.

All dependency names, versions, and install strategies are included under `dependencies/`, but platform-specific Python wheels, Conda binaries, CUDA drivers, and model weights are not bundled into the archive. The first environment creation still needs a network download so the same package can install the correct builds for macOS, Linux, and Windows.

Source and data are already packaged, so there is no need to `git clone` again.

## 4. Fastest Install Path

### 4.1 macOS / Linux

Open a terminal and enter the handoff directory:

```bash
cd /path/to/MorphAgent_handoff_20260719
bash scripts/setup.sh
```

After install, launch:

```bash
bash scripts/start_ui.sh
```

To use a different environment name:

```bash
MORPHAGENT_ENV_NAME=my_morphagent bash scripts/setup.sh
MORPHAGENT_ENV_NAME=my_morphagent bash scripts/start_ui.sh
```

### 4.2 Windows PowerShell

Prefer opening PowerShell from **Anaconda Prompt / Miniforge Prompt**, then run:

```powershell
cd C:\path\to\MorphAgent_handoff_20260719
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\start_ui.ps1
```

### 4.3 Manual Install

Without the scripts:

```bash
conda create -n morphagent -c conda-forge python=3.10 pip numpy "pyqt=5" qtpy -y
conda activate morphagent
# Core Qt / numpy also come from conda-forge in scripts/setup.sh and setup_windows.ps1.
python -m pip install -r dependencies/requirements-demo-ui.txt
python -m pip install -e MorphAgent
python scripts/verify_install.py --ui-smoke
python MorphAgent/launch_ui.py
```

`requirements-demo-ui.txt` is the recommended dependency set for handoff review: it runs the UI, LLM/VLM API routes, the Tau demo with existing masks, Features, and Evidence, without downloading segmentation stacks.

## 5. Optional: Segmentation for custom data

Masks under each sample’s `segmentation/` folder are always reused when present. When masks are missing, the UI pipeline defaults to **Allen** (`SEGMENTATION_BACKEND=allen`, conda env `morphagent_allen`). If that environment is not installed, segmentation is skipped for those samples and the run continues.

```bash
conda env create -f dependencies/environment-allen-optional.yml
```

Cellpose/PyTorch remains available as an alternate backend (`SEGMENTATION_BACKEND=cellpose`) via `dependencies/requirements-segmentation-optional.txt` when you explicitly need it.

## 5b. Dataset folder layout

```text
project_or_data_root/
  dataset/                 # or place sample folders directly at the root
    sample_a/
      image.tif            # primary image
      slices/              # optional VLM-ready views
      segmentation/        # optional masks (reused when present)
    sample_b/
      ...
  expert_knowledge/        # optional
  deep_research/           # optional prepared reports
  RAG/                     # optional literature PDFs/XML
```

Recommend **≥5 samples**. Fewer than 5 shows a Configure warning (validation uniqueness) but does not block Run.

## 6. First Launch and API Configuration

The window maximizes automatically on launch.

1. On the home page, click **Start a discovery run**.
2. Under **1 · Data**, click **Load demo dataset**, or browse to a folder that contains `dataset/<sample>/*.tif`.
3. The system fills in the demo samples (`WT_1`–`WT_5` wild-type, `MU_1`–`MU_5` mutant), paired `metadata.csv`, the data description, and a default biological question.
4. Under **3 · Model API**, fill Base URL, API key, and Model (leave fields blank — no invented defaults). Credentials are applied automatically when you click **Run MorphAgent** (no separate Save step). Optional VLM fields can stay empty to reuse the LLM connection. Free tip: [Free AI APIs 2026](https://aicosthub.com/guides/free-ai-apis-2026).
5. Under **4 · Analysis**, optionally keep **Enable feature validation** on and point at a metadata CSV (`sample_id` + group/label). The demo already ships `demo/data/metadata.csv`.
6. **Use the same connection for image scoring** is unchecked by default; check it to hide separate VLM fields.
7. Advanced **Config** defaults to temperature **0** (reproducible Code + VLM).

Do not forward the generated `.env` to others. For a second handoff, delete `MorphAgent/.env` and keep only `.env.example`.

The install scripts create a local `.env` from `.env.example` on the recipient’s machine, with only the API key still missing. The handoff package itself does not include that file.

## 7. Running the Bundled Tau Demo

Recommended settings:

- Data: **Load demo dataset**
- Biological question: use the auto-filled Tau aggregation question
- Analysis route: Code + VLM
- Knowledge sources: Expert notes, Deep research, Literature / RAG (demo digests prepared folders only — no auto PubMed / auto deep-research API)
- Scale: **1 round × 5 candidates · target 5** (advanced **Config** panel for temperature, rounds, workers)
- Masks: internal reuse when present; Allen only if missing (demo already has masks)
- Reproducibility: fixed on by the UI
- Runtime: a **1-round demo run typically takes about 5–30 minutes**, depending on how fast the configured model responds

**Demo vs custom knowledge paths**

| Source | Deep research | Literature / RAG |
|--------|---------------|------------------|
| Load demo dataset | Digest `demo/data/deep_research/` only | Precomputed RAG cache + local `RAG/` |
| Your own dataset | `--auto-deep-research` when checked | `--auto-literature-retrieval --pubmed-max-results 10` when checked |

Auto steps log failures and continue so a knowledge fetch miss does not abort the whole run.

After clicking **Run MorphAgent** in the lower right, the Run page shows **Live run** with:

```text
Inspect → Prepare → Plan → Quantify → Validate → Export
```

Expect roughly **5–30 minutes** for one demo round; slower APIs or higher concurrency contention sit toward the upper end. For a quick UI review without waiting, load `completed_demo_run` from Home instead.

Logs are continuously written to `ui_console.log` in the current results directory. Default results location:

```text
MorphAgent/demo/data/results/run_ui_YYYYMMDD_HHMMSS/
```

After the run finishes:

- **Features**: inspect feature cards, Code/VLM routes, status, and validation scores.
- **Evidence**: select a feature and inspect related measurements, validation, and sources (shared run-level preview folders are not attached as per-feature images).

## 8. Inspect Existing Results Without Rerunning

This is the fastest UI review path and does not need an API:

1. Launch the UI.
2. On Home, click **Load a previous run**.
3. Select:

```text
MorphAgent/demo/data/results/completed_demo_run
```

4. The UI opens Features directly.
5. Use the left navigation to continue to Evidence.

This directory contains 10 feature cards, two rounds of validation results, feature outputs for 5 samples, knowledge summaries, segmentation summaries, and image previews.

## 9. Post-Install Self-Check

```bash
conda activate morphagent
python scripts/verify_install.py --ui-smoke
```

The self-check confirms:

- Core Python/Qt modules can be imported;
- All 10 demo samples, originals, VLM slices, and masks are present;
- The completed run can be parsed into 10 feature cards;
- Home/Configure/Run/Features/Evidence can initialize in offscreen mode;
- Whether the optional Torch/Cellpose segmentation components are installed.

To run repository tests:

```bash
conda activate morphagent
python -m pip install -r dependencies/requirements-test.txt
cd MorphAgent
python -m pytest -q tests
```

Default tests cover only the current main program and Qt UI. `segmentation_allen/` uses a separate legacy Python 3.6 environment and should not be collected by the default Python 3.10 test run.

## 10. Common Issues

### Run button is disabled

Usually one of Data, Biological question, or API is incomplete. Click **Load demo dataset** first, then fill Model API (applied automatically on Run).

### API returns 404

If Base URL is missing `/v1`, MorphAgent now retries once after appending `/v1`. Still confirm the model name matches the provider console exactly.

### Code route works, but VLM route fails

Confirm the VLM Base URL / API key / model, and that the endpoint accepts image inputs. You can also temporarily choose **Code only**.

### Failure when Allen segmentation is missing

Install `dependencies/environment-allen-optional.yml` if custom samples have no masks. Without it, those samples skip segmentation and the run continues.

### Literature / RAG PDF step is slow or mentions PaddleX

Demo and UI default to **lightweight PDF text extract** (PyMuPDF) → LLM summary. PaddleX is optional and **not** required for a smooth demo. Custom Literature / RAG runs download PubMed PDFs then use the lite path (`RAG_PDF_BACKEND=lite` by default). Only if you need layout OCR for scanned PDFs: install PaddleX (see `requirements-extra-optional.txt`) and set `RAG_PDF_BACKEND=paddlex`.

### UI fails to open or Qt plugin errors

Confirm you launch with Python from the `morphagent` environment, not system Python:

```bash
conda run --no-capture-output -n morphagent python MorphAgent/launch_ui.py
```

On Linux headless servers you can only run `--ui-smoke` offscreen checks; an interactive window cannot be shown.

## 11. Run Strategy Summary

- The UI does not rewrite the scientific pipeline; it calls `main.py` in the same directory.
- `.env` is the shared model configuration source for both UI and CLI.
- The backend runs as a separate subprocess; merged stdout/stderr is shown live and written to `ui_console.log`.
- Each run uses a separate results directory; artifacts are kept on failure or cancel.
- The bundled demo reuses existing masks by default, so Allen/Cellpose are not triggered.
- Demo knowledge digests prepared folders only; custom data can auto deep-research / PubMed (max 10).
- The precomputed RAG cache avoids PDF parsing on the first demo run.
- Reproducibility is fixed on, with random seed 42.
- Code ReAct retries default to `CODE_MAX_RETRIES=3`.
- The home page can load completed results directly for quick Features/Evidence review.

## 12. Delivery Security Notes

- This package does not include an API key.
- `MorphAgent/.env` is not in this package.
- Logs and manifests do not store secrets.
- After the recipient first saves API settings, a local `.env` is created; delete it before sharing again.
- `SHA256SUMS.txt` can be used to verify files were not corrupted in transit.
