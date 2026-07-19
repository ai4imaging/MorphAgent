# MorphAgent UI Handoff Notes

This directory is a reviewable handoff package for the MorphAgent graphical interface. It includes the current UI source, backend pipeline, dependency manifests, install/launch/self-check scripts, Tau neuron demo data provided by the instructor, one completed run result, and the final English demo video.

## 1. Package Contents

```text
MorphAgent_handoff_20260719/
├── MorphAgent/                         # Current runnable source
│   ├── launch_ui.py                    # UI entry point
│   ├── main.py                         # Full backend pipeline entry point
│   ├── morphagent_ui/                  # Home/Configure/Run/Features/Evidence
│   ├── envs/                           # Original repo dependency files
│   ├── demo/data/                      # 5 Tau demo samples and knowledge materials
│   └── demo/data/results/
│       └── completed_demo_run/         # Completed results, ready to load and inspect
├── dependencies/                       # Dependency manifests organized by purpose
├── scripts/                            # One-click install, launch, and self-check
├── demo_video/
│   ├── MorphAgent_demo_english.mp4     # ~4 min 40 sec English demo
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
4. On the Configure page, enter your own API credentials, then run the full pipeline with **Use bundled Tau demo**.
5. Install Cellpose/PyTorch segmentation dependencies only if you need to regenerate segmentation masks.

This lets you verify the UI and result views first, then external APIs, and only then the longer full computation path.

## 3. System Requirements

- macOS, Linux, or Windows 10/11.
- Miniforge, Mambaforge, Miniconda, or Anaconda recommended.
- Python 3.10; the install scripts create a dedicated environment named `morphagent`.
- UI + bundled demo: at least 8 GB RAM and 5 GB free disk recommended; GPU not required.
- Regenerating Cellpose masks: NVIDIA GPU recommended; CPU works but is much slower.
- A full new experiment needs a reachable OpenAI Chat Completions–compatible API.
- Models used for the Code + VLM route must support both text and image input; if the model does not support images, use Code only or configure a separate VLM.

All dependency names, versions, and install strategies are included under `dependencies/`, but platform-specific Python wheels, Conda binaries, CUDA drivers, and model weights are not bundled into the archive. The first environment creation still needs a network download so the same package can install the correct builds for macOS, Linux, and Windows.

Source and data are already packaged, so there is no need to `git clone` again.

## 4. Fastest Install Path

### 4.1 macOS / Linux

Open a terminal and enter the handoff directory:

```bash
cd /path/to/MorphAgent_handoff_20260719
bash scripts/setup_macos_linux.sh
```

After install, launch:

```bash
bash scripts/start_ui.sh
```

To use a different environment name:

```bash
MORPHAGENT_ENV_NAME=my_morphagent bash scripts/setup_macos_linux.sh
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
conda create -n morphagent python=3.10 pip -y
conda activate morphagent
python -m pip install -r dependencies/requirements-demo-ui.txt
python -m pip install -e MorphAgent
python scripts/verify_install.py --ui-smoke
python MorphAgent/launch_ui.py
```

`requirements-demo-ui.txt` is the recommended dependency set for handoff review: it runs the UI, LLM/VLM API routes, the Tau demo with existing masks, Features, and Evidence, without downloading the larger Cellpose/PyTorch segmentation stack.

## 5. Optional: Full Segmentation Support

If you need **Regenerate Cellpose masks** in Configure, install on top of the base environment:

```bash
conda activate morphagent
python -m pip install -r dependencies/requirements-segmentation-optional.txt
python scripts/verify_install.py --ui-smoke
```

Or use the full environment provided by the repo:

```bash
conda env create -f dependencies/environment-full.yml
conda activate morphagent
python -m pip install -e MorphAgent
```

Allen `aicssegmentation` is a legacy, separate optional backend. Refer to `dependencies/environment-allen-optional.yml` only when explicitly needed. It is not required for the bundled Tau demo.

## 6. First Launch and API Configuration

The window maximizes automatically on launch.

1. On the home page, click **Start a discovery run**.
2. Under **1 · Data**, click **Use bundled Tau demo**.
3. The system fills in 5 samples, the data description, and a default biological question.
4. Under **3 · Model API**, fill in:
   - Base URL: template prefilled with `https://api.gpugeek.com/v1`; change this if you switch providers to their OpenAI-compatible endpoint.
   - API key: the recipient’s own key.
   - Model: template prefilled with `Vendor2/GPT-4o`; when switching providers, use the exact model name required by that provider.
5. If the same model supports image input, keep **Use the same connection for image scoring**.
6. Click **Save API configuration**. Settings are stored only on the local machine in `MorphAgent/.env`.

Do not forward the generated `.env` to others. For a second handoff, delete `MorphAgent/.env` and keep only `.env.example`.

The install scripts create a local `.env` from `.env.example` on the recipient’s machine, with only the API key still missing. The handoff package itself does not include that file.

## 7. Running the Bundled Tau Demo

Recommended settings:

- Data: Use bundled Tau demo
- Biological question: use the auto-filled Tau aggregation question
- Analysis route: Code + VLM
- Mask preparation: Reuse existing masks
- Knowledge sources: select Expert notes, Deep research, and Literature / RAG
- Fixed scale: 2 rounds × 5 candidates, target 10
- Reproducibility: fixed on by the UI; no user choice required

After clicking **Run MorphAgent** in the lower right, the Run page shows:

```text
Inspect → Prepare → Plan → Quantify → Validate → Export
```

Logs are continuously written to `ui_console.log` in the current results directory. Default results location:

```text
MorphAgent/demo/data/results/run_ui_YYYYMMDD_HHMMSS/
```

After the run finishes:

- **Features**: inspect feature cards, Code/VLM routes, status, and validation scores.
- **Evidence**: select a feature and inspect related measurements, validation, sources, and image context.

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
- All 5 demo samples, originals, VLM slices, and masks are present;
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

Usually one of Data, Biological question, or API is incomplete. Click **Use bundled Tau demo** first, then save Model API.

### API returns 404

Check whether Base URL is missing `/v1`, and confirm the model name matches the provider console exactly.

### Code route works, but VLM route fails

The current model may not support image input. Configure a separate multimodal VLM, or temporarily choose **Code only**.

### Failure after selecting Regenerate Cellpose masks

Install `requirements-segmentation-optional.txt` first. GPU drivers and the PyTorch build must match the recipient machine; that is why segmentation components are not shipped as offline binaries.

### RAG PDF parsing warns that PaddleX is missing

The bundled demo already includes a precomputed RAG cache, so normal use of **Use bundled Tau demo** does not need PaddleX. Only when switching to new PDF corpora and re-parsing should you consult `requirements-extra-optional.txt`.

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
- The bundled demo reuses existing masks by default, so Cellpose is not triggered.
- The precomputed RAG cache avoids PDF parsing on the first demo run.
- Reproducibility is fixed on, with random seed 42.
- The home page can load completed results directly for quick Features/Evidence review.

## 12. Delivery Security Notes

- This package does not include an API key.
- `MorphAgent/.env` is not in this package.
- Logs and manifests do not store secrets.
- After the recipient first saves API settings, a local `.env` is created; delete it before sharing again.
- `SHA256SUMS.txt` can be used to verify files were not corrupted in transit.
