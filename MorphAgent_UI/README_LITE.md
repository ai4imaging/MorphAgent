# MorphAgent Desktop UI

Install and open the **new Codex-style desktop workspace** in one Conda
environment. The standard setup includes Qt6/WebEngine and the lightweight
analysis dependencies; no separate browser or second environment is required.

Everything the UI needs lives in this one folder: the analysis pipeline, the UI
package, the bundled demo, the installation scripts and the frontend assets.

```text
MorphAgent_UI/
├── launch_desktop_ui.py   # default desktop workspace entry point
├── launch_web_ui.py       # optional browser entry point
├── launch_ui.py           # legacy Qt5 interface (still available)
├── main.py                # analysis pipeline shared with the CLI
├── src/morphagent_ui/     # UI code, artwork, and paper knowledge base
├── design-preview/        # frontend used by desktop/browser workspaces
├── scripts/               # macOS/Linux and Windows setup and launch
├── dependencies/          # analysis + Qt6/WebEngine requirements
├── demo/                  # bundled data, summaries, and completed results
└── .env                   # legacy/CLI configuration only
```

The environment name remains `morphagent_lite`. Installation uses an editable
package rooted at this folder, so subsequent code updates use the same source.

UI flow: **Settings → Design → confirm configuration → inline progress → Visualize**.
**Compute** applies all available feature scripts from an uploaded previous run
to a new dataset. Both workflows require confirmation before execution and save
results automatically with a download link; there is no separate Run or Save page.
Enter your own API connection in Settings; the new workspace keeps keys only in
memory for the current session and does not read saved `.env` keys.
See [README.md](README.md) for the complete desktop workflow.

## Supported

- Qt6 desktop workspace, with the old Qt5 UI retained as a separate launcher
- Bundled Tau demo dataset
- Code / VLM API runs
- PDF/DOCX/TXT/Markdown knowledge attachments beside the dataset upload
- Required feature count, Code/VLM routes, and Ultra fast / Fast / Detailed modes
- Saved-code computation on new data, automatic exports, and feature histograms
- Reuse existing masks (`SEGMENTATION_BACKEND=none`)

## Lightweight knowledge and optional backends

The new workspace uses explicitly uploaded knowledge only. It does not offer
background Deep Research / Literature toggles or perform live literature search.
PDF text extraction uses `pypdf`; scanned documents need OCR before upload.

The retained **legacy Qt5 interface** has a separate lightweight path. When its
Deep Research or Literature / RAG option is enabled:

1. use a supplied/prepared summary txt when present;
2. otherwise ask the configured LLM to synthesize the corresponding summary from the biological question and dataset description;
3. cache it under `<project>/.knowledge_precomputed/` and inject it into feature-planning prompts;
4. if the model call fails, log the error and continue without that source rather than aborting the run.

The literature synthesis is explicitly instructed not to claim live retrieval or invent citations.

The lightweight setup skips these heavy or optional backends:

- Heavy PDF parsing (`pymupdf`)
- Auto PubMed / literature download
- Separate deep-research provider/model
- Automatic Allen / Cellpose segmentation

Prepared demo files:

- `demo/precomputed/expert_knowledge_summary.txt`
- `demo/precomputed/deep_research_summary.txt`
- `demo/precomputed/rag_knowledge_summary.txt`

For the full command-line workflow and its optional dependencies, see the
[CLI guide](../MorphAgent_CLI/README.md). The UI selects
lightweight knowledge processing for its child pipeline; direct CLI use retains
the full document/retrieval route.

## Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) / Anaconda (`conda` on PATH)
- Network for the first pip install
- An OS/CPU/Python combination with PySide6 6.10.2 wheels; macOS 13+ for the pinned
  macOS wheels. Windows/Linux still need real-machine display/driver testing.

## Install

### macOS / Linux

```bash
git clone https://github.com/ai4imaging/MorphAgent.git
cd MorphAgent/MorphAgent_UI
bash scripts/setup.sh
bash scripts/start_ui.sh
```

### Windows

1. Open `MorphAgent_UI\scripts\` in Explorer.
2. Double-click **`setup_windows.bat`**.
3. Double-click **`start_ui_windows.bat`**.

Or from Anaconda Prompt / PowerShell inside `MorphAgent_UI\`:

```powershell
.\scripts\setup_windows.bat
.\scripts\start_ui_windows.bat
```

Already installed? Run the same setup command once to add the desktop packages
to your existing `morphagent_lite` environment. **Do not recreate it for this
upgrade.** Setup does not delete experiment history or results.

Only if you intentionally want to delete and recreate the selected environment:

```bash
MORPHAGENT_RECREATE_ENVS=1 bash scripts/setup.sh
```

## Segmentation behavior

- Samples with `dataset/<sample>/segmentation/*` masks → reused.
- Samples without masks → skipped (`skipped_no_backend`); Code/VLM still run without `seg`.
- For automatic segmentation, install the optional dependencies described in the [CLI guide](../MorphAgent_CLI/README.md).

## What setup does

1. Best-effort Anaconda ToS accept (so plugins / libmamba can stay enabled)
2. `conda create -n morphagent_lite python=3.10 pip` from **defaults** only (tiny solve)
3. `pip install -r dependencies/requirements-lite.txt` (numpy / scipy / PyQt5 / … — **not** via conda). This also includes `requirements-desktop.txt`, installing **PySide6 6.10.2 + matching Essentials/Addons/WebEngine/shiboken6** automatically.
4. `pip install -e .` (this folder)
5. Preserves existing API values while updating legacy/CLI defaults in `.env` (`CONDA_ENV=morphagent_lite`, `SEGMENTATION_BACKEND=none`). The new workspace still requires users to enter their API connection in Settings.
6. Checks Qt6/WebEngine and legacy Qt5 in separate processes, analysis imports,
   desktop source files and bundled demo outputs. Installation failure stops the
   script; success is not printed until these checks pass.

If you installed an earlier nested version, rerun `bash scripts/setup.sh` (or
`setup_windows.bat`) once to update the editable-package path. It reuses the
existing environment. Existing root `.env` values are preserved for legacy/CLI
use; they are not loaded by the new desktop workspace.

If the tiny `conda create` fails (ToS / old solver), setup retries once with classic solver — still **only** `python` + `pip`. It never runs a conda-forge mega-solve of PyQt/numpy/scipy (that path has crashed old `conda.exe` with `0xc0000005`).

If pip PyQt fails, setup retries `pip install PyQt5` (not `conda install` from conda-forge).

## Windows / conda troubleshooting

| Symptom | What Lite does |
|---------|----------------|
| Old conda (e.g. 23.7) forced classic solver on a huge conda-forge list | Lite does **not** conda-install the science stack; only python+pip |
| `CONDA_NO_PLUGINS` + `CONDA_SOLVER=classic` always on | Removed as default; classic is fallback for create only |
| SSL / `CondaSSLError` while fetching indexes | Create uses defaults; science packages come from pip (`--retries 5`) |
| `conda.exe` exit `0xc0000005` during solve | Avoided by not solving PyQt+numpy+… through conda |

Recommended: Miniconda **≥ 23.9** (libmamba default). If create still fails, upgrade conda or retry on a stable network.

## Verify

```bash
conda run -n morphagent_lite python scripts/verify_install.py
```

The import/file checks do not guarantee end-to-end API availability or scientific
run success. The old Qt5 interface remains available from `MorphAgent_UI/`:

```bash
conda activate morphagent_lite
python launch_ui.py
```

Do not import Qt5 and Qt6 into the same Python process. Keeping both packages in
the same Conda environment is supported by these separate launchers.
