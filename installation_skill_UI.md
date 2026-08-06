# MorphAgent UI — Agent Operating Guide (installation_skill_UI)

> Audience: an autonomous coding agent (Codex / Claude Code / Cursor / similar)
> that must **install and launch the MorphAgent desktop UI** on a new machine.
> Follow the steps in order. Each step has a concrete verification you MUST run
> before moving on. Do not skip verifications. Prefer failing loudly and fixing,
> over guessing.
>
> Related guide: [`installation_skill.md`](installation_skill.md) covers the
> **CLI / `main.py` pipeline** (Cellpose-SAM unified env). This file covers the
> **Qt desktop UI** under `MorphAgent_UI/`. Use this file when the user asks to
> install / start the GUI, demo app, or “MorphAgent UI”.

## 0. What the UI is (mental model)

`MorphAgent_UI/` is a handoff package that wraps the same MorphAgent pipeline
(`main.py`) behind a Qt app: **Home → Configure → Run → Features → Evidence**.

Layout (paths relative to the **git repo root**):

```text
MorphAgent/                         # git clone root
├── installation_skill.md           # CLI agent guide
├── installation_skill_UI.md        # THIS file
└── MorphAgent_UI/                  # UI handoff root — all scripts run FROM HERE
    ├── scripts/
    │   ├── setup.sh / setup_windows.bat|.ps1
    │   ├── start_ui.sh / start_ui_windows.bat|.ps1
    │   └── verify_install.py
    ├── dependencies/               # requirements-demo-ui.txt, optional Allen yml
    └── MorphAgent/                 # nested runnable app (NOT the git root)
        ├── launch_ui.py            # UI entry point
        ├── main.py                 # pipeline the UI launches
        ├── morphagent_ui/          # Qt pages + free-demo API helper
        └── demo/data/              # 10 Tau samples + completed_demo_run
```

**Critical invariants:**

1. **Working directory for install/launch scripts is always `MorphAgent_UI/`.**
   Relative paths like `scripts/setup.sh` mean `MorphAgent_UI/scripts/setup.sh`.
2. **No local LLM/VLM weights.** All model calls go through an OpenAI-compatible
   HTTP API (configured in the UI or via `MorphAgent_UI/MorphAgent/.env`).
3. **Two conda envs for the UI path (plus optional Allen):**
   - `morphagent` — Qt desktop UI + agent (`main.py` orchestration, LangChain).
   - `morphagent_sandbox` — **frozen** scientific stack for agent-generated
     `extract()` code only. Runtime fix scripts must **not** pin/change versions
     or reinstall numpy/scikit-image/scipy/….
   - `morphagent_allen` — optional Allen segmentation.
4. **UI segmentation default is Allen** (`SEGMENTATION_BACKEND=allen`).
5. **UI code / both runs disable the VLM critic** (`--disable-critic-agent`).

## 1. Prerequisites

- macOS, Linux, or Windows 10/11 with a display (the UI is a desktop Qt app).
- **conda** available: Miniforge / Miniconda / Anaconda.
- Network for the first install (conda-forge + pip downloads).
- Optional for a live pipeline run: OpenAI-compatible LLM + multimodal VLM keys.
  You can still open the UI and browse the completed demo run without keys.

```bash
# MUST pass
command -v conda && conda --version
```

If `conda` is missing: install Miniforge
(https://github.com/conda-forge/miniforge) or Miniconda, then open a new shell
(Windows: Anaconda/Miniforge Prompt). On Windows the `.bat` wrappers also
auto-search common install paths when PATH is empty.

## 2. Enter the UI handoff root

From the **git repo root** (the directory that contains both `MorphAgent_UI/`
and this file):

```bash
cd MorphAgent_UI
pwd   # must end with .../MorphAgent_UI
ls scripts/setup.sh scripts/start_ui.sh MorphAgent/launch_ui.py
```

On Windows PowerShell:

```powershell
cd MorphAgent_UI
Get-Location
dir scripts\setup_windows.bat, scripts\start_ui_windows.bat, MorphAgent\launch_ui.py
```

Expected: all listed files exist. If `MorphAgent/launch_ui.py` is missing, you
are in the wrong directory (do not confuse git-root `MorphAgent` naming with the
nested `MorphAgent_UI/MorphAgent/` app tree — there is **no** top-level app
folder named only `MorphAgent/` for the UI; the runnable tree is nested).

> Note: the git repo root is often also named `MorphAgent` after clone. The UI
> app lives at `MorphAgent/MorphAgent_UI/MorphAgent/`. Always `cd` into
> `MorphAgent_UI` before running scripts.

## 3. Install environments (MUST)

### 3a. macOS / Linux

```bash
cd MorphAgent_UI
bash scripts/setup.sh
```

What this does:

- Creates / updates conda env **`morphagent`** (Python 3.10) with conda-forge
  **PyQt5 / qtpy**, then pip-installs `dependencies/requirements-demo-ui.txt`
  and editable `MorphAgent_UI/MorphAgent` (UI + agent).
- Creates / updates conda env **`morphagent_sandbox`** with the frozen
  scientific stack from `dependencies/requirements-sandbox.txt` (feature
  `extract()` only — no Qt / LangChain).
- By default also creates optional **`morphagent_allen`** (Allen segmentation).
  Skip with: `MORPHAGENT_INSTALL_ALLEN=0 bash scripts/setup.sh`
- Custom UI env name: `MORPHAGENT_ENV_NAME=my_morphagent bash scripts/setup.sh`
- Custom sandbox name: `MORPHAGENT_SANDBOX_ENV_NAME=my_sandbox bash scripts/setup.sh`

### 3b. Windows (preferred: double-click / `.bat`)

From Explorer, open `MorphAgent_UI\scripts\` and double-click:

1. **`setup_windows.bat`** — one-click install (ExecutionPolicy Bypass, conda
   auto-discovery, UTF-8 for pip, **conda `pyqt=5` only** — pip PyQt5 is
   filtered out on purpose).

Or from Anaconda/Miniforge Prompt / PowerShell, still inside `MorphAgent_UI\`:

```powershell
.\scripts\setup_windows.bat
```

Do **not** use Explorer “Run with PowerShell” on the `.ps1` alone — Windows
blocks unsigned scripts. Always go through the `.bat`.

### 3.1 Verify install (MUST pass)

```bash
cd MorphAgent_UI
conda run --no-capture-output -n morphagent python scripts/verify_install.py --ui-smoke
```

Expected: prints `[OK]` lines for imports, demo data (10 samples), completed
run cards, and an offscreen UI smoke test. Non-zero exit → fix before launch.

Lightweight import-only check if you need a faster signal:

```bash
conda run -n morphagent python - <<'PY'
import importlib
need = ["PyQt5", "qtpy", "numpy", "pandas", "skimage", "langchain_core",
        "langchain_openai", "langgraph", "openai", "dotenv"]
missing = []
for m in need:
    try: importlib.import_module(m)
    except Exception as e: missing.append((m, repr(e)))
from PyQt5 import QtCore
print("PyQt5", QtCore.PYQT_VERSION_STR, "MISSING:", missing or "none")
assert not missing
print("UI ENV OK")
PY
```

## 4. Launch the UI (MUST)

### 4a. macOS / Linux

```bash
cd MorphAgent_UI
bash scripts/start_ui.sh
```

### 4b. Windows

Double-click `MorphAgent_UI\scripts\start_ui_windows.bat`, or:

```powershell
cd MorphAgent_UI
.\scripts\start_ui_windows.bat
```

Both paths run:

```text
conda run --no-capture-output -n morphagent python MorphAgent/launch_ui.py
```

### 4.1 Verify launch (MUST)

- A MorphAgent window appears with navigation: Home / Configure / Run /
  Features / Evidence (exact labels may vary slightly).
- If the process exits immediately, re-run `verify_install.py --ui-smoke` and
  read the traceback. Common causes: wrong cwd, missing `morphagent` env, broken
  Qt (Windows: re-run `setup_windows.bat` so conda `pyqt=5` is the only Qt stack).

Headless agents without a display: you cannot interactively drive the GUI, but
you **must** still complete install + `verify_install.py --ui-smoke`. Tell the
user the UI is installed and they should run `scripts/start_ui.sh` (or the
Windows `.bat`) on a machine with a desktop session.

## 5. First actions inside the UI (for you or the user)

Recommended order:

1. **Load a previous run** (Home) → open  
   `MorphAgent_UI/MorphAgent/demo/data/results/completed_demo_run`  
   → browse **Features** and **Evidence**. **No API key required.**
2. **Configure** → **Load demo dataset** (10 Tau samples `WT_1`–`WT_5` /
   `MU_1`–`MU_5`, metadata already wired for validation).
3. **API credentials** on Configure (required only for a new Run):
   - Fill Base URL / API key / Model, **or**
   - Click **Use free restricted API** (token-limited; scale locks to
     **1 round × 5 candidates · target 5**).
   - Credentials are applied on **Run MorphAgent** and written to
     `MorphAgent_UI/MorphAgent/.env` (never commit this file).
4. Click **Run MorphAgent**. Demo scale defaults to a small pilot; free-API mode
   stays locked at 1×5.

Models for Code + VLM must accept **image** inputs (or set method to Code only /
configure a separate VLM).

## 6. Custom data layout (if not using the demo)

Select the **parent folder that contains `dataset/`**:

```text
<selected_folder>/
  dataset/
    sample_1/
      image.tif          # or .tiff / .png / …
      slices/            # optional 2D views for VLM
      segmentation/      # optional masks; reused when present
    sample_2/
      ...
  metadata.csv           # optional; enable Feature validation in Configure
```

- UI auto-segmentation for missing masks: **Allen** via `morphagent_allen`.
  If that env is missing, segmentation is skipped and the run still continues.
- Demo masks are already present under each sample’s `segmentation/`.

## 7. How the UI invokes the pipeline (agent notes)

When Run is clicked, the UI builds a `main.py` command from Configure and injects
env vars such as:

- `SEGMENTATION_BACKEND=allen`
- `SEGMENTATION_CONDA_ENV=morphagent_allen`
- `CONDA_ENV=morphagent_sandbox` (feature code sandbox; never the Qt UI env)
- `ENABLE_CRITIC_AGENT=false` when method is `code` or `both`
- scale: `NUM_ROUNDS` / `FEATURES_PER_ITERATION` / `TARGET_FEATURE_COUNT`

Do **not** re-enable the critic for UI demos unless the user explicitly asks.
Do **not** switch the UI path to Cellpose unless asked — CLI Cellpose setup is
documented in `installation_skill.md`, not here.
Do **not** let fix agents reinstall numpy/scikit-image into `morphagent` or
`morphagent_sandbox`; the UI blocks version pins and core-stack mutations.

## 8. Troubleshooting / known pitfalls

- **`conda: command not found`** → install Miniforge/Miniconda; on Windows prefer
  `setup_windows.bat` (auto-finds common install roots).
- **PowerShell “running scripts is disabled”** → use `setup_windows.bat` /
  `start_ui_windows.bat`, never raw Explorer “Run with PowerShell” on `.ps1`.
- **PyQt5 / Qt import errors on Windows** → pip PyQt5 conflicts with conda Qt.
  Re-run `setup_windows.bat` (it strips pip PyQt5 and reaffirms conda `pyqt=5`).
- **`verify_install.py` fails on demo data** → ensure you are in
  `MorphAgent_UI/` and `MorphAgent/demo/data/dataset` has 10 sample folders.
- **UI opens but Run fails with 401** → fix Base URL / API key / model on
  Configure, or use **Use free restricted API** for a limited smoke run.
- **Allen env missing** → demo with bundled masks still works; for custom data
  without masks, re-run setup without `MORPHAGENT_INSTALL_ALLEN=0`, or accept
  skipped segmentation.
- **Wrong directory** → if `scripts/setup.sh` is not found, you are not in
  `MorphAgent_UI/`.

## 9. From-scratch checklist (copy/paste)

```bash
# 0. repo root → UI handoff
cd MorphAgent_UI

# 1. install (macOS/Linux)
bash scripts/setup.sh
# Windows instead:
#   .\scripts\setup_windows.bat

# 2. verify (MUST)
conda run --no-capture-output -n morphagent python scripts/verify_install.py --ui-smoke

# 3. launch
bash scripts/start_ui.sh
# Windows instead:
#   .\scripts\start_ui_windows.bat
```

Inside the UI: **Load a previous run** (no key) → then **Load demo dataset** +
API (or free restricted API) → **Run MorphAgent**.

Done. If install verification and launch succeed, the MorphAgent UI is correctly
installed and ready for demo or custom runs.
