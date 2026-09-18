# MorphAgent desktop workspace

This folder is self-contained: the analysis pipeline (`main.py`), the UI package
(`src/morphagent_ui/`), the frontend (`design-preview/`), the bundled Tau demo
(`demo/`), the installer (`scripts/`) and its requirements all live here. The
headless command-line pipeline is packaged separately in
[`../MorphAgent_CLI/`](../MorphAgent_CLI/). See [README_LITE.md](README_LITE.md)
for the folder layout and what the installer does.

This entry point opens the **new Codex-style interface** in a maximized native
window. Qt WebEngine embeds its own Chromium engine; Chrome/Safari/Edge need not
be installed or launched. Design, Compute, Visualize, Help, API settings, and
history are shared with the browser UI. Results save automatically; there is no
separate Save page.

## Standard installation (new or existing environment)

From the repository root:

```bash
cd MorphAgent_UI
bash scripts/setup.sh
bash scripts/start_ui.sh
```

On Windows, run `MorphAgent_UI\scripts\setup_windows.bat`, then
`MorphAgent_UI\scripts\start_ui_windows.bat`.

Setup now includes **PySide6 and WebEngine automatically**. It creates
`morphagent_lite` if missing, or updates that same environment if it already
exists. The default start script opens this new desktop workspace, not the
legacy Qt5 interface. A separate desktop environment is not needed.

Do not enable `MORPHAGENT_RECREATE_ENVS=1` or Windows `-Recreate` for a normal
upgrade: those explicit options remove and rebuild the selected environment.

## Optional: add only the desktop packages to an already-working environment

From the repository root, in Terminal or Windows Anaconda Prompt:

```bash
conda activate morphagent_lite
python -m pip install -r MorphAgent_UI/dependencies/requirements-desktop.txt
python -m pip check
```

This shorter upgrade adds PySide6 6.10.2, matching Essentials/Addons (including
WebEngine), and shiboken6. It does not reinstall the scientific stack, change Python, remove
PyQt5, or create another Conda environment. Do not import the legacy PyQt5 UI and
the new PySide6 window in the same Python process. These are separate launchers.

The pinned macOS wheel requires macOS 13 or later. The installation must have a
wheel compatible with your OS/CPU/Python. Qt WebEngine increases the environment
size and still needs platform-specific testing (GPU/display scaling included).
An embedded engine reduces browser-version differences, not every OS issue.
No system-wide Qt installation or C++ build tools are required when wheels exist.

This extra command is **not necessary after the updated standard setup**.
Setup verifies Qt6/WebEngine and legacy Qt5 imports in separate Python processes;
it does not make a paid API call or run a scientific experiment.

## Launch

```bash
cd MorphAgent_UI
conda activate morphagent_lite
python launch_desktop_ui.py
```

Or, from any working directory:

```bash
bash /path/to/MorphAgent/MorphAgent_UI/scripts/start_ui.sh
```

Windows: `MorphAgent_UI\scripts\start_ui_windows.bat`.
The earlier `start_desktop_ui.sh` and `start_desktop_ui_windows.bat` names remain
aliases for the same desktop launcher.

No browser is opened. The window starts maximized with normal system window
controls. Ctrl/Cmd +/- adjusts zoom; Ctrl/Cmd+0 resets it. The internal service
uses an automatically assigned free port and only listens on `127.0.0.1`.

Only one instance can own a workspace. If the browser version is still running,
stop its terminal with Ctrl+C before starting the desktop version. An in-progress
run will be stopped by that action; wait for it to finish if you need its results.
Desktop startup never kills another launcher or overwrites its history.

## Workflow and exit

- Use **Design**, **Compute**, and **Visualize** in the left sidebar. Settings
  stays at the bottom. There is no duplicate navigation across the top.
- Settings still requires your own API connection; no bundled/default credentials.
  Keys stay in memory for this service session. They must be entered after restart.
  Applied keys display `********`; **Change** lets you replace them without
  exposing the saved credential.
- **Design** → add a dataset, question and required **Feature number** (1–500) → submit
  → review the configuration → **Confirm and run**. Nothing starts before that
  confirmation. Missing feature count or API key opens a blocking reminder dialog.
  Back to edit / Escape preserves the draft without running anything.
  The confirmation always offers an optional **Deep Research** checkbox, which
  asks the configured model to write a background brief from your question. It
  works on its own or alongside uploaded knowledge documents.
  After confirmation the question moves to the top of this same page, with live
  run progress and output below. **Estimated remaining** updates from workload
  and observed loop progress; it is approximate and may change after API delays.
- Knowledge now lives beside **Add data**, not in Settings. **Upload knowledge**
  becomes **Knowledge attached** after adding PDF/DOCX/TXT/Markdown files. Click it
  to preview extracted text, download originals, add or remove files. Removing
  the last attachment restores the upload label; past run inputs are preserved.
- **Compute** → **Upload features** → choose a run's `feature/` folder → **Add data**
  for a new target dataset → submit → **Confirm and run**. The features to extract
  are already fixed by the previous run, so Compute asks no question.
  The previous-run picker starts at `.web_workspace/exports/`. Both exported
  `feature/` folders and older raw run folders are accepted. Cancel
  keeps your current attachments. Click the feature count to inspect the read-only
  list. All available code runs automatically; there are no selection checkboxes.
  VLM-only definitions are shown as unavailable for standalone computation.
  Missing API settings open the same blocking dialog as Design. Execution
  uses the saved scripts unchanged, without model calls or feature redesign; the
  source run's question is inherited as run context for VLM rescoring. Use trusted
  scripts and compatible images and masks. Progress, approximate ETA, and logs
  stay on Compute.
- Completed runs automatically save their timestamped folder and ZIP under
  `.web_workspace/exports/`. **Download results (.zip)** appears directly in
  Design or Compute and opens a native Save dialog, without silently overwriting a
  chosen file. Raw results remain under `.web_workspace/runs/<timestamp>/results/`.
  Packaging failures show **Retry save** inline.
- Upload uses a native folder picker. Loading previous results and selecting a
  Compute target also use native folder pickers, not pasted path prompts.
- **Visualize** accepts a run's `value/feature_value.csv` independently of History. It shows
  retained features with **All / Code / VLM** filters and a distribution histogram for the
  selected feature, without per-sample tables or extra validation panels.
- **Help** opens Ask MorphAgent. After entering your Model API settings, ask about
  the paper, supplementary material, figures, or implementation. Answers use
  bounded excerpts from the maintained manuscript and first-party code, with
  source labels. The conversation stays in this UI session and is not saved to
  History or disk.
- **History** lists actual analysis/computation runs. Opening an imported bundle
  for Compute or Visualize does not add a visible history row. Click **Design**
  or **Compute** to return to its independent draft, or to its active progress.
- Each **History** row has a trash button. Confirmation lists the workspace-owned
  run directory and exports that will move to the system **Trash / Recycle Bin**.
  Cancel/Escape changes nothing. Original datasets, external imported folders,
  and files still used by another run are preserved. Running/saving records and
  sources used by active computations cannot be deleted. Files can be recovered
  from the system Trash before it is emptied; there is no permanent-delete fallback.
- Closing during analysis asks before stopping it. Cancel keeps the analysis and
  window open. Confirm stops owned workers, preserves partial artifacts, and
  releases the local port and workspace lock. Keep the terminal open while using
  the app; force-killing it bypasses this cleanup.

The in-app renderer uses a nonpersistent profile and blocks external pages from
navigating inside the trusted UI. Deliberately clicked external http(s) links may
open in the system browser. The local view bypasses system proxies; model calls
continue to use Python's existing proxy environment.

This is a Python desktop launcher, **not yet a standalone .app/.exe installer**.
The browser fallback remains `python launch_web_ui.py`. The older, different Qt5
interface remains `python launch_ui.py`. See [README_WEB.md](README_WEB.md) for shared dataset and result formats,
and [README_LITE.md](README_LITE.md) for the packaging details of this folder.

## Engineering verification

Run Qt6 tests separately from tests that import the legacy PyQt5 UI:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p test_desktop_window.py -v
```

The macOS desktop is tested locally. Windows/Linux launch helpers are provided,
but those platforms need real-machine verification before claiming support there.
