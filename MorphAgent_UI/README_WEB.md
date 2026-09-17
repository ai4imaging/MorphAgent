# Codex-style local workspace

Prefer a desktop window without opening a browser? Use
`python launch_desktop_ui.py` after the one-time installation in
[README.md](README.md). Both entry points use this same workflow
and results; the desktop entry uses its own Qt WebEngine renderer.

The new browser UI calls the same root `main.py` and `reuse_code.py` used by the
desktop UI. It is not a simulated workflow. The original desktop launcher remains
available and unchanged.

## Start with your installed environment

From `MorphAgent_UI/`:

```bash
conda activate morphagent_lite
python launch_web_ui.py
```

Or use the helper script:

```bash
bash scripts/start_web_ui.sh
```

Windows (Anaconda Prompt):

```bat
conda activate morphagent_lite
python launch_web_ui.py
```

Alternatively run `MorphAgent_UI\scripts\start_web_ui_windows.bat`.
Open **http://127.0.0.1:8766**. Keep the terminal running. Ctrl+C stops the service
and its active analysis. Closing only the browser does not stop the analysis.
If the port is busy, use `python launch_web_ui.py --port 8767`.

For a fresh installation, follow the existing UI setup instructions first. The
updated `dependencies/requirements-lite.txt` includes `pypdf` for PDF references.
An existing installation needs only `python -m pip install 'pypdf>=5,<7'` for this
new PDF support. TXT/Markdown and DOCX extraction need no extra packages.

## First real run

1. Open **Settings → Model API**. Enter an OpenAI-compatible Base URL, key, and
   model; click **Apply settings**. There is no default/free API. The web runtime
   does not read `.env` or inherited model credentials. Connections stay in memory
   for this service session only; enter them again after restarting the service.
   Applied keys remain visibly masked as `********`. Use **Change** to replace
   one. The mask is display-only; the service never sends the stored key back.
2. In **Analysis**, choose Code, VLM, or both. Ultra fast =
   1 loop, Fast = 5, Detailed = 20. The pipeline can stop
   early according to its existing stopping rules. Reproducibility is enabled.
3. In **Design**, click **Upload knowledge** beside **Add data** to attach PDF,
   Word **.docx**, UTF-8 TXT or Markdown files (up to 20 MB each). The button becomes
   **Knowledge attached**. Click it again to inspect the list, preview extracted
   text, download originals, add files or remove them. Removing the last attachment
   restores **Upload knowledge**. Settings no longer has a Knowledge tab/switch;
   attached files automatically participate in the next discovery run. Text is
   extracted locally and fed into the existing expert-knowledge planning path.
   Scanned PDFs must be OCR'd first. No live literature search is implied. With no
   attachments, this workspace does not enable the background-knowledge route.
   Removal does not change knowledge already copied into a past/active run; local
   originals remain recoverable under `.web_workspace/references/`.
4. In **Design**, enter the biological question and choose the Tau demo or import
   your folder. Use `dataset/<sample>/image.tif`, one folder per sample. Existing
   `segmentation/` masks are reused; this Lite runtime does not generate new masks.
   Fill in **Feature number** beside the other composer controls. It starts blank
   and is required: enter a whole number from 1 to 500 (the existing UI target
   range). This becomes `--target-feature-count`; candidates per round are
   `ceil(feature number / selected loops)`, at least one. This is a planning target,
   not a guarantee of the number ultimately retained after validation.
5. Click the submit arrow in **Design**, or press Enter in the question field.
   A missing/invalid count or missing API connection opens a prominent blocking
   dialog; **Open Model API** takes you to the connection settings. After local
   preflight succeeds, **Review run configuration** lists the question, mode/loops,
   route, dataset, knowledge filenames, and feature count. The dialog also offers an
   optional **Deep Research** checkbox, which can be combined with attached knowledge
   files; when checked, the configured model writes a background brief from your
   biological question and the planner uses it. Only **Confirm and run**
   starts the analysis. The same page now shows your question at the top, with
   progress and logs below; there is no separate Run tab. **Back to edit**, close, or Escape leaves
   the draft unchanged and starts nothing. API keys are never shown in this summary.
   Real stdout/stderr streams into **Live output**. Model calls may take minutes;
   elapsed time and **Estimated remaining** remain visible while waiting. The
   estimate uses workload, elapsed time and loop/stage progress. It is approximate,
   can increase after slow model calls or retries, and is not a completion deadline.
   Failures stay failed in history.
6. **View results** (or History → **Visualize**) shows retained features only.
   Use **All / Code / VLM** to filter by extraction method, then click
   a feature to see its distribution histogram. Individual sample
   values and validation details are not displayed. From a fresh Design/Compute draft,
   **Visualize → Upload feature_value.csv** opens the default export folder: select
   `<run timestamp>/value/feature_value.csv`. No API key or new analysis is needed.
   Missing/non-finite values are excluded, not converted to zero.
7. On completion, the UI automatically creates a compact timestamped folder and
   ZIP. **Results saved → Download results (.zip)** appears on the same page.
   No extra Save click is required. Packaging failures preserve raw results and
   show **Retry save**. There is no separate Save tab. See the structure below.

## Help: ask about the paper

The **Help** page uses the same session-only Model API connection as Design.
Ask a question about the paper or code; Ask MorphAgent retrieves relevant
excerpts from the bundled manuscript/code knowledge base and replies with
source labels. Chat history is kept in memory for the current page session only.

When PDFs in the repository's `manuscript/` folder change, local Help refreshes
its workspace cache on the next question. Before distributing the updated UI,
refresh the bundled knowledge from `MorphAgent_UI/`:

```bash
PYTHONPATH=src python scripts/build_reviewer_knowledge.py \
  --manuscript-dir manuscript --code-root . \
  --output src/morphagent_ui/reviewer_knowledge/knowledge.json
```

This build includes the manuscript, SI, algorithm, prompt, supplementary tables,
feature lists, and selected first-party code. It excludes the cover letter,
virtual environments, result folders, and credentials. The running Help service
reloads the knowledge file after it is rebuilt.

## Compute saved features on new data

1. Open **Compute** and click **Upload features**. In the desktop app the
   native picker opens `.web_workspace/exports/`: select the timestamp-named
   run's `feature/` folder. The browser fallback asks for that folder path.
   Unzip a received ZIP first. Older raw `results/` directories remain supported. Loading does
   not execute anything; imported completion is labelled unverified.
2. Click **N features available** to inspect the read-only feature list. Every
   reusable feature runs automatically; no checkboxes or selection/clear controls
   are shown. Saved `extract.py` scripts are labelled **Replays saved code**, and
   saved VLM features — which never had a script — are labelled **Rescored by the
   VLM** and are scored again from their saved descriptions. Features with neither
   are labelled **Nothing reusable saved**. Search only filters the view.
3. Click **Add data** to upload a new target folder, or use the Tau demo. Enter
   your question in the composer. These inputs are independent of Design.
4. Submit with the arrow or Enter. Missing source, data, question, selected code,
   or API settings opens a blocking reminder. **Review run configuration** lists
   the question, Compute mode, source run, target data, and selected feature names.
   Only **Confirm and run** starts execution; cancelling preserves your draft.
   API configuration is required consistently with Design. Replaying saved code
   makes **no LLM/VLM calls** and receives no model credentials; selecting a saved
   VLM feature does score the new images through the VLM, so those runs need VLM
   credentials and are billed like a Design run. The prompt is saved as context
   and passed to VLM scoring; it never rewrites the historical scripts.
   All available individual extractors run, not a merged round script. Use
   trusted code; images and masks must match its requirements. No feature is
   redesigned and no new validation is performed.
5. Stay on Compute: the question appears at the top with progress, live output,
   elapsed time and an approximate remaining time beneath. Results are saved
   automatically and an inline download is provided. **New computation** starts
   another draft. A check mark means the process completed with
   usable measurements; Compute additionally requires every selected sample/feature
   measurement to be finite. A cross denotes failed, interrupted, stopped, empty,
   or partially completed runs; the text specifies which. A process completing
   does not establish biological validity, and missing measurements stay missing.

## Sidebar and History

The workflow navigation is on the left: **Design**, **Compute**, **Visualize**.
Settings stays at the bottom. The top bar does not duplicate these page buttons.
History lists actual Design/Compute runs, including failed attempts. Imported
bundles opened for inspection/computation are hidden from this list, and there
is no bundled-demo shortcut in the sidebar. Importing does not start an analysis.
Click **Design** or **Compute** to return to its independent draft. If that
workflow has an active run, its progress is shown instead. Viewing History never
overwrites draft inputs. There is no extra **New design** button on result pages.

Use the trash button on a History row to delete the run. The confirmation lists
the exact workspace-owned run folder and recorded export folders/ZIPs that will
be removed from their original paths. Cancel or Escape changes nothing.
Original datasets, external imported folders and files referenced by another
History entry are preserved. Running/saving jobs and sources used by active
Compute jobs cannot be deleted.

Files go to the operating system **Trash / Recycle Bin**, using the installed Qt
desktop dependency. They appear together in a `MorphAgent-deleted-run-*` folder
with a `manifest.json` recording the old History entry and original paths. No
workspace-local recycle bin is used. Recover files through the system Trash
before emptying it. If the system cannot recycle the files, deletion fails and
the files and History entry are restored; there is no permanent-delete fallback.
Folders left over from earlier record-only deletions are not bulk-deleted.

## Saved bundle

New runs are named `YYYYMMDD_HHMMSS_microseconds`, avoiding same-second collisions.
Completed runs automatically write to `.web_workspace/exports/`:

```text
<run timestamp>.zip
<run timestamp>/
  feature/
    feature_descriptions.csv
    <feature name>/
      code/
        extract.py (or definition.json)
  value/
    feature_value.csv
```

The descriptions table includes the run status and code availability. The value
table contains each retained feature's sample ID, value, description, and Code/VLM
method. Dropped features and per-feature value files are omitted. For VLM
features (or absent scripts), `code/definition.json` describes why no executable
script is available; no code is invented. Repeated saves make a numbered snapshot
instead of overwriting an existing export. Original raw results are preserved.
Raw pipeline output lives in `.web_workspace/runs/<run timestamp>/results/`.
Partially completed reuse runs are also packaged, explicitly labelled partial.

## Data and credentials

- Web API credentials stay only in process memory and are never written to `.env`,
  workspace JSON, logs, or exports. The star mask indicates a key already entered
  in this session. Leaving the replacement field blank keeps it; changing Base URL
  requires re-entering its key. Restarting clears the session and its masks.
- The legacy Qt5 desktop UI and CLI retain their separate configuration paths;
  this session-only behavior applies to the new browser and Qt6 workspace launchers.
- `.web_workspace/` stores imported datasets, extracted references, history, and
  isolated run inputs/results. It is git-ignored. Input datasets are copied before
  execution; originals are preserved. Copies need additional disk space.
  Launch checks space for the input copy plus a 1 GiB output reserve; larger
  datasets/runs may require substantially more space while running.
- **Compute → Upload features** accepts an exported `feature/` folder (or an
  older raw results folder). **Visualize → Upload feature_value.csv** accepts the
  exported `value/feature_value.csv` file.
- The server binds only to `127.0.0.1`, checks Host/Origin, and uses a per-session
  API token. It is a single-user local tool, not a remotely exposed web service.
- Enabled documents and relevant images are sent to the configured model during
  analysis. Generated code executes locally, as in the original UI.
- A stopped/crashed process can leave partial artifacts. They are retained for
  inspection; history loading is not automatic resumption of an interrupted run.

`design-preview/` can still be served statically for design review, but static
mode is explicitly marked **Design preview** and cannot perform real analysis.
