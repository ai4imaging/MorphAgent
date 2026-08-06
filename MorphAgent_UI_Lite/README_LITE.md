# MorphAgent UI Lite

**Tau demo trial.** Single-environment handoff: install lightly, open the Qt UI, Load demo, and run one Code/VLM round with prepared knowledge summaries.

| | **Lite** | Full `MorphAgent_UI` |
|--|----------|----------------------|
| Scope | Tau demo + Code/VLM API | Full research workflow |
| Env | **`morphagent_lite` only** | `morphagent` + sandbox (+ optional Allen) |
| Knowledge | Prepared txt, or configured-LLM synthesis from the biological question | Live PDF / PubMed / deep-research generation |
| Segmentation | Reuse masks if present; **never auto-seg** | Allen when masks missing |
| Install | `conda create` (Python) then slim pip | Conda + broader science stack |

UI flow: **Home → Configure → Run → Features → Evidence**.

## Supported

- Qt desktop UI
- Bundled Tau demo dataset
- Code / VLM API runs
- Expert notes via prepared txt
- Deep Research / Literature toggles: prepared txt first; otherwise two configured-LLM calls synthesize summaries from the biological question + dataset description
- Reuse existing masks (`SEGMENTATION_BACKEND=none`)

## Intentionally skipped (no popup)

Lite does **not** run heavy knowledge pipelines. When Deep Research or Literature / RAG is enabled:

1. use a supplied/prepared summary txt when present;
2. otherwise ask the configured LLM to synthesize the corresponding summary from the biological question and dataset description;
3. cache it under `<project>/.knowledge_precomputed/` and inject it into feature-planning prompts;
4. if the model call fails, log the error and continue without that source rather than aborting the run.

The literature synthesis is explicitly instructed not to claim live retrieval or invent citations.

- PDF parsing (`pymupdf`)
- Auto PubMed / literature download
- Separate deep-research provider/model
- Automatic Allen / Cellpose segmentation

Prepared demo files:

- `MorphAgent/demo/precomputed/expert_knowledge_summary.txt`
- `MorphAgent/demo/precomputed/deep_research_summary.txt`
- `MorphAgent/demo/precomputed/rag_knowledge_summary.txt`

For live literature fetch, PDF ingestion, or Allen segmentation, use [`MorphAgent_UI`](../MorphAgent_UI/).

## Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) / Anaconda (`conda` on PATH)
- Network for the first pip install

## Install

### macOS / Linux

```bash
cd MorphAgent_UI_Lite
bash scripts/setup.sh
bash scripts/start_ui.sh
```

### Windows

1. Open `MorphAgent_UI_Lite\scripts\` in Explorer.
2. Double-click **`setup_windows.bat`**.
3. Double-click **`start_ui_windows.bat`**.

Or from Anaconda Prompt / PowerShell inside `MorphAgent_UI_Lite\`:

```powershell
.\scripts\setup_windows.bat
.\scripts\start_ui_windows.bat
```

Recreate the env:

```bash
MORPHAGENT_RECREATE_ENVS=1 bash scripts/setup.sh
```

## Segmentation behavior

- Samples with `dataset/<sample>/segmentation/*` masks → reused.
- Samples without masks → skipped (`skipped_no_backend`); Code/VLM still run without `seg`.
- For automatic Allen segmentation, use the full [`MorphAgent_UI`](../MorphAgent_UI/) package.

## What setup does

1. Best-effort Anaconda ToS accept (so plugins / libmamba can stay enabled)
2. `conda create -n morphagent_lite python=3.10 pip` from **defaults** only (tiny solve)
3. `pip install -r dependencies/requirements-lite.txt` (numpy / scipy / PyQt5 / … — **not** via conda)
4. `pip install -e MorphAgent`
5. Writes Lite defaults into `MorphAgent/.env` (`CONDA_ENV=morphagent_lite`, `SEGMENTATION_BACKEND=none`)

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
