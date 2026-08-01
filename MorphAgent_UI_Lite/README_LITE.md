# MorphAgent UI Lite

Single-environment handoff of the MorphAgent desktop UI.

| Full `MorphAgent_UI` | **Lite** |
|----------------------|----------|
| `morphagent` + `morphagent_sandbox` + optional `morphagent_allen` | **`morphagent_lite` only** |
| Auto Allen when masks are missing | **Never auto-segments**; reuse user masks if present |
| Conda + pip hybrid for Qt/science | `conda create` (Python only) then **mostly pip** |

UI flow is unchanged: **Home → Configure → Run → Features → Evidence**.

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
- For automatic Allen segmentation, use the full [`MorphAgent_UI`](../MorphAgent_UI/) package or Docker Compose there.

## What setup does

1. `conda create -n morphagent_lite python=3.10 pip`
2. `pip install -r dependencies/requirements-lite.txt` (includes PyQt5)
3. `pip install -e MorphAgent`
4. Writes Lite defaults into `MorphAgent/.env` (`CONDA_ENV=morphagent_lite`, `SEGMENTATION_BACKEND=none`)

If pip PyQt fails on your platform, setup retries with `conda install -c conda-forge pyqt=5`.

## Verify

```bash
conda run -n morphagent_lite python scripts/verify_install.py
```
