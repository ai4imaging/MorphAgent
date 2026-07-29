# Environments

MorphAgent ships **one** unified environment for everything, plus an optional
legacy environment for the Allen segmentation backend.

| File | Env name | Python | Purpose |
|------|----------|--------|---------|
| `environment.yml` | `morphagent` | 3.10 | **Main env** — agent (LangChain/LangGraph/OpenAI client) + sandbox scientific stack + Cellpose-SAM + pymupdf (lite PDF text) + PubMed retrieval |
| `requirements.txt` | — | 3.10 | pip mirror of `environment.yml` for venv/pyenv/uv users |
| `requirements-ui.txt` | — | 3.10 | Focused standalone Qt graphical workflow |
| `requirements-optional.txt` | — | 3.10 | Optional extras: PaddleX OCR + local Qwen3-VL |
| `environment_allen.yml` | `morphagent_allen` | 3.6 | **Optional/legacy** — Allen `aicssegmentation` backend only |

There is **no local LLM/VLM** in this build: every model call goes through an
OpenAI-compatible HTTP API. Configure keys/URLs in `../config.py` or via the
environment variables listed in `../.env.example`. `socksio` is included so
the OpenAI/httpx client can honor a configured SOCKS proxy.

## 1. Unified environment (recommended)

```bash
# from the repo root
conda env create -f envs/environment.yml
conda activate morphagent
```

Or, if you manage your own Python 3.10 environment:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r envs/requirements.txt
```

> **Slow download / behind a slow PyPI?** The `torch` + CUDA wheels are several
> GB. You can point pip at a faster mirror without editing the yaml, e.g.:
> ```bash
> PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
> PIP_EXTRA_INDEX_URL=https://pypi.org/simple \
> conda env create -f envs/environment.yml
> ```

Quick sanity check after install:

```bash
python - <<'PY'
import langchain_core, langgraph, openai, cellpose, torch
import numpy, pandas, skimage, cv2, tifffile, mahotas, bs4, requests
import fitz  # pymupdf
print("torch CUDA available:", torch.cuda.is_available())
print("cellpose:", cellpose.version)
print("pymupdf OK")
print("OK")
PY
```

### Graphical workflow (optional, recommended)

Install the focused Qt interface on top of the unified environment:

```bash
conda activate morphagent
pip install -e .
python launch_ui.py  # automatically reads ../.env
```

QtPy and PyQt5 are included in the unified environment and editable package
dependencies. `requirements-ui.txt` remains available for adding only the
standalone UI dependencies to an existing environment. This default does not
install or show napari. If interactive layer/canvas inspection is needed,
install `pip install -e ".[napari]"` and launch with
`python launch_ui.py --with-napari`; the editable package also registers
`morphagent_ui/napari.yaml` with napari's plugin engine.

## 2. Optional extras

Install on top of the `morphagent` env only if you need them:

```bash
conda activate morphagent
pip install -r envs/requirements-optional.txt
```

- **Optional PaddleX OCR** — default PDF path is pymupdf (lite). For layout OCR:
  ```bash
  pip install "paddlex[ocr]==3.3.10" paddlepaddle==3.0.0
  export RAG_PDF_BACKEND=paddlex
  export PADDLEX_DEVICE=cpu   # or gpu:0 after installing paddlepaddle-gpu
  ```
- **Local Qwen3-VL** — only if you want to run a VLM locally
  (`--vlm-api-provider qwen`) instead of an API VLM. Requires a suitable GPU.

The bundled teacher reference demo also reuses existing segmentation masks, so
it does not require Cellpose or a local GPU during that run. Keep the full
environment for new datasets that still need automatic segmentation.

## 3. Allen segmentation (optional / legacy)

```bash
conda env create -f envs/environment_allen.yml
conda activate morphagent_allen
pip install -e segmentation_allen        # vendored aicssegmentation
```

Then tell MorphAgent to use it for segmentation:

```bash
export SEGMENTATION_CONDA_ENV=morphagent_allen
```

> Python 3.6 is end-of-life; the pins in `environment_allen.yml` reflect a
> known-working install. Only create this env if you specifically need the
> Allen backend — the default Cellpose-SAM path needs none of it.
