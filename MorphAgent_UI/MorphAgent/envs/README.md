# Environments

MorphAgent ships **one** unified environment for everything, plus an optional
legacy environment for the Allen segmentation backend.

| File | Env name | Python | Purpose |
|------|----------|--------|---------|
| `environment.yml` | `morphagent` | 3.10 | **Main env** — agent (LangChain/LangGraph/OpenAI client) + sandbox scientific stack + Cellpose-SAM + PaddleX (CPU PDF parsing) + PubMed retrieval |
| `requirements.txt` | — | 3.10 | pip mirror of `environment.yml` for venv/pyenv/uv users |
| `requirements-optional.txt` | — | 3.10 | Optional extras: GPU paddlepaddle + local Qwen3-VL |
| `environment_allen.yml` | `morphagent_allen` | 3.6 | **Optional/legacy** — Allen `aicssegmentation` backend only |

There is **no local LLM/VLM** in this build: every model call goes through an
OpenAI-compatible HTTP API. Configure keys/URLs in `../config.py` or via the
environment variables listed in `../.env.example`.

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
import paddlex, paddle
print("torch CUDA available:", torch.cuda.is_available())
print("cellpose:", cellpose.version)
print("paddlex OK")
print("OK")
PY
```

## 2. Optional extras

Install on top of the `morphagent` env only if you need them:

```bash
conda activate morphagent
pip install -r envs/requirements-optional.txt
```

- **GPU PDF parsing** — the unified env ships CPU `paddlepaddle`. For faster
  PDF parsing on a CUDA machine:
  ```bash
  pip uninstall -y paddlepaddle && pip install paddlepaddle-gpu==3.0.0
  export PADDLEX_DEVICE=gpu:0
  ```
- **Local Qwen3-VL** — only if you want to run a VLM locally
  (`--vlm-api-provider qwen`) instead of an API VLM. Requires a suitable GPU.

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
