# MorphAgent — Agent Operating Guide (installation_skill)

> Audience: an autonomous coding agent (like you) that must **set up and run
> MorphAgent from scratch** on a new machine. Follow the steps in order. Each
> step has a concrete verification you MUST run before moving on. Do not skip
> verifications. Prefer failing loudly and fixing, over guessing.

## 0. What MorphAgent is (mental model)

MorphAgent is an LLM/VLM agent that **designs and extracts quantitative
morphological features** from microscopy images. Given a dataset (one folder
per sample) and a natural-language task, it: understands the dataset → (optionally)
segments cells → plans a feature list → **writes and executes Python feature
code** in a sandbox + **scores visual features with a VLM** → validates → writes
a `features.csv`.

**Critical invariant — there is NO local model in this build.** Every LLM and VLM
call goes through an **OpenAI-compatible HTTP API**. If you find yourself trying
to download model weights, STOP — that is only for the optional advanced local
Qwen3-VL path, which you should not use unless the user explicitly asks.

Frameworks: `langchain-core` (messages/tools), `langchain-openai` (`ChatOpenAI`),
`langgraph` (the agent graph). Segmentation default: **Cellpose-SAM**.

## 1. Environment setup

The repo ships a **single unified conda environment** (`morphagent`, Python 3.10)
that contains the agent, the code sandbox, and Cellpose-SAM. There is also an
**optional, separate** legacy env (`morphagent_allen`, Python 3.6) only for the
Allen `aicssegmentation` backend.

```bash
# from the repo root
conda env create -f envs/environment.yml      # creates env `morphagent`
conda activate morphagent
```

- **Slow/interrupted download?** The `torch` + CUDA wheels are several GB. Use a
  faster mirror WITHOUT editing the yaml (env vars only):
  ```bash
  PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  PIP_EXTRA_INDEX_URL=https://pypi.org/simple \
  conda env create -f envs/environment.yml
  ```
- If the target env name is already taken by an unrelated env, create under a
  different name with `conda env create -n <name> -f envs/environment.yml` and
  then set `CONDA_ENV=<name>` and `SEGMENTATION_CONDA_ENV=<name>` (see step 2).
- venv/uv users: `pip install -r envs/requirements.txt` into a Python 3.10 env.

**Do NOT re-pin the langchain packages.** They are a deliberately self-consistent
langchain **1.x** set (`langchain-core==1.4.9`, `langchain-openai==1.3.5`,
`langgraph==1.2.9`, `openai>=2.45,<3`). The old umbrella `langchain` /
`langchain-community` packages are intentionally absent — the code never imports
them, and their 0.3.x pins would create an unsatisfiable `langchain-core`
conflict. If you ever see a `langchain-core` ResolutionImpossible error, it means
someone re-added those packages; remove them again.

### 1.1 Verify the environment (MUST pass)

```bash
conda activate morphagent   # or your chosen env name
python - <<'PY'
import importlib
ext=['langchain_core','langchain_openai','langgraph','openai','cellpose','torch',
     'numpy','pandas','skimage','cv2','tifffile','mahotas','statsmodels','h5py',
     'bs4','lxml','scipy','sklearn','matplotlib','seaborn','PIL','yaml','dotenv']
missing=[]
for m in ext:
    try: importlib.import_module(m)
    except Exception as e: missing.append((m,repr(e)))
import torch, cellpose
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('cellpose', getattr(cellpose,'version',getattr(cellpose,'__version__','?')))
print('MISSING:', missing or 'none')
assert not missing, missing
print('ENV OK')
PY
```

Expected: prints a torch version, a cellpose version (4.x), `MISSING: none`,
`ENV OK`. `cuda True` if a GPU is present (Cellpose-SAM and code features run much
faster on GPU; VLM scoring only needs the API and works CPU-only).

Then confirm the app itself imports and the CLI starts:

```bash
cd <repo root>
python main.py --help        # must print usage without tracebacks
```

## 2. Configure the API (LLM + VLM)

No keys are committed. Configure via environment variables (preferred — keeps
secrets out of git) or by editing `config.py`.

```bash
cp .env.example .env
# edit .env, then:
source .env
```

Required variables (both an LLM and a VLM endpoint; they may be the same):

```bash
export LLM_BASE_URL="https://api.openai.com/v1"   # any OpenAI-compatible gateway
export LLM_API_KEY="sk-..."
export LLM_MODEL="gpt-4o"                          # text model: planning / code / review

export VLM_BASE_URL="https://api.openai.com/v1"   # multimodal endpoint
export VLM_API_KEY="sk-..."
export VLM_MODEL="gpt-4o"                          # vision model: scores image features
```

- Any OpenAI Chat Completions–compatible service works (OpenAI, Azure OpenAI,
  DeepSeek, Together, OpenRouter, self-hosted vLLM/Ollama/LM Studio gateway).
- If a gateway needs custom HTTP headers, set `DEFAULT_LLM_HEADERS` /
  `DEFAULT_VLM_HEADERS` in `config.py`.
- Alternative to env vars: edit the `USER CONFIGURATION` block at the top of
  `config.py` (`DEFAULT_LLM_*` / `DEFAULT_VLM_*`).
- If you created the conda env under a non-default name, also:
  `export CONDA_ENV=<name>` and `export SEGMENTATION_CONDA_ENV=<name>` (these
  tell the sandbox and the Cellpose-SAM subprocess which env to run in).

### 2.1 Verify credentials (MUST pass before a real run)

```bash
python - <<'PY'
from config import make_chat_llm, settings
from langchain_core.messages import HumanMessage
print("LLM endpoint:", settings.llm_base_url, "model:", settings.llm_model)
llm = make_chat_llm()
print("reply:", llm.invoke([HumanMessage(content="reply with the single word: ok")]).content)
PY
```

Expected: a short model reply (e.g. `ok`). A 401/timeout means the key/base URL
is wrong — fix before running the pipeline (a full run is expensive to fail late).

## 3. Prepare / validate the input data

MorphAgent's layout rule: **one dataset = one directory; each subdirectory is one
sample.** The sample list comes from **scanning subdirectories**, NOT from any
manifest file.

```
INPUT/                         # pass this to --data-root
├── dataset_index.txt          # free-text dataset description (see below)
├── sample_0001/
│   ├── image.tif              # PRIMARY file(s): raw data, input to code features (`extract(img, seg)`)
│   ├── slices/                # SECONDARY (optional): 2D PNGs preferred by VLM scoring
│   │   └── slice_0000_channel_0.png
│   └── segmentation/          # masks (auto-generated OR bring-your-own)
│       ├── mask_cell.tif
│       └── mask_nucleus.tif
├── sample_0002/
└── ...
```

Key facts to check:
- **Primary files** are the images directly in each sample dir (`.tif/.tiff/.png/
  .jpg/.jpeg/.bmp/.gif`). These are `img` in generated `extract(img, seg)` code.
- **`seg` is a dict keyed by mask file stem**: `segmentation/mask_cell.tif` →
  `seg["mask_cell"]`. Code accesses masks by name, never by index. Masks may be
  binary or instance-label maps.
- **Bring-your-own segmentation**: drop masks into each sample's `segmentation/`
  and the pipeline skips auto-segmentation by default.
- **Description file** (under `data_root`, resolved in order): `dataset_index.txt`
  → `README.md` → `README.txt` → `dataset_description.json` → `description.txt`
  (or pass `--description`). It is free text for the LLM — describe dimensions
  (2D/3D/multi-channel/z-stack), what each channel is, naming conventions.
- Optional knowledge folders live under the **project root** (the parent that
  contains `dataset/`): `expert_knowledge/`, `deep_research/`, `RAG/`. All
  optional. PDFs use lightweight PyMuPDF text extract by default; `.md/.txt/.xml`
  are read directly. Optional PaddleX OCR: see `envs/requirements-optional.txt`.
  You can also let MorphAgent **generate** these contents:
  - `--auto-deep-research` — one API call to `DEEP_RESEARCH_MODEL` writes a
    markdown report into `deep_research/`.
  - `--auto-literature-retrieval` — searches PubMed/Europe PMC and downloads
    open-access PDFs into `RAG/` (needs outbound internet; a proxy via
    `HTTPS_PROXY` works).

Verify the data is discoverable:

```bash
python - <<'PY'
from pathlib import Path
from utils_helpers import read_dataset_index, find_description_file
root=Path("/path/to/INPUT")            # if INPUT has a dataset/ subdir, use that
samples=read_dataset_index(root)
print("num samples:", len(samples), "first few:", samples[:3])
print("description:", find_description_file(root))
PY
```

Expected: a non-zero sample count and a resolved description path. If 0 samples,
the layout is wrong (are there per-sample subdirectories?).

## 4. Run the pipeline

```bash
conda activate morphagent
source .env

python main.py "Generate unbiased morphological features for these microscopy images" \
  --data-root /path/to/INPUT \
  --method both \
  --features-per-iteration 10 \
  --target-feature-count 200
```

- **Positional arg `user_query`** (required): the natural-language task; it feeds
  the feature-planning prompt.
- `--method {code,vlm,both}`: restrict to code features, VLM features, or both.
- `--num-rounds N`, `--resume`: multi-round runs; `--resume` continues an existing
  `--results-dir` without redoing finished rounds.
- Knowledge toggles: `--disable-expert-knowledge` / `--disable-deep-research` /
  `--disable-rag` (all on by default; they no-op if the folders are absent).
- Segmentation toggles: `--disable-segmentation`, `--segmentation-skip-if-present`
  (default), `--segmentation-run-even-if-present`.
- VLM: `--vlm-api-provider online` (default, recommended). `qwen` = local model,
  advanced only, needs extra GPU deps you must install yourself — avoid unless
  explicitly requested.
- Smoke-test cheaply first: small `--target-feature-count` (e.g. 20) and a couple
  of sample folders, to confirm the API + sandbox + segmentation path all work,
  before committing to a large run.

### Outputs

Written to `<project_root>/results/run_<timestamp>/` (or `--results-dir`):
`features.csv`, `retained_features.csv`, `feature_registry.json`,
`feature_plan.json`, `dataset_description.txt`, `*_summary.txt` (knowledge),
`segmentation_summary.json`.

## 5. Optional components

- **Cellpose-SAM segmentation** (default): runs inside the main pipeline (GPU
  recommended). Runs in `SEGMENTATION_CONDA_ENV` (defaults to the same env).
- **Allen `aicssegmentation`** (legacy, optional): needs its own env.
  ```bash
  conda env create -f envs/environment_allen.yml   # morphagent_allen (py3.6)
  conda activate morphagent_allen
  pip install -e segmentation_allen
  python segmentation_allen/check_installation.py
  export SEGMENTATION_CONDA_ENV=morphagent_allen
  ```
  See `segmentation_allen/README_SEGMENTATION.md`.
- **PDF parsing**: default is pymupdf lite extract → LLM. Optional PaddleX OCR:
  `pip install -r envs/requirements-optional.txt` (uncomment paddlex lines) then
  `export RAG_PDF_BACKEND=paddlex`.
- **Local Qwen3-VL** (advanced, optional): `pip install -r envs/requirements-optional.txt`
  then `--vlm-api-provider qwen`. Requires a suitable GPU.
- **Auto Deep Research / Literature Retrieval**: `--auto-deep-research` and
  `--auto-literature-retrieval` (see README). Configure `DEEP_RESEARCH_*` /
  `NCBI_EMAIL` in `.env` or `config.py`.

## 6. Troubleshooting / known pitfalls

- **`langchain-core` ResolutionImpossible** → someone re-added `langchain` /
  `langchain-community` / `langchain-experimental`. Remove them; keep only the
  `langchain-core` / `langchain-openai` / `langgraph` 1.x set.
- **`torch.cuda.is_available()` is False** → CPU-only machine. VLM scoring still
  works (API). Code features and Cellpose-SAM run but slower; large jobs want a GPU.
- **VLM 401 / timeout** → check `VLM_BASE_URL` / `VLM_API_KEY` / `VLM_MODEL`; the
  VLM endpoint must accept image inputs (multimodal).
- **0 samples found** → data layout wrong; each sample must be its own subdirectory
  under `data_root`.
- **PDF knowledge ignored / empty extract** → install `pymupdf` (in the unified
  env). Scanned/image-only PDFs need optional PaddleX (`RAG_PDF_BACKEND=paddlex`)
  or convert reports to `.md/.txt`.
- **Literature download failed but search worked** → network restriction on the
  server (no outbound HTTP/FTP to NCBI/EBI, or region blocking). Run on a
  machine with internet (`HTTPS_PROXY` works) or drop PDFs into `RAG/` manually.
- **Segmentation runs in the wrong/missing env** → set `SEGMENTATION_CONDA_ENV` to
  the env that has Cellpose-SAM (or `morphagent_allen` for the Allen backend).

## 7. From-scratch checklist (copy/paste)

```bash
# 1. env
conda env create -f envs/environment.yml          # add PIP_INDEX_URL=... to speed up
conda activate morphagent
python main.py --help                             # sanity

# 2. api
cp .env.example .env && $EDITOR .env && source .env
# verify creds with the snippet in step 2.1

# 3. data — arrange INPUT/ as in step 3, then verify with the snippet in step 3

# 4. run (start small)
python main.py "your feature task" --data-root /path/to/INPUT \
  --method both --target-feature-count 20
```

Done. If all four verifications passed (env import, `--help`, credential ping,
sample discovery), MorphAgent is correctly installed and ready for real runs.
