# MorphAgent Demo

A tiny, self-contained example that runs the **full MorphAgent pipeline** on
**10** neuron microscopy samples (`WT_1`–`WT_5` wild-type, `MU_1`–`MU_5` mutant)
for **1 round × 5 features**. Open the notebook and run the cells top to bottom:

```bash
conda activate morphagent          # the unified env from envs/environment.yml
jupyter lab                        # open demo/morphagent_demo.ipynb
```

## What's inside

```
demo/
├── morphagent_demo.ipynb     # step-by-step runnable notebook
├── data/                     # project root passed to --data-root
│   ├── dataset/              # 10 samples: WT_1–5 + MU_1–5 (image.tif + slices/ + segmentation/)
│   ├── metadata.csv          # paired group labels for feature validation
│   ├── expert_knowledge/     # notes + example image (summarized by the LLM)
│   ├── deep_research/         # report.md (plain text — no PaddleX needed)
│   └── RAG/                   # a few example PDFs
├── precomputed/              # RAG summary used to seed the offline cache
└── results/                  # created when you run the notebook
```

## Dataset layout (also for custom data)

Each sample is one folder under `dataset/`:

```
dataset/
  WT_1/
    image.tif            # primary image (code route)
    slices/              # optional VLM-ready views
    segmentation/        # optional masks; reused when present
  WT_2/
    ...
```

Recommend **≥5 samples** (enough unique values for validation). Fewer than 5
triggers a UI warning but does not block Run.

## Why it runs cheaply and offline (no GPU / no heavy PDF OCR)

- **Segmentation is reused.** Every sample already ships masks under
  `segmentation/`, so the pipeline marks them `skipped_user_seg` and never
  needs Allen or Cellpose. No GPU required.
- **deep_research** is provided as `report.md` (read directly by the LLM).
  Loading the demo dataset does **not** pass `--auto-deep-research`.
- **RAG** is served from a pre-seeded cache on the happy path. Fresh PDF
  ingest uses lightweight PyMuPDF text extract -> LLM (not PaddleX). The demo
  path does **not** pass `--auto-literature-retrieval`.

## Requirements

- The `morphagent` conda environment (see `../envs/environment.yml`).
  PDF ingest uses **pymupdf** (lite extract); PaddleX is optional only.
- A working **LLM** and **VLM** API (OpenAI-compatible). Fill them in the
  notebook's API configuration cell (or set `LLM_*` / `VLM_*` env vars /
  edit `../config.py`) before running.

## Custom datasets (outside this demo)

When you choose your own folder in the UI (not Load demo dataset):

- **Deep research** checked -> `--auto-deep-research` (uses the LLM API).
- **Literature / RAG** checked -> `--auto-literature-retrieval --pubmed-max-results 10`
  (PubMed -> lite PDF text -> LLM). Failures are logged and the run continues with
  whatever local knowledge files already exist.
- Missing masks -> Allen via `morphagent_allen` when available; otherwise the
  sample is skipped gracefully (install optional Allen env only if needed).

See the top-level `README.md` and `installation_skill.md` for full details.
