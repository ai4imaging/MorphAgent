# MorphAgent Demo

A tiny, self-contained example that runs the **full MorphAgent pipeline** on 5
neuron microscopy samples for **2 rounds × 5 features**. Open the notebook and
run the cells top to bottom:

```bash
conda activate morphagent          # the unified env from envs/environment.yml
jupyter lab                        # open demo/morphagent_demo.ipynb
```

## What's inside

```
demo/
├── morphagent_demo.ipynb     # step-by-step runnable notebook
├── data/                     # project root passed to --data-root
│   ├── dataset/              # 5 samples: image.tif + slices/ + segmentation/
│   ├── expert_knowledge/     # notes + example image (summarized by the LLM)
│   ├── deep_research/         # report.md (plain text — no PaddleX needed)
│   └── RAG/                   # a few example PDFs
├── precomputed/              # RAG summary used to seed the offline cache
└── results/                  # created when you run the notebook
```

## Why it runs cheaply and offline (no GPU / no PaddleX)

- **Segmentation is reused.** Every sample already ships masks under
  `segmentation/`, so the pipeline marks them `skipped_user_seg` and never calls
  Cellpose-SAM. No GPU required.
- **deep_research** is provided as `report.md` (read directly by the LLM).
- **RAG** is served from a pre-seeded cache (the notebook writes it) so raw PDF
  parsing with PaddleX is skipped.

## Requirements

- The `morphagent` conda environment (see `../envs/environment.yml`).
- A working **LLM** and **VLM** API (OpenAI-compatible). Configure `LLM_*` /
  `VLM_*` environment variables or edit `../config.py` before running.

See the top-level `README.md` and `installation_skill.md` for full details.
