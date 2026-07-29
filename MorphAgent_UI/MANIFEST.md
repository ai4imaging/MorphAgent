# MorphAgent handoff manifest

## Source baseline

- Upstream repository: `https://github.com/10cbvkw/MorphAgent.git`
- Upstream base commit: `03bf8b57d607acaa34a23f23819ac6e8bdcbbf6c`
- Delivery date: 2026-07-19
- The delivered source includes the locally reviewed Qt UI and integration changes made after the upstream base commit.
- The upstream `.git` directory and the local secret `.env` are intentionally excluded.

## Main entry points

| Purpose | Entry point |
|---|---|
| Focused desktop UI | `MorphAgent/launch_ui.py` |
| Optional napari-integrated UI | `MorphAgent/launch_ui.py --with-napari` |
| Scientific pipeline CLI | `MorphAgent/main.py` |
| UI package | `MorphAgent/morphagent_ui/` |
| Teacher demo notebook | `MorphAgent/demo/morphagent_demo.ipynb` |
| Handoff verification | `scripts/verify_install.py` |

## Runtime strategy

1. `launch_ui.py` loads the repository-local `.env`.
2. The Qt UI collects only the required data, biological question, model connection, route, mask policy, and knowledge-source choices.
3. `RunConfig.build_command()` converts the UI state into the real `main.py` CLI arguments.
4. `PipelineWorker` starts `main.py` as a background subprocess with unbuffered output.
5. Logs stream to the Run page and `ui_console.log`.
6. Each run writes to a timestamped directory under the selected project root.
7. Features and Evidence read the persisted CSV/JSON/image artifacts; Home can load an existing run without executing the pipeline.

## Dependency tiers

| Tier | File | Use |
|---|---|---|
| Recommended review/demo | `dependencies/requirements-demo-ui.txt` | UI, API LLM/VLM, bundled masks and all result pages |
| Cellpose segmentation | `dependencies/requirements-segmentation-optional.txt` | Regenerate masks |
| Full upstream environment | `dependencies/environment-full.yml` / `requirements-full.txt` | Unified install including segmentation + pymupdf for RAG / auto literature / PDF deep-research |
| Optional PDF OCR / local VLM | `dependencies/requirements-extra-optional.txt` | Optional PaddleX OCR or local Qwen; not needed for the bundled demo |
| Legacy Allen backend | `dependencies/environment-allen-optional.yml` | Optional isolated legacy environment |
| Reviewer tests | `dependencies/requirements-test.txt` | pytest and Qt tests |

## Bundled demo

- Ten Tau-neuron samples (`WT_1`–`WT_10`).
- Each sample contains `image.tif`, a VLM-ready slice, and segmentation masks.
- Expert notes, a deep-research report, three RAG PDFs, and a precomputed RAG cache are included.
- `MorphAgent/demo/data/results/completed_demo_run` contains a completed two-round run with 10 feature cards.

## Demonstration video

- `demo_video/MorphAgent_demo_english.mp4` (compressed for repository size)
- English narration and burned-in English subtitles.
- The experiment segment is accelerated 15×.
- A three-second silent visual hold separates the accelerated experiment from the next narration.
- Sidecar subtitles are provided as `MorphAgent_demo_english.srt`.

