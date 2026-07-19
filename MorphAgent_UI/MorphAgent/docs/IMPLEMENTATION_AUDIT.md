# MorphAgent UI implementation audit

This audit separates manuscript-derived product requirements, existing repository behavior, and interface additions. It is not evidence that a biological experiment has been reproduced.

## Requirement coverage

| Requirement | Implementation | Verification boundary |
|---|---|---|
| Read and reflect the current manuscript | Six stages mirror inspect/prepare/plan/dual-route quantification/validation/export; feature cards expose biological interpretation, visual signature, channel/mask needs, operators, statistics, and route rationale | Compared against the complete local manuscript during design; the unpublished DOCX is not copied into this repository |
| Use Nellie as a UI reference without copying it | Reuses metadata-first preflight, progressive enablement, background work, and optional napari layers; replaces Nellie branding/layout with five task-focused destinations and manuscript-specific review | The reference repository remains unchanged |
| Fast complete workflow | Project/dataset picker → scan → biological question → pilot → actionable preflight → one-click launch → live stages → Features + Evidence review; or Home → completed-run loader for result-only debugging | Widget workflow and previous-run tests plus subprocess transport fixture; real scientific execution still needs user data, credentials, compute, and the core environment |
| Preserve the current scientific implementation | UI constructs and launches `main.py` as a subprocess instead of reimplementing agent/segmentation/extraction/validation logic | Command-construction tests and subprocess fixture verify the integration boundary |
| Code/VLM route clarity | Aqua CODE and violet VLM labels, route selection, separate source counts, rationale fields, semantic-score caveat | Visual inspection of Configure, Run, Features, and Evidence screens |
| Data and runtime readiness | Mirrors nested `dataset/` resolution, direct sample folders, primary-code/VLM-source precedence, masks, empty samples, API/runtime checks, and resume markers | Synthetic dataset/model tests cover nested roots, route counts, blocking issues, and completed rounds |
| Long-running observability | QThread worker, streamed stdout/stderr, monotonic stage parser, elapsed time, progress, artifact snapshots, persistent console log | Stage parser and subprocess tests; cancellation preserves artifacts by design |
| Resume | Requires explicit results directory with `round_N/round_results.json`; reconstructs safe settings from UI manifest and delegates to `--resume` | Resume-marker unit test |
| Auditable results | Reads registry first, then round plans, then matrix headers; Features presents equal-width table/detail panes while Evidence uses equal review columns, a three-column selector, a compact name/description summary, and curated measurements, validation, provenance, and images without presenting scripts/logs as evidence; the first curated source previews by default | Registry/plan fallback, responsive three-column evidence tests, first-source preview test, previous-run loading test, and real-run screenshot inspection |
| Distinct visual identity | Ink/navy surfaces, aqua code, violet VLM, restrained evidence treatment, original resolution-transition cell artwork | Five rendered destinations inspected; tested core contrast pairs are 7.46:1–17.12:1 |
| Focused desktop shell and optional napari integration | `launch_ui.py` defaults to a standalone Qt window; `--with-napari` opts into the viewer, npe2 manifest, and viewer-bound image action | Standalone window test and screenshot verify that layer controls/canvas are absent; `npe2 validate` covers the optional plugin manifest |
| Installation and handoff | `.env.example`, editable UI extra, README fast path, detailed UI guide, screenshots, packaged resources | `pip wheel --no-deps --no-build-isolation` succeeds and wheel contents include manifest, UI modules, and hero asset |

## Explicit non-claims

- The UI test suite does not prove that a particular provider endpoint, GPU, Cellpose-SAM installation, generated feature, or VLM score is scientifically correct.
- No paid API call, automatic segmentation, generated-code experiment, or full manuscript benchmark was run during UI validation because this host does not have the complete MorphAgent runtime/credentials/data configured.
- VLM scores remain semantic model assessments rather than calibrated physical measurements.
- The current generated-code Conda boundary is not an operating-system security sandbox.

## Verification commands

Run from the repository root after installing the UI extra:

```bash
QT_QPA_PLATFORM=offscreen python -W error::ResourceWarning -m unittest discover -s tests -v
npe2 validate morphagent_ui/napari.yaml
python -m pip wheel . --no-deps --no-build-isolation --wheel-dir /tmp/morphagent-wheel
git diff --check
```
