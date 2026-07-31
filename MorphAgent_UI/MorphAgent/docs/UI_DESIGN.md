# MorphAgent UI design specification

## Product promise

MorphAgent turns a microscopy dataset and a biological question into a compact, auditable library of biologically grounded scalar features. The interface must make this scientific contract visible:

```text
dataset + biological question + optional knowledge/metadata
  -> inspect -> prepare -> plan feature cards
  -> quantify by code and/or VLM
  -> validate statistically and visually
  -> export complete + retained features + audit evidence
```

The application is not a generic chatbot, code IDE, segmentation program, or benchmark dashboard. Its core product object is the feature card.

## Fast path

A first-time user should be able to launch a defensible pilot in four decisions:

1. Select the project/dataset directory.
2. State the biological question.
3. Complete one model connection and select Code + VLM, Code only, or VLM only.
4. Press **Run MorphAgent**; actionable validation is shown only when something blocks the run.

Description, metadata, masks, and knowledge folders are auto-detected. Model credentials are stored only in the git-ignored repository `.env`; secret values are masked and never copied to run artifacts.

## Information architecture

The application has five destinations:

| Destination | Purpose |
|---|---|
| Home | Product identity, one primary new-discovery entry, and one secondary completed-run loader for result debugging |
| Configure | Dataset, biological question, model API, route, and optional inputs |
| Run | Six-stage progress, current work, cost/activity, logs, cancel/retry |
| Features | Equal-width searchable feature table plus the selected feature's biological interpretation, route, validation state, and structured design fields |
| Evidence | Equal-width review columns: three-column feature selector with a compact name/description summary, plus curated measurements, validation, provenance, segmentation context, and shared image previews; first source previews by default |

The default application is a focused standalone Qt window that opens maximized while retaining the native title bar and window controls. napari's image, label, points, tracks, scale, contrast, 2D/3D, and time controls are available only in the explicit `--with-napari` inspection mode, which also opens maximized and avoids unused viewer chrome during ordinary configuration and execution.

## Workflow stages

| Stage | Current code evidence | Completion evidence |
|---|---|---|
| Inspect | sample discovery, dataset understanding, cell-context detection | samples found + dataset description available |
| Prepare | knowledge summaries, segmentation, first-sample visualization, slices | summaries/masks/previews written or explicitly skipped |
| Plan | LangGraph planner and `feature_plan.json` | valid non-empty feature-card list persisted |
| Quantify | code extractors and VLM scoring | cumulative `features.csv` and per-feature audit folders |
| Validate | deterministic validation/metadata-aware analysis | registry and retained CSV/report |
| Export | `round_results.json`, run artifacts | requested rounds complete and output index refreshes |

Tabs and stages use icon + text + state, never color alone. States are `pending`, `running`, `complete`, `warning`, `failed`, and `cancelled`.

## Configuration model

### Required

- Dataset/project path
- Biological question
- Results directory
- Valid LLM endpoint/key/model in the repository `.env`, entered in Configure or edited directly

### Contextual

- VLM endpoint/key/model for `vlm`/`both`
- Supported GPU or precomputed masks when automatic Cellpose segmentation is enabled
- A Python interpreter that can import the MorphAgent environment

### Optional and auto-detected

- Dataset description
- Metadata CSV
- `expert_knowledge/`
- `deep_research/`
- `RAG/`
- Existing `segmentation/` masks
- Existing run directory for resume

### Run scale

The bundled teacher demo uses the repository-grounded reference scale: two rounds, five candidates per round, target ten features. Low-frequency custom-run defaults (counts, ratios, workers, and concurrency) are read from `.env` so the first-run interface contains only decisions users routinely need.

## Feature card

A card contains:

- name and status;
- biological interpretation;
- expected visual signature;
- target object and spatial scale/context;
- required channels and masks;
- route badge (`CODE` with angle-bracket icon, `VLM` with aperture icon);
- candidate operators and summary statistic;
- execution attempts / warning state;
- CV, redundancy and available metadata association;
- code artifact or VLM 0–100 semantic score/rationale;
- directly linked feature image/evidence paths when the pipeline produces them;
- audit history and decision (`retained`, `rejected`, `warning`).

Shared first-sample visualizations are shown on Evidence as run-level context, not feature-specific evidence. The interface labels this boundary unless a future pipeline artifact links an image directly to a feature.

VLM scores are always labelled **semantic score**, never shown as physical measurements.

## Visual system

### Palette

| Token | Hex | Use |
|---|---|---|
| Ink 950 | `#07111F` | application background |
| Ink 900 | `#0B1626` | dock background |
| Slate 850 | `#111F33` | cards and inputs |
| Slate 700 | `#29405E` | borders and inactive controls |
| Text | `#E6F6FF` | primary text |
| Muted | `#9CB0C8` | secondary text |
| Aqua | `#22D3EE` | primary action, code route, active focus |
| Violet | `#A78BFA` | VLM route and semantic evidence |
| Coral | `#FB7185` | selected evidence / signed data accent |
| Success | `#34D399` | complete/retained |
| Warning | `#FBBF24` | user review required |
| Error | `#F87171` | blocking failure |

Coral and blue/coral diverging palettes remain data encodings rather than navigation colors.

### Typography

- UI: `Inter`, `Fira Sans`, or the operating-system sans-serif fallback.
- Identifiers, file paths, command previews and metrics: `Fira Code`, `SFMono-Regular`, or monospace fallback.
- No oversized landing-page typography inside a scientific dock. Product title 24–28 px; page title 18–20 px; body 13–14 px.

### Shape and effects

- 8 px card radius and 6 px control radius.
- 1 px visible borders; aqua 2 px focus ring.
- No floating glass cards, cartoon agents, emojis, pulsing decoration, or layout-shifting hover transforms.
- Minimal aqua/violet glow only on the hero and active stage, never behind dense tables.

### Brand image

`morphagent_ui/resources/morphagent_hero.png` is original project artwork: a diagonal wide-field-to-resolved cell transition with cyan networks, violet bundles and coral puncta feeding evidence callouts. It directly represents resolution-aware, biologically grounded feature design and is distinct from Nellie's logo.

## Long-running task behavior

- Show `stage 3 of 6`, round, current feature/sample when available, elapsed time, and coarse overall progress.
- Stream stdout/stderr into an expandable monospace log with search/copy/open-log actions.
- **Cancel** terminates the complete process group and records `cancelled`; it never deletes artifacts.
- Errors show stage, last readable message, log path, and actions: retry, edit configuration, resume completed rounds, open output.
- The UI writes `ui_run_manifest.json` before launch with non-secret configuration, command arguments, timestamp and state.
- Resume is enabled only for an explicit results directory containing completed `round_N/round_results.json` markers.

## Safety and reproducibility

- Generated code executes with the current user's permissions; show this warning before first code/both run.
- Display a mask/GPU readiness warning before automatic segmentation.
- Disclose that `slices/` and possibly `segmentation/` are written under samples.
- Never store API keys in settings or manifests.
- Default to reproducible mode in UI presets.
- Keep failed/rejected features and tracebacks in the audit view.

## Initial implementation boundary

The first UI release launches the existing `main.py` as a subprocess and observes its authoritative output artifacts. It does not duplicate scientific code. A later refactor may extract `main()` into a reusable pipeline service once behavior is covered by tests.
