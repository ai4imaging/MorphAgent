# Superseded unified Results page design

> Superseded on 2026-07-19 after user review. The current interface restores separate Features and Evidence destinations; Evidence shows the complete run catalog, including shared images.

The former Features and Evidence destinations are merged into one Results page. The persistent left column is the feature table; the right column is a compact summary and feature-linked evidence browser. The groups are structural rather than color-coded. Each artifact is a real file written by MorphAgent or referenced by the feature registry.

Selecting a row immediately rebuilds the right-hand groups using its exact name, round, method, and registry source paths. Code features prioritize their generated implementation block; VLM features prioritize their selected score and rationale. Shared matrices and registry files remain valid sources, but their in-app previews are pruned to the selected column or record so unrelated features never appear.

The teacher-demo run contains only shared first-sample overview images. Because it does not produce per-feature overlays, those images are omitted from feature evidence instead of being repeated with a misleading implication. A directly linked image can still appear when a future registry records one in the feature's `source_paths`.

A single click previews an artifact inside the UI. Text, JSON, CSV, log, and Python files render as readable text; supported raster images render as scaled previews. Double click and one clearly labelled “Open externally” button open the selected file in the system application. Unsupported and oversized files receive an explicit message instead of failing silently. Empty groups are omitted so the page does not feel dense.

Tests cover five-destination navigation, selection synchronization, feature-isolated CSV previews, shared-image exclusion, expanded artifact discovery, and the no-selection/empty state. Visual QA uses Code and VLM features from the teacher-provided Tau demo run and preserves the accepted dark background with aqua selection feedback.
