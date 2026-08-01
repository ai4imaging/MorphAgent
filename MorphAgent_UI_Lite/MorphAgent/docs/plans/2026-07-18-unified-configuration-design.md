# Unified Configure implementation plan

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:test-driven-development` and execute each task in order.

**Goal:** Remove the separate Settings destination and make Configure the single place for run inputs and repository-local API configuration.

**Architecture:** Configure remains a short vertical workflow: Data, Biological question, Model API, and Analysis. The API section reads and updates the git-ignored repository `.env`; secrets use password inputs, are never copied to commands/manifests/logs, and blank secret fields preserve an existing key. Low-frequency discovery/performance values are read from `.env` into `RunConfig` but are not duplicated as first-run controls.

**Tech stack:** Python, Qt via qtpy, python-dotenv, unittest.

### Task 1: Specify the merged navigation and API section

**Files:**
- Modify: `tests/test_ui_widget.py`

1. Write a failing widget test requiring four destinations and no Settings item.
2. Require a `3 · Model API` step in Configure and `4 · Analysis` after it.
3. Run the focused widget tests and confirm they fail because Settings is still separate.

### Task 2: Specify safe `.env` persistence

**Files:**
- Modify: `tests/test_ui_widget.py`
- Create: `morphagent_ui/environment.py`

1. Write a failing test with a temporary repository `.env`.
2. Require existing base URL/model values to load, an existing secret to remain masked, a replacement key to persist, and same-API VLM values to reuse LLM values.
3. Implement narrowly scoped read/update helpers using python-dotenv-compatible syntax and update the current process environment after saving.

### Task 3: Read low-frequency scale defaults from `.env`

**Files:**
- Modify: `tests/test_ui_models.py`
- Modify: `morphagent_ui/models.py`

1. Write a failing test for `FEATURES_PER_ITERATION`, `TARGET_FEATURE_COUNT`, `NUM_ROUNDS`, `CODE_VLM_RATIO`, `KNOWLEDGE_DEPENDENCY`, `CODE_PARALLEL_WORKERS`, and `VLM_ONLINE_CONCURRENCY`.
2. Add validated environment-backed default factories.
3. Keep the teacher reference-demo preset authoritative at 2 × 5, target 10.

### Task 4: Merge Settings into Configure

**Files:**
- Modify: `morphagent_ui/widgets/configure.py`
- Modify: `morphagent_ui/main.py`
- Modify: `launch_ui.py`

1. Add the Model API card with LLM base URL, password key, model, and a visible “reuse for image scoring” choice.
2. Reveal separate VLM inputs only when reuse is disabled.
3. Save explicitly to `.env`, clear secret text after saving, refresh preflight immediately, and show a non-secret status.
4. Remove Settings from navigation/page construction and map legacy `settings` demo requests to Configure.

### Task 5: Verify and document

**Files:**
- Modify: `README.md`
- Modify: `docs/UI_GUIDE.md`
- Modify: `docs/UI_DESIGN.md`
- Modify: `docs/IMPLEMENTATION_AUDIT.md`

1. Update all destination counts and configuration instructions.
2. Run focused tests, the full suite, and `git diff --check`.
3. Render Configure against a temporary non-secret `.env` and inspect section hierarchy and visibility.
