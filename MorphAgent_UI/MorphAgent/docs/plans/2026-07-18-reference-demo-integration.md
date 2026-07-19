# Reference Demo Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the Qt pilot and local setup follow the teacher-provided five-sample Tau demo instead of the ad-hoc BBBC021 pilot.

**Architecture:** Keep the scientific pipeline in `main.py` unchanged. Add the teacher demo under `demo/`, teach `RunConfig` how to load its paths, query, preset, and precomputed RAG cache, and expose that action through one Configure-page button. Keep API credentials external and document the gateway token cap discovered during the baseline run.

**Tech Stack:** Python 3.10, QtPy/PyQt5, unittest, MorphAgent's existing RAG hash/cache utility.

### Task 1: Encode the teacher pilot contract

**Files:**
- Modify: `tests/test_ui_models.py`
- Modify: `morphagent_ui/models.py`

1. Add a failing test asserting that `RunPreset.PILOT` means `both`, 5 candidates per round, 10 target features, and 2 rounds.
2. Run the focused test and confirm the old 20/20/1 values fail.
3. Change only the pilot preset values and labels.
4. Re-run the focused test.

### Task 2: Load and prepare the bundled reference demo

**Files:**
- Modify: `tests/test_ui_models.py`
- Modify: `morphagent_ui/models.py`
- Add: `demo/` from the teacher-provided `github_demo/` source, excluding generated results.

1. Add a failing temp-directory test for `RunConfig.apply_reference_demo()`.
2. Assert it selects `<repo>/demo/data`, fills the Tau query, enables all three knowledge sources, reuses segmentation, clears metadata/results, and writes one valid `.rag_cache` file using `demo/precomputed/rag_knowledge_summary.txt`.
3. Implement the smallest method using `knowledge.rag._compute_rag_folder_hash`.
4. Copy the source demo assets into `MorphAgent/demo/` without altering `github_demo/`.
5. Run the focused model tests.

### Task 3: Expose one UI action

**Files:**
- Modify: `tests/test_ui_widget.py`
- Modify: `morphagent_ui/widgets/configure.py`

1. Add a failing widget test that clicks `Load reference demo` and verifies the scanned 5-sample configuration and teacher query.
2. Add the button beside `Scan dataset` and connect it to `RunConfig.apply_reference_demo()`.
3. Refresh all fields and preflight after loading.
4. Update Configure-page copy from the old 20-feature manuscript pilot to the five-sample teacher demo.
5. Run widget tests with `QT_QPA_PLATFORM=offscreen`.

### Task 4: Capture environment compatibility discovered by reproduction

**Files:**
- Modify: `.env.example`
- Modify: `envs/environment.yml`
- Modify: `envs/requirements.txt`
- Modify: `README.md`
- Modify: `docs/UI_GUIDE.md`

1. Add `socksio==1.0.0` so `httpx` works when the host exports a SOCKS proxy.
2. Add documented `LLM_MAX_TOKENS=16384` and `MERGE_MAX_TOKENS=16384` safe gateway defaults; keep keys empty.
3. Document the reference-demo button, RAG cache, 2×5 settings, and expected output location.
4. Do not change validation or generated-code semantics in this task.

### Task 5: Verify end to end

**Files:**
- Verify: `demo/data/`, `demo/precomputed/`, UI tests, source status.

1. Run all UI tests in the `morphagent` environment.
2. Run `python main.py -h` import/CLI smoke test.
3. Run the bundled demo preflight and assert 5 samples, 5 primary images, 5 VLM views, and 25 masks.
4. Inspect the teacher baseline output already written under `github_demo/results/demo_run/`; report VLM success and the first-round generated-code failures separately.
5. Do not commit automatically because the working tree already contains the user's uncommitted UI work.
