# Ask MorphAgent Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a secure, paper-and-code-grounded reviewer chat workflow to both MorphAgent UI distributions.

**Architecture:** A prebuilt JSON corpus stores source-labelled manuscript, supplementary, and code chunks. A pure-Python retriever selects bounded evidence, an OpenAI-compatible client assembles evidence-grounded messages, and a Qt background worker feeds a dedicated chat page opened from Home after API setup.

**Tech Stack:** Python 3.10, qtpy/PyQt5, OpenAI Python client, python-dotenv, standard-library JSON/XML/ZIP/tokenization, unittest.

### Task 1: Build and validate the reviewer knowledge bundle

**Files:**
- Create: `scripts/build_reviewer_knowledge.py`
- Create: `MorphAgent_UI_Docker/MorphAgent/morphagent_ui/reviewer_knowledge/knowledge.json`
- Create: `MorphAgent_UI_Lite/MorphAgent/morphagent_ui/reviewer_knowledge/knowledge.json`
- Create: `MorphAgent_UI_Docker/MorphAgent/tests/test_reviewer_chat.py`
- Create: `MorphAgent_UI_Lite/MorphAgent/tests/test_reviewer_chat.py`

1. Write failing tests that load a small temporary bundle, preserve source labels, reject malformed entries, and rank a manuscript/code chunk for matching questions.
2. Run each new test module with `morphagent_lite/bin/python -m unittest -v tests.test_reviewer_chat` and confirm failure because the reviewer-chat module is absent.
3. Implement DOCX XML extraction, optional PDF extraction, source-code filtering/chunking, deterministic JSON output, and the minimal bundle loader/retriever.
4. Generate the two distribution-local bundles from the archive supplied through `--submission-zip` and their own code trees.
5. Re-run the new tests and confirm green.

### Task 2: Implement evidence-grounded prompt and API client

**Files:**
- Create: `MorphAgent_UI_Docker/MorphAgent/morphagent_ui/reviewer_chat.py`
- Create: `MorphAgent_UI_Lite/MorphAgent/morphagent_ui/reviewer_chat.py`
- Modify: both `tests/test_reviewer_chat.py`

1. Add failing tests for positive/evidence-bound prompt rules, limitation handling language, bounded conversation history, source-labelled context, response parsing, and one-time `/v1` fallback on a 404.
2. Verify the tests fail for the expected missing behavior.
3. Implement `KnowledgeChunk`, `ReviewerKnowledgeBase`, `build_system_prompt`, `build_chat_messages`, and `ReviewerChatClient` with injected client factories for offline tests.
4. Re-run both test modules and confirm green.

### Task 3: Add API setup and chat widgets

**Files:**
- Create: `MorphAgent_UI_Docker/MorphAgent/morphagent_ui/widgets/ask.py`
- Create: `MorphAgent_UI_Lite/MorphAgent/morphagent_ui/widgets/ask.py`
- Modify: both `morphagent_ui/widgets/__init__.py`
- Modify: both `morphagent_ui/theme.py`
- Modify: both `tests/test_ui_widget.py`

1. Add failing Qt tests for required resolved credentials, masked saved key behavior, welcome copy, message rendering, send-disabled busy state, and error recovery.
2. Verify red in offscreen Qt mode.
3. Implement `AskApiDialog`, `ChatWorker`, message-card rendering, chat composer, Enter/Shift+Enter handling, sending/error states, and back signal.
4. Add only the minimal chat-specific stylesheet roles needed for message alignment and composer readability.
5. Re-run the focused Qt tests and confirm green.

### Task 4: Wire Home and the main shell

**Files:**
- Modify: both `morphagent_ui/widgets/home.py`
- Modify: both `morphagent_ui/main.py`
- Modify: both `tests/test_ui_widget.py`

1. Add failing tests asserting an `Ask MorphAgent` Home button below the previous-run action, a preserved five-row sidebar, API-dialog gating, a hidden sixth stacked page, and Back-to-Home behavior.
2. Verify red.
3. Add `ask_morphagent_requested`, construct the chat page with the repository root/bundle, and implement open/back handlers without adding a navigation row.
4. Verify green in full and Lite packages.

### Task 5: Documentation, security, and regression verification

**Files:**
- Modify: `MorphAgent_UI_Docker/README_UI.md`
- Modify: both `MorphAgent/README.md`
- Modify: relevant dependency manifests only if a truly new runtime dependency is required.

1. Document the reviewer workflow, provider privacy boundary, local corpus behavior, and the fact that no API key/chat transcript is bundled.
2. Run all new tests in both packages.
3. Run existing focused UI/model suites and compare failures with the recorded baseline.
4. Construct both full widgets offscreen and verify Home → dialog → chat → Home navigation without network calls.
5. Search tracked changes and knowledge bundles for `.env`, `sk-`, known private keys, absolute user paths, and submission metadata that should not ship.
6. Review `git diff --check`, tracked file sizes, and branch status.
