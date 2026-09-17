# Web UI runtime integration plan

**Goal:** connect the accepted Codex-style UI to the shared MorphAgent pipeline,
with real configuration, data imports, logs, history, feature evidence and exports.

**Architecture:** a loopback-only Python HTTP service serves the existing frontend.
It reuses `RunConfig`, dataset scanning, feature-card parsing and the original
`main.py` / `reuse_code.py` subprocesses. The desktop UI remains available. No
scientific algorithms or extra source tree are copied.

**Tech stack:** Python standard library HTTP/threading/subprocess; existing
python-dotenv and scientific runtime; pypdf for PDF text; standard-library DOCX
XML parsing. Vanilla HTML/CSS/JS stays unchanged in visual direction.

## Tasks and verification

1. Add failing `tests/test_web_service.py` for mode mapping, masked credentials,
   document parsing, dataset traversal protection, process lifecycle, history,
   per-feature evidence filtering, and authenticated HTTP endpoints.
2. Implement `src/morphagent_ui/web_service.py` and `web_documents.py`; run the
   tests with `.venv-ui-check/bin/python -m pytest tests/test_web_service.py -q`.
   Data inputs are copied per run; logs are redacted; process exit and scientific
   artifact availability are separate states. Cancellation targets only its child.
3. Connect `MorphAgent_UI/design-preview/app.js` through `runtime.js` to the API.
   Keep static preview mode when opened without the Python service. Replace all
   example output with actual run results in connected mode.
4. Add `launch_web_ui.py`, macOS/Linux and Windows launch scripts, and usage docs.
   Update the existing installation requirements for PDF text extraction.
5. Exercise all pages in agent-browser, including a real local fixture subprocess,
   file upload, saved-result reload, API masking, errors and cancellation. Then
   run the original pipeline with the bundled restricted API at 1×5 if available.
   Record genuine provider/dependency failures; never substitute simulated success.

## Acceptance / safety

- Ultra fast=1, Fast=5, Detailed=20, fixed seed and temperature zero; free demo
  remains restricted to 1×5, own credentials allow other presets.
- API keys stay server-side and out of logs, manifests, responses and downloads.
- Bind only to 127.0.0.1; validate Host/Origin and require a per-session API token.
- Only registered result files can be previewed/downloaded; block traversal,
  hidden/secret paths, symlink escapes and unsupported uploaded files.
- Uploaded knowledge is explicitly sent to the configured model only on Run.
- No commit/push; preserve unrelated dirty-worktree changes.
