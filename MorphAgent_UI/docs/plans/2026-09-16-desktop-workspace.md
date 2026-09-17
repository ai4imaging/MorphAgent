# Desktop Workspace Implementation Plan

**Goal:** Launch the current Codex-style workspace in a maximized Qt window, in the existing morphagent_lite environment, without changing scientific behavior or stored runs.

**Architecture:** Reuse WorkspaceService and its loopback-only HTTP API. A desktop runtime owns an automatically assigned local port, server thread, workspace lock, and shutdown. A PySide6 QWebEngineView renders the existing assets; it uses an off-the-record profile and native file/save dialogs. The existing browser and legacy PyQt5 launchers remain available.

**Tech stack:** Existing Python 3.10 environment; PySide6 6.10.2 (including WebEngine); existing HTML/CSS/JS; unittest/pytest. No API calls required for engineering verification.

## Constraints
- No replacement of existing environments or scientific dependencies. pip dry-run resolves only PySide6, Essentials, Addons, and shiboken6.
- Do not import PyQt5 and PySide6 into one process. Keep old launch_ui.py unchanged.
- Preserve the dirty worktree, datasets, and history; no commits or push requested.
- No web browser opens automatically in desktop mode. Bind only loopback, keep Host/Origin/token protections, deny external navigation inside the trusted view.
- Window close during an active analysis must ask before stopping; cleanup must stop this runtime's own subprocesses and release its workspace lock.

## Tasks
1. Add failing lifecycle and URL policy tests in tests/test_desktop_runtime.py. Implement src/morphagent_ui/desktop_runtime.py with automatic port and owned cleanup. Run tests against temporary workspaces.
2. Add Qt tests for transient profile, native directory/file selection, download routing, navigation policy, and close confirmation. Implement src/morphagent_ui/desktop_window.py and launch_desktop_ui.py. Test Qt in a separate process from legacy PyQt5 tests.
3. Add a pinned optional requirements-desktop.txt, pyproject desktop extra, and macOS/Linux/Windows launch helpers. Update README_WEB.md and add a focused desktop launch guide.
4. Run existing web/backend regressions. Launch real Qt with temporary workspace and exercise Data, Settings, Reuse, Visualize, and downloads without paid API calls. Inspect a screenshot from the actual Qt renderer.
5. Launch the desktop against the user's existing workspace only after checking no active browser process/run exists. Report the current-environment command and any platform verification limits.

## Verification record
- [x] Dependency dry-run: only four Qt6 packages would be added on macOS arm64/Python 3.10.
- [x] Runtime tests RED then GREEN: ephemeral loopback port, workspace lock, failed startup, cancellation and stdout cleanup, exact-origin policy, Qt-free stage parser.
- [x] Eight real Qt tests: transient profile and empty API, folder upload, native dialogs, download/save, active-close cancel, system-quit confirmation, Data → fixture CLI → completed run → export ZIP. Fixture CLI is a plumbing test, not a new scientific/API result.
- [x] Existing web regressions: 72 selected backend/legacy-controller tests and five JavaScript UI tests pass.
- [x] Actual Qt screenshot inspected at .web_workspace/desktop-preview.png (temporary test workspace, not user results).
- [x] Installation/launch instructions updated in root README, README_WEB.md and README_DESKTOP.md; optional pinned desktop requirements and launch helpers added.

## Outcome / limitations
- Installed only PySide6 6.10.2 / Essentials / Addons / shiboken6 into the existing morphagent_lite environment. pip check passes. Existing PyQt5 5.15.11 still imports in its separate process.
- Extracted StageDetector into stages.py so the web runtime never imports the Qt5 controller just to parse logs. Added a closing guard for shutdown and closed subprocess stdout handles. Scientific algorithms and legacy UI layout were not changed.
- macOS arm64/Python 3.10 locally verified. Windows/Linux real-machine tests and standalone .app/.exe packaging are not performed.
- macOS Qt/Chromium emitted graphics fallback and input-method diagnostics, but rendering, uploads/downloads and all tests completed. No sandbox-disabling or global proxy changes were used.
- No API calls, original dataset edits, commits or pushes during this implementation.
