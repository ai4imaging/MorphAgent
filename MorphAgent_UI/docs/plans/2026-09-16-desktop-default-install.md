# Desktop dependencies in the standard installer

User-approved scope: the normal setup/launch commands should install and open
the new Qt6 workspace in the existing `morphagent_lite` environment. No new
environment, no scientific-pipeline changes, no removal of legacy PyQt5.

- [x] Add the pinned desktop requirement to the shared install dependency list.
- [x] Point macOS/Linux and Windows default launchers to `launch_desktop_ui.py`.
- [x] Verify Qt6/WebEngine and legacy Qt5 in separate Python processes.
- [x] Update installation and upgrade documentation to the two-command workflow.
- [x] Run setup/launcher regression tests, installed dependency verification,
      shell syntax checks and desktop tests. Do not recreate the user's env.

Windows execution and a clean-machine full installation cannot be claimed from
macOS script/static checks. Existing results, .env secrets and running jobs must
not be modified by verification.

## Verification results

- Initial regression run: 8 expected failures (missing dependency inclusion,
  old launcher targets, missing isolated Qt checks), then corrected.
- Final focused + existing regression run: 84 passed.
- Real Qt desktop interaction suite in `morphagent_lite`: 8 passed. Mac graphics
  fallback/input-method diagnostics are nonfatal; all interactions completed.
- Updated verifier: Qt6 6.10.2/WebEngine and Qt5 5.15.14 imports, analysis modules,
  source and demo checks passed.
- Offline pip dry-run of the complete updated requirements: satisfied by the
  existing environment, exit 0; `pip check`: no broken requirements.
- Actual `bash scripts/start_ui.sh --help`: new desktop entry point, exit 0.
- Bash syntax and `git diff --check`: passed.
- No environment recreation, dependency reinstall, model API calls, push or PR.
- Windows scripts checked statically, not executed on Windows. A new-machine
  network installation was not performed; installer flow was tested with an
  external conda recorder, including installation failure propagation.
