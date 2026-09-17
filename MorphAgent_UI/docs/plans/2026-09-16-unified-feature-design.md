# Unified Feature design, retained key masks, ETA and automatic outputs

- [x] Show retained session keys as a read-only star mask with a Change action;
      never copy a real key back into the page or submit the mask as a credential.
- [x] Replace separate Data/Run tabs with Feature design. On confirmation, replace
      the composer with the submitted question at the top and the live run below.
      Keep review-before-run, history, Reuse, Visualize and Save intact.
- [x] Share the existing workload/observed-progress estimator through a Qt-free
      module, show explicitly approximate remaining time, and track loop progress.
- [x] Automatically package completed outputs under exports/<timestamp>/ with a
      matching ZIP. Show the path and a download action in the run view. Keep raw
      results, preserve failed/partial labels, and make packaging failures retryable.
- [x] Verify frontend contracts, subprocess/save behavior, legacy timing tests,
      and real Qt interactions/screenshots without a paid analysis.

Use the current environment and accepted neutral style. Do not interrupt the
user's existing UI, alter scientific algorithms, or push changes without a request.

Verification (2026-09-16): 103 Python regression tests, 16 frontend contract tests,
and 11 real PySide6/WebEngine desktop tests passed. The desktop test exercises
applied primary/VLM key masks, replacement, confirmation, same-page runtime/ETA,
delayed automatic packaging, native ZIP download, and an 800px window. CLI output
uses a fixture script, not a paid model or a scientific accuracy test. Existing
user sessions, environments, datasets, and history were left intact.
