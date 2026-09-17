# Compute: previous run → new data → confirmed execution

Keep the accepted neutral desktop style. Reuse becomes Compute; remove the Save
destination, retaining automatic bundles and inline download/retry actions.
The user explicitly requests implementation, so work in the existing dirty
workspace without committing, resetting, or isolating away the current UI.

- [x] Read portable timestamp bundles and raw timestamp/results directories.
      Use only scripts inside the selected folder; never follow external paths.
- [x] Present a composer like Feature design, with previous-run and dataset
      attachments, an independent question, and expandable feature selection.
      Default to selecting available code features, never VLM-only definitions.
- [x] Default the native history picker to .web_workspace/exports. Cancel keeps
      the existing selection. Loading a run alone must never execute its code.
- [x] Validate prompt, source, features, data, and user API connection; show the
      same blocking alerts and configuration confirmation before launch. Prompt
      is recorded as context, not used to silently rewrite historical extractors.
- [x] Execute selected saved scripts unchanged with no planner/VLM calls, on the
      Compute page. Put the question first, then progress/ETA/logs; automatically
      save and download the new results. Keep Feature design drafts independent.
- [x] Verify portable bundle round-trip, path safety, API/confirmation guards,
      default picker location, same-page progress, automatic download, and narrow
      layouts in real Qt. Use fixture scripts only, not a paid model invocation.

Alternative considered: keep the existing history-list dashboard. The requested
upload-first composer is preferred because it matches Feature design and accepts
portable bundles received from another machine as well as local runs.

Verification: 106 Python regression tests, 17 frontend behavior tests, and 12
real Qt/WebEngine tests passed. Tests used temporary datasets and fixture APIs;
portable extractor tests executed a saved image-size measurement on a new TIFF.
No live model calls or user experiment runs were started. Existing user data and
the dirty worktree were preserved. Screenshots in `.web_workspace/compute-*.png`
show fixture UI only. Windows/Linux were not exercised on real hardware.
