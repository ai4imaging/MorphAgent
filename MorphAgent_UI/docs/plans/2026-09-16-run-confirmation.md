# Required feature count and explicit run confirmation

Keep the accepted neutral composer. Add an initially empty, required Feature
number input. Reuse the legacy target range (1–500) and pass the value to the
CLI target-feature-count. Distribute candidate planning over the selected 1/5/20
rounds with ceil(target / rounds), at least one candidate per round. This is a
planning target, not a promise of the final number retained after validation.

Submission must not launch a subprocess. First validate the count, data, question,
and user-supplied model connections. Missing/invalid inputs use a prominent modal
with an action leading to the field or API settings. Preflight checks remain local.
Then show a configuration modal with the question, mode/rounds, route, dataset,
knowledge filenames, and feature count. Confirm and run starts the frozen reviewed
payload once; Back/Escape makes no run and preserves inputs. Never expose API keys.

- [x] Add failing frontend and backend contract tests.
- [x] Add composer input, warning modal, confirmation summary and submit gating.
- [x] Validate/map the required count server-side and record it in run metadata.
- [x] Update desktop integration tests and startup/workflow documentation.
- [x] Verify regression tests and actual Qt modal/responsive interactions.

Validation: 97 Python regression tests, 14 frontend tests, and 10 separate real
Qt6 tests passed. Missing/invalid count and missing primary/separate VLM key are
blocked; the confirmation excludes credentials and escapes user text; cancelling
preserves the draft; the launched payload is the reviewed snapshot; repeated
confirmation cannot create a second run. Real desktop tests exercised input
events, Enter, Escape, API-settings navigation, a fixture CLI launch/export, and
an 800-pixel viewport. Inspected composer, API warning, and configuration screenshots.
Qt logged its existing macOS compositor/input-method fallback warnings, but all
assertions passed. No scientific run or paid model request was made.

Do not stop the user's UI, install dependencies, or make paid model calls.
