# Knowledge attachments in the Data composer

Approved UX: move knowledge out of Settings and beside Add data. Empty state is
Upload knowledge; after upload it is Knowledge attached. Click again to inspect,
add or remove files; removing the last file restores the empty label.

Preserve the existing neutral design. Use an accessible dialog for attachment
management. File names open a clearly labelled extracted-text preview (the actual
model input); keep original PDF/DOCX/TXT download available. No new dependencies.

- [x] Remove the Settings knowledge tab/switch and add composer attachment status.
- [x] Add attachment list, upload feedback, preview/download, removal and empty state.
- [x] Derive per-run external-knowledge enablement from actual attached files.
- [x] Add authenticated, bounded preview/original access; removal stays recoverable.
- [x] Test frontend transitions, safe API access, run inputs and real desktop UI.

Verification: 87 Python regression tests, 10 frontend tests, and 9 separate real
Qt6 desktop tests passed. The desktop flow covers PDF/TXT uploads, extracted-text
preview, escaped document content, original-file download, removal, focus return,
and an 800-pixel-wide window without horizontal page overflow. Desktop screenshots
of the attached state and management dialog were inspected. No paid API calls or
scientific analysis were performed, and the user's running workspace was not stopped.

Do not stop the user's UI or run paid analysis. Use isolated test workspaces.
