# MorphAgent workspace · UI design preview

**For real analysis**, run `python launch_web_ui.py` from the repository root
with `morphagent_lite` activated. See [the runtime guide](../README_WEB.md).
The same visual interface is connected to the original backend by `runtime.js`.
The instructions below are only for the standalone, offline design preview.

A standalone, Codex-inspired interface prototype. It does not replace the desktop
UI or execute the MorphAgent pipeline. No packages, API keys, or model calls are
needed to preview it.

## Open the preview

From the repository root:

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory design-preview
```

Open **http://127.0.0.1:8765** in your browser. Press `Ctrl+C` in the terminal to
stop the preview server. You can also open `index.html` directly in a browser.

## What you can try

- **Design:** add a local image folder, or load the Tau sample and edit the question.
- **Settings → Model API:** edit connection fields; use a separate VLM connection.
- **Settings → Analysis:** choose Code + VLM, Code, or VLM; select Ultra fast
  (1 loop), Fast (5 loops), or Detailed (20 loops).
- **Upload knowledge**, beside Add data: attach PDF, Word, TXT, or Markdown;
  click **Knowledge attached** to inspect, add or remove references.
- **Design, after confirmation:** play an explicitly simulated workflow
  on the same page (static preview only; no analysis takes place).
- **Compute:** preview the previous-run and target-data composer. Loading and
  executing saved scripts requires the desktop/browser runtime, not this offline preview.
- **Visualize:** browse original demo images. There is no separate Save page;
  real results are saved and downloaded from the runtime's run monitor.

All settings and selections are held in memory and reset on reload. API keys are
not saved, exported, or sent. Documents are not parsed or uploaded. Image previews
are shared dataset context, not feature-specific heatmaps.

## Files

- `index.html` — entry point
- `styles.css` — responsive layout and visual styles
- `app.js` — client-side prototype state and interactions
- `assets/tau-*.png` — sample images from the existing local Tau demo
