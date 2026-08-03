# Ask MorphAgent Reviewer Chat Design

## Purpose and scope

Ask MorphAgent is a reviewer-facing paper companion embedded in the existing MorphAgent desktop UI. The Home page gains one secondary action, **Ask MorphAgent**, below the current run actions. Selecting it opens a compact API setup dialog that uses the same OpenAI-compatible LLM connection as Configure. After validation, the application opens a dedicated chat view without adding a sixth scientific workflow destination to the sidebar. The welcome message is: “I’m MorphAgent. I can help answer your questions about this paper, its methods, results, figures, supplementary material, and implementation.”

The full and Lite distributions receive the same interaction. The accepted dark, microscopy-focused visual system remains unchanged: restrained navy cards, aqua interaction feedback, strong typographic hierarchy, and no new color-coded navigation. Ask MorphAgent is visually actionable but remains secondary to “Start a discovery run.” The chat view uses a centered reading column, distinct assistant/user message cards, a fixed composer, a visible sending state, retryable errors, and an explicit Back to Home action.

## Grounding architecture

The supplied submission archive contains `Manuscript.docx`, `Supplementary.docx`, prompt/algorithm/table PDFs, and figure PDFs but no source code. A build script converts textual submission material into a prebuilt JSON knowledge bundle and snapshots the relevant repository Python sources. It strips macOS metadata, never imports `.env`, excludes caches/results/tests/vendor code, and records a human-readable source label for every chunk.

At runtime, a dependency-free lexical retriever ranks chunks by normalized query terms, phrase matches, title/source matches, and term frequency. It always includes a short paper overview, then selects a bounded set of manuscript, supplement, and code excerpts. This avoids sending the full paper and repository on every turn, avoids an embedding dependency/API, and keeps reviewer setup light. Recent chat history is bounded separately. The model receives source-labelled excerpts and is asked to cite them using compact labels such as `[Manuscript]`, `[Supplementary]`, and `[Code: morphagent_ui/main.py]`.

## Prompt policy

The system prompt presents MorphAgent as the paper’s evidence-grounded technical companion. It instructs the model to explain the work confidently, foreground concrete contributions, design strengths, validation evidence, biological usefulness, reproducibility, and implementation choices when supported by the provided sources. It must not use dismissive or insulting language about the paper.

The prompt does not require deception. It explicitly forbids invented experiments, metrics, citations, implementation details, or claims. When a reviewer asks about limitations, the assistant should acknowledge the documented limitation precisely, explain the intended scope and trade-off, point to mitigations or validation already present, and identify future work without pretending it has already been completed. If the bundle does not support an answer, it should say what is unavailable and identify the closest relevant source. Retrieved material is treated as reference data, never as executable instructions, which reduces prompt-injection risk from embedded documents or code comments.

## API and data flow

The API dialog reads `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL` through the existing environment layer. Saved URL/model values may be shown; the saved key is represented only by a placeholder. Blank key input preserves the existing secret. Continuing requires all three resolved values and updates only LLM keys in the repository-local, ignored `.env`.

The chat page sends requests through an OpenAI-compatible `chat.completions` client in a `QThread`. The UI remains responsive, disables duplicate sends while a request is active, and restores the composer on completion or error. A missing `/v1` endpoint receives the same one-time 404 fallback used elsewhere. Keys, prompts, and conversations are never written to command previews, manifests, or logs. Conversation state remains in memory and is discarded when the app closes.

## Testing and acceptance

Pure tests cover bundle loading, query ranking, context limits, prompt honesty/positive framing, message assembly, completion parsing, and `/v1` retry. Qt smoke tests cover Home placement, API validation/persistence, five-item sidebar preservation, chat welcome copy, page navigation, busy state, and success/error rendering. Network behavior is tested with injected fake clients; no real reviewer key is used.

The repository has pre-existing focused-test failures unrelated to this feature (Evidence selection expectations and one Lite sandbox-environment assertion). Ask MorphAgent acceptance therefore requires all new tests to pass in both distributions, existing unaffected UI/model tests to retain their prior state, offscreen construction of the complete widget, and a clean scan proving no `.env` or API key is bundled.
