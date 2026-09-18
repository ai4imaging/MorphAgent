"""LLM-authored fallback segmentation, verified by the VLM.

The paper's pipeline segments with Cellpose-SAM or Allen, which the Lite / UI
install deliberately does not ship (heavy CUDA + conda environments). Until now
a dataset without masks simply ran without segmentation, so every mask-dependent
feature was lost.

This module fills that gap. When a dataset carries no masks and no backend is
available it:

1. asks the LLM how many masks the research question actually needs, and what
   each one means (planning is grounded in the user query, the dataset
   description and the real image statistics);
2. asks the LLM to write a classical ``segment(img)`` (threshold + morphology)
   that returns those masks;
3. runs it on one sample and shows the raw image plus a mask overlay to the VLM,
   which accepts or rejects it with concrete feedback;
4. repeats steps 2-3 for at most ``max_rounds`` attempts, keeping the last
   executable code even if the VLM never accepts it;
5. applies the accepted code to every sample.

Masks land in ``sample_dir/segmentation/<mask_name>.tif`` with a ``README.txt``
holding their semantics, which is exactly the contract user-supplied masks
follow. Everything downstream — ``data_path_selector`` building the ``seg``
dict, ``collect_data_statistics`` describing the keys, the feature planner
reading ``mask_order_description`` — therefore picks the masks up unchanged.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# How many masks the planner may ask for. More than a handful stops being
# useful for classical thresholding and blows up the VLM check.
MAX_MASKS = 4
DEFAULT_MAX_ROUNDS = 3

_NAME_SAFE = re.compile(r"[^a-z0-9_]+")


@dataclass
class MaskPlan:
    """One mask the planner decided the analysis needs."""

    name: str
    semantics: str
    target: str = ""
    channel_hint: str = ""
    rationale: str = ""


@dataclass
class AutoSegmentationOutcome:
    """What the fallback produced, for the caller's summary and prompts."""

    applied: bool = False
    reason: str = ""
    masks: List[MaskPlan] = field(default_factory=list)
    results: Dict[str, str] = field(default_factory=dict)
    semantics: Dict[str, str] = field(default_factory=dict)
    rounds_used: int = 0
    vlm_status: str = "not_run"  # passed | rejected | unverified | not_run
    strategy: str = ""
    code: Optional[str] = None

    def summary(self) -> Dict[str, Any]:
        """A JSON-serialisable record for ``segmentation_summary.json``."""
        return {
            "applied": self.applied,
            "reason": self.reason,
            "rounds_used": self.rounds_used,
            "vlm_status": self.vlm_status,
            "strategy": self.strategy,
            "masks": [
                {
                    "name": mask.name,
                    "semantics": mask.semantics,
                    "target": mask.target,
                    "channel_hint": mask.channel_hint,
                    "rationale": mask.rationale,
                }
                for mask in self.masks
            ],
            "per_sample": self.results,
        }


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _log(message: str) -> None:
    """Print indented so ``StageDetector`` keeps treating this as part of Step 2.4."""
    print(f"  {message}", flush=True)


def _to_display_plane(array: Any) -> Any:
    """Import the runner's plane reduction so prompt, validator and overlay agree."""
    try:
        from tools.auto_segmentation_runner import to_display_plane
    except Exception:  # noqa: BLE001 - direct execution / packaging differences
        from .auto_segmentation_runner import to_display_plane  # type: ignore

    return to_display_plane(array)


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Pull the first balanced JSON object out of an LLM response."""
    if not text:
        return None
    start = text.find("{")
    while start >= 0:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start : index + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(parsed, dict):
                        return parsed
                    break
        start = text.find("{", start + 1)
    return None


def _sanitise_mask_name(raw: str, index: int, taken: set) -> str:
    """Normalise a planned name into a filename stem the pipeline will accept.

    ``list_segmentation_files`` drops stems that look like preview overlays
    (``rgb``, ``color``, ``overlay``, …), so a name the planner picked could
    silently make the mask invisible. Anything rejected falls back to a
    positional name.
    """
    try:
        from tools.image_io import is_segmentation_mask_filename
    except Exception:  # noqa: BLE001
        from .image_io import is_segmentation_mask_filename  # type: ignore

    name = _NAME_SAFE.sub("_", str(raw or "").strip().lower()).strip("_")
    if not name:
        name = f"mask_{index}"
    if not name.startswith("mask"):
        name = f"mask_{name}"
    name = re.sub(r"_+", "_", name).strip("_")
    if not is_segmentation_mask_filename(f"{name}.tif"):
        name = f"mask_{index}"
    base = name
    suffix = 2
    while name in taken:
        name = f"{base}_{suffix}"
        suffix += 1
    taken.add(name)
    return name


def _sandbox_python() -> Path:
    """Reuse the interpreter that runs generated feature code."""
    try:
        from config import settings
        from tools.code_executor import _find_conda_python

        env = getattr(settings, "conda_env", "") or ""
        if env:
            found = _find_conda_python(env)
            if found is not None:
                return found
    except Exception:  # noqa: BLE001 - any resolution failure falls back below
        pass
    return Path(sys.executable)


def _runner_path() -> Path:
    return Path(__file__).resolve().parent / "auto_segmentation_runner.py"


def _reference_stats(image_path: Path, dataset_description: Optional[str]) -> Dict[str, Any]:
    """Collect the image facts the planner and the code generator are shown."""
    stats: Dict[str, Any] = {}
    try:
        from tools.data_statistics import collect_data_statistics

        stats = collect_data_statistics(
            image_path=image_path,
            segmentation_paths=None,
            dataset_description=dataset_description,
        )
    except Exception as exc:  # noqa: BLE001
        _log(f"[WARN] Could not collect image statistics: {exc}")

    plane_shape: Optional[Tuple[int, ...]] = None
    try:
        from tools.image_io import load_image_array

        plane = _to_display_plane(load_image_array(image_path))
        plane_shape = tuple(int(size) for size in plane.shape)
    except Exception as exc:  # noqa: BLE001
        _log(f"[WARN] Could not determine the image plane shape: {exc}")
    stats["plane_shape"] = plane_shape
    return stats


def _stats_block(stats: Dict[str, Any], image_path: Path) -> str:
    """Render the image facts as a prompt section."""
    plane_shape = stats.get("plane_shape")
    lines = [
        f"- File: `{image_path.name}`",
        f"- Array shape: {stats.get('image_shape', 'Unknown')}",
        f"- Dtype: {stats.get('image_dtype', 'Unknown')}",
        f"- Intensity range: min={stats.get('image_min', 'Unknown')}, max={stats.get('image_max', 'Unknown')}",
    ]
    if plane_shape:
        lines.append(
            f"- Required mask shape: {tuple(plane_shape)} "
            "(height, width of the single 2-D plane every mask must match)"
        )
    channel_information = str(stats.get("channel_information") or "").strip()
    if channel_information:
        lines.append("- Channel layout:")
        lines.extend(f"  {line}" for line in channel_information.splitlines() if line.strip())
    return "\n".join(lines)


def _truncate(text: Optional[str], limit: int) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[:limit] + "\n… (truncated)"


# ---------------------------------------------------------------------------
# step 1 - plan which masks the question needs
# ---------------------------------------------------------------------------


_PLAN_PROMPT = """You are a bioimage analysis expert preparing a dataset for automated \
morphological profiling.

This dataset ships **no segmentation masks**, and no deep-learning segmentation backend \
(Cellpose-SAM, Allen) is installed. A lightweight classical segmentation \
(thresholding, morphology, watershed) will be written from scratch and applied to every \
sample, then used for feature quantification.

Your job is to decide **which masks this specific analysis needs** — not a generic set.

====================
Research question
====================
{user_query}

====================
Dataset description
====================
{dataset_description}

====================
Reference image (first sample)
====================
{image_stats}

====================
How to decide
====================
1. Read the research question and pick only the compartments the measurements genuinely \
need. One mask is often enough; never propose more than {max_masks}.
2. Each mask must be plausibly obtainable from intensity thresholding plus morphology on \
one of the channels above. Do not propose targets that need a trained model (for example \
individual organelles inside a dense cytoplasm, cell-cycle phase, or lineage).
3. If the question is about whole-object shape or texture and no compartment structure is \
needed, a single foreground mask is the right answer.
4. Prefer instance labels (one integer per object) when per-object statistics matter, and \
a binary mask when only region statistics matter.
5. Name each mask in snake_case starting with `mask_` (for example `mask_nucleus`, \
`mask_cell`, `mask_foreground`). Avoid the words rgb, color, overlay, preview, \
visualization and summary — those names are reserved for preview images.

====================
Output format
====================
Reply with JSON only, no prose:

{{
  "strategy": "one or two sentences on the overall segmentation approach",
  "masks": [
    {{
      "name": "mask_nucleus",
      "target": "what physical object this mask covers",
      "semantics": "short label used in downstream prompts, e.g. 'nucleus (instance labels)'",
      "channel_hint": "which channel to threshold and why",
      "rationale": "why the research question needs this mask"
    }}
  ]
}}
"""


def _fallback_plan() -> Tuple[List[MaskPlan], str]:
    """A single foreground mask: always meaningful, never worse than no masks."""
    return (
        [
            MaskPlan(
                name="mask_foreground",
                semantics="foreground objects (whole-image intensity threshold)",
                target="stained foreground objects",
                rationale="planning fell back to a generic foreground mask",
            )
        ],
        "Threshold the brightest structural channel to obtain a foreground mask.",
    )


def _plan_masks(
    user_query: str,
    dataset_description: Optional[str],
    stats: Dict[str, Any],
    image_path: Path,
    artifacts_dir: Optional[Path],
) -> Tuple[List[MaskPlan], str]:
    """Ask the LLM which masks to build; fall back to a foreground mask on any failure."""
    prompt = _PLAN_PROMPT.format(
        user_query=_truncate(user_query, 4000) or "(not provided)",
        dataset_description=_truncate(dataset_description, 8000) or "(not provided)",
        image_stats=_stats_block(stats, image_path),
        max_masks=MAX_MASKS,
    )

    response = ""
    try:
        from config import get_code_temperature, make_chat_llm, settings
        from langchain_core.messages import HumanMessage

        kwargs: Dict[str, Any] = {"temperature": get_code_temperature()}
        if getattr(settings, "reproduce_mode", False):
            kwargs["seed"] = settings.reproduce_seed
        llm = make_chat_llm(**kwargs)
        raw = llm.invoke([HumanMessage(content=prompt)])
        response = raw.content if hasattr(raw, "content") else str(raw)
    except Exception as exc:  # noqa: BLE001
        _log(f"[WARN] Mask planning LLM call failed ({exc}); using a foreground mask")

    if artifacts_dir is not None:
        _write_text(artifacts_dir / "plan_prompt.txt", prompt)
        _write_text(artifacts_dir / "plan_response.txt", response)

    parsed = _extract_json_object(response)
    if not parsed:
        return _fallback_plan()

    entries = parsed.get("masks")
    if not isinstance(entries, list) or not entries:
        return _fallback_plan()

    taken: set = set()
    masks: List[MaskPlan] = []
    for index, entry in enumerate(entries[:MAX_MASKS], start=1):
        if not isinstance(entry, dict):
            continue
        name = _sanitise_mask_name(entry.get("name", ""), index, taken)
        target = str(entry.get("target") or "").strip()
        semantics = str(entry.get("semantics") or "").strip() or target or f"segmentation mask `{name}`"
        masks.append(
            MaskPlan(
                name=name,
                semantics=semantics,
                target=target,
                channel_hint=str(entry.get("channel_hint") or "").strip(),
                rationale=str(entry.get("rationale") or "").strip(),
            )
        )

    if not masks:
        return _fallback_plan()

    strategy = str(parsed.get("strategy") or "").strip()
    return masks, strategy


# ---------------------------------------------------------------------------
# step 2 - write the segmentation code
# ---------------------------------------------------------------------------


_CODE_PROMPT = """You are writing a classical segmentation routine for a bioimage dataset \
that has no masks and no deep-learning segmentation backend available.

====================
Research question
====================
{user_query}

====================
Dataset description
====================
{dataset_description}

====================
Reference image (first sample)
====================
{image_stats}

====================
Masks to produce
====================
{mask_block}

Overall strategy agreed during planning: {strategy}

====================
Contract (must be followed exactly)
====================
Write a single self-contained Python module that defines:

    def segment(img):
        ...
        return {{{example_keys}}}

- `img` is the raw array loaded straight from the sample file, with the shape and dtype \
listed above. Handle that layout explicitly; do not assume a different one.
- Return a dict whose keys are exactly: {key_list}.
- Every value must be a 2-D numpy array of shape {plane_shape}: either a boolean/0-1 \
binary mask, or an instance-label map with one positive integer per object and 0 for \
background.
- Import everything you use **inside** `segment` or at module top level from numpy, \
scipy and scikit-image only. Never import cellpose, torch, tensorflow, aicsimageio or \
aicssegmentation, and never download anything.
- No file I/O, no printing, no plotting, no `if __name__ == "__main__"` block.
- Be defensive: cast to float, guard against all-zero or constant channels, and fall back \
to a percentile threshold if Otsu degenerates. `segment` must never raise.

====================
Quality requirements
====================
- Normalise intensities robustly (for example percentile clipping) before thresholding, \
since absolute intensity scales vary between samples.
- Smooth lightly before thresholding, then clean up with morphological opening/closing and \
small-object removal so the mask follows real object outlines.
- For instance labels, separate touching objects with a distance-transform watershed \
(`scipy.ndimage.distance_transform_edt` + `skimage.segmentation.watershed` on local \
maxima) rather than returning one merged blob.
- Nested compartments must stay consistent: a nucleus mask should lie inside its cell mask.
- A mask that is empty or covers the whole frame is a failure. Choose thresholds that leave \
a sensible fraction of the image as foreground.

Reply with the Python code only, inside a single ```python code block.
{feedback_block}"""


def _mask_block(masks: List[MaskPlan]) -> str:
    lines = []
    for index, mask in enumerate(masks, start=1):
        lines.append(f"{index}. `{mask.name}`")
        lines.append(f"   - Target: {mask.target or mask.semantics}")
        lines.append(f"   - Downstream label: {mask.semantics}")
        if mask.channel_hint:
            lines.append(f"   - Channel hint: {mask.channel_hint}")
        if mask.rationale:
            lines.append(f"   - Needed because: {mask.rationale}")
    return "\n".join(lines)


def _feedback_block(attempts: List[Dict[str, Any]]) -> str:
    """Replay earlier attempts so the LLM fixes them instead of repeating them."""
    if not attempts:
        return ""
    parts = [
        "\n====================",
        "Previous attempts (fix these problems)",
        "====================",
    ]
    for attempt in attempts:
        parts.append(f"\n--- Attempt {attempt['round']} ---")
        code = _truncate(attempt.get("code"), 6000)
        if code:
            parts.append("Code:\n```python\n" + code + "\n```")
        if attempt.get("error"):
            parts.append("It failed to run:\n" + _truncate(attempt["error"], 3000))
        if attempt.get("mask_stats"):
            parts.append("Mask statistics:\n" + json.dumps(attempt["mask_stats"], indent=2))
        if attempt.get("vlm_feedback"):
            parts.append(
                "A vision model compared the masks against the raw image and rejected them:\n"
                + _truncate(attempt["vlm_feedback"], 3000)
            )
    parts.append(
        "\nWrite a materially different implementation that addresses the feedback above. "
        "Do not resubmit the same thresholds."
    )
    return "\n".join(parts)


def _extract_code(text: str) -> Optional[str]:
    """Pull the Python module out of the response, tolerating missing fences."""
    if not text:
        return None

    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    for block in blocks:
        if "def segment" in block:
            return block.strip() + "\n"
    if blocks:
        candidate = blocks[0].strip()
        if candidate:
            return candidate + "\n"

    index = text.find("def segment")
    if index < 0:
        return None
    # Keep any imports that precede the function definition.
    head = text[:index]
    import_start = len(head)
    for match in re.finditer(r"^(?:import|from)\s+\S+", head, re.MULTILINE):
        import_start = min(import_start, match.start())
    return text[import_start:].strip() + "\n"


def _generate_segment_code(
    user_query: str,
    dataset_description: Optional[str],
    stats: Dict[str, Any],
    image_path: Path,
    masks: List[MaskPlan],
    strategy: str,
    attempts: List[Dict[str, Any]],
) -> Tuple[Optional[str], str, str]:
    """Return ``(code, prompt, response)``; ``code`` is None when extraction fails."""
    plane_shape = stats.get("plane_shape")
    key_list = ", ".join(f'"{mask.name}"' for mask in masks)
    prompt = _CODE_PROMPT.format(
        user_query=_truncate(user_query, 4000) or "(not provided)",
        dataset_description=_truncate(dataset_description, 8000) or "(not provided)",
        image_stats=_stats_block(stats, image_path),
        mask_block=_mask_block(masks),
        strategy=strategy or "(not recorded)",
        example_keys=", ".join(f'"{mask.name}": ...' for mask in masks),
        key_list=key_list,
        plane_shape=tuple(plane_shape) if plane_shape else "(height, width)",
        feedback_block=_feedback_block(attempts),
    )

    response = ""
    try:
        from config import get_code_temperature, make_chat_llm, settings
        from langchain_core.messages import HumanMessage

        kwargs: Dict[str, Any] = {"temperature": get_code_temperature()}
        if getattr(settings, "reproduce_mode", False):
            kwargs["seed"] = settings.reproduce_seed
        llm = make_chat_llm(**kwargs)
        raw = llm.invoke([HumanMessage(content=prompt)])
        response = raw.content if hasattr(raw, "content") else str(raw)
    except Exception as exc:  # noqa: BLE001
        _log(f"[WARN] Segmentation code generation failed: {exc}")
        return None, prompt, response

    return _extract_code(response), prompt, response


# ---------------------------------------------------------------------------
# step 3 - run the code
# ---------------------------------------------------------------------------


def _run_segment_code(
    code: str,
    image_path: Path,
    out_dir: Path,
    mask_names: List[str],
    *,
    preview_dir: Optional[Path] = None,
    strict: bool,
    timeout: int,
    work_dir: Path,
) -> Dict[str, Any]:
    """Execute the generated ``segment`` in the sandbox interpreter.

    Returns the runner's JSON payload, or ``{"success": False, "error": ...}``.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    code_path = work_dir / "segment_code.py"
    names_path = work_dir / "mask_names.json"
    code_path.write_text(code, encoding="utf-8")
    names_path.write_text(json.dumps(mask_names), encoding="utf-8")

    command = [
        str(_sandbox_python()),
        str(_runner_path()),
        "--code",
        str(code_path),
        "--image",
        str(Path(image_path).resolve()),
        "--out-dir",
        str(Path(out_dir).resolve()),
        "--names-json",
        str(names_path),
    ]
    if preview_dir is not None:
        command.extend(["--preview-dir", str(Path(preview_dir).resolve())])
    if strict:
        command.append("--strict")

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"segmentation timed out after {timeout}s"}
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "error": f"could not launch the segmentation runner: {exc}"}

    payload = _extract_json_object(completed.stdout or "")
    if payload is None:
        detail = (completed.stderr or completed.stdout or "").strip()
        return {
            "success": False,
            "error": f"runner produced no JSON (exit {completed.returncode}): {detail[-2000:]}",
        }
    if completed.stderr:
        payload.setdefault("stderr", completed.stderr[-2000:])
    return payload


# ---------------------------------------------------------------------------
# step 4 - VLM verification
# ---------------------------------------------------------------------------


_VLM_PROMPT = """You are verifying an automatically generated segmentation before it is used \
for quantitative morphological profiling.

====================
Images (in this order)
====================
{image_manifest}

In every overlay the mask interior is tinted and its outline drawn in a solid colour on top \
of the same raw image shown first.

====================
What each mask should be
====================
{mask_block}

Analysis context: {user_query}

====================
How to judge
====================
Accept a mask when the outlines follow the intended objects well enough for shape, \
intensity and texture measurements. Small boundary imprecision and a few missed faint \
objects are acceptable.

Reject a mask when it:
- covers background instead of the intended object, or the wrong structure entirely;
- is essentially empty, or floods most of the frame;
- merges clearly separate objects into single blobs when per-object measurement is intended;
- is fragmented into noise speckle rather than coherent objects;
- is nested inconsistently (for example a nucleus mask extending outside its cell).

====================
Output format
====================
Reply with JSON only, no prose:

{{
  "masks": {{
{per_mask_schema}
  }},
  "passed": true,
  "feedback": "if anything was rejected, say concretely what is wrong and how the thresholding or morphology should change"
}}

`passed` must be true only when every mask is acceptable.
"""


def _vlm_freeform(prompt: str, image_paths: List[str], log_file: Optional[Path] = None) -> str:
    """Ask the configured VLM a free-form question about images.

    Mirrors the critic agent's dual online / local-Qwen handling in
    ``tools.code_executor.evaluate_result_with_critic``.
    """
    from config import settings
    from tools.vlm_client import get_vlm_client

    client = get_vlm_client()
    client._load_model()
    processed = client._preprocess_images(image_paths, log_file)
    resolved = [str(Path(path).resolve()) for path in processed]

    try:
        if settings.vlm_api_provider == "online":
            content = client._images_to_content(resolved)
            content.append({"type": "text", "text": prompt})
            return client._chat_with_retry(content, log_file)

        import torch
        from config import get_vlm_temperature

        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": path} for path in resolved],
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = client._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        device = next(client._model.parameters()).device
        device_inputs = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }
        del inputs
        with torch.no_grad():
            generated = client._model.generate(
                **device_inputs,
                max_new_tokens=768,
                do_sample=False,
                temperature=get_vlm_temperature(),
            )
        trimmed = [
            output[len(prompt_ids) :]
            for prompt_ids, output in zip(device_inputs["input_ids"], generated)
        ]
        return client._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
    finally:
        if getattr(client, "_temp_dirs", None):
            client.cleanup_temp_files()


def _verify_with_vlm(
    masks: List[MaskPlan],
    preview: Dict[str, str],
    user_query: str,
    artifacts_dir: Optional[Path],
    round_number: int,
) -> Tuple[str, str]:
    """Show the raw image and one overlay per mask to the VLM.

    Returns ``(status, feedback)`` where status is ``passed``, ``rejected`` or
    ``unverified``. ``unverified`` means the VLM itself could not be reached, so
    the caller must not spend another round on it.
    """
    original = preview.get("original")
    ordered = [mask for mask in masks if preview.get(mask.name)]
    if not original or not ordered:
        return "unverified", "no preview images were produced, so the masks could not be checked"

    image_paths = [original] + [preview[mask.name] for mask in ordered]
    manifest_lines = ["1. The raw image, contrast-stretched, with no mask."]
    for index, mask in enumerate(ordered, start=2):
        manifest_lines.append(f"{index}. Overlay of `{mask.name}` on the same raw image.")

    prompt = _VLM_PROMPT.format(
        image_manifest="\n".join(manifest_lines),
        mask_block=_mask_block(ordered),
        user_query=_truncate(user_query, 1500) or "(not provided)",
        per_mask_schema=",\n".join(
            f'    "{mask.name}": {{"passed": true, "reason": "what the outlines actually follow"}}'
            for mask in ordered
        ),
    )

    try:
        response = _vlm_freeform(prompt, image_paths)
    except Exception as exc:  # noqa: BLE001
        _log(f"[WARN] VLM check unavailable ({exc}); accepting the current masks unverified")
        return "unverified", f"the VLM could not be reached: {exc}"

    if artifacts_dir is not None:
        _write_text(artifacts_dir / f"vlm_check_round_{round_number}.txt", prompt + "\n\n---\n\n" + response)

    parsed = _extract_json_object(response)
    if not parsed:
        _log("[WARN] VLM response could not be parsed; accepting the current masks unverified")
        return "unverified", "the VLM response could not be parsed"

    per_mask = parsed.get("masks") if isinstance(parsed.get("masks"), dict) else {}
    verdicts: List[str] = []
    all_passed = True
    for mask in ordered:
        entry = per_mask.get(mask.name)
        if isinstance(entry, dict):
            passed = bool(entry.get("passed", True))
            reason = str(entry.get("reason") or "").strip()
        else:
            passed = True
            reason = ""
        if not passed:
            all_passed = False
        verdicts.append(f"{mask.name}: {'OK' if passed else 'REJECTED'}{f' — {reason}' if reason else ''}")

    if "passed" in parsed:
        all_passed = all_passed and bool(parsed.get("passed"))

    feedback = str(parsed.get("feedback") or "").strip()
    detail = "\n".join(verdicts)
    for line in verdicts:
        _log(f"  [VLM] {line}")

    if all_passed:
        return "passed", detail
    return "rejected", (feedback + "\n\nPer-mask verdicts:\n" + detail).strip()


# ---------------------------------------------------------------------------
# step 5 - apply to every sample
# ---------------------------------------------------------------------------


def _write_text(path: Path, text: str) -> None:
    """Best-effort artifact write; never break the run over a log file."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text or "", encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        _log(f"[WARN] Could not write {path.name}: {exc}")


def _write_semantics_readme(seg_dir: Path, masks: List[MaskPlan]) -> None:
    """Record ``stem: meaning`` lines.

    ``tools.data_statistics._parse_segmentation_semantics`` reads this file, so
    the planner's wording reaches the code-generation prompt the same way a
    user-written README would.
    """
    lines = [f"{mask.name}: {mask.semantics}" for mask in masks]
    lines.append("")
    lines.append("# Generated by MorphAgent auto-segmentation (LLM-written, VLM-checked).")
    _write_text(seg_dir / "README.txt", "\n".join(lines))


def _apply_to_all_samples(
    code: str,
    sample_ids: List[str],
    data_root: Path,
    dataset_description: Optional[str],
    masks: List[MaskPlan],
    work_dir: Path,
    previews_dir: Optional[Path],
    timeout: int,
) -> Dict[str, str]:
    """Run the accepted code on every sample; a per-sample failure never aborts."""
    from utils_helpers import find_image_paths

    mask_names = [mask.name for mask in masks]
    results: Dict[str, str] = {}
    total = len(sample_ids)

    for index, sample_id in enumerate(sample_ids, start=1):
        sample_dir = data_root / sample_id
        if not sample_dir.is_dir():
            _log(f"[{index}/{total}] [WARN] {sample_id}: sample directory is missing")
            results[sample_id] = "failed"
            continue

        try:
            image_paths = find_image_paths(sample_dir, dataset_description)
        except Exception as exc:  # noqa: BLE001
            _log(f"[{index}/{total}] [WARN] {sample_id}: image lookup failed ({exc})")
            results[sample_id] = "failed"
            continue
        if not image_paths:
            _log(f"[{index}/{total}] [WARN] {sample_id}: no image file found")
            results[sample_id] = "failed"
            continue

        # Only the first sample keeps previews, so the artifacts stay small.
        sample_preview = previews_dir / sample_id if previews_dir is not None and index == 1 else None
        payload = _run_segment_code(
            code,
            Path(image_paths[0]),
            sample_dir / "segmentation",
            mask_names,
            preview_dir=sample_preview,
            strict=False,
            timeout=timeout,
            work_dir=work_dir,
        )

        if not payload.get("success"):
            _log(f"[{index}/{total}] [WARN] {sample_id}: {payload.get('error', 'unknown error')}")
            results[sample_id] = "failed"
            continue

        written = payload.get("written") or []
        if not written:
            _log(f"[{index}/{total}] [WARN] {sample_id}: no masks were written")
            results[sample_id] = "failed"
            continue

        _write_semantics_readme(sample_dir / "segmentation", masks)
        problems = payload.get("problems") or []
        if problems:
            _log(f"[{index}/{total}] [OK] {sample_id}: {len(written)} mask(s), with warnings: {'; '.join(problems)}")
        else:
            _log(f"[{index}/{total}] [OK] {sample_id}: {len(written)} mask(s) written")
        results[sample_id] = "success"

    return results


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def dataset_has_segmentation(sample_ids: List[str], data_root: Path) -> bool:
    """True when any sample already carries masks, using the pipeline's own scan."""
    try:
        from tools.segmentation import list_segmentation_files
    except Exception:  # noqa: BLE001
        from .segmentation import list_segmentation_files  # type: ignore

    for sample_id in sample_ids:
        try:
            if list_segmentation_files(data_root / sample_id):
                return True
        except Exception:  # noqa: BLE001
            continue
    return False


def run_auto_segmentation(
    sample_ids: List[str],
    data_root: Path,
    user_query: str,
    dataset_description: Optional[str] = None,
    results_dir: Optional[Path] = None,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    timeout: Optional[int] = None,
) -> AutoSegmentationOutcome:
    """Plan, write, VLM-check and apply a fallback segmentation.

    Never raises: any failure returns ``applied=False`` so the caller simply
    continues without masks, which is the pre-existing behaviour.
    """
    try:
        return _run_auto_segmentation(
            sample_ids=sample_ids,
            data_root=Path(data_root),
            user_query=user_query,
            dataset_description=dataset_description,
            results_dir=Path(results_dir) if results_dir is not None else None,
            max_rounds=max(1, int(max_rounds or DEFAULT_MAX_ROUNDS)),
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001 - segmentation is best-effort
        import traceback

        _log(f"[WARN] Auto-segmentation aborted: {exc}")
        traceback.print_exc()
        return AutoSegmentationOutcome(applied=False, reason=f"aborted: {exc}")


def _run_auto_segmentation(
    sample_ids: List[str],
    data_root: Path,
    user_query: str,
    dataset_description: Optional[str],
    results_dir: Optional[Path],
    max_rounds: int,
    timeout: Optional[int],
) -> AutoSegmentationOutcome:
    from utils_helpers import find_image_paths

    if not sample_ids:
        return AutoSegmentationOutcome(applied=False, reason="no samples to segment")

    if timeout is None:
        try:
            from config import settings

            timeout = int(getattr(settings, "code_sandbox_timeout", 300))
        except Exception:  # noqa: BLE001
            timeout = 300

    artifacts_dir = results_dir / "auto_segmentation" if results_dir is not None else None
    if artifacts_dir is not None:
        try:
            artifacts_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            _log(f"[WARN] Could not create the artifacts directory ({exc}); continuing without it")
            artifacts_dir = None

    # Pick the first sample that actually has an image as the trial sample.
    reference_id: Optional[str] = None
    reference_image: Optional[Path] = None
    for sample_id in sample_ids:
        sample_dir = data_root / sample_id
        if not sample_dir.is_dir():
            continue
        try:
            found = find_image_paths(sample_dir, dataset_description)
        except Exception:  # noqa: BLE001
            found = []
        if found:
            reference_id = sample_id
            reference_image = Path(found[0])
            break
    if reference_image is None:
        return AutoSegmentationOutcome(applied=False, reason="no readable image found in any sample")

    _log("[Auto-Seg] No masks in this dataset and no segmentation backend available.")
    _log(f"[Auto-Seg] Writing a classical segmentation with the LLM (trial sample: {reference_id}).")

    stats = _reference_stats(reference_image, dataset_description)
    masks, strategy = _plan_masks(user_query, dataset_description, stats, reference_image, artifacts_dir)
    _log(
        f"[Auto-Seg] Planned {len(masks)} mask(s): "
        + ", ".join(f"{mask.name} ({mask.semantics})" for mask in masks)
    )
    if artifacts_dir is not None:
        _write_text(
            artifacts_dir / "mask_plan.json",
            json.dumps(
                {
                    "strategy": strategy,
                    "trial_sample": reference_id,
                    "masks": [mask.__dict__ for mask in masks],
                },
                indent=2,
                ensure_ascii=False,
            ),
        )

    mask_names = [mask.name for mask in masks]
    attempts: List[Dict[str, Any]] = []
    accepted_code: Optional[str] = None
    accepted_status = "not_run"
    fallback_code: Optional[str] = None
    rounds_used = 0

    scratch = Path(tempfile.mkdtemp(prefix="morphagent_autoseg_"))
    try:
        for round_number in range(1, max_rounds + 1):
            rounds_used = round_number
            _log(f"[Auto-Seg] Round {round_number}/{max_rounds}: generating segmentation code…")

            code, prompt, response = _generate_segment_code(
                user_query,
                dataset_description,
                stats,
                reference_image,
                masks,
                strategy,
                attempts,
            )
            round_dir = artifacts_dir / f"round_{round_number}" if artifacts_dir is not None else None
            if round_dir is not None:
                _write_text(round_dir / "code_prompt.txt", prompt)
                _write_text(round_dir / "code_response.txt", response)

            if not code:
                _log("[Auto-Seg] Could not extract code from the response.")
                attempts.append(
                    {"round": round_number, "code": "", "error": "no Python code could be extracted"}
                )
                continue

            if round_dir is not None:
                _write_text(round_dir / "segment_code.py", code)

            trial_out = scratch / f"round_{round_number}" / "masks"
            trial_preview = (round_dir / "preview") if round_dir is not None else (
                scratch / f"round_{round_number}" / "preview"
            )
            payload = _run_segment_code(
                code,
                reference_image,
                trial_out,
                mask_names,
                preview_dir=trial_preview,
                strict=True,
                timeout=timeout,
                work_dir=scratch / f"round_{round_number}" / "work",
            )
            if round_dir is not None:
                _write_text(round_dir / "run_result.json", json.dumps(payload, indent=2, ensure_ascii=False))

            if not payload.get("success"):
                error = str(payload.get("error", "unknown error"))
                traceback_text = str(payload.get("traceback") or "")
                _log(f"[Auto-Seg] Trial run failed: {error.splitlines()[0][:200]}")
                attempts.append(
                    {
                        "round": round_number,
                        "code": code,
                        "error": (error + "\n" + traceback_text).strip(),
                    }
                )
                continue

            mask_stats = payload.get("masks") or {}
            _log(
                "[Auto-Seg] Trial masks: "
                + ", ".join(
                    f"{name} ({info.get('object_count', 0)} objects, "
                    f"{float(info.get('foreground_fraction', 0.0)) * 100:.1f}% foreground)"
                    for name, info in mask_stats.items()
                )
            )

            # Keep the newest executable code: if the VLM never accepts anything,
            # this is what gets applied, per the "use it anyway" requirement.
            fallback_code = code

            status, feedback = _verify_with_vlm(
                masks, payload.get("preview") or {}, user_query, round_dir, round_number
            )
            if status in {"passed", "unverified"}:
                accepted_code = code
                accepted_status = status
                _log(
                    "[Auto-Seg] VLM accepted the masks."
                    if status == "passed"
                    else "[Auto-Seg] VLM check unavailable; keeping the current masks."
                )
                break

            _log(f"[Auto-Seg] VLM rejected the masks in round {round_number}.")
            attempts.append(
                {
                    "round": round_number,
                    "code": code,
                    "mask_stats": mask_stats,
                    "vlm_feedback": feedback,
                }
            )

        if accepted_code is None and fallback_code is not None:
            accepted_code = fallback_code
            accepted_status = "rejected"
            _log(
                f"[Auto-Seg] No attempt passed the VLM check in {max_rounds} rounds; "
                "applying the last working segmentation anyway."
            )

        if accepted_code is None:
            if artifacts_dir is not None:
                _write_text(
                    artifacts_dir / "summary.json",
                    json.dumps(
                        AutoSegmentationOutcome(
                            applied=False,
                            reason="no attempt produced runnable segmentation code",
                            masks=masks,
                            rounds_used=rounds_used,
                            strategy=strategy,
                        ).summary(),
                        indent=2,
                        ensure_ascii=False,
                    ),
                )
            _log("[Auto-Seg] Giving up; the run continues without masks.")
            return AutoSegmentationOutcome(
                applied=False,
                reason="no attempt produced runnable segmentation code",
                masks=masks,
                rounds_used=rounds_used,
                strategy=strategy,
            )

        _log(f"[Auto-Seg] Applying the segmentation to all {len(sample_ids)} sample(s)…")
        results = _apply_to_all_samples(
            accepted_code,
            sample_ids,
            data_root,
            dataset_description,
            masks,
            scratch / "apply",
            (artifacts_dir / "previews") if artifacts_dir is not None else None,
            timeout,
        )

        succeeded = sum(1 for value in results.values() if value == "success")
        _log(f"[Auto-Seg] Done: {succeeded}/{len(sample_ids)} sample(s) segmented.")

        outcome = AutoSegmentationOutcome(
            applied=succeeded > 0,
            reason="" if succeeded else "the segmentation could not be applied to any sample",
            masks=masks,
            results=results,
            semantics={mask.name: mask.semantics for mask in masks},
            rounds_used=rounds_used,
            vlm_status=accepted_status,
            strategy=strategy,
            code=accepted_code,
        )
        if artifacts_dir is not None:
            _write_text(artifacts_dir / "segment_code.py", accepted_code)
            _write_text(
                artifacts_dir / "summary.json",
                json.dumps(outcome.summary(), indent=2, ensure_ascii=False),
            )
        return outcome
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
