"""Reuse completed MorphAgent code features on a new dataset without LLM/VLM calls."""

from __future__ import annotations

import json
import math
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from tools.code_executor import CodeExecutor, ExtractionResult


ProgressCallback = Callable[[str], None]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}


@dataclass(frozen=True)
class ReuseRoundSpec:
    round_number: int
    source_round_dir: Path
    feature_names: tuple[str, ...]
    feature_defs: tuple[dict[str, Any], ...]
    merged_code_path: Path
    skipped_vlm_names: tuple[str, ...] = ()


@dataclass
class ReuseSummary:
    source_results: Path
    data_root: Path
    output_dir: Path
    rounds: list[ReuseRoundSpec] = field(default_factory=list)
    skipped_rounds: list[dict[str, Any]] = field(default_factory=list)
    feature_columns: list[str] = field(default_factory=list)
    sample_ids: list[str] = field(default_factory=list)
    features_csv: Path | None = None


def _normalize_feature_key(name: str) -> str:
    text = (name or "").lower().strip()
    text = re.sub(r"[\s/()]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def _get_merged_feature_value(sample_features: Any, feature_display_name: str) -> Any:
    if not isinstance(sample_features, dict):
        return np.nan
    if feature_display_name in sample_features:
        return sample_features[feature_display_name]
    norm_display = _normalize_feature_key(feature_display_name)
    if not norm_display:
        return np.nan
    if norm_display in sample_features:
        return sample_features[norm_display]
    candidates: list[tuple[int, Any]] = []
    for key, value in sample_features.items():
        norm_key = _normalize_feature_key(str(key))
        if norm_key == norm_display:
            return value
        if norm_display.startswith(norm_key) or norm_key.startswith(norm_display):
            candidates.append((len(norm_key), value))
    if not candidates:
        return np.nan
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _emit(callback: ProgressCallback | None, message: str) -> None:
    print(message)
    if callback is not None:
        callback(message)


def resolve_dataset_root(data_root: str | Path) -> Path:
    root = Path(data_root).expanduser().resolve()
    candidate = root / "dataset"
    return candidate if candidate.is_dir() else root


def list_sample_ids(data_root: str | Path) -> list[str]:
    dataset = resolve_dataset_root(data_root)
    if not dataset.is_dir():
        return []
    return sorted(
        item.name
        for item in dataset.iterdir()
        if item.is_dir() and not item.name.startswith(".")
    )


def _image_extensions() -> set[str]:
    try:
        from config import settings

        return {str(item).lower() for item in settings.image_extensions}
    except Exception:
        return set(IMAGE_EXTENSIONS)


def find_primary_image_paths(sample_dir: Path, *_args: Any, **_kwargs: Any) -> list[str]:
    """Deterministic primary-image lookup used by reuse (no LLM selection)."""

    extensions = _image_extensions()
    if not sample_dir.is_dir():
        return []
    paths = sorted(
        path
        for path in sample_dir.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )
    return [str(path) for path in paths]


def find_segmentation_paths(sample_dir: Path) -> list[Path]:
    """Collect segmentation masks without importing LLM-backed selectors."""

    extensions = _image_extensions()
    seg_dir = sample_dir / "segmentation"
    if not seg_dir.is_dir():
        return []
    try:
        from tools.segmentation import list_segmentation_files

        return [path for _name, path in list_segmentation_files(sample_dir)]
    except Exception:
        return sorted(
            path
            for path in seg_dir.iterdir()
            if path.is_file() and path.suffix.lower() in extensions
        )


def execute_reused_merged_code(
    merged_code: str,
    feature_names: list[str],
    sample_ids: list[str],
    data_root: Path,
    results_dir: Path,
    *,
    conda_env: str | None = None,
) -> ExtractionResult:
    """Execute historical extract_all() code without DataPathSelector / LLM init."""

    merged_dir = results_dir / "merged_features"
    merged_dir.mkdir(parents=True, exist_ok=True)
    extract_py_path = merged_dir / "extract_all.py"
    extract_py_path.write_text(merged_code, encoding="utf-8")

    executor = CodeExecutor(data_root, conda_env=conda_env)
    all_values: dict[str, dict[str, Any]] = {}
    all_errors: dict[str, str] = {}
    log_path = merged_dir / "execution_log.txt"
    with log_path.open("w", encoding="utf-8") as log_fp:
        for sample_id in sample_ids:
            sample_dir = data_root / sample_id
            if not sample_dir.is_dir():
                message = f"Sample directory does not exist: {sample_dir}"
                all_errors[sample_id] = message
                log_fp.write(f"{sample_id}: [ERROR] {message}\n")
                continue
            image_paths = find_primary_image_paths(sample_dir)
            if not image_paths:
                message = f"No primary image found for sample {sample_id}"
                all_errors[sample_id] = message
                log_fp.write(f"{sample_id}: [ERROR] {message}\n")
                continue
            seg_paths = find_segmentation_paths(sample_dir)
            success, result_value, error_msg = executor.execute_single_sample(
                extract_py_path,
                Path(image_paths[0]),
                seg_paths,
            )
            if not success:
                message = error_msg or "Unknown error"
                all_errors[sample_id] = message
                log_fp.write(f"{sample_id}: [ERROR] {message}\n")
                continue
            if isinstance(result_value, dict):
                payload = result_value
            else:
                try:
                    payload = json.loads(result_value) if isinstance(result_value, str) else {"unknown": result_value}
                except Exception:
                    payload = {feature_names[0]: result_value} if feature_names else {"unknown": result_value}
                if not isinstance(payload, dict):
                    payload = {feature_names[0]: payload} if feature_names else {"unknown": payload}
            all_values[sample_id] = payload
            log_fp.write(f"{sample_id}: [OK] {payload}\n")
    print(
        f"  [Reuse] Execution completed: {len(all_values)}/{len(sample_ids)} samples succeeded"
    )
    return ExtractionResult(values=all_values, errors=all_errors)


def _round_number(path: Path) -> int | None:
    try:
        return int(path.name.split("_", 1)[1])
    except (IndexError, ValueError):
        return None


def _merged_code_path(round_dir: Path) -> Path | None:
    preferred = round_dir / "merged_features" / "extract_all.py"
    if preferred.is_file():
        return preferred
    fallback = round_dir / "merged_feature_code.py"
    if fallback.is_file():
        return fallback
    return None


def _load_feature_plan(round_dir: Path) -> dict[str, Any]:
    plan_path = round_dir / "feature_plan.json"
    if not plan_path.is_file():
        return {}
    try:
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def discover_reuse_rounds(source_results: str | Path) -> tuple[list[ReuseRoundSpec], list[dict[str, Any]]]:
    """Discover reusable code rounds from a completed MorphAgent results directory."""

    root = Path(source_results).expanduser().resolve()
    rounds: list[ReuseRoundSpec] = []
    skipped: list[dict[str, Any]] = []
    if not root.is_dir():
        return rounds, skipped

    round_dirs = sorted(
        (path for path in root.glob("round_*") if path.is_dir() and _round_number(path) is not None),
        key=lambda path: _round_number(path) or 0,
    )
    for round_dir in round_dirs:
        number = _round_number(round_dir) or 0
        plan = _load_feature_plan(round_dir)
        features = plan.get("features", []) if isinstance(plan.get("features"), list) else []
        code_defs: list[dict[str, Any]] = []
        vlm_names: list[str] = []
        for raw in features:
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name") or "").strip()
            method = str(raw.get("method") or "").strip().lower()
            if not name:
                continue
            if method == "code":
                code_defs.append(dict(raw))
            elif method == "vlm":
                vlm_names.append(name)
        merged = _merged_code_path(round_dir)
        if not code_defs:
            skipped.append(
                {
                    "round": number,
                    "reason": "no_code_features",
                    "path": str(round_dir),
                    "skipped_vlm": vlm_names,
                }
            )
            continue
        if merged is None:
            skipped.append(
                {
                    "round": number,
                    "reason": "missing_merged_code",
                    "path": str(round_dir),
                    "code_features": [item.get("name") for item in code_defs],
                    "skipped_vlm": vlm_names,
                }
            )
            continue
        rounds.append(
            ReuseRoundSpec(
                round_number=number,
                source_round_dir=round_dir,
                feature_names=tuple(str(item.get("name")) for item in code_defs),
                feature_defs=tuple(code_defs),
                merged_code_path=merged,
                skipped_vlm_names=tuple(vlm_names),
            )
        )
    return rounds, skipped


def summarize_source_results(source_results: str | Path) -> dict[str, Any]:
    rounds, skipped = discover_reuse_rounds(source_results)
    return {
        "source_results": str(Path(source_results).expanduser().resolve()),
        "reusable_rounds": len(rounds),
        "skipped_rounds": len(skipped),
        "code_feature_count": sum(len(item.feature_names) for item in rounds),
        "rounds": [
            {
                "round": item.round_number,
                "code_features": list(item.feature_names),
                "merged_code": str(item.merged_code_path),
                "skipped_vlm": list(item.skipped_vlm_names),
            }
            for item in rounds
        ],
        "skipped": skipped,
    }


def diagnose_reuse_inputs(
    source_results: str | Path | None,
    data_root: str | Path | None,
) -> list[str]:
    """Return human-readable blockers for reuse preflight."""

    blockers: list[str] = []
    if source_results is None or not str(source_results).strip():
        blockers.append("Choose a completed MorphAgent results folder.")
    else:
        source = Path(source_results).expanduser()
        if not source.is_dir():
            blockers.append("The history results folder does not exist.")
        else:
            rounds, _skipped = discover_reuse_rounds(source)
            if not rounds:
                blockers.append(
                    "No reusable code rounds with merged feature code were found in this results folder."
                )
    if data_root is None or not str(data_root).strip():
        blockers.append("Choose a dataset folder.")
    else:
        root = Path(data_root).expanduser()
        if not root.exists() or not root.is_dir():
            blockers.append("The dataset folder does not exist.")
        else:
            samples = list_sample_ids(root)
            if not samples:
                blockers.append(
                    "No sample folders were found. Expected dataset/<sample>/*.tif (or select the parent that contains dataset/)."
                )
            else:
                dataset = resolve_dataset_root(root)
                empty = [
                    sample
                    for sample in samples
                    if not find_primary_image_paths(dataset / sample)
                ]
                if empty and len(empty) == len(samples):
                    blockers.append(
                        "Sample folders were found, but none contain primary images for the code route."
                    )
    if (
        source_results
        and data_root
        and str(source_results).strip()
        and str(data_root).strip()
    ):
        try:
            source_resolved = Path(source_results).expanduser().resolve()
            data_resolved = Path(data_root).expanduser().resolve()
            if source_resolved == data_resolved:
                blockers.append("History results and the new dataset cannot be the same folder.")
        except OSError:
            pass
    return blockers


def _default_output_dir(data_root: Path) -> Path:
    dataset = resolve_dataset_root(data_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return dataset.parent / "results" / f"reuse_ui_{timestamp}"


def _copy_merged_code(source: Path, destination_round: Path) -> Path:
    merged_dir = destination_round / "merged_features"
    merged_dir.mkdir(parents=True, exist_ok=True)
    target = merged_dir / "extract_all.py"
    shutil.copy2(source, target)
    # Keep a root-level copy for parity with discovery runs.
    shutil.copy2(source, destination_round / "merged_feature_code.py")
    return target


def _write_round_plan(destination_round: Path, feature_defs: Iterable[dict[str, Any]], round_number: int) -> None:
    payload = {
        "template_name": "feature_reuse",
        "round": round_number,
        "features": list(feature_defs),
        "reuse": True,
        "method_filter": "code",
    }
    (destination_round / "feature_plan.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _append_round_to_csv(
    features_csv: Path,
    sample_ids: list[str],
    feature_names: list[str],
    extraction: ExtractionResult,
) -> list[str]:
    if features_csv.is_file():
        frame = pd.read_csv(features_csv)
        if "sample_id" not in frame.columns:
            frame.insert(0, "sample_id", sample_ids)
        else:
            frame = frame.set_index("sample_id").reindex(sample_ids).reset_index()
    else:
        frame = pd.DataFrame({"sample_id": sample_ids})

    written: list[str] = []
    for feature_name in feature_names:
        column = feature_name
        if column in frame.columns:
            existing = frame[column]
            if existing.notna().sum() > 0:
                column = f"{feature_name}_new_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        values: list[Any] = []
        for sample_id in sample_ids:
            sample_features = extraction.values.get(sample_id)
            if isinstance(sample_features, dict):
                value = _get_merged_feature_value(sample_features, feature_name)
            else:
                value = np.nan
            if value is None:
                value = np.nan
            elif isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                value = np.nan
            values.append(value)
        frame[column] = values
        written.append(column)
    cols = ["sample_id"] + [col for col in frame.columns if col != "sample_id"]
    frame = frame[cols]
    frame.to_csv(features_csv, index=False, encoding="utf-8")
    return written


def _build_registry_from_source(
    source_results: Path,
    output_dir: Path,
    reused_feature_names: list[str],
    round_dirs: dict[int, Path],
) -> dict[str, Any]:
    reused = set(reused_feature_names)
    source_registry_path = source_results / "feature_registry.json"
    entries: list[dict[str, Any]] = []
    if source_registry_path.is_file():
        try:
            payload = json.loads(source_registry_path.read_text(encoding="utf-8"))
            raw_entries = payload.get("entries", []) if isinstance(payload, dict) else []
        except (OSError, UnicodeError, json.JSONDecodeError):
            raw_entries = []
        for entry in raw_entries:
            if not isinstance(entry, dict):
                continue
            method = str(entry.get("method") or "").strip().lower()
            name = str(entry.get("actual_column_name") or entry.get("name") or "").strip()
            if method != "code" or name not in reused:
                continue
            copied = dict(entry)
            round_number = int(copied.get("latest_round") or copied.get("first_round") or 0)
            round_dir = round_dirs.get(round_number, output_dir)
            copied["source_paths"] = {
                "round_dir": str(round_dir),
                "feature_plan_path": str(round_dir / "feature_plan.json"),
                "raw_features_csv": str(output_dir / "features.csv"),
                "reused_from": str(source_results),
            }
            entries.append(copied)

    if not entries:
        # Fallback: build retained cards from reused plan metadata when registry is absent.
        for round_number, round_dir in sorted(round_dirs.items()):
            plan = _load_feature_plan(round_dir)
            for feature in plan.get("features", []) if isinstance(plan.get("features"), list) else []:
                if not isinstance(feature, dict):
                    continue
                name = str(feature.get("name") or "").strip()
                if not name or name not in reused:
                    continue
                entries.append(
                    {
                        "feature_id": f"round_{round_number}:{name}",
                        "name": name,
                        "canonical_name": name,
                        "method": "code",
                        "description": feature.get("description", ""),
                        "category": feature.get("category", "other"),
                        "first_round": round_number,
                        "latest_round": round_number,
                        "current_status": "retained",
                        "live": True,
                        "actual_column_name": name,
                        "source_paths": {
                            "round_dir": str(round_dir),
                            "feature_plan_path": str(round_dir / "feature_plan.json"),
                            "raw_features_csv": str(output_dir / "features.csv"),
                            "reused_from": str(source_results),
                        },
                        "decision_history": [
                            {
                                "feature_id": f"round_{round_number}:{name}",
                                "feature_name": name,
                                "status": "retained",
                                "reason_codes": ["reused_without_revalidation"],
                                "explanation": "Reused historical code feature without running validation.",
                                "compared_feature_ids": [],
                                "validation_score": None,
                                "actual_column_name": name,
                                "llm_reviewed": False,
                            }
                        ],
                    }
                )

    live_ids = [
        str(entry.get("feature_id"))
        for entry in entries
        if entry.get("live") or str(entry.get("current_status", "")).lower() == "retained"
    ]
    feature_id_to_column = {
        str(entry.get("feature_id")): str(entry.get("actual_column_name") or entry.get("name"))
        for entry in entries
        if entry.get("feature_id")
    }
    all_names = [str(entry.get("actual_column_name") or entry.get("name")) for entry in entries]
    return {
        "version": 1,
        "entries": entries,
        "live_feature_ids": live_ids,
        "feature_id_to_column": feature_id_to_column,
        "all_raw_feature_names": all_names,
        "all_historical_feature_names": all_names,
        "reuse": True,
        "reused_from": str(source_results),
    }


def _write_retained_csv(features_csv: Path, registry: dict[str, Any]) -> Path | None:
    if not features_csv.is_file():
        return None
    live_columns = [
        registry.get("feature_id_to_column", {}).get(feature_id, feature_id)
        for feature_id in registry.get("live_feature_ids", [])
    ]
    live_columns = [name for name in live_columns if name]
    frame = pd.read_csv(features_csv)
    keep = ["sample_id"] + [col for col in live_columns if col in frame.columns]
    if len(keep) <= 1:
        # If source marked everything dropped, still keep the full matrix as retained for browsing.
        keep = list(frame.columns)
    retained = frame[keep]
    out = features_csv.parent / "retained_features.csv"
    retained.to_csv(out, index=False, encoding="utf-8")
    return out


def run_code_reuse(
    source_results: str | Path,
    data_root: str | Path,
    output_dir: str | Path | None = None,
    *,
    code_parallel_workers: int = 1,
    conda_env: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> ReuseSummary:
    """Execute historical merged code features on a new dataset and write a new results tree."""

    blockers = diagnose_reuse_inputs(source_results, data_root)
    if blockers:
        raise ValueError("; ".join(blockers))

    source = Path(source_results).expanduser().resolve()
    data_input = Path(data_root).expanduser().resolve()
    dataset_root = resolve_dataset_root(data_input)
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None and str(output_dir).strip()
        else _default_output_dir(data_input)
    )
    destination.mkdir(parents=True, exist_ok=True)

    rounds, skipped = discover_reuse_rounds(source)
    sample_ids = list_sample_ids(data_input)
    summary = ReuseSummary(
        source_results=source,
        data_root=dataset_root,
        output_dir=destination,
        rounds=list(rounds),
        skipped_rounds=list(skipped),
        sample_ids=list(sample_ids),
    )

    _emit(progress_callback, f"[Reuse] Source results: {source}")
    _emit(progress_callback, f"[Reuse] Dataset root: {dataset_root}")
    _emit(progress_callback, f"[Reuse] Output directory: {destination}")
    _emit(
        progress_callback,
        f"[Reuse] Found {len(rounds)} reusable code round(s), {len(sample_ids)} sample(s)",
    )
    for item in skipped:
        _emit(
            progress_callback,
            f"[Reuse] Skipping round {item.get('round')}: {item.get('reason')}",
        )

    features_csv = destination / "features.csv"
    all_results: dict[str, dict[str, Any]] = {sample_id: {} for sample_id in sample_ids}
    written_columns: list[str] = []
    round_dirs: dict[int, Path] = {}

    total = len(rounds)
    for index, round_spec in enumerate(rounds, start=1):
        _emit(
            progress_callback,
            f"[Reuse] Round {index}/{total} · round_{round_spec.round_number} · "
            f"{len(round_spec.feature_names)} code feature(s)",
        )
        if round_spec.skipped_vlm_names:
            _emit(
                progress_callback,
                f"[Reuse] Skipping {len(round_spec.skipped_vlm_names)} VLM feature(s) in this round",
            )

        out_round = destination / f"round_{round_spec.round_number}"
        out_round.mkdir(parents=True, exist_ok=True)
        round_dirs[round_spec.round_number] = out_round
        _write_round_plan(out_round, round_spec.feature_defs, round_spec.round_number)
        local_code = _copy_merged_code(round_spec.merged_code_path, out_round)
        merged_code = local_code.read_text(encoding="utf-8")

        # Intentionally avoid tools.code_executor.execute_merged_code(): that path
        # constructs DataPathSelector, which eagerly initializes an LLM client.
        _ = code_parallel_workers  # reserved for a future deterministic parallel path
        try:
            extraction = execute_reused_merged_code(
                merged_code,
                list(round_spec.feature_names),
                sample_ids,
                dataset_root,
                out_round,
                conda_env=conda_env,
            )
        except Exception as exc:  # noqa: BLE001 - isolate round failures
            _emit(progress_callback, f"[Reuse] [ERROR] Round {round_spec.round_number} failed: {exc}")
            extraction = ExtractionResult(values={}, errors={sid: str(exc) for sid in sample_ids})

        columns = _append_round_to_csv(
            features_csv,
            sample_ids,
            list(round_spec.feature_names),
            extraction,
        )
        written_columns.extend(columns)

        sample_results: dict[str, dict[str, Any]] = {}
        for sample_id in sample_ids:
            sample_payload: dict[str, Any] = {}
            sample_features = extraction.values.get(sample_id)
            for feature_name, column in zip(round_spec.feature_names, columns):
                if isinstance(sample_features, dict):
                    value = _get_merged_feature_value(sample_features, feature_name)
                else:
                    value = np.nan
                if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
                    value = None
                sample_payload[column] = value
                all_results[sample_id][column] = value
            sample_results[sample_id] = sample_payload

        round_payload = {
            "round": round_spec.round_number,
            "reuse": True,
            "feature_plan": {
                "features": list(round_spec.feature_defs),
            },
            "sample_results": sample_results,
            "errors": extraction.errors,
            "source_merged_code": str(round_spec.merged_code_path),
            "skipped_vlm_features": list(round_spec.skipped_vlm_names),
            "validation": None,
        }
        (out_round / "round_results.json").write_text(
            json.dumps(round_payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        success_count = sum(1 for sample_id in sample_ids if sample_id not in extraction.errors)
        _emit(
            progress_callback,
            f"[Reuse] [OK] Round {round_spec.round_number} complete · "
            f"{success_count}/{len(sample_ids)} samples succeeded",
        )

    registry = _build_registry_from_source(source, destination, written_columns, round_dirs)
    (destination / "feature_registry.json").write_text(
        json.dumps(registry, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    retained_path = _write_retained_csv(features_csv, registry)

    manifest = {
        "mode": "code_reuse",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_results": str(source),
        "data_root": str(dataset_root),
        "selected_data_root": str(data_input),
        "results_dir": str(destination),
        "sample_ids": sample_ids,
        "feature_columns": written_columns,
        "rounds": [
            {
                "round": item.round_number,
                "code_features": list(item.feature_names),
                "merged_code": str(item.merged_code_path),
                "skipped_vlm": list(item.skipped_vlm_names),
            }
            for item in rounds
        ],
        "skipped_rounds": skipped,
        "features_csv": str(features_csv),
        "retained_features_csv": str(retained_path) if retained_path else None,
        "llm_calls": False,
        "vlm_calls": False,
        "notes": "Historical merged code features were executed on the new dataset without planning, knowledge, LLM, or VLM steps.",
    }
    (destination / "reuse_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (destination / "ui_run_manifest.json").write_text(
        json.dumps(
            {
                "mode": "code_reuse",
                "query": "Reuse history code features",
                "data_root": str(data_input),
                "results_dir": str(destination),
                "source_results": str(source),
                "method": "code",
                "command": [
                    "python",
                    "-u",
                    "reuse_code.py",
                    "--source-results",
                    str(source),
                    "--data-root",
                    str(data_input),
                    "--results-dir",
                    str(destination),
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary.feature_columns = written_columns
    summary.features_csv = features_csv
    _emit(progress_callback, f"[Reuse] [DONE] Wrote {len(written_columns)} code feature column(s)")
    _emit(progress_callback, f"[Reuse] Final feature file: {features_csv}")
    return summary
