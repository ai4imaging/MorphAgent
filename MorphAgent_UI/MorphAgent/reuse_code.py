#!/usr/bin/env python3
"""CLI entry point for reusing completed MorphAgent code features on a new dataset.

This path never imports the LangGraph planner, knowledge modules, LLM clients, or VLM
scoring. It only loads historical merged code and executes it against the selected data.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _configure_stdio_utf8() -> None:
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reuse historical MorphAgent code features on a new dataset (no LLM/VLM).",
    )
    parser.add_argument(
        "--source-results",
        required=True,
        help="Completed MorphAgent results directory containing round_*/merged feature code.",
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="New dataset root (project folder with dataset/ or the dataset folder itself).",
    )
    parser.add_argument(
        "--results-dir",
        default="",
        help="Output directory for the reuse run (default: <data>/results/reuse_ui_<timestamp>).",
    )
    parser.add_argument(
        "--code-parallel-workers",
        type=int,
        default=1,
        help="Number of parallel workers for merged-code execution (default: 1).",
    )
    parser.add_argument(
        "--conda-env",
        default="",
        help="Optional conda environment name used to execute feature code.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_stdio_utf8()
    # Ensure repository root imports resolve when launched as a script.
    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    args = build_parser().parse_args(argv)
    from tools.code_reuse import diagnose_reuse_inputs, run_code_reuse

    blockers = diagnose_reuse_inputs(args.source_results, args.data_root)
    if blockers:
        print("[Reuse] Preflight failed:", file=sys.stderr)
        for item in blockers:
            print(f"  - {item}", file=sys.stderr)
        return 2

    try:
        summary = run_code_reuse(
            source_results=args.source_results,
            data_root=args.data_root,
            output_dir=args.results_dir or None,
            code_parallel_workers=max(1, int(args.code_parallel_workers)),
            conda_env=args.conda_env.strip() or None,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[Reuse] [ERROR] {exc}", file=sys.stderr)
        return 1

    print(f"[DONE] Reuse complete · {len(summary.feature_columns)} code columns")
    if summary.features_csv is not None:
        print(f"Final feature file: {summary.features_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
