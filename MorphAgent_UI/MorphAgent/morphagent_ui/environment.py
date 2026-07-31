"""Read and safely update the repository-local model configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

from dotenv import dotenv_values, set_key


MODEL_ENV_KEYS = (
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "LLM_MODEL",
    "VLM_BASE_URL",
    "VLM_API_KEY",
    "VLM_MODEL",
)

# Persisted UI run-scale knobs (own API only). Free demo API ignores these.
RUN_SCALE_ENV_KEYS = (
    "NUM_ROUNDS",
    "FEATURES_PER_ITERATION",
    "TARGET_FEATURE_COUNT",
    "CODE_PARALLEL_WORKERS",
    "VLM_ONLINE_CONCURRENCY",
    "UI_TEMPERATURE",
)


def repository_env_path(repository_root: str | Path) -> Path:
    return Path(repository_root).expanduser().resolve() / ".env"


def _ensure_env_file(path: Path) -> None:
    if path.exists():
        return
    example = path.with_name(".env.example")
    initial = example.read_text(encoding="utf-8") if example.is_file() else ""
    path.write_text(initial, encoding="utf-8")


def _read_env_keys(repository_root: str | Path, keys: tuple[str, ...]) -> dict[str, str]:
    path = repository_env_path(repository_root)
    file_values = dotenv_values(path) if path.is_file() else {}
    return {
        name: str(file_values.get(name) or os.environ.get(name, ""))
        for name in keys
    }


def _save_env_keys(
    repository_root: str | Path,
    values: Mapping[str, str],
    keys: tuple[str, ...],
) -> Path:
    path = repository_env_path(repository_root)
    _ensure_env_file(path)
    for name in keys:
        if name not in values:
            continue
        value = str(values[name]).strip()
        set_key(str(path), name, value, quote_mode="always")
        os.environ[name] = value
    return path


def read_model_environment(repository_root: str | Path) -> dict[str, str]:
    """Return resolved model values without logging or displaying secrets."""

    return _read_env_keys(repository_root, MODEL_ENV_KEYS)


def save_model_environment(
    repository_root: str | Path,
    values: Mapping[str, str],
) -> Path:
    """Update only model keys, preserving comments and unrelated `.env` values."""

    return _save_env_keys(repository_root, values, MODEL_ENV_KEYS)


def read_run_scale_environment(repository_root: str | Path) -> dict[str, str]:
    """Return saved UI run-scale values from `.env` / process env."""

    return _read_env_keys(repository_root, RUN_SCALE_ENV_KEYS)


def save_run_scale_environment(
    repository_root: str | Path,
    values: Mapping[str, str],
) -> Path:
    """Persist run-scale knobs used by the Configure page (own API)."""

    return _save_env_keys(repository_root, values, RUN_SCALE_ENV_KEYS)
