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


def repository_env_path(repository_root: str | Path) -> Path:
    return Path(repository_root).expanduser().resolve() / ".env"


def read_model_environment(repository_root: str | Path) -> dict[str, str]:
    """Return resolved model values without logging or displaying secrets."""

    path = repository_env_path(repository_root)
    file_values = dotenv_values(path) if path.is_file() else {}
    return {
        name: str(file_values.get(name) or os.environ.get(name, ""))
        for name in MODEL_ENV_KEYS
    }


def save_model_environment(
    repository_root: str | Path,
    values: Mapping[str, str],
) -> Path:
    """Update only model keys, preserving comments and unrelated `.env` values."""

    path = repository_env_path(repository_root)
    if not path.exists():
        example = path.with_name(".env.example")
        initial = example.read_text(encoding="utf-8") if example.is_file() else ""
        path.write_text(initial, encoding="utf-8")

    for name in MODEL_ENV_KEYS:
        if name not in values:
            continue
        value = str(values[name]).strip()
        set_key(str(path), name, value, quote_mode="always")
        os.environ[name] = value
    return path
