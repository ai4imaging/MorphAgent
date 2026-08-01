"""Policy for installs into the MorphAgent UI Lite extract environment.

Lite runs UI + extract() in morphagent_lite. Runtime install may only *add*
missing non-core packages — never pin/change versions of the frozen core set.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence, Tuple

# Top-level names as they appear in pip / ImportError messages (Lite keep-list).
CORE_SCIENCE_PACKAGES = frozenset(
    {
        "numpy",
        "scipy",
        "pandas",
        "scikit-image",
        "skimage",
        "scikit-learn",
        "sklearn",
        "opencv-python",
        "opencv-python-headless",
        "cv2",
        "pillow",
        "pil",
        "matplotlib",
        "tifffile",
        "imageio",
        "imagecodecs",
        "tqdm",
    }
)

# Import name -> preferred pip distribute name (for messaging only).
_IMPORT_TO_PIP = {
    "skimage": "scikit-image",
    "sklearn": "scikit-learn",
    "cv2": "opencv-python-headless",
    "pil": "pillow",
}

_VERSION_SPEC_RE = re.compile(
    r"(==|!=|<=|>=|~=|===|<|>)"  # pip version operators
    r"|"
    r"(?:=[0-9])",  # conda-style pkg=1.2
)

_UNINSTALL_RE = re.compile(r"\b(pip3?|conda)\b[^\n]*\buninstall\b", re.IGNORECASE)
_FORCE_RE = re.compile(
    r"--force-reinstall|--no-deps\b|pip\s+install\s+[^\n]*\s+-U\b|pip\s+install\s+[^\n]*\s+--upgrade\b",
    re.IGNORECASE,
)
_PIP_INSTALL_RE = re.compile(
    r"\b(?:python\s+-m\s+)?pip3?\s+install\b([^\n\\;|&]*)",
    re.IGNORECASE,
)
_CONDA_INSTALL_RE = re.compile(
    r"\bconda\s+install\b([^\n\\;|&]*)",
    re.IGNORECASE,
)


def normalize_package_name(name: str) -> str:
    return (name or "").strip().lower().replace("_", "-")


def is_core_science_package(name: str) -> bool:
    raw = (name or "").strip()
    if not raw:
        return False
    top = raw.split(".")[0]
    candidates = {
        normalize_package_name(raw),
        normalize_package_name(top),
        top.lower(),
        raw.lower(),
    }
    core_norms = {normalize_package_name(p) for p in CORE_SCIENCE_PACKAGES}
    core_norms |= {p.lower() for p in CORE_SCIENCE_PACKAGES}
    return bool(candidates & core_norms)


def pip_name_for_import(name: str) -> str:
    top = (name or "").split(".")[0].lower()
    return _IMPORT_TO_PIP.get(top, top)


def _tokens_from_install_args(args: str) -> List[str]:
    tokens: List[str] = []
    for raw in args.replace("\\\n", " ").split():
        if raw.startswith("-"):
            continue
        if raw in {"-y", "-n", "--yes", "--name"}:
            continue
        # Skip conda env name after -n
        tokens.append(raw.strip("\"'"))
    return tokens


def _package_root(token: str) -> str:
    # numpy==1.2 / scikit-image>=0.22 / pkg[extra]==1.0
    cleaned = token.split("[", 1)[0]
    for sep in ("==", "!=", "<=", ">=", "~=", "===", "=", "<", ">"):
        if sep in cleaned:
            cleaned = cleaned.split(sep, 1)[0]
            break
    return cleaned.strip()


def validate_package_install_request(package_name: str) -> Tuple[bool, str]:
    """Allow only unpinned, non-core package names for auto-install."""

    name = (package_name or "").strip()
    if not name:
        return False, "empty package name"
    if _VERSION_SPEC_RE.search(name) or any(ch in name for ch in "<>="):
        return False, f"version pins are forbidden ({name!r}); use the preinstalled sandbox stack"
    root = _package_root(name)
    if is_core_science_package(root):
        return (
            False,
            f"core science package {root!r} is frozen in the sandbox — "
            "adapt code to the installed API (e.g. skimage.feature.graycomatrix) "
            "instead of reinstalling",
        )
    return True, ""


def validate_install_script(script: str) -> Tuple[bool, str, str]:
    """Validate a fix-agent install script.

    Returns (ok, reason, sanitized_script). When ok is False, sanitized_script
    is empty and the caller must skip execution.
    """

    text = (script or "").strip()
    if not text:
        return True, "empty script", ""

    if _UNINSTALL_RE.search(text):
        return False, "uninstall commands are forbidden in the code sandbox", ""
    if _FORCE_RE.search(text):
        return False, "force-reinstall / upgrade flags are forbidden in the code sandbox", ""
    if _VERSION_SPEC_RE.search(text):
        return False, "version pins are forbidden; sandbox stack versions are fixed at setup", ""

    # Collect referenced packages from pip/conda install lines.
    packages: List[str] = []
    for match in _PIP_INSTALL_RE.finditer(text):
        packages.extend(_tokens_from_install_args(match.group(1)))
    for match in _CONDA_INSTALL_RE.finditer(text):
        packages.extend(_tokens_from_install_args(match.group(1)))

    # Channel / env flags that look like packages
    skip = {"-c", "conda-forge", "defaults", "pip", "python"}
    for token in packages:
        if token.lower() in skip or token.startswith("-"):
            continue
        root = _package_root(token)
        if not root or root.lower() in skip:
            continue
        ok, reason = validate_package_install_request(root)
        if not ok:
            return False, reason, ""

    return True, "ok", text


def blocked_install_guidance(reason: str) -> str:
    return (
        "Sandbox install was blocked by MorphAgent UI policy "
        f"({reason}). "
        "Do NOT change numpy/scikit-image/scipy versions. "
        "Rewrite extract() to use the already-installed packages and current APIs "
        "(scikit-image uses graycomatrix/graycoprops, not greycomatrix/greycoprops). "
        "Only truly missing non-core packages may be installed without a version pin."
    )
