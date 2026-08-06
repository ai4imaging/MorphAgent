#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Check the installation status of the Allen segmentation backend.

Run this inside the conda env where you installed the vendored
`aicssegmentation` package (see README / environment_allen.yml)::

    python check_installation.py
"""

import sys
import os

print("=" * 60)
print("Allen Segmentation installation check")
print("=" * 60)

print(f"\nPython version: {sys.version}")
print(f"Python executable: {sys.executable}")

print("\n" + "=" * 60)
print("Check the aicssegmentation package")
print("=" * 60)

try:
    import aicssegmentation
    print("aicssegmentation is installed")
    print(f"  location: {aicssegmentation.__file__}")
    print(f"  version : {getattr(aicssegmentation, '__version__', 'unknown')}")
except ImportError as e:
    print(f" aicssegmentation is NOT installed: {e}")
    print("\nAttempting to install the vendored copy next to this script...")
    import subprocess
    # The aicssegmentation package is vendored in this same directory.
    aics_path = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(aics_path, "aicssegmentation")):
        print(f"Found vendored aicssegmentation at: {aics_path}")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", aics_path],
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        if result.stderr:
            print("stderr:")
            print(result.stderr)
    else:
        print(f" Could not find a vendored aicssegmentation under: {aics_path}")

print("\n" + "=" * 60)
print("Check core modules")
print("=" * 60)

modules_to_check = [
    "aicssegmentation.core.pre_processing_utils",
    "aicssegmentation.core.MO_threshold",
    "aicssegmentation.core.seg_dot",
]

for module in modules_to_check:
    try:
        __import__(module)
        print(f" {module}")
    except ImportError as e:
        print(f" {module}: {e}")

print("\n" + "=" * 60)
print("Check other dependencies")
print("=" * 60)

deps = ["numpy", "scipy", "skimage", "tifffile"]
for dep in deps:
    try:
        mod = __import__(dep)
        print(f" {dep}: {mod.__version__ if hasattr(mod, '__version__') else 'installed'}")
    except ImportError:
        print(f" {dep}: NOT installed")

# Optional: aicsimageio (not required for TIFF driver)
try:
    import aicsimageio  # noqa: F401
    print("aicsimageio: installed (optional)")
except ImportError:
    print("· aicsimageio: not installed (OK — TIFF path uses tifffile)")

print("\n" + "=" * 60)
