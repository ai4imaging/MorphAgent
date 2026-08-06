"""Tool modules for MorphAgent.

Segmentation, code execution and VLM helpers are imported directly by the
modules that need them to keep this package import lightweight.
"""

from .vlm_client import VLMClient

__all__ = ["VLMClient"]
