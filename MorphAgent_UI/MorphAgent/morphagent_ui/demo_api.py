"""Obfuscated free/demo OpenAI-compatible credentials for MorphAgent UI testing.

The payload is XOR-obfuscated (not strong cryptography). Anyone with the source
can recover it; the goal is to keep the key out of plain text and bind decode
to the MorphAgent UI package path so casual copy-paste reuse is less convenient.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from pathlib import Path
from typing import TypedDict


class FreeDemoCredentials(TypedDict):
    base_url: str
    api_key: str
    model: str


# Soft binder + obfuscated JSON {"base_url","api_key","model"}.
_BINDER = b"MorphAgent.UI.demo.gpugeek.v1"
_BLOB = (
    b"T5bW8Fr_XRA6Pb85yUtW4OJBDCxRruoixNPm2xMKaJAa19v8BuwzR2Rz_HOCfEnx"
    b"6xAMIU7_7zqIgvPXGV1jzlvQhKAZqjIBIzPwN5gWR-6jAVloTv_qftvf5J9WQy-"
    b"WW9DR_QugIDMtP_lsmREN08JmGzcR7ec="
)

# Free/demo gateway is token-limited — keep UI scale at demo size.
FREE_DEMO_ROUNDS = 1
FREE_DEMO_CANDIDATES = 5
FREE_DEMO_TARGET = 5

FREE_DEMO_NOTICE = (
    "Free restricted API for MorphAgent testing only. "
    "Token quota is limited. Scale is locked to 1 round × 5 candidates · target 5. "
    "Use your own API key to remove this limit."
)


def _package_bound() -> bool:
    """Decode only when this module lives under MorphAgent's morphagent_ui package."""

    here = Path(__file__).resolve()
    parts = [part.lower() for part in here.parts]
    joined = "/".join(parts)
    return "morphagent" in joined and "morphagent_ui" in parts and here.name == "demo_api.py"


def _derive_key() -> bytes:
    return hmac.new(_BINDER, b"credentials", hashlib.sha256).digest()


def load_free_demo_credentials() -> FreeDemoCredentials:
    if not _package_bound():
        raise RuntimeError("Free demo API credentials are only available inside MorphAgent UI.")

    key = _derive_key()
    raw = base64.urlsafe_b64decode(_BLOB)
    plain = bytes(byte ^ key[index % len(key)] for index, byte in enumerate(raw))
    data = json.loads(plain.decode("utf-8"))
    base_url = str(data.get("base_url", "")).strip()
    api_key = str(data.get("api_key", "")).strip()
    model = str(data.get("model", "")).strip()
    if not (base_url and api_key and model):
        raise RuntimeError("Free demo API credentials are incomplete.")
    return {"base_url": base_url, "api_key": api_key, "model": model}


def is_free_demo_connection(base_url: str, api_key: str = "") -> bool:
    """True when the form still points at the free demo gateway (and key, if given)."""

    try:
        creds = load_free_demo_credentials()
    except RuntimeError:
        return False
    url = (base_url or "").strip().rstrip("/")
    expected = creds["base_url"].rstrip("/")
    if url != expected:
        return False
    key = (api_key or "").strip()
    return not key or key == creds["api_key"]
