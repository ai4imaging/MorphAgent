"""OpenAI-compatible base URL helpers (e.g. missing ``/v1`` → 404)."""
from __future__ import annotations

from typing import Any, Optional
from urllib.parse import urlparse, urlunparse


def normalize_base_url(base_url: str) -> str:
    return (base_url or "").strip().rstrip("/")


def with_v1_suffix(base_url: str) -> Optional[str]:
    """Return ``base_url + '/v1'`` when ``/v1`` is not already the last path segment.

    Returns None when the input is empty or already ends with ``/v1``.
    """
    raw = normalize_base_url(base_url)
    if not raw:
        return None
    parsed = urlparse(raw)
    path = (parsed.path or "").rstrip("/")
    if path.endswith("/v1") or path == "v1":
        return None
    new_path = f"{path}/v1" if path else "/v1"
    return urlunparse(parsed._replace(path=new_path))


def is_http_404_error(exc: BaseException) -> bool:
    """Best-effort detection of HTTP 404 from OpenAI / httpx / LangChain errors."""
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "status", None)
    if status == 404:
        return True

    response = getattr(exc, "response", None)
    if response is not None and getattr(response, "status_code", None) == 404:
        return True

    body = getattr(exc, "body", None)
    if isinstance(body, dict) and str(body.get("code", "")).lower() in {"404", "not_found"}:
        return True

    text = str(exc).lower()
    if "404" not in text:
        return False
    markers = (
        "not found",
        "error code: 404",
        "status code: 404",
        "status_code=404",
        "/chat/completions",
        "httpstatuserror",
        "notfounderror",
    )
    return any(m in text for m in markers)


def ensure_openai_base_url(base_url: str, *, tried_v1: bool = False) -> tuple[str, bool]:
    """Return ``(url, switched)`` — if ``tried_v1`` is False and /v1 is missing, append it."""
    current = normalize_base_url(base_url)
    if tried_v1:
        return current, False
    candidate = with_v1_suffix(current)
    if candidate is None:
        return current, False
    return candidate, True
