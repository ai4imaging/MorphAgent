"""Best-effort geographic launch counting for the MorphAgent UI."""

from __future__ import annotations

from functools import partial
from threading import Thread
from urllib.request import Request, urlopen


VISITOR_PAGE_ID = "ai4imaging-morphagent-ui"
VISITOR_BADGE_URL = (
    "https://vbr.nathanchung.dev/badge"
    f"?page_id={VISITOR_PAGE_ID}&hit=true"
)
VISITOR_MAP_URL = f"https://vbr.nathanchung.dev/info/{VISITOR_PAGE_ID}"
DEFAULT_TIMEOUT_SECONDS = 2.0
USER_AGENT = "MorphAgent-UI/1.0"


def register_visit(*, opener=urlopen, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> None:
    """Register one UI launch with the hosted visitor-map endpoint."""
    request = Request(
        VISITOR_BADGE_URL,
        headers={"User-Agent": USER_AGENT},
        method="GET",
    )
    with opener(request, timeout=timeout) as response:
        response.read(1)


def _register_visit_safely(*, opener=urlopen, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> None:
    """Keep analytics failures completely separate from application startup."""
    try:
        register_visit(opener=opener, timeout=timeout)
    except Exception:
        return


def start_visit_registration(
    *,
    opener=urlopen,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    thread_factory=Thread,
):
    """Start one non-blocking visitor registration and return its daemon thread."""
    target = partial(_register_visit_safely, opener=opener, timeout=timeout)
    thread = thread_factory(
        target=target,
        name="morphagent-visitor-analytics",
        daemon=True,
    )
    thread.start()
    return thread
