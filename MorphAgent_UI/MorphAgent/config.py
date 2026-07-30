"""Global configuration management"""
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional, List, Set, Dict, Any
from pathlib import Path

import copy as _copy

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)


# ============================================================================
# USER CONFIGURATION — EDIT HERE (or set the matching environment variables)
# ============================================================================
# MorphAgent (public build) reaches every model through an OpenAI-compatible
# HTTP API. There is NO local model in this build. You can point these at any
# provider that speaks the OpenAI Chat Completions protocol: OpenAI, Azure
# OpenAI, DeepSeek, Together, vLLM/Ollama/LM Studio (self-hosted gateway), etc.
#
# Two ways to configure (env vars take precedence and keep secrets out of git):
#   export LLM_BASE_URL="https://api.openai.com/v1"
#   export LLM_API_KEY="sk-..."
#   export LLM_MODEL="gpt-4o"
#   export VLM_BASE_URL="https://api.openai.com/v1"   # multimodal endpoint
#   export VLM_API_KEY="sk-..."
#   export VLM_MODEL="gpt-4o"
# ...or simply replace the fallback strings on the right-hand side below.

# --- LLM: the "brain" (plans features, writes/fixes code, reviews) ----------
DEFAULT_LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
DEFAULT_LLM_API_KEY = os.getenv("LLM_API_KEY", "")
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")

# --- VLM: the multimodal "eyes" (scores visual features from images) --------
# Falls back to the LLM endpoint if you do not set VLM_* separately.
DEFAULT_VLM_BASE_URL = os.getenv("VLM_BASE_URL", DEFAULT_LLM_BASE_URL)
DEFAULT_VLM_API_KEY = os.getenv("VLM_API_KEY", DEFAULT_LLM_API_KEY)
DEFAULT_VLM_MODEL = os.getenv("VLM_MODEL", "gpt-4o")

# Some gateways require extra HTTP headers (e.g. a custom User-Agent). Set to
# a dict like {"User-Agent": "MorphAgent/1.0"} if needed, otherwise None.
DEFAULT_LLM_HEADERS: Optional[Dict[str, str]] = None
DEFAULT_VLM_HEADERS: Optional[Dict[str, str]] = None

# --- Deep Research model: writes a literature-grounded report in ONE call ----
# There is NO local deep-research model in this build. The auto deep-research
# step makes a single call to an OpenAI-compatible endpoint. Point it at a
# web-search / research-capable model for best results (e.g. Perplexity
# "sonar" / "sonar-deep-research", OpenAI "gpt-4o-search-preview", or any
# strong chat model). Falls back to the main LLM if left unset.
DEFAULT_DEEP_RESEARCH_BASE_URL = os.getenv("DEEP_RESEARCH_BASE_URL", DEFAULT_LLM_BASE_URL)
DEFAULT_DEEP_RESEARCH_API_KEY = os.getenv("DEEP_RESEARCH_API_KEY", DEFAULT_LLM_API_KEY)
DEFAULT_DEEP_RESEARCH_MODEL = os.getenv("DEEP_RESEARCH_MODEL", DEFAULT_LLM_MODEL)

# --- Literature retrieval (PubMed / Europe PMC) ------------------------------
# NCBI is polite-access friendly; providing an email (and optionally a free
# NCBI API key) raises rate limits. No key is strictly required.
DEFAULT_NCBI_EMAIL = os.getenv("NCBI_EMAIL", "")
DEFAULT_NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")
# ============================================================================


# LLM provider presets. By default there is a single "DEFAULT" preset wired to
# the values above. Add your own named presets here and switch between them at
# runtime with `--api-provider <NAME>` (case-insensitive).
API_PROVIDER_PRESETS: Dict[str, Dict[str, Any]] = {
    "DEFAULT": {
        "llm_base_url": DEFAULT_LLM_BASE_URL,
        "llm_api_key": DEFAULT_LLM_API_KEY,
        "llm_model": DEFAULT_LLM_MODEL,
        "llm_default_headers": DEFAULT_LLM_HEADERS,
    },
    # Example — copy, rename, and fill in to register another endpoint:
    # "DEEPSEEK": {
    #     "llm_base_url": "https://api.deepseek.com/v1",
    #     "llm_api_key": os.getenv("DEEPSEEK_API_KEY", ""),
    #     "llm_model": "deepseek-chat",
    # },
}


# VLM provider presets (keys matched case-insensitively):
#   ONLINE / API — OpenAI-compatible multimodal API (default & recommended)
#   QWEN         — local self-hosted Qwen3-VL (ADVANCED; deps NOT installed by
#                  the default environment, requires GPU + modelscope + torch)
VLM_PROVIDER_PRESETS: Dict[str, Dict[str, Any]] = {
    "ONLINE": {
        "engine": "api",
        "base_url": DEFAULT_VLM_BASE_URL,
        "api_key": DEFAULT_VLM_API_KEY,
        "model": DEFAULT_VLM_MODEL,
        "default_headers": DEFAULT_VLM_HEADERS,
    },
    "API": {
        "engine": "api",
        "base_url": DEFAULT_VLM_BASE_URL,
        "api_key": DEFAULT_VLM_API_KEY,
        "model": DEFAULT_VLM_MODEL,
        "default_headers": DEFAULT_VLM_HEADERS,
    },
    "QWEN": {
        "engine": "local",
    },
}


def get_vlm_temperature() -> float:
    """VLM sampling temperature; forced to 0.0 in reproduce mode."""
    if settings.reproduce_mode:
        return 0.0
    return settings.vlm_temperature


def get_code_temperature() -> float:
    """Code-generation LLM temperature; forced to 0.0 in reproduce mode."""
    if settings.reproduce_mode:
        return 0.0
    return settings.code_temperature


def apply_vlm_provider(provider: str) -> None:
    """Select the VLM backend from ``--vlm-api-provider``.

    ``online``/``api`` = OpenAI-compatible multimodal API (default & recommended).
    ``qwen`` = local self-hosted Qwen3-VL (advanced; extra deps required).

    For any API-engine preset the canonical internal value is kept as
    ``"online"`` so downstream branch checks (``vlm_api_provider == "online"``)
    keep working regardless of the alias the user typed.
    """
    name = (provider or "ONLINE").strip().upper()
    preset = VLM_PROVIDER_PRESETS.get(name)
    if preset is None:
        print(
            f"  ⚠️  Unknown vlm_api_provider '{provider}', falling back to the online API."
            f" Options: {sorted(VLM_PROVIDER_PRESETS.keys())}"
        )
        preset = VLM_PROVIDER_PRESETS["ONLINE"]
        name = "ONLINE"
    if preset.get("engine") == "api":
        settings.vlm_api_provider = "online"
        settings.vlm_online_base_url = preset["base_url"]
        settings.vlm_online_api_key = preset["api_key"]
        settings.vlm_online_model = preset["model"]
        settings.vlm_online_default_headers = preset.get("default_headers")
    else:
        settings.vlm_api_provider = "qwen"


def apply_api_provider(provider: str) -> None:
    """Select the LLM endpoint from ``--api-provider`` (a key in
    ``API_PROVIDER_PRESETS``; unknown/empty falls back to ``DEFAULT``)."""
    name = (provider or "DEFAULT").strip().upper()
    preset = API_PROVIDER_PRESETS.get(name)
    if preset is None:
        print(
            f"  ⚠️  Unregistered api_provider '{provider}', using DEFAULT (environment variables / config defaults)."
            f" Registered: {sorted(API_PROVIDER_PRESETS.keys())}"
        )
        preset = API_PROVIDER_PRESETS["DEFAULT"]
        name = "DEFAULT"
    settings.llm_base_url = preset["llm_base_url"]
    settings.llm_api_key = preset["llm_api_key"]
    settings.llm_model = preset["llm_model"]
    settings.llm_default_headers = preset.get("llm_default_headers")
    settings.llm_provider_name = name.lower()


def _get_retry_after_seconds(exc: Exception) -> Optional[int]:
    """Try to extract the retry-after seconds from the exception object or its text."""
    retry_after = getattr(exc, "retry_after", None)
    if isinstance(retry_after, (int, float)) and retry_after > 0:
        return int(retry_after)

    message = str(exc)
    match = re.search(r"['\"]retry_after['\"]\s*:\s*(\d+)", message)
    if match:
        return int(match.group(1))
    return None


def _is_timeout_like_error(exc: Exception) -> bool:
    """Determine whether the exception is a timeout / 524 / connection-timeout type."""
    if isinstance(exc, (APITimeoutError, APIConnectionError)):
        return True

    message = str(exc).lower()
    timeout_markers = (
        "timeout",
        "timed out",
        "time out",
        "error code: 524",
        "status': 524",
        "status\": 524",
        "origin_response_timeout",
    )
    return any(marker in message for marker in timeout_markers)


def _should_retry_llm_error(exc: Exception) -> bool:
    """Determine whether the exception is worth retrying."""
    if isinstance(exc, (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError)):
        return True
    if isinstance(exc, APIError):
        return _is_timeout_like_error(exc)
    return False


def _is_context_length_error(exc: Exception) -> bool:
    """Determine whether the exception likely means the prompt is too long / the request body is too large (can be mitigated by compressing the prompt).

    Some gateways often return an over-long request as a 400 'Invalid Request'
    with vague information, so BadRequestError(400) is uniformly treated as a
    candidate for compression-and-retry, aided by common textual markers.
    """
    if isinstance(exc, BadRequestError):
        return True
    status = getattr(exc, "status_code", None)
    if status == 400 or status == 413:
        return True
    message = str(exc).lower()
    markers = (
        "context length",
        "maximum context",
        "context_length_exceeded",
        "too long",
        "string too long",
        "maximum number of tokens",
        "reduce the length",
        "request entity too large",
        "invalid request",
        "error code: 400",
        "error code: 413",
    )
    return any(marker in message for marker in markers)


def _message_text(content: Any) -> str:
    """Extract a plain-text length measure of a message (works with both string and multimodal list content)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return ""


def _truncate_text(text: str, budget: int) -> str:
    """Keep the head and tail while compressing the middle so the text does not exceed budget (approximately)."""
    if budget <= 0 or len(text) <= budget:
        return text
    head = (budget * 2) // 3
    tail = max(0, budget - head)
    removed = len(text) - head - tail
    marker = f"\n\n...[truncated {removed} characters to fit the context length limit]...\n\n"
    return text[:head] + marker + (text[-tail:] if tail > 0 else "")


def _clone_message_with_content(msg: Any, new_content: Any) -> Any:
    """Produce a new message with its content replaced, avoiding modifying the original object where possible."""
    for method in ("model_copy", "copy"):
        fn = getattr(msg, method, None)
        if callable(fn):
            try:
                return fn(update={"content": new_content})
            except Exception:
                pass
    try:
        cloned = _copy.copy(msg)
        cloned.content = new_content
        return cloned
    except Exception:
        return msg


def _cap_message(msg: Any, cap: int) -> Any:
    """Compress a single message's text content to at most cap characters (preserving non-text parts such as images)."""
    content = getattr(msg, "content", None)
    if isinstance(content, str):
        if len(content) <= cap:
            return msg
        return _clone_message_with_content(msg, _truncate_text(content, cap))
    if isinstance(content, list):
        text_total = sum(
            len(str(item.get("text", "")))
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
        if text_total <= cap:
            return msg
        ratio = (cap / text_total) if text_total else 1.0
        new_content: List[Any] = []
        changed = False
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = str(item.get("text", ""))
                per_budget = max(200, int(len(text) * ratio))
                if len(text) > per_budget:
                    new_item = dict(item)
                    new_item["text"] = _truncate_text(text, per_budget)
                    new_content.append(new_item)
                    changed = True
                else:
                    new_content.append(item)
            else:
                new_content.append(item)
        if changed:
            return _clone_message_with_content(msg, new_content)
    return msg


def _compress_messages(messages: Any, total_budget: int) -> Any:
    """Compress the message list so the total text does not exceed total_budget; returns (new_messages, changed)."""
    if not isinstance(messages, (list, tuple)):
        return messages, False
    lengths = [len(_message_text(getattr(m, "content", ""))) for m in messages]
    total = sum(lengths)
    if total <= total_budget:
        return messages, False
    hi = max(lengths) if lengths else 0
    lo = 0
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if sum(min(length, mid) for length in lengths) <= total_budget:
            lo = mid
        else:
            hi = mid - 1
    cap = max(lo, 200)
    new_messages = []
    changed_any = False
    for msg in messages:
        capped = _cap_message(msg, cap)
        new_messages.append(capped)
        if capped is not msg:
            changed_any = True
    return new_messages, changed_any


# Decreasing compression budgets (in characters) when the prompt is too long, progressively more aggressive (used only as truncation fallback after semantic compaction fails)
PROMPT_COMPRESSION_BUDGETS = (80000, 40000, 16000)

# Sentinel object returned when compression retries still fail
_COMPRESSION_FAILED = object()


# ===== Semantic compaction (LLM summary compression, similar to Claude Code /compact) =====
# Only summarize non-system messages whose text length exceeds this threshold via the LLM (small messages are not worth summarizing)
SEMANTIC_COMPACT_TRIGGER_CHARS = 8000
# Number of characters kept verbatim at the end of each message (the current task/output-format requirements are usually at the end and must be preserved as-is)
SEMANTIC_COMPACT_TAIL_CHARS = 4000
# Maximum input characters per chunk when summarizing in chunks (ensures the summary call itself is not over-long)
SEMANTIC_COMPACT_CHUNK_CHARS = 24000
# Proactive compression threshold (characters): if the total message characters exceed this value before calling the LLM, do semantic compaction first.
# Default 0 = disable proactive compression (keep only the reactive compression that "only compresses when a 400/413 is actually hit").
# Lesson learned: late-round prompts are routinely 100k+ characters, so setting the threshold too low causes almost every call to be needlessly pre-compressed,
# running several extra summary LLM calls, seriously slowing things down and being lossy for the context. Only enable it via an environment variable with a high threshold when truly needed.
PROACTIVE_COMPACT_THRESHOLD_CHARS = int(os.getenv("PROMPT_PROACTIVE_COMPACT_CHARS", "0"))

# The summarizer's system prompt: strongly constrains "preserve identifiers/thresholds/formats verbatim, only compress verbose descriptions"
_COMPACT_SYSTEM_PROMPT = (
    "You compress accumulated PROMPT CONTEXT for a microscopy feature-engineering agent so it fits "
    "the model context window. Rewrite the provided context into a compact but information-preserving form.\n"
    "STRICT RULES:\n"
    "1. Preserve VERBATIM every feature name / identifier as a deduplicated list — these are required to "
    "avoid proposing duplicate features; never omit, rename, paraphrase, or merge them.\n"
    "2. Preserve all required output formats / JSON schemas, numeric thresholds, constraints, file paths, "
    "mask names, channel names, and biological terms exactly.\n"
    "3. If a prior summary is provided, keep ALL of its facts and extend it (do not drop earlier information).\n"
    "4. You MAY aggressively shorten verbose prose, repeated explanations, long natural-language descriptions, "
    "and redundant restatements.\n"
    "Output ONLY the compressed context text, with no preamble or commentary."
)

# Marker injected before the compressed message (inspired by the compaction marker in Claude Code / hermes-agent)
_COMPACT_SUMMARY_MARKER = (
    "[CONTEXT COMPACTION — the following is a compressed summary of the early accumulated context in this prompt, for reference only; "
    "the complete and authoritative current task instructions are in the 'VERBATIM ORIGINAL' section at the end]\n"
)
_COMPACT_TAIL_MARKER = "\n\n[VERBATIM ORIGINAL — the following is the verbatim original text at the end of this prompt (current task/output requirements)]\n"


def _chunk_text(text: str, chunk_chars: int) -> List[str]:
    if chunk_chars <= 0:
        return [text]
    return [text[i : i + chunk_chars] for i in range(0, len(text), chunk_chars)]


def _summarize_text_via_llm(raw_llm: Any, text: str) -> str:
    """Use one or more LLM calls to produce a chunked rolling summary of over-long text.

    raw_llm should be the underlying ChatOpenAI (not wrapped by this class's compression) to avoid recursive compression.
    The chunks are small enough that the summary call itself will not trigger the over-long condition again.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    chunks = _chunk_text(text, SEMANTIC_COMPACT_CHUNK_CHARS)
    running_summary = ""
    for index, chunk in enumerate(chunks, start=1):
        if running_summary:
            user_content = (
                f"Existing compressed summary (please extend it without losing any of its information/identifiers):\n"
                f"{running_summary}\n\n"
                f"New segment to merge in (chunk {index}/{len(chunks)}):\n{chunk}"
            )
        else:
            user_content = f"Context segment to compress (chunk {index}/{len(chunks)}):\n{chunk}"
        response = raw_llm.invoke(
            [
                SystemMessage(content=_COMPACT_SYSTEM_PROMPT),
                HumanMessage(content=user_content),
            ]
        )
        running_summary = response.content if hasattr(response, "content") else str(response)
        if not isinstance(running_summary, str):
            running_summary = str(running_summary)
    return running_summary


class RetryableChatLLM:
    """Adds a unified timeout/524 retry strategy on top of ChatOpenAI."""

    def __init__(
        self,
        llm: Any,
        provider_name: str,
        timeout_seconds: int,
        max_attempts: int,
        base_retry_delay_seconds: int,
        max_retry_delay_seconds: int,
        llm_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._llm = llm
        self._provider_name = provider_name or "unknown"
        self._timeout_seconds = timeout_seconds
        self._max_attempts = max(1, max_attempts)
        self._base_retry_delay_seconds = max(1, base_retry_delay_seconds)
        self._max_retry_delay_seconds = max(self._base_retry_delay_seconds, max_retry_delay_seconds)
        self._llm_params = dict(llm_params or {})
        self._v1_fallback_tried = False

    def _total_message_chars(self, messages: Any) -> int:
        if not isinstance(messages, (list, tuple)):
            return 0
        total = 0
        for msg in messages:
            content = getattr(msg, "content", None)
            total += len(_message_text(content))
        return total

    def _maybe_proactive_compact(self, args: tuple) -> tuple:
        """Proactive compression before the call: when the total message characters exceed the threshold, do semantic compaction first, then send.

        Any failure/exception is swallowed and the original args are returned (there is still the post-call reactive compression fallback), guaranteeing no crash.
        """
        if PROACTIVE_COMPACT_THRESHOLD_CHARS <= 0:
            # Proactive compression is disabled: send directly, and if over-long, let the post-call reactive compression handle it
            return args
        if not args or not isinstance(args[0], (list, tuple)):
            return args
        messages = args[0]
        total = self._total_message_chars(messages)
        if total <= PROACTIVE_COMPACT_THRESHOLD_CHARS:
            return args
        try:
            new_messages, changed = self._semantic_compress_messages(messages)
        except Exception as exc:
            print(
                f"  ⚠️  Proactive compression failed ({type(exc).__name__}); sending the original prompt, "
                f"with reactive compression as a fallback if needed (provider={self._provider_name})"
            )
            return args
        if not changed:
            return args
        new_total = self._total_message_chars(new_messages)
        print(
            f"  🧠 Proactive compression: detected a prompt of about {total} characters (> threshold {PROACTIVE_COMPACT_THRESHOLD_CHARS}), "
            f"LLM-summarized it down to about {new_total} characters before calling (provider={self._provider_name})"
        )
        return (new_messages,) + tuple(args[1:])

    def _rebuild_llm(self, base_url: str) -> None:
        from langchain_openai import ChatOpenAI

        params = dict(self._llm_params)
        params["base_url"] = base_url
        self._llm_params = params
        self._llm = ChatOpenAI(**params)

    def _maybe_retry_with_v1_base_url(self, exc: Exception) -> bool:
        """If the gateway returned 404 because ``/v1`` was missing, rebuild and retry once."""
        from utils_modules.openai_base_url import is_http_404_error, with_v1_suffix

        if self._v1_fallback_tried or not is_http_404_error(exc):
            return False
        current = str(self._llm_params.get("base_url") or settings.llm_base_url or "")
        candidate = with_v1_suffix(current)
        if not candidate:
            return False
        self._v1_fallback_tried = True
        print(
            f"  ⚠️  LLM API returned 404 (provider={self._provider_name}); "
            f"retrying with base_url={candidate}"
        )
        self._rebuild_llm(candidate)
        try:
            settings.llm_base_url = candidate
        except Exception:
            pass
        return True

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        """Uniformly handle retries for LLM timeout/524/connection errors; proactively compress over-long prompts before the call, and compress-then-retry when the prompt is too long (400/413)."""
        args = self._maybe_proactive_compact(args)
        for attempt in range(1, self._max_attempts + 1):
            try:
                return self._llm.invoke(*args, **kwargs)
            except Exception as exc:
                if self._maybe_retry_with_v1_base_url(exc):
                    try:
                        return self._llm.invoke(*args, **kwargs)
                    except Exception as retry_exc:
                        exc = retry_exc

                # Prompt too long / request body too large: progressively compress the messages and retry to avoid crashing outright
                if _is_context_length_error(exc) and args:
                    compressed_result = self._invoke_with_compression(args, kwargs, exc)
                    if compressed_result is not _COMPRESSION_FAILED:
                        return compressed_result
                    # Compression still failed: let the logic below decide whether an ordinary retry is still worthwhile
                    if not _should_retry_llm_error(exc) or attempt >= self._max_attempts:
                        raise

                if not _should_retry_llm_error(exc) or attempt >= self._max_attempts:
                    raise

                retry_after = _get_retry_after_seconds(exc)
                delay = min(
                    self._max_retry_delay_seconds,
                    self._base_retry_delay_seconds * (2 ** (attempt - 1)),
                )
                if retry_after is not None:
                    delay = max(delay, min(retry_after, self._max_retry_delay_seconds))

                error_kind = "timeout" if _is_timeout_like_error(exc) else "transient error"
                print(
                    f"  ⚠️  LLM {error_kind} from {self._provider_name} "
                    f"(attempt {attempt}/{self._max_attempts}, timeout={self._timeout_seconds}s): "
                    f"{type(exc).__name__}"
                )
                print(f"     Waiting {delay} seconds before retry...")
                time.sleep(delay)

        raise RuntimeError("Unexpected retry loop fallthrough")

    def _semantic_compress_messages(self, messages: Any) -> Any:
        """Semantic compaction: protect the system message, keep the verbatim tail of the user message, and LLM-summarize the early accumulated content.

        Returns (new_messages, changed). Multimodal (list) content is not handled here (left to the truncation fallback).
        """
        if not isinstance(messages, (list, tuple)):
            return messages, False
        new_messages = []
        changed = False
        for msg in messages:
            role = getattr(msg, "type", "") or ""
            content = getattr(msg, "content", None)
            # Protect system messages; non-string (multimodal) content is left untouched here
            if role == "system" or not isinstance(content, str):
                new_messages.append(msg)
                continue
            if len(content) <= SEMANTIC_COMPACT_TRIGGER_CHARS:
                new_messages.append(msg)
                continue
            tail = content[-SEMANTIC_COMPACT_TAIL_CHARS:]
            head = content[:-SEMANTIC_COMPACT_TAIL_CHARS]
            try:
                summary = _summarize_text_via_llm(self._llm, head)
            except Exception as exc:
                print(
                    f"  ⚠️  Semantic compaction summary call failed ({type(exc).__name__}); keeping this message as-is, "
                    f"leaving it to the truncation fallback (provider={self._provider_name})"
                )
                new_messages.append(msg)
                continue
            if not summary.strip():
                new_messages.append(msg)
                continue
            new_text = _COMPACT_SUMMARY_MARKER + summary + _COMPACT_TAIL_MARKER + tail
            # Only adopt it when it is actually shorter, to avoid a summary that ends up longer
            if len(new_text) < len(content):
                new_messages.append(_clone_message_with_content(msg, new_text))
                changed = True
            else:
                new_messages.append(msg)
        return new_messages, changed

    def _invoke_with_compression(self, args: tuple, kwargs: dict, original_exc: Exception) -> Any:
        """Compression retry when the prompt is too long; returns the sentinel if everything fails.

        Order (inspired by the Claude Code /compact strategy):
          1) Semantic compaction (LLM summary: protect system, keep the verbatim tail, chunked rolling summary);
          2) If semantic compression had no effect or is still over-long → fall back to progressive truncation (80k→40k→16k characters), guaranteeing no crash.
        """
        messages = args[0]
        rest = args[1:]
        last_exc = original_exc

        # 1) Semantic compaction
        semantic_messages, semantic_changed = self._semantic_compress_messages(messages)
        if semantic_changed:
            print(
                f"  🧠 Detected an over-long prompt ({type(last_exc).__name__}); applied LLM summary-style compression to the accumulated context"
                f" (protecting system + keeping the verbatim tail), retrying (provider={self._provider_name})..."
            )
            try:
                return self._llm.invoke(semantic_messages, *rest, **kwargs)
            except Exception as exc:
                last_exc = exc
                if not _is_context_length_error(exc):
                    print(
                        f"  ❌ A non-length error occurred after semantic compression ({type(exc).__name__}); no further compression"
                    )
                    return _COMPRESSION_FAILED
            # Still over-long after semantic compression: continue with the truncation fallback based on the compressed messages
            messages = semantic_messages

        # 2) Truncation fallback (progressively more aggressive)
        for budget in PROMPT_COMPRESSION_BUDGETS:
            new_messages, changed = _compress_messages(messages, budget)
            if not changed:
                continue
            print(
                f"  ⚠️  Semantic compression was insufficient to meet the context limit; falling back to truncating to about {budget} characters and retrying"
                f" (provider={self._provider_name})..."
            )
            try:
                return self._llm.invoke(new_messages, *rest, **kwargs)
            except Exception as exc:
                last_exc = exc
                if not _is_context_length_error(exc):
                    break
                continue
        print(
            f"  ❌ Prompt compression retries still failed (provider={self._provider_name}): {type(last_exc).__name__}"
        )
        return _COMPRESSION_FAILED

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)


def make_chat_llm(**kwargs: Any):
    """Create a ChatOpenAI instance, automatically applying the current api-provider's base_url / key / headers."""
    from langchain_openai import ChatOpenAI

    params: Dict[str, Any] = {
        "model": settings.llm_model,
        "base_url": settings.llm_base_url,
        "api_key": settings.llm_api_key,
        "max_tokens": settings.llm_max_tokens,
        "timeout": settings.llm_request_timeout_seconds,
        "max_retries": 0,
    }
    if settings.llm_default_headers:
        params["default_headers"] = settings.llm_default_headers
    params.update(kwargs)
    llm = ChatOpenAI(**params)
    return RetryableChatLLM(
        llm=llm,
        provider_name=settings.llm_provider_name,
        timeout_seconds=settings.llm_request_timeout_seconds,
        max_attempts=settings.llm_timeout_max_attempts,
        base_retry_delay_seconds=settings.llm_retry_base_delay_seconds,
        max_retry_delay_seconds=settings.llm_retry_max_delay_seconds,
        llm_params=params,
    )


@dataclass
class MorphAgentConfig:
    """MorphAgent global configuration"""
    
    # LLM Settings (Brain). Defaults come from the USER CONFIGURATION block /
    # environment variables at the top of this file. Switch endpoints at runtime
    # with `--api-provider <NAME>` (see API_PROVIDER_PRESETS).
    llm_model: str = DEFAULT_LLM_MODEL
    llm_base_url: Optional[str] = DEFAULT_LLM_BASE_URL
    llm_api_key: Optional[str] = DEFAULT_LLM_API_KEY
    llm_provider_name: str = os.getenv("LLM_PROVIDER_NAME", "default")
    llm_default_headers: Optional[Dict[str, str]] = field(default_factory=lambda: DEFAULT_LLM_HEADERS)
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "65535"))  # Token limit
    llm_request_timeout_seconds: int = int(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "600"))
    llm_timeout_max_attempts: int = int(os.getenv("LLM_TIMEOUT_MAX_ATTEMPTS", "3"))
    llm_retry_base_delay_seconds: int = int(os.getenv("LLM_RETRY_BASE_DELAY_SECONDS", "10"))
    llm_retry_max_delay_seconds: int = int(os.getenv("LLM_RETRY_MAX_DELAY_SECONDS", "120"))
    # The LLM's max_tokens when merging feature code; explicitly raised to a larger limit by default to reduce the risk of long functions being truncated
    # To adjust, override it via the MERGE_MAX_TOKENS environment variable
    merge_max_tokens: Optional[int] = int(os.getenv("MERGE_MAX_TOKENS", "180000"))

    # VLM Settings (Eyes). The public build defaults to the online multimodal
    # API. The fields below with `vlm_model_path` / `vlm_device` only matter for
    # the ADVANCED local Qwen3-VL path (`--vlm-api-provider qwen`).
    vlm_engine: str = "remote"  # "remote" (API, default) or "local"
    vlm_model_path: str = os.getenv("VLM_MODEL_PATH", "Qwen/Qwen3-VL-8B-Instruct")  # local-only
    vlm_device: str = os.getenv("VLM_DEVICE", "cuda")  # local-only
    vlm_max_images: int = int(os.getenv("VLM_MAX_IMAGES", "50"))  # Maximum number of images for the VLM
    vlm_max_tokens: int = int(os.getenv("VLM_MAX_TOKENS", "65535"))  # VLM Token limit
    vlm_image_resize_max: int = int(os.getenv("VLM_IMAGE_RESIZE_MAX", "512"))  # Maximum image resize dimension (long side)
    vlm_supported_formats: Set[str] = field(default_factory=lambda: {'.png', '.jpg', '.jpeg'})  # Image formats supported by the VLM

    # VLM Provider Settings: online=OpenAI-compatible multimodal API (default); qwen=local Qwen3-VL (advanced)
    vlm_api_provider: str = os.getenv("VLM_API_PROVIDER", "online")
    vlm_online_base_url: Optional[str] = DEFAULT_VLM_BASE_URL
    vlm_online_api_key: Optional[str] = DEFAULT_VLM_API_KEY
    vlm_online_model: str = DEFAULT_VLM_MODEL
    vlm_online_default_headers: Optional[Dict[str, str]] = field(default_factory=lambda: DEFAULT_VLM_HEADERS)
    vlm_online_max_tokens: int = int(os.getenv("VLM_ONLINE_MAX_TOKENS", "4096"))
    vlm_online_request_timeout: int = int(os.getenv("VLM_ONLINE_REQUEST_TIMEOUT", "150"))
    vlm_online_max_attempts: int = int(os.getenv("VLM_ONLINE_MAX_ATTEMPTS", "3"))
    # Thread-level hard wall-clock timeout (seconds): even if the server's "trickle-style" slow response defeats the httpx read timeout,
    # the request is forcibly abandoned and retried when the time is up, preventing a single sample from hanging the whole round of VLM scoring indefinitely.
    vlm_online_hard_timeout: int = int(os.getenv("VLM_ONLINE_HARD_TIMEOUT", "180"))
    vlm_temperature: float = float(os.getenv("VLM_TEMPERATURE", "0.0"))

    # Reproducibility (--reproduce)
    reproduce_mode: bool = False
    reproduce_seed: int = int(os.getenv("REPRODUCE_SEED", "42"))
    reproduce_cache_dir: Optional[str] = None  # Set at runtime to data_root/.morphagent_repro_cache
    
    # Segmentation Settings (Optional). UI/demo default backend is Allen
    # (classic aicssegmentation) in the isolated `morphagent_allen` env.
    # Do NOT fall back to CONDA_ENV here — that is the agent env (`morphagent`)
    # and is why Allen previously ran without aicsimageio.
    segmentation_backend: str = os.getenv("SEGMENTATION_BACKEND", "allen")  # "allen" or "cellpose"
    cellpose_model: str = "cyto2"  # Cellpose model name
    segmentation_conda_env: str = os.getenv("SEGMENTATION_CONDA_ENV", "morphagent_allen")
    
    # Data Settings. Point this at your dataset with `--data-root`; the default
    # is a `data/` folder next to this file (see README for the expected layout).
    data_root: str = os.getenv("DATA_ROOT", str(Path(__file__).parent / "data" / "dataset"))
    
    # Dataset Description Files (sorted by priority)
    dataset_description_files: List[str] = field(default_factory=lambda: [
        "dataset_index.txt",
        "README.md",
        "README.txt",
        "dataset_description.json",
        "description.txt"
    ])
    
    # Knowledge Base Settings
    vector_db_path: Optional[str] = os.getenv("VECTOR_DB_PATH", "./knowledge/vector_db")
    expert_examples_path: Optional[str] = os.getenv("EXPERT_EXAMPLES_PATH", "./knowledge/expert_examples")

    # Auto Deep Research (single API call -> markdown report in the
    # deep_research/ folder). See DEFAULT_DEEP_RESEARCH_* at the top of the file.
    deep_research_base_url: Optional[str] = DEFAULT_DEEP_RESEARCH_BASE_URL
    deep_research_api_key: Optional[str] = DEFAULT_DEEP_RESEARCH_API_KEY
    deep_research_model: str = DEFAULT_DEEP_RESEARCH_MODEL
    deep_research_max_tokens: int = int(os.getenv("DEEP_RESEARCH_MAX_TOKENS", "8000"))

    # Auto Literature Retrieval (PubMed / Europe PMC -> PDFs into the RAG folder)
    ncbi_email: str = DEFAULT_NCBI_EMAIL
    ncbi_api_key: str = DEFAULT_NCBI_API_KEY
    pubmed_max_results: int = int(os.getenv("PUBMED_MAX_RESULTS", "8"))
    pubmed_min_year: int = int(os.getenv("PUBMED_MIN_YEAR", "0"))
    pubmed_open_access_only: bool = bool(os.getenv("PUBMED_OPEN_ACCESS_ONLY", "True").lower() == "true")

    # Device hint for optional PaddleX OCR when RAG_PDF_BACKEND=paddlex.
    paddlex_device: str = os.getenv("PADDLEX_DEVICE", "cpu")
    
    # Prompt Templates
    prompt_templates_dir: Path = Path(__file__).parent / "knowledge" / "prompts"
    
    # Code Execution Settings
    code_sandbox_timeout: int = int(os.getenv("CODE_SANDBOX_TIMEOUT", "300"))  # Code execution timeout (seconds)
    code_max_memory: int = int(os.getenv("CODE_MAX_MEMORY", "2"))  # GB
    code_max_retries: int = int(os.getenv("CODE_MAX_RETRIES", "3"))  # Maximum number of retries (default 3)
    code_error_rate_threshold: float = float(os.getenv("CODE_ERROR_RATE_THRESHOLD", "0.5"))  # Error rate threshold (50%)
    code_temperature: float = float(os.getenv("CODE_TEMPERATURE", "0.0"))  # Temperature for code generation (default 0.0 for reproducibility)
    enable_critic_agent: bool = bool(os.getenv("ENABLE_CRITIC_AGENT", "True").lower() == "true")  # Whether to enable the VLM critic agent to evaluate code (enabled by default)
    # Conda env used to EXECUTE agent-generated feature code (the "sandbox").
    # UI handoff: isolated from the Qt/agent env `morphagent` (see MorphAgent_UI setup).
    conda_env: str = os.getenv("CONDA_ENV", "morphagent_sandbox")
    conda_base_paths: List[Path] = field(default_factory=lambda: [
        p for p in [
            Path(os.environ["CONDA_BASE"]) if os.getenv("CONDA_BASE") else None,
            Path(os.environ["CONDA_PREFIX"]).parent.parent if os.getenv("CONDA_PREFIX") else None,
            Path.home() / "miniconda3",
            Path.home() / "anaconda3",
            Path("/opt/conda"),
        ] if p is not None
    ])  # Candidate Conda installation root paths (by priority; can be overridden with CONDA_BASE)
    python_versions: List[str] = field(default_factory=lambda: [
        "python3.10", "python3.11", "python3.9", "python3.12", "python3", "python"
    ])  # List of Python versions (by priority)
    
    # Image Processing Settings
    image_extensions: Set[str] = field(default_factory=lambda: {
        ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"
    })  # Supported image extensions
    
    # Illumination Correction Settings (Singh et al., J. Microscopy 2014)
    enable_illumination_correction: bool = bool(os.getenv("ENABLE_ILLUMINATION_CORRECTION", "False").lower() == "true")  # Whether to enable illumination correction
    illumination_correction_median_window: int = int(os.getenv("ILLUMINATION_CORRECTION_MEDIAN_WINDOW", "150"))  # Median filter window size (pixels)
    illumination_correction_downsample_factor: int = int(os.getenv("ILLUMINATION_CORRECTION_DOWNSAMPLE_FACTOR", "4"))  # Downsampling factor (used to speed up ICF computation)
    illumination_correction_group_by_channel: bool = bool(os.getenv("ILLUMINATION_CORRECTION_GROUP_BY_CHANNEL", "True").lower() == "true")  # Whether to compute ICF grouped by channel (True: per channel; False: all images together)
    
    # Feature Extraction Settings
    target_feature_count: int = int(os.getenv("TARGET_FEATURE_COUNT", "1000"))
    features_per_iteration: int = int(os.getenv("FEATURES_PER_ITERATION", "10"))
    feature_categories: List[str] = field(default_factory=lambda: [
        "morphology", "intensity", "texture", "distribution", "spatial", "other"
    ])  # List of feature categories
    feature_methods: List[str] = field(default_factory=lambda: ["code", "vlm"])  # List of feature extraction methods

    # Data Organization Settings (extracted from the dataset description; these are just defaults)
    # These values should be auto-extracted from the dataset description file and should not be hard-coded
    primary_file_patterns: List[str] = field(default_factory=lambda: [])  # Primary file patterns (extracted from the description)
    secondary_dir_patterns: List[str] = field(default_factory=lambda: [])  # Secondary directory patterns (extracted from the description)
    channel_names: List[str] = field(default_factory=lambda: [])  # Channel names (extracted from the description)

    def __post_init__(self):
        """Post-processing: validate and create the necessary directories"""
        if self.llm_api_key is None:
            print("Warning: LLM_API_KEY is not set.")
        
        # Ensure the directories exist
        self.prompt_templates_dir.mkdir(parents=True, exist_ok=True)
        Path(self.vector_db_path).mkdir(parents=True, exist_ok=True) if self.vector_db_path else None
        Path(self.expert_examples_path).mkdir(parents=True, exist_ok=True) if self.expert_examples_path else None
    
    def update_from_dataset_description(self, description: str):
        """Update the configuration from the dataset description (auto-detect channel names, file structure, etc.)
        
        Args:
            description: The dataset description text
        """
        # Logic for extracting information from the description can be added here
        # For example: extracting channel names, file naming patterns, etc.
        # Left empty for now; can later be extracted via an LLM or regular expressions
        pass


# Global singleton configuration
settings = MorphAgentConfig()
