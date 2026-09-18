"""Process-wide locks for the resources feature-level parallelism now shares.

Running feature code generation and VLM scoring on several threads turned a
number of single-threaded assumptions into races. The locks that guard shared
*process* resources live here rather than in the modules that happen to trip
over them, so their scope and ordering can be read in one place.

Lock ordering: each lock below is a leaf. Never acquire one while holding
another, and the set stays deadlock-free by construction.
"""
from __future__ import annotations

import contextlib
import io
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, List, Optional, Sequence

# Generated code is installed into one shared Conda sandbox. Two concurrent
# `pip install` calls into the same environment can leave it half-written, so
# installs are serialised even though the executions around them are parallel.
_SANDBOX_INSTALL_LOCK = threading.Lock()

# Installs are launched from two places that cannot see each other's state: this
# process (the fix scripts) and the sandbox subprocesses (auto-install on
# ImportError). They therefore coordinate through a directory on disk, whose
# path is shared with the generated wrapper in ``code_executor``.
SANDBOX_INSTALL_LOCK_PATH = str(
    Path(__import__("tempfile").gettempdir()) / "morphagent_sandbox_install.lock"
)

# Longest a process will wait for its turn before installing anyway. Proceeding
# unserialised risks a garbled environment; waiting forever on a lock whose owner
# has died stalls the whole run, and only one of those is recoverable.
_INSTALL_LOCK_TIMEOUT = 120.0
_INSTALL_LOCK_STALE_AFTER = 300.0


@contextlib.contextmanager
def sandbox_install_mutex() -> Iterator[bool]:
    """Serialise installs into the shared sandbox, across threads and processes.

    Yields whether the cross-process claim succeeded; a false value means the
    caller timed out waiting and is proceeding anyway, which is deliberate.
    """
    import os
    import time

    with _SANDBOX_INSTALL_LOCK:
        held = False
        deadline = time.time() + _INSTALL_LOCK_TIMEOUT
        while time.time() < deadline:
            try:
                os.mkdir(SANDBOX_INSTALL_LOCK_PATH)
                held = True
                break
            except FileExistsError:
                try:
                    age = time.time() - os.path.getmtime(SANDBOX_INSTALL_LOCK_PATH)
                    if age > _INSTALL_LOCK_STALE_AFTER:
                        os.rmdir(SANDBOX_INSTALL_LOCK_PATH)  # owner died holding it
                        continue
                except Exception:
                    pass
                time.sleep(0.5)
            except Exception:
                break
        try:
            yield held
        finally:
            if held:
                try:
                    os.rmdir(SANDBOX_INSTALL_LOCK_PATH)
                except Exception:
                    pass

# The critic scores through the process-wide VLM client. With local weights that
# client is a single GPU model; even online it shares one endpoint adaptation
# state. Serialising critic calls keeps code generation threads from colliding.
CRITIC_LOCK = threading.Lock()

# `get_data_path_selector()` hands every thread the same instance, including its
# one LLM client. This guards both the singleton's creation and its use.
PATH_SELECTOR_LOCK = threading.RLock()

# Threads append to the same execution log. Interleaved writes do not corrupt
# the file on POSIX, but they do tear lines apart and make a failed run much
# harder to read, which matters more than the negligible lock cost.
_LOG_LOCK = threading.Lock()


def append_log(path: Optional[Path | str], text: str) -> None:
    """Append one chunk to a shared log file, whole, or give up silently.

    Logging must never be the reason an analysis fails, so an unwritable path is
    swallowed exactly as the call sites did before.
    """
    if not path or not text:
        return
    try:
        with _LOG_LOCK:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
    except Exception:
        pass


def run_ordered(
    tasks: Sequence[Any],
    worker: Callable[[Any], Any],
    max_workers: int,
    on_result: Optional[Callable[[Any], None]] = None,
) -> List[Any]:
    """Run ``worker`` over ``tasks`` concurrently, returning results in task order.

    Two properties matter more here than the speedup. Results are returned in the
    order the tasks were given, never the order the threads happened to finish, so
    nothing downstream can drift with timing. And ``on_result`` fires as each
    result lands, so a long run keeps reporting progress instead of going quiet
    until the last task returns.

    ``worker`` is expected to handle its own failures; an exception it does let
    escape propagates once the pool drains, exactly as it would serially.
    """
    tasks = list(tasks)
    if not tasks:
        return []

    workers = max(1, min(int(max_workers), len(tasks)))
    if workers == 1:
        results = []
        for task in tasks:
            result = worker(task)
            if on_result is not None:
                on_result(result)
            results.append(result)
        return results

    from concurrent.futures import ThreadPoolExecutor, as_completed

    slots: List[Any] = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker, task): position for position, task in enumerate(tasks)}
        for future in as_completed(futures):
            position = futures[future]
            result = future.result()
            if on_result is not None:
                on_result(result)
            slots[position] = result
    return slots


class _ThreadRoutedStdout:
    """A ``sys.stdout`` stand-in that can divert one thread's writes to a buffer.

    The pipeline reports progress with ``print``. Once features are generated on
    several threads those reports arrive interleaved, so a feature's traceback
    lands in the middle of another feature's success message. Threads that opt in
    write to their own buffer instead, which the worker flushes as one block when
    the feature finishes; every other thread reaches the real stdout untouched.
    """

    def __init__(self, base):
        self._base = base
        self._local = threading.local()

    def _buffer(self):
        return getattr(self._local, "buffer", None)

    def route(self, buffer: Optional[io.StringIO]) -> None:
        self._local.buffer = buffer

    def write(self, data):
        target = self._buffer()
        if target is None:
            return self._base.write(data)
        return target.write(data)

    def flush(self):
        if self._buffer() is None:
            self._base.flush()

    def __getattr__(self, name):
        return getattr(self._base, name)


# Installing the proxy is refcounted so nested or repeated parallel sections
# cannot restore stdout while an outer section is still using it.
_stdout_proxy: Optional[_ThreadRoutedStdout] = None
_stdout_depth = 0
_stdout_lock = threading.Lock()


@contextlib.contextmanager
def thread_grouped_stdout() -> Iterator[None]:
    """Install per-thread stdout routing for the duration of a parallel section."""
    global _stdout_proxy, _stdout_depth
    with _stdout_lock:
        if _stdout_depth == 0:
            _stdout_proxy = _ThreadRoutedStdout(sys.stdout)
            sys.stdout = _stdout_proxy
        _stdout_depth += 1
        original = _stdout_proxy._base if _stdout_proxy else sys.stdout
    try:
        yield
    finally:
        with _stdout_lock:
            _stdout_depth -= 1
            if _stdout_depth == 0:
                sys.stdout = original
                _stdout_proxy = None


@contextlib.contextmanager
def collected_output() -> Iterator[List[str]]:
    """Collect this thread's prints, yielding a one-element list with the text.

    Falls back to printing straight through when no parallel section is active,
    so the same worker body works serially.
    """
    holder: List[str] = [""]
    proxy = _stdout_proxy
    if proxy is None:
        yield holder
        return
    buffer = io.StringIO()
    proxy.route(buffer)
    try:
        yield holder
    finally:
        proxy.route(None)
        holder[0] = buffer.getvalue()
