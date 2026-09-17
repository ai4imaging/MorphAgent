"""Lifecycle for the desktop's private, loopback-only workspace service."""
from contextlib import ExitStack
from pathlib import Path
import sys
import threading
import time
from urllib.parse import urlsplit

from .web_http import create_server
from .web_service import ACTIVE, WorkspaceService, workspace_lock


def is_workspace_url(url: str, origin: str) -> bool:
    """Only the exact local origin may navigate the trusted application view."""
    try:
        target, local = urlsplit(url), urlsplit(origin)
        return (target.scheme == local.scheme == 'http'
                and target.hostname == local.hostname == '127.0.0.1'
                and target.port == local.port
                and target.username is None and target.password is None)
    except ValueError:
        return False


class DesktopRuntime:
    def __init__(self, repository_root, *, port=0, python_executable=sys.executable):
        self.repo = Path(repository_root).resolve()
        self.port = port
        self.python = python_executable
        self.service = None
        self.server = None
        self.thread = None
        self._resources = ExitStack()
        self._stopped = False

    def __enter__(self):
        try:
            self._resources.enter_context(workspace_lock(self.repo))
            self.service = WorkspaceService(self.repo, self.python)
            self.server = create_server(self.service, self.port)
            self._resources.callback(self.server.server_close)
            self.url = f'http://127.0.0.1:{self.server.server_port}'
            self.thread = threading.Thread(target=self.server.serve_forever,
                                           kwargs={'poll_interval':.1}, daemon=True,
                                           name='MorphAgent-local-server')
            self.thread.start()
            return self
        except BaseException:
            self._resources.close()
            raise

    def has_active_runs(self):
        if not self.service:
            return False
        with self.service.lock:
            return any(j['status'] in ACTIVE or j.get('exportStatus') == 'saving'
                       for j in self.service.jobs.values())

    def stop(self, timeout=10):
        """Stop only owned workers; keep the workspace locked until they finish."""
        if self._stopped:
            return
        if self.service:
            with self.service.lock:
                self.service.closing = True
        if self.thread and self.thread.is_alive():
            self.server.shutdown()
            self.thread.join(timeout=2)
        if self.service:
            with self.service.lock:
                keys = [k for k,j in self.service.jobs.items() if j['status'] in ACTIVE]
            for key in keys:
                self.service.cancel_run(key)
            deadline = time.monotonic()+timeout
            while self.has_active_runs() and time.monotonic() < deadline:
                time.sleep(.05)
            if self.has_active_runs():
                raise RuntimeError('Analysis is still stopping or saving results. Please wait, then close again.')
            self.service.connection.clear()
            self.service.secrets.clear()
        self._resources.close()
        self._stopped = True

    def __exit__(self, *_):
        self.stop()
