#!/usr/bin/env python3
"""Launch the Codex-style workspace, connected to the original MorphAgent CLI."""
import argparse
import sys
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))


def main():
    from morphagent_ui.web_service import workspace_lock
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    try:
        with workspace_lock(ROOT):
            serve(args, parser)
    except RuntimeError as exc:
        parser.exit(1, str(exc) + '\n')


def serve(args, parser):
    from morphagent_ui.web_service import WorkspaceService, ACTIVE
    from morphagent_ui.web_http import create_server
    service = WorkspaceService(ROOT)
    try:
        server = create_server(service, args.port)
    except OSError as exc:
        parser.exit(1, f'Cannot open port {args.port}: {exc}\nChoose another port with --port 8767.\n')
    url = f'http://127.0.0.1:{server.server_port}'
    print(f'MorphAgent workspace: {url}\nPython: {sys.executable}\nKeep this terminal open. Press Ctrl+C to stop.', flush=True)
    if not args.no_browser:
        threading.Timer(.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=.25)
    except KeyboardInterrupt:
        print('\nStopping local workspace and its active runs…', flush=True)
    finally:
        service.closing = True
        for key, job in service.jobs.items():
            if job['status'] in ACTIVE:
                service.cancel_run(key)
        server.server_close()
        # Keep the workspace lock while workers persist cancellation or finish
        # packaging completed results. Raw outputs survive an interrupted export.
        import time
        deadline = time.monotonic() + 7
        while (service.processes or any(j.get('exportStatus') == 'saving' for j in service.jobs.values())) and time.monotonic() < deadline:
            time.sleep(.1)


if __name__ == '__main__':
    main()
