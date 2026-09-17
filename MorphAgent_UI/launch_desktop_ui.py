#!/usr/bin/env python3
"""Open the current workspace in a Qt6 desktop window (not the legacy Qt5 UI)."""
import argparse
import signal
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=0,help='Internal loopback port (default: automatically find a free port)')
    args=parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error('--port must be between 0 and 65535')
    try:
        from morphagent_ui.desktop_window import create_application, WorkspaceWindow
        from PySide6.QtCore import QCoreApplication, QEvent, QTimer
    except ImportError as exc:
        parser.exit(1,'Desktop dependencies are missing. In your current environment run:\n'
                    '  python -m pip install -r dependencies/requirements-desktop.txt\n'
                    f'Import error: {exc}\n')
    from morphagent_ui.desktop_runtime import DesktopRuntime
    app=create_application()
    try:
        with DesktopRuntime(ROOT,port=args.port) as runtime:
            window=WorkspaceWindow(runtime)
            window.showMaximized()
            print(f'MorphAgent desktop · Python: {sys.executable}\nClose the window to stop the local service.',flush=True)
            previous=signal.signal(signal.SIGINT,lambda *_:window.close())
            timer=QTimer()
            timer.timeout.connect(lambda:None)
            timer.start(250)  # Allow Python signals while the Qt event loop is idle.
            try:
                return app.exec()
            finally:
                timer.stop()
                signal.signal(signal.SIGINT,previous)
                window.deleteLater()
                QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
    except (RuntimeError,OSError) as exc:
        parser.exit(1,f'{exc}\nIf a browser workspace is already running, stop its terminal with Ctrl+C first.\n')


if __name__ == '__main__':
    raise SystemExit(main())
