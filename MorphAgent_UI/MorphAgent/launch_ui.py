#!/usr/bin/env python3
"""Launch the focused MorphAgent desktop app or its optional napari mode."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path


def load_repository_environment(env_path: str | Path | None = None) -> bool:
    """Load the repository .env as the UI's single configuration source."""
    from dotenv import dotenv_values

    path = Path(env_path) if env_path is not None else Path(__file__).resolve().parent / ".env"
    if not path.is_file():
        return False
    values = dotenv_values(path)
    for name, value in values.items():
        if value is not None:
            os.environ[name] = value
    return bool(values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the MorphAgent desktop interface")
    parser.add_argument(
        "--demo-page",
        choices=["home", "configure", "run", "results", "features", "evidence", "settings"],
        default=None,
        help="Visual-only state for screenshots; legacy results/settings names map to Features/Configure",
    )
    parser.add_argument(
        "--with-napari",
        action="store_true",
        help="Embed MorphAgent beside napari's image canvas and layer controls (optional).",
    )
    return parser


def create_standalone_window(demo_page: str | None = None):
    """Create the focused Qt window without napari's unused viewer chrome."""
    from qtpy.QtWidgets import QApplication, QMainWindow

    from morphagent_ui.main import MorphAgentWidget

    application = QApplication.instance() or QApplication([sys.argv[0]])
    application.setApplicationName("MorphAgent")
    application.setOrganizationName("MorphAgent")
    window = QMainWindow()
    window.setWindowTitle("MorphAgent · biologically grounded microscopy profiling")
    widget = MorphAgentWidget()
    if demo_page:
        widget.show_demo(demo_page)
    window.setCentralWidget(widget)
    window.setMinimumSize(1100, 720)
    window.resize(1320, 860)
    return application, window, widget


def launch_standalone(demo_page: str | None = None) -> int:
    application, window, _widget = create_standalone_window(demo_page)
    window.showMaximized()
    return application.exec_()


def launch_with_napari(demo_page: str | None = None) -> int:
    """Launch the optional viewer-integrated workspace for layer inspection."""
    try:
        import napari
    except ImportError as exc:
        raise SystemExit(
            "napari mode is not installed. Run `pip install -e \".[napari]\"` "
            "or launch the focused default window without --with-napari."
        ) from exc

    from morphagent_ui.main import MorphAgentWidget

    viewer = napari.Viewer(title="MorphAgent · biologically grounded microscopy profiling")
    widget = MorphAgentWidget(viewer=viewer)
    if demo_page:
        widget.show_demo(demo_page)
    dock = viewer.window.add_dock_widget(widget, area="right", name="MorphAgent")
    dock.setMinimumWidth(1060)
    viewer.window._qt_window.showMaximized()
    napari.run()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    load_repository_environment()
    args = build_parser().parse_args(argv)
    if args.with_napari:
        return launch_with_napari(args.demo_page)
    return launch_standalone(args.demo_page)


if __name__ == "__main__":
    raise SystemExit(main())
