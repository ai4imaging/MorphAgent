"""Main navigation shell for MorphAgent."""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import QEvent, Qt
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import (
    QAction,
    QApplication,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .controller import RunController
from .environment import save_ui_preference_environment
from .models import RunConfig
from .theme import (
    UI_FONT_SCALE_STEP,
    build_stylesheet,
    clamp_ui_font_scale,
    detect_display_scale,
    resolve_ui_font_scale,
)
from .widgets.configure import ConfigurePage
from .widgets.home import HomePage
from .widgets.results import EvidencePage, FeaturesPage
from .widgets.run import RunPage


NAVIGATION = (
    ("Home", "Overview and quick start"),
    ("Configure", "Dataset, question, and preflight"),
    ("Run", "Stages, logs, and artifacts"),
    ("Features", "Feature library and biological interpretation"),
    ("Evidence", "Complete run artifacts and image previews"),
)


class MorphAgentWidget(QWidget):
    """A napari-compatible widget that also works in a standalone window."""

    def __init__(self, viewer=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self.config = RunConfig()
        self.controller = RunController(self)
        self._repository_root = Path(__file__).resolve().parents[1]
        self._display_font_scale = detect_display_scale()
        self._font_scale = resolve_ui_font_scale(
            repository_root=self._repository_root,
        )

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.sidebar = QWidget()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setStyleSheet(
            "QWidget#Sidebar { background: #0B1626; border-right: 1px solid #29405E; }"
        )
        side_layout = QVBoxLayout(self.sidebar)
        side_layout.setContentsMargins(10, 16, 10, 12)
        side_layout.setSpacing(12)
        self.mark = QLabel("M")
        self.mark.setAlignment(Qt.AlignCenter)
        brand = QLabel("MorphAgent")
        brand.setProperty("role", "title")
        brand_row = QHBoxLayout()
        brand_row.addWidget(self.mark)
        brand_row.addWidget(brand)
        brand_row.addStretch(1)
        side_layout.addLayout(brand_row)

        self.navigation = QListWidget()
        self.navigation.setObjectName("Navigation")
        self.navigation.setSpacing(2)
        for index, (title, tip) in enumerate(NAVIGATION):
            item = QListWidgetItem(f"{index + 1:02d}  {title}")
            item.setToolTip(tip)
            item.setData(Qt.AccessibleTextRole, f"{title}: {tip}")
            self.navigation.addItem(item)
        side_layout.addWidget(self.navigation, 1)

        zoom_row = QHBoxLayout()
        zoom_row.setSpacing(6)
        self.font_smaller_btn = QToolButton()
        self.font_smaller_btn.setText("A-")
        self.font_smaller_btn.setToolTip("Smaller text (Ctrl/Cmd -)")
        self.font_larger_btn = QToolButton()
        self.font_larger_btn.setText("A+")
        self.font_larger_btn.setToolTip("Larger text (Ctrl/Cmd + or =)")
        self.font_scale_label = QLabel("")
        self.font_scale_label.setProperty("role", "muted")
        self.font_scale_label.setAlignment(Qt.AlignCenter)
        self.font_scale_label.setToolTip("Text size. Shortcuts: Ctrl/Cmd + or = enlarge, - shrink, 0 reset. Ctrl/Cmd + mouse wheel also works.")
        zoom_row.addWidget(self.font_smaller_btn, 1)
        zoom_row.addWidget(self.font_larger_btn, 1)
        side_layout.addLayout(zoom_row)
        side_layout.addWidget(self.font_scale_label)
        version = QLabel("v0.1 · manuscript UI")
        version.setProperty("role", "muted")
        version.setAlignment(Qt.AlignCenter)
        side_layout.addWidget(version)
        root.addWidget(self.sidebar)

        self.pages = QStackedWidget()
        self.home_page = HomePage()
        self.configure_page = ConfigurePage(self.config)
        self.run_page = RunPage(self.controller)
        self.features_page = FeaturesPage(viewer)
        self.evidence_page = EvidencePage(viewer)
        for page in (
            self.home_page,
            self.configure_page,
            self.run_page,
            self.features_page,
            self.evidence_page,
        ):
            self.pages.addWidget(page)
        root.addWidget(self.pages, 1)
        self._font_actions: list[QAction] = []
        self._connect()
        self.navigate(0)
        self._install_font_controls()
        self._apply_font_scale(persist=False)
        self.setMinimumSize(1050, 700)
        self.setFocusPolicy(Qt.StrongFocus)

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().showEvent(event)
        self._attach_font_actions_to_window()

    def _scaled_px(self, base: int) -> int:
        return max(8, int(round(base * self._font_scale)))

    def _apply_font_scale(self, *, persist: bool = True) -> None:
        """Keep MorphAgent styles scoped to this dock (not the whole QApplication)."""

        self._font_scale = clamp_ui_font_scale(self._font_scale)
        self.setStyleSheet(build_stylesheet(self._font_scale))
        mark_size = self._scaled_px(34)
        self.mark.setFixedSize(mark_size, mark_size)
        self.mark.setStyleSheet(
            "background: #0891B2; color: #F0FDFF; border: 1px solid #22D3EE; "
            f"border-radius: {self._scaled_px(9)}px; font-size: {self._scaled_px(18)}px; font-weight: 800;"
        )
        self.sidebar.setFixedWidth(self._scaled_px(188))
        if hasattr(self, "font_scale_label"):
            self.font_scale_label.setText(f"Text {self._font_scale:.0%}")
        if hasattr(self.configure_page, "demo_guide"):
            self.configure_page.demo_guide.setStyleSheet(
                f"font-size: {self._scaled_px(22)}px;"
            )
        if hasattr(self.run_page, "log_view"):
            self.run_page.log_view.setStyleSheet(
                'font-family: "Fira Code", "SFMono-Regular", monospace; '
                f"font-size: {self._scaled_px(11)}px;"
            )
        if persist:
            try:
                save_ui_preference_environment(
                    self._repository_root,
                    {"UI_FONT_SCALE": f"{self._font_scale:.2f}"},
                )
            except Exception:
                pass

    def _install_font_controls(self) -> None:
        """Buttons + window actions + key/wheel filter (QShortcut alone is flaky on macOS)."""

        self.font_smaller_btn.clicked.connect(self._zoom_font_out)
        self.font_larger_btn.clicked.connect(self._zoom_font_in)

        # QAction on the top-level window is more reliable than QShortcut on a child.
        context = (
            Qt.ApplicationShortcut if self.viewer is None else Qt.WindowShortcut
        )
        specs = (
            (
                "Zoom text in",
                [
                    QKeySequence(QKeySequence.ZoomIn),
                    QKeySequence(Qt.CTRL | Qt.Key_Equal),
                    QKeySequence(Qt.CTRL | Qt.Key_Plus),
                    QKeySequence(Qt.META | Qt.Key_Equal),
                    QKeySequence(Qt.META | Qt.Key_Plus),
                ],
                self._zoom_font_in,
            ),
            (
                "Zoom text out",
                [
                    QKeySequence(QKeySequence.ZoomOut),
                    QKeySequence(Qt.CTRL | Qt.Key_Minus),
                    QKeySequence(Qt.CTRL | Qt.Key_Underscore),
                    QKeySequence(Qt.META | Qt.Key_Minus),
                    QKeySequence(Qt.META | Qt.Key_Underscore),
                ],
                self._zoom_font_out,
            ),
            (
                "Reset text zoom",
                [
                    QKeySequence(Qt.CTRL | Qt.Key_0),
                    QKeySequence(Qt.META | Qt.Key_0),
                ],
                self._zoom_font_reset,
            ),
        )
        self._font_actions = []
        for name, sequences, slot in specs:
            action = QAction(name, self)
            action.setShortcuts([seq for seq in sequences if not seq.isEmpty()])
            action.setShortcutContext(context)
            action.triggered.connect(slot)
            self.addAction(action)
            self._font_actions.append(action)

        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
        self._attach_font_actions_to_window()

    def _attach_font_actions_to_window(self) -> None:
        host = self.window()
        if host is None or host is self:
            return
        for action in self._font_actions:
            if action not in host.actions():
                host.addAction(action)

    def eventFilter(self, obj, event):  # noqa: N802 - Qt API
        if not self._is_font_zoom_target(obj):
            return super().eventFilter(obj, event)

        # Let our QActions win over line-edits that would otherwise eat the keys.
        if event.type() == QEvent.ShortcutOverride and self._is_font_zoom_chord(event):
            event.accept()
            return True

        if event.type() == QEvent.KeyPress and self._handle_font_zoom_key(event):
            return True

        if event.type() == QEvent.Wheel and self._handle_font_zoom_wheel(event):
            return True

        return super().eventFilter(obj, event)

    def _is_font_zoom_target(self, obj) -> bool:
        if obj is self:
            return True
        try:
            if isinstance(obj, QWidget) and (obj is self.window() or self.isAncestorOf(obj)):
                return True
        except RuntimeError:
            return False
        return False

    def _is_font_zoom_chord(self, event) -> bool:
        modifiers = event.modifiers()
        if not (modifiers & (Qt.ControlModifier | Qt.MetaModifier)):
            return False
        if modifiers & Qt.AltModifier:
            return False
        return event.key() in (
            Qt.Key_Plus,
            Qt.Key_Equal,
            Qt.Key_Minus,
            Qt.Key_Underscore,
            Qt.Key_0,
        )

    def _handle_font_zoom_key(self, event) -> bool:
        if not self._is_font_zoom_chord(event):
            return False
        key = event.key()
        if key in (Qt.Key_Plus, Qt.Key_Equal):
            self._zoom_font_in()
            return True
        if key in (Qt.Key_Minus, Qt.Key_Underscore):
            self._zoom_font_out()
            return True
        if key == Qt.Key_0:
            self._zoom_font_reset()
            return True
        return False

    def _handle_font_zoom_wheel(self, event) -> bool:
        modifiers = event.modifiers()
        if not (modifiers & (Qt.ControlModifier | Qt.MetaModifier)):
            return False
        delta = event.angleDelta().y()
        if delta == 0:
            delta = event.pixelDelta().y()
        if delta > 0:
            self._zoom_font_in()
            return True
        if delta < 0:
            self._zoom_font_out()
            return True
        return False

    def _zoom_font_in(self) -> None:
        self._font_scale = clamp_ui_font_scale(self._font_scale + UI_FONT_SCALE_STEP)
        self._apply_font_scale()

    def _zoom_font_out(self) -> None:
        self._font_scale = clamp_ui_font_scale(self._font_scale - UI_FONT_SCALE_STEP)
        self._apply_font_scale()

    def _zoom_font_reset(self) -> None:
        self._font_scale = self._display_font_scale
        self._apply_font_scale()

    def _connect(self) -> None:
        self.navigation.currentRowChanged.connect(self.pages.setCurrentIndex)
        self.home_page.new_run_requested.connect(self._new_run)
        self.home_page.previous_run_requested.connect(self._load_previous_results)
        self.configure_page.run_requested.connect(self._start_run)
        self.run_page.cancel_requested.connect(self.controller.cancel)
        self.run_page.edit_requested.connect(lambda: self.navigate(1))
        self.run_page.review_requested.connect(self._show_results)
        self.features_page.feature_selected.connect(self.evidence_page.select_feature)
        self.controller.run_finished.connect(lambda _success, _path: self._refresh_results())

    def navigate(self, index: int) -> None:
        self.navigation.setCurrentRow(index)
        self.pages.setCurrentIndex(index)

    def _new_run(self) -> None:
        if self.controller.running:
            self.navigate(2)
            return
        # Keep the user's run scale (and any values saved in .env). Only the
        # free restricted API path locks scale to 1 round × 5 candidates.
        self.config.resume = False
        self.config.results_dir = ""
        self.configure_page.load_from_config()
        self.configure_page.load_run_scale_settings()
        self.configure_page.refresh_preflight(scan=False)
        self.navigate(1)

    def _load_previous_results(self, results_dir: str) -> None:
        path = Path(results_dir).expanduser()
        if not path.is_dir():
            QMessageBox.warning(self, "Run folder not found", "Choose an existing MorphAgent results folder.")
            return
        self.features_page.load_results(str(path))
        self.evidence_page.set_results(str(path), self.features_page.cards)
        if not self.features_page.cards and not self.evidence_page.artifacts:
            QMessageBox.warning(
                self,
                "No MorphAgent results found",
                "This folder does not contain feature cards or recognizable run artifacts.",
            )
            return
        self.config.results_dir = str(path)
        self.navigate(3)

    def _start_run(self, config: RunConfig, dataset_summary) -> None:
        try:
            results_dir = self.controller.start(config, dataset_summary)
        except (RuntimeError, OSError) as exc:
            QMessageBox.critical(self, "Could not launch MorphAgent", str(exc))
            return
        self.run_page.begin(results_dir)
        self.navigate(2)

    def _refresh_results(self) -> None:
        results_dir = self.run_page.results_dir or self.config.results_dir
        if not results_dir:
            return
        self.features_page.load_results(results_dir)
        self.evidence_page.set_results(results_dir, self.features_page.cards)

    def _show_results(self) -> None:
        self._refresh_results()
        self.navigate(3)

    def show_demo(self, page: str = "home") -> None:
        """Explicit screenshot-only state; never used by a scientific run."""
        requested = page.lower()
        normalized = "features" if requested == "results" else requested
        if normalized == "settings":
            normalized = "configure"
        pages = {name.lower(): index for index, (name, _tip) in enumerate(NAVIGATION)}
        if normalized == "run":
            self.run_page.show_demo_state()
        elif normalized == "features":
            self.features_page.show_demo_state()
        elif normalized == "evidence":
            self.features_page.show_demo_state()
            self.evidence_page.set_results("", self.features_page.cards)
        self.navigate(pages.get(normalized, 0))
