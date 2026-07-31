"""Main navigation shell for MorphAgent."""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QShortcut,
    QStackedWidget,
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
        version = QLabel("v0.1 · manuscript UI")
        version.setProperty("role", "muted")
        version.setAlignment(Qt.AlignCenter)
        version.setToolTip("Font size: Ctrl + / Ctrl − / Ctrl 0")
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
        self._connect()
        self.navigate(0)
        self._apply_font_scale(persist=False)
        self._install_font_shortcuts()
        self.setMinimumSize(1050, 700)

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

    def _install_font_shortcuts(self) -> None:
        bindings = (
            (QKeySequence.ZoomIn, self._zoom_font_in),
            (QKeySequence("Ctrl+="), self._zoom_font_in),
            (QKeySequence("Ctrl++"), self._zoom_font_in),
            (QKeySequence.ZoomOut, self._zoom_font_out),
            (QKeySequence("Ctrl+-"), self._zoom_font_out),
            (QKeySequence("Ctrl+0"), self._zoom_font_reset),
        )
        self._font_shortcuts = []
        for sequence, slot in bindings:
            shortcut = QShortcut(sequence, self)
            shortcut.setContext(Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(slot)
            self._font_shortcuts.append(shortcut)

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
