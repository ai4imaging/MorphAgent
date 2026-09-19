"""Qt6 shell for the existing workspace, without importing the legacy Qt5 UI."""
import json
from pathlib import Path
import sys
import threading

if 'PyQt5.QtCore' in sys.modules or 'PyQt6.QtCore' in sys.modules:
    raise RuntimeError('Start the desktop workspace in a fresh process; do not mix Qt bindings.')

from PySide6.QtCore import Qt, QTimer, QUrl, QStandardPaths, Signal, Slot
from PySide6.QtGui import QDesktopServices, QKeySequence, QShortcut
from PySide6.QtNetwork import QNetworkProxy
from PySide6.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile, QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView

from .desktop_runtime import is_workspace_url


def create_application(argv=None):
    app = QApplication.instance()
    if app is None:
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
        app = QApplication(argv if argv is not None else [sys.argv[0]])
        app.setApplicationName('MorphAgent')
        app.setOrganizationName('MorphAgent')
    # The GUI only loads localhost. Do not route it through the system proxy.
    # Python/API subprocess proxy environment variables remain untouched.
    QNetworkProxy.setApplicationProxy(QNetworkProxy(QNetworkProxy.ProxyType.NoProxy))
    return app


class WorkspacePage(QWebEnginePage):
    FOLDER_PROMPTS = {
        'Choose the feature folder from a saved run:':'Choose a saved feature folder',
        'Paste the local results folder path (contains features.csv):': 'Choose saved results',
        'Paste the new dataset folder path (contains dataset/):': 'Choose target dataset',
        'Paste the dataset folder path (contains dataset/):': 'Choose dataset folder',
    }

    def __init__(self, profile, origin, parent=None):
        super().__init__(profile, parent)
        self.origin = origin

    def acceptNavigationRequest(self, url, navigation_type, is_main_frame):
        if is_workspace_url(url.toString(), self.origin):
            return True
        if (is_main_frame and navigation_type == self.NavigationType.NavigationTypeLinkClicked
                and url.scheme() in ('http','https')):
            QDesktopServices.openUrl(url)
        return False

    def chooseFiles(self, mode, old_files, accepted_mime_types):
        if mode == self.FileSelectionMode.FileSelectUploadFolder:
            folder = QFileDialog.getExistingDirectory(self.view(), 'Choose dataset folder',
                                                      old_files[0] if old_files else '')
            # Hand the page the path and no files. Returning the folder would make
            # the renderer enumerate every file and upload it over loopback HTTP,
            # which copies the whole dataset and stops scaling in the thousands.
            if folder:
                self.deliver_folder(folder)
            return []
        return super().chooseFiles(mode, old_files, accepted_mime_types)

    def deliver_folder(self, folder):
        script = 'window.morphagentFolderChosen && window.morphagentFolderChosen(%s)' % json.dumps(folder)
        # chooseFiles runs while the renderer waits; defer so it is not reentrant.
        QTimer.singleShot(0, lambda: self.runJavaScript(script))

    def view(self):
        return QWebEngineView.forPage(self)

    def javaScriptPrompt(self, security_origin, message, default_value):
        if not is_workspace_url(security_origin.toString(), self.origin):
            return False, ''
        if message in self.FOLDER_PROMPTS:
            folder = QFileDialog.getExistingDirectory(self.view(), self.FOLDER_PROMPTS[message], default_value)
            return bool(folder), folder
        if message == 'Choose feature_value.csv to visualize:':
            file, _ = QFileDialog.getOpenFileName(self.view(), 'Choose feature_value.csv', default_value, 'CSV files (*.csv)')
            return bool(file), file
        return super().javaScriptPrompt(security_origin, message, default_value)

    def javaScriptConsoleMessage(self, level, message, line_number, source_id):
        # Never mirror page inputs/API error details into a permanent console log.
        pass


class WorkspaceWindow(QMainWindow):
    shutdown_finished = Signal(str)

    def __init__(self, runtime):
        super().__init__()
        self.runtime = runtime
        self.closed = False
        self._closing = False
        self.setWindowTitle('MorphAgent · Workspace')
        self.resize(1400,900)
        self.setMinimumSize(800,600)
        self.view = QWebEngineView(self)
        self.view.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self.setCentralWidget(self.view)
        self.profile = QWebEngineProfile(self)  # unnamed = off-the-record
        self.profile.setHttpCacheType(QWebEngineProfile.HttpCacheType.MemoryHttpCache)
        self.page = WorkspacePage(self.profile, runtime.url, self.view)
        self.view.setPage(self.page)
        self.page.settings().setAttribute(QWebEngineSettings.WebAttribute.JavascriptCanOpenWindows, False)
        self.page.settings().setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, False)
        self.profile.downloadRequested.connect(self.save_download)
        self.view.loadFinished.connect(self._loaded)
        self.page.renderProcessTerminated.connect(self._renderer_stopped)
        self.shutdown_finished.connect(self._shutdown_done)
        self._shortcuts = []
        for key, callback in ((QKeySequence.StandardKey.Quit,self.close),
                              (QKeySequence.StandardKey.ZoomIn,lambda:self.zoom(.1)),
                              (QKeySequence.StandardKey.ZoomOut,lambda:self.zoom(-.1)),
                              ('Ctrl+0',lambda:self.view.setZoomFactor(1)),
                              ('Ctrl+R',self.view.reload)):
            shortcut=QShortcut(QKeySequence(key),self)
            shortcut.activated.connect(callback)
            self._shortcuts.append(shortcut)
        self.statusBar().showMessage('Opening workspace…')
        self.view.load(QUrl(runtime.url))

    def zoom(self, delta):
        self.view.setZoomFactor(max(.6,min(2,self.view.zoomFactor()+delta)))

    def _loaded(self, ok):
        if not self._closing:
            self.statusBar().showMessage('' if ok else 'Could not load the workspace. Press Ctrl/Cmd+R to retry.')

    def _renderer_stopped(self, *_):
        if not self._closing:
            self.statusBar().showMessage('Display process stopped. Press Ctrl/Cmd+R to reload; analysis continues locally.')

    def save_download(self, download):
        # Downloads only originate from our trusted local page (including its blobs).
        origin = download.url().toString()
        if origin.startswith('blob:'):
            origin=origin[5:]
        if not is_workspace_url(origin, self.runtime.url):
            download.cancel()
            return
        folder = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DownloadLocation)
        name = Path(download.suggestedFileName()).name or 'MorphAgent-results.zip'
        path, _ = QFileDialog.getSaveFileName(self, 'Save MorphAgent results', str(Path(folder)/name))
        if not path:
            download.cancel()
            return
        target=Path(path)
        download.setDownloadDirectory(str(target.parent))
        download.setDownloadFileName(target.name)
        download.isFinishedChanged.connect(lambda:self._download_finished(download))
        download.accept()
        self.statusBar().showMessage('Saving '+target.name+'…')

    def _download_finished(self, download):
        if download.isFinished():
            if download.state() == download.DownloadState.DownloadCompleted:
                self.statusBar().showMessage('Saved '+download.downloadFileName(),7000)
            elif download.state() == download.DownloadState.DownloadInterrupted:
                self.statusBar().showMessage('Save failed: '+download.interruptReasonString())
            else:
                self.statusBar().showMessage('Save cancelled.',5000)

    def closeEvent(self, event):
        if self.closed:
            event.accept()
            return
        event.ignore()
        if self._closing:
            return
        if self.runtime.has_active_runs():
            answer=QMessageBox.question(self, 'An analysis is running',
                'Stop the analysis and close MorphAgent?\nCompleted and partial results will be kept.',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel)
            if answer != QMessageBox.StandardButton.Yes:
                return
        self._closing=True
        self.view.setEnabled(False)
        self.statusBar().showMessage('Closing workspace and stopping owned analysis processes…')
        threading.Thread(target=self._stop_runtime,daemon=True,name='MorphAgent-shutdown').start()

    def _stop_runtime(self):
        try:
            self.runtime.stop()
            message=''
        except Exception as exc:
            message=str(exc)
        self.shutdown_finished.emit(message)

    @Slot(str)
    def _shutdown_done(self, error):
        if error:
            self._closing=False
            self.view.setEnabled(True)
            QMessageBox.warning(self,'Still stopping',error)
            return
        self.closed=True
        self.view.stop()
        self.close()
