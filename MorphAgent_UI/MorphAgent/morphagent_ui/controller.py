"""Background execution bridge between the desktop UI and MorphAgent CLI."""

from __future__ import annotations

import os
import re
import signal
import subprocess
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from qtpy.QtCore import QObject, QThread, Signal

from .models import DatasetSummary, RunConfig, find_completed_rounds


@dataclass(frozen=True)
class StageSpec:
    key: str
    title: str
    description: str
    progress: int


STAGES = (
    StageSpec("inspect", "Inspect", "Understand samples and biological context", 8),
    StageSpec("prepare", "Prepare", "Reuse masks or create them with Allen when missing", 24),
    StageSpec("plan", "Plan", "Propose biologically grounded feature cards", 40),
    StageSpec("quantify", "Quantify", "Run generated code and/or semantic scoring", 60),
    StageSpec("validate", "Validate", "Screen variation, redundancy, and evidence", 82),
    StageSpec("export", "Export", "Persist matrices, registry, and audit trail", 96),
)


class StageDetector:
    """Translate the existing CLI's stable top-level messages into UI stages."""

    _rules = (
        (re.compile(r"^Step 2\.[45]:"), 1),
        (re.compile(r"^Step 3(?:\.5)?:"), 2),
        (re.compile(r"^Step 4:"), 3),
        (re.compile(r"^Step (?:5(?:\.5)?|6):"), 4),
        (re.compile(r"^(?:✅ Round \d+ complete!|🎉 All \d+ rounds .*complete!)"), 5),
        (re.compile(r"^Final feature file:"), 5),
        (re.compile(r"^Step [12]:"), 0),
    )

    def __init__(self) -> None:
        self.index = 0

    def feed(self, raw_line: str) -> int | None:
        line = raw_line.rstrip()
        # Nested code-route messages intentionally begin with spaces and reuse
        # "Step 2/3/4"; only unindented top-level stages are eligible.
        if not line or line[0].isspace():
            return None
        for pattern, index in self._rules:
            if pattern.search(line):
                if index < self.index:
                    return None
                changed = index != self.index
                self.index = index
                return index if changed or index == 0 else None
        return None


def artifact_snapshot(results_dir: str | Path) -> dict[str, Any]:
    root = Path(results_dir)
    names = (
        "features.csv",
        "retained_features.csv",
        "feature_registry.json",
        "segmentation_summary.json",
    )
    files = {name: (root / name).is_file() for name in names}
    rounds = find_completed_rounds(root)
    return {
        "results_dir": str(root),
        "files": files,
        "completed_rounds": rounds,
        "artifact_count": sum(files.values()) + len(rounds),
    }


class PipelineWorker(QThread):
    log_line = Signal(str)
    stage_changed = Signal(int, str, str)
    progress_changed = Signal(int)
    artifacts_changed = Signal(dict)
    run_succeeded = Signal(int, str)
    run_failed = Signal(str, int, str)
    run_cancelled = Signal(str)

    def __init__(self, config: RunConfig, dataset: DatasetSummary | None = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.config = config
        self.dataset = dataset
        self._cancel_requested = threading.Event()
        self._process: subprocess.Popen[str] | None = None
        self._kill_timer: threading.Timer | None = None

    @property
    def results_dir(self) -> Path:
        return Path(self.config.results_dir).expanduser().resolve()

    def cancel(self) -> None:
        self._cancel_requested.set()
        process = self._process
        if process is None or process.poll() is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except (ProcessLookupError, OSError):
            return
        self._kill_timer = threading.Timer(5.0, self._force_kill)
        self._kill_timer.daemon = True
        self._kill_timer.start()

    def _force_kill(self) -> None:
        process = self._process
        if process is None or process.poll() is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except (ProcessLookupError, OSError):
            pass

    def run(self) -> None:
        detector = StageDetector()
        results_dir = self.results_dir
        results_dir.mkdir(parents=True, exist_ok=True)
        self.config.write_manifest(results_dir, self.dataset)
        log_path = results_dir / "ui_console.log"
        self.stage_changed.emit(0, STAGES[0].key, STAGES[0].title)
        self.progress_changed.emit(STAGES[0].progress)
        self.artifacts_changed.emit(artifact_snapshot(results_dir))

        try:
            with log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
                log_handle.write(f"[{datetime.now().isoformat(timespec='seconds')}] UI launch\n")
                env = os.environ.copy()
                env.update(self.config.pipeline_environment())
                self._process = subprocess.Popen(
                    self.config.build_command(),
                    cwd=self.config.repository_root,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    start_new_session=(os.name == "posix"),
                )
                assert self._process.stdout is not None
                last_snapshot: dict[str, Any] | None = None
                with self._process.stdout as stdout:
                    for raw_line in stdout:
                        line = raw_line.rstrip("\r\n")
                        log_handle.write(line + "\n")
                        self.log_line.emit(line)
                        stage_index = detector.feed(line)
                        if stage_index is not None:
                            stage = STAGES[stage_index]
                            self.stage_changed.emit(stage_index, stage.key, stage.title)
                            self.progress_changed.emit(stage.progress)
                        current = artifact_snapshot(results_dir)
                        if current != last_snapshot:
                            last_snapshot = current
                            self.artifacts_changed.emit(current)
                return_code = self._process.wait()
        except (OSError, subprocess.SubprocessError) as exc:
            self.run_failed.emit(str(exc), -1, str(log_path))
            return
        finally:
            if self._kill_timer is not None:
                self._kill_timer.cancel()

        self.artifacts_changed.emit(artifact_snapshot(results_dir))
        if self._cancel_requested.is_set():
            self.run_cancelled.emit(str(results_dir))
        elif return_code == 0:
            self.progress_changed.emit(100)
            self.run_succeeded.emit(return_code, str(results_dir))
        else:
            self.run_failed.emit(f"MorphAgent exited with status {return_code}.", return_code, str(log_path))


class RunController(QObject):
    log_line = Signal(str)
    stage_changed = Signal(int, str, str)
    progress_changed = Signal(int)
    artifacts_changed = Signal(dict)
    state_changed = Signal(str, str)
    run_finished = Signal(bool, str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.worker: PipelineWorker | None = None

    @property
    def running(self) -> bool:
        return bool(self.worker and self.worker.isRunning())

    def start(self, config: RunConfig, dataset: DatasetSummary | None = None) -> str:
        if self.running:
            raise RuntimeError("A MorphAgent run is already active.")
        if not config.results_dir.strip():
            root = Path(config.data_root).expanduser().resolve()
            dataset_root = root / "dataset" if (root / "dataset").is_dir() else root
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config.results_dir = str(dataset_root.parent / "results" / f"run_ui_{timestamp}")
        results_dir = str(Path(config.results_dir).expanduser().resolve())
        self.worker = PipelineWorker(config, dataset, self)
        self.worker.log_line.connect(self.log_line)
        self.worker.stage_changed.connect(self.stage_changed)
        self.worker.progress_changed.connect(self.progress_changed)
        self.worker.artifacts_changed.connect(self.artifacts_changed)
        self.worker.run_succeeded.connect(lambda _code, path: self._finish(True, "complete", path))
        self.worker.run_failed.connect(lambda message, _code, log: self._finish(False, message, log))
        self.worker.run_cancelled.connect(lambda path: self._finish(False, "cancelled", path))
        self.state_changed.emit("running", "MorphAgent is inspecting the dataset.")
        self.worker.start()
        return results_dir

    def cancel(self) -> None:
        if self.worker and self.worker.isRunning():
            self.state_changed.emit("cancelling", "Stopping after the current process boundary…")
            self.worker.cancel()

    def _finish(self, success: bool, message: str, path: str) -> None:
        if success:
            self.state_changed.emit("complete", "Run complete. Results and audit artifacts are ready.")
        elif message == "cancelled":
            self.state_changed.emit("cancelled", "Run cancelled. Existing artifacts were preserved.")
        else:
            self.state_changed.emit("failed", message)
        self.run_finished.emit(success, path)
