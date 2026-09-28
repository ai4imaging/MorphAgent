"""Keep the results folder current while a run is still in flight.

A run can be stopped from the workspace or killed outright, and whatever finished
by then is worth keeping - vision scores in particular cost real API calls. Every
finished sample is folded into the feature table and written by replacing the file
in one step, so an interrupted write leaves the previous snapshot rather than a
truncated one.
"""
from __future__ import annotations

import csv
import os
import signal
import tempfile
import threading
from pathlib import Path


def write_csv_atomically(path, fieldnames, rows) -> None:
    """Replace `path` in a single step so a kill cannot truncate it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=str(path.parent), suffix='.tmp')
    try:
        with os.fdopen(descriptor, 'w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(fieldnames), extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


class FeatureTableCheckpoint:
    """The feature table as it stands, rewritten whenever a sample lands."""

    def __init__(self, path, sample_ids, columns=()):
        self.path = Path(path)
        self._samples = [str(sample) for sample in sample_ids]
        self._columns: list[str] = []
        self._values: dict[str, dict] = {sample: {} for sample in self._samples}
        # Re-entrant: a stop signal interrupts the very thread that may hold this.
        self._lock = threading.RLock()
        self.add_columns(columns)

    def adopt_existing(self) -> None:
        """Carry forward the columns that earlier rounds already wrote."""
        if not self.path.is_file() or self.path.is_symlink():
            return
        try:
            with self.path.open(encoding='utf-8-sig', newline='') as stream:
                reader = csv.DictReader(stream)
                names = [name for name in (reader.fieldnames or []) if name != 'sample_id']
                rows = list(reader)
        except (OSError, csv.Error):
            return
        with self._lock:
            self.add_columns(names)
            for row in rows:
                sample = str(row.get('sample_id', ''))
                if sample not in self._values:
                    continue
                for name in names:
                    value = row.get(name, '')
                    if value != '' and value is not None:
                        self._values[sample][name] = value

    def add_columns(self, names) -> None:
        with self._lock:
            for name in names:
                if name and name not in self._columns:
                    self._columns.append(name)

    def record(self, sample_id, values) -> None:
        """Store one finished sample, then refresh the file on disk."""
        with self._lock:
            sample = str(sample_id)
            if sample not in self._values:
                self._samples.append(sample)
                self._values[sample] = {}
            self.add_columns(values)
            self._values[sample].update(values)
            self.save()

    def save(self) -> None:
        with self._lock:
            if not self._columns:
                return
            rows = [
                {'sample_id': sample,
                 **{name: self._values[sample].get(name, '') for name in self._columns}}
                for sample in self._samples
            ]
            write_csv_atomically(self.path, ['sample_id', *self._columns], rows)


    def merge_into(self, target) -> None:
        """Fold this round's measurements into the cumulative table.

        Only for use once nothing else will write: a running pipeline renames a
        feature whose column already holds values, so the round must not appear
        in the cumulative table until the round itself is done with it.
        """
        target = Path(target)
        existing_names, existing_rows = [], {}
        if target.is_file() and not target.is_symlink():
            try:
                with target.open(encoding='utf-8-sig', newline='') as stream:
                    reader = csv.DictReader(stream)
                    existing_names = [n for n in (reader.fieldnames or []) if n != 'sample_id']
                    existing_rows = {str(r.get('sample_id', '')): r for r in reader}
            except (OSError, csv.Error):
                return
        with self._lock:
            added = [name for name in self._columns if name not in existing_names]
            if not added:
                return
            samples = list(existing_rows) or self._samples
            for sample in self._samples:
                if sample not in existing_rows:
                    existing_rows[sample] = {'sample_id': sample}
                    samples.append(sample)
            rows = []
            for sample in samples:
                row = dict(existing_rows.get(sample, {'sample_id': sample}))
                for name in added:
                    row[name] = self._values.get(sample, {}).get(name, '')
                rows.append(row)
            write_csv_atomically(target, ['sample_id', *existing_names, *added], rows)


def save_when_stopped(save, announce=print) -> None:
    """Persist results when a run is stopped instead of allowed to finish.

    The workspace stops a run with SIGTERM and follows with SIGKILL a few seconds
    later. The default action for SIGTERM ends the process without unwinding, so
    `finally` blocks never run and the snapshot has to be taken in the handler.
    """

    if threading.current_thread() is not threading.main_thread():
        return

    def handle(number, _frame):
        try:
            save()
            announce('[Checkpoint] Stopped early; results up to this point are saved.')
        except Exception as exc:  # noqa: BLE001 - a failed save must still exit
            announce(f'[Checkpoint] Could not save on stop: {type(exc).__name__}: {exc}')
        finally:
            os._exit(128 + number)

    for name in ('SIGTERM', 'SIGINT'):
        number = getattr(signal, name, None)
        if number is None:
            continue
        try:
            signal.signal(number, handle)
        except (ValueError, OSError):
            pass
