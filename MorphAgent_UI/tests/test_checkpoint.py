"""Results must survive a run being stopped, so these tests really stop one."""
from __future__ import annotations

import csv
import json
import os
import signal
import subprocess
import sys
import time
import unittest
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY))

from tools.checkpoint import FeatureTableCheckpoint, write_csv_atomically  # noqa: E402


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


class AtomicWriteTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parent / '_checkpoint_tmp'
        self.root.mkdir(exist_ok=True)
        self.addCleanup(lambda: __import__('shutil').rmtree(self.root, ignore_errors=True))

    def test_a_failed_write_leaves_the_previous_table_readable(self):
        target = self.root / 'features.csv'
        write_csv_atomically(target, ['sample_id', 'a'], [{'sample_id': 's1', 'a': 1}])

        class Exploding(dict):
            def get(self, *args, **kwargs):
                raise RuntimeError('row blew up')

        with self.assertRaises(RuntimeError):
            write_csv_atomically(target, ['sample_id', 'a'], [Exploding()])
        self.assertEqual(read_csv(target), [{'sample_id': 's1', 'a': '1'}])
        self.assertEqual(list(self.root.glob('*.tmp')), [])


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parent / '_checkpoint_tmp2'
        self.root.mkdir(exist_ok=True)
        self.addCleanup(lambda: __import__('shutil').rmtree(self.root, ignore_errors=True))

    def test_every_recorded_sample_is_on_disk_immediately(self):
        checkpoint = FeatureTableCheckpoint(self.root / 'partial.csv', ['s1', 's2'])
        checkpoint.record('s1', {'area': 4.0})
        rows = read_csv(self.root / 'partial.csv')
        self.assertEqual(rows[0], {'sample_id': 's1', 'area': '4.0'})
        # The sample still waiting keeps an empty cell rather than disappearing.
        self.assertEqual(rows[1], {'sample_id': 's2', 'area': ''})

    def test_adopting_an_existing_table_keeps_earlier_rounds(self):
        table = self.root / 'features.csv'
        write_csv_atomically(table, ['sample_id', 'old'], [{'sample_id': 's1', 'old': 7}])
        checkpoint = FeatureTableCheckpoint(table, ['s1'])
        checkpoint.adopt_existing()
        checkpoint.record('s1', {'new': 9})
        self.assertEqual(read_csv(table), [{'sample_id': 's1', 'old': '7', 'new': '9'}])

    def test_merging_a_round_adds_its_columns_without_touching_the_others(self):
        table = self.root / 'features.csv'
        write_csv_atomically(table, ['sample_id', 'round_one'],
                             [{'sample_id': 's1', 'round_one': 1},
                              {'sample_id': 's2', 'round_one': 2}])
        checkpoint = FeatureTableCheckpoint(self.root / 'partial.csv', ['s1', 's2'])
        checkpoint.record('s1', {'round_two': 10})
        checkpoint.merge_into(table)
        rows = read_csv(table)
        self.assertEqual(rows[0], {'sample_id': 's1', 'round_one': '1', 'round_two': '10'})
        # s2 was never measured this round, so its new cell is blank, not wrong.
        self.assertEqual(rows[1], {'sample_id': 's2', 'round_one': '2', 'round_two': ''})

    def test_merging_does_not_duplicate_a_column_that_already_exists(self):
        table = self.root / 'features.csv'
        write_csv_atomically(table, ['sample_id', 'area'], [{'sample_id': 's1', 'area': 5}])
        checkpoint = FeatureTableCheckpoint(self.root / 'partial.csv', ['s1'])
        checkpoint.record('s1', {'area': 99})
        checkpoint.merge_into(table)
        self.assertEqual(read_csv(table), [{'sample_id': 's1', 'area': '5'}])


class MergedExtractionCallbackTests(unittest.TestCase):
    """Discovery hands each finished sample to the round's checkpoint."""

    def test_each_sample_is_reported_as_the_merged_code_finishes_it(self):
        import tools.code_executor as code_executor
        import tools.data_path_selector as data_path_selector

        samples = ['s1', 's2', 's3']
        values = {'s1': {'area': 1.0}, 's2': {'area': 2.0}, 's3': {'area': 3.0}}
        seen = []

        class Selector:
            def select_data_paths(self, sample_dir, *args, **kwargs):
                return {'image_paths': [str(Path(sample_dir) / 'image.tif')],
                        'segmentation_paths': []}

        original_selector = data_path_selector.get_data_path_selector
        original_execute = code_executor.CodeExecutor.execute_single_sample
        data_path_selector.get_data_path_selector = lambda *a, **k: Selector()
        code_executor.CodeExecutor.execute_single_sample = (
            lambda self, script, image, masks=None: (True, values[Path(image).parent.name], None))
        self.addCleanup(setattr, data_path_selector, 'get_data_path_selector', original_selector)
        self.addCleanup(setattr, code_executor.CodeExecutor, 'execute_single_sample', original_execute)

        root = Path(__file__).resolve().parent / '_checkpoint_tmp5'
        for sample in samples:
            (root / sample).mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: __import__('shutil').rmtree(root, ignore_errors=True))

        result = code_executor.execute_merged_code(
            'def extract_all(img, seg): return {}', ['area'], samples, root,
            lambda directory, _description: [str(Path(directory) / 'image.tif')],
            results_dir=root / 'results', num_workers=1,
            on_sample=lambda sample, measured: seen.append((sample, measured)))

        self.assertEqual([sample for sample, _ in seen], samples)
        self.assertEqual(seen[1], ('s2', {'area': 2.0}))
        self.assertEqual(result.values['s3'], {'area': 3.0})


class StopSignalTests(unittest.TestCase):
    """A handler that calls os._exit cannot be exercised in-process."""

    def setUp(self):
        self.root = Path(__file__).resolve().parent / '_checkpoint_tmp3'
        self.root.mkdir(exist_ok=True)
        self.addCleanup(lambda: __import__('shutil').rmtree(self.root, ignore_errors=True))

    def test_a_terminated_run_still_writes_what_it_measured(self):
        script = self.root / 'run.py'
        script.write_text(f'''
import sys, time
sys.path.insert(0, {str(REPOSITORY)!r})
from tools.checkpoint import FeatureTableCheckpoint, save_when_stopped

checkpoint = FeatureTableCheckpoint({str(self.root / "out.csv")!r}, ["s1", "s2", "s3"])
save_when_stopped(checkpoint.save)
checkpoint.record("s1", {{"area": 1.0}})
print("ready", flush=True)
time.sleep(60)
''', encoding='utf-8')
        process = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, text=True)
        self.addCleanup(process.kill)
        self.assertEqual(process.stdout.readline().strip(), 'ready')
        process.send_signal(signal.SIGTERM)
        self.assertEqual(process.wait(timeout=10), 128 + signal.SIGTERM)
        self.assertEqual(read_csv(self.root / 'out.csv')[0], {'sample_id': 's1', 'area': '1.0'})


class StoppedComputeRunTests(unittest.TestCase):
    """Stop a real Compute run part-way and read what it left behind."""

    def setUp(self):
        self.root = Path(__file__).resolve().parent / '_checkpoint_tmp4'
        self.root.mkdir(exist_ok=True)
        self.addCleanup(lambda: __import__('shutil').rmtree(self.root, ignore_errors=True))

    def build_source(self, sample_count):
        import numpy as np
        import tifffile
        source = self.root / 'source'
        (source / 'round_1' / 'features' / 'slow').mkdir(parents=True)
        (source / 'round_1' / 'features' / 'slow' / 'extract.py').write_text(
            'def extract(img, seg):\n'
            '    import time\n'
            '    time.sleep(0.4)\n'
            '    import numpy as np\n'
            '    return float(np.asarray(img).mean())\n', encoding='utf-8')
        (source / 'round_1' / 'feature_plan.json').write_text(
            json.dumps({'features': [{'name': 'slow', 'method': 'code', 'description': 'slow'}]}),
            encoding='utf-8')
        (source / 'features.csv').write_text('sample_id,slow\na,1\n', encoding='utf-8')
        for index in range(sample_count):
            sample = self.root / 'data' / 'dataset' / f's{index:02d}'
            sample.mkdir(parents=True)
            tifffile.imwrite(sample / 'image.tif', np.full((8, 8), index + 1, dtype=np.uint8))
        return source

    def test_stopping_mid_run_keeps_the_samples_already_measured(self):
        source = self.build_source(25)
        output = self.root / 'out'
        script = (
            'import sys;'
            f'sys.path.insert(0, {str(REPOSITORY)!r});'
            f'sys.path.insert(0, {str(REPOSITORY / "src")!r});'
            'from tools.selected_reuse import run_selected_reuse;'
            f'run_selected_reuse({str(source)!r}, {str(self.root / "data")!r}, {str(output)!r}, ["slow"])'
        )
        process = subprocess.Popen([sys.executable, '-u', '-c', script],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   start_new_session=os.name != 'nt')
        self.addCleanup(process.kill)
        measured = 0
        deadline = time.time() + 90
        while time.time() < deadline:
            line = process.stdout.readline()
            if not line:
                break
            if line.startswith('[Compute] Progress'):
                measured = int(line.split()[2].split('/')[0])
                if measured >= 3:
                    break
        self.assertGreaterEqual(measured, 3, 'run never got going')
        os.killpg(process.pid, signal.SIGTERM) if os.name != 'nt' else process.terminate()
        process.wait(timeout=20)

        rows = read_csv(output / 'features.csv')
        filled = [row for row in rows if row['slow'] not in ('', None)]
        self.assertGreaterEqual(len(filled), 3)
        self.assertLess(len(filled), 25, 'the run was supposed to be stopped early')
        summary = json.loads((output / 'reuse_manifest.json').read_text(encoding='utf-8'))
        self.assertTrue(summary['stopped_early'])
        self.assertFalse(summary['complete'])
        # The plan is on disk too, so the workspace can still describe the columns.
        self.assertTrue((output / 'round_1' / 'feature_plan.json').is_file())


if __name__ == '__main__':
    unittest.main()
