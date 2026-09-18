"""Tests for running features concurrently.

Feature-level parallelism only pays off if it cannot change results, so these
cover the invariants that make that true: ordering independent of timing, state
that is genuinely per-thread, and shared resources nobody can corrupt.
"""
from __future__ import annotations

import io
import random
import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.concurrency import (  # noqa: E402
    append_log,
    collected_output,
    run_ordered,
    thread_grouped_stdout,
)


class RunOrderedTests(unittest.TestCase):
    def test_results_follow_task_order_not_completion_order(self):
        """The whole design rests on this: later tasks may finish first."""

        def worker(task):
            # Reverse the natural timing so completion order is the opposite
            # of submission order.
            time.sleep((10 - task) * 0.01)
            return task * 10

        results = run_ordered(range(10), worker, max_workers=4)
        self.assertEqual(results, [task * 10 for task in range(10)])

    def test_random_durations_still_return_in_order(self):
        def worker(task):
            time.sleep(random.random() * 0.02)
            return task

        for _ in range(5):
            self.assertEqual(run_ordered(range(12), worker, max_workers=4), list(range(12)))

    def test_serial_and_parallel_agree(self):
        def worker(task):
            return task ** 2

        tasks = list(range(20))
        self.assertEqual(
            run_ordered(tasks, worker, max_workers=1),
            run_ordered(tasks, worker, max_workers=8),
        )

    def test_callback_sees_every_result_as_it_lands(self):
        seen = []
        lock = threading.Lock()

        def record(result):
            with lock:
                seen.append(result)

        run_ordered(range(8), lambda task: task, max_workers=3, on_result=record)
        self.assertEqual(sorted(seen), list(range(8)))

    def test_concurrency_never_exceeds_the_limit(self):
        active = 0
        peak = 0
        lock = threading.Lock()

        def worker(task):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.02)
            with lock:
                active -= 1
            return task

        run_ordered(range(12), worker, max_workers=2)
        self.assertLessEqual(peak, 2)
        self.assertGreater(peak, 1, "expected the pool to actually overlap work")

    def test_worker_limit_never_exceeds_task_count(self):
        peak = 0
        active = 0
        lock = threading.Lock()

        def worker(task):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.01)
            with lock:
                active -= 1
            return task

        run_ordered([1, 2], worker, max_workers=8)
        self.assertLessEqual(peak, 2)

    def test_empty_task_list(self):
        self.assertEqual(run_ordered([], lambda task: task, max_workers=4), [])

    def test_a_failing_task_does_not_lose_the_others(self):
        """The real worker catches its own errors; this pins the contract."""

        def worker(task):
            try:
                if task == 3:
                    raise ValueError("boom")
                return ("ok", task)
            except ValueError as exc:
                return ("failed", str(exc))

        results = run_ordered(range(6), worker, max_workers=3)
        self.assertEqual(results[3], ("failed", "boom"))
        self.assertEqual([r[0] for r in results], ["ok", "ok", "ok", "failed", "ok", "ok"])


class ThreadGroupedStdoutTests(unittest.TestCase):
    def test_each_thread_output_stays_in_one_block(self):
        captured = io.StringIO()
        original = sys.stdout
        sys.stdout = captured
        try:
            def worker(task):
                with collected_output() as output:
                    for line in range(4):
                        # Interleave aggressively: without routing these would
                        # be shuffled together in the shared stream.
                        print(f"task{task}-line{line}")
                        time.sleep(0.005)
                return output[0]

            with thread_grouped_stdout():
                blocks = run_ordered(range(3), worker, max_workers=3)
        finally:
            sys.stdout = original

        for task, block in enumerate(blocks):
            expected = "".join(f"task{task}-line{line}\n" for line in range(4))
            self.assertEqual(block, expected)
        self.assertEqual(captured.getvalue(), "", "routed output must not also reach stdout")

    def test_stdout_is_restored_even_when_a_worker_raises(self):
        original = sys.stdout
        with self.assertRaises(RuntimeError):
            with thread_grouped_stdout():
                raise RuntimeError("boom")
        self.assertIs(sys.stdout, original)

    def test_unrouted_threads_still_reach_the_real_stdout(self):
        captured = io.StringIO()
        original = sys.stdout
        sys.stdout = captured
        try:
            with thread_grouped_stdout():
                print("visible")
        finally:
            sys.stdout = original
        self.assertEqual(captured.getvalue(), "visible\n")

    def test_collected_output_works_without_an_active_section(self):
        """The same worker body has to run serially too."""
        captured = io.StringIO()
        original = sys.stdout
        sys.stdout = captured
        try:
            with collected_output() as output:
                print("serial")
        finally:
            sys.stdout = original
        self.assertEqual(output[0], "")
        self.assertEqual(captured.getvalue(), "serial\n")


class AppendLogTests(unittest.TestCase):
    def test_concurrent_appends_keep_lines_whole(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "execution_log.txt"
            payloads = [f"{'x' * 200}-{index}\n" for index in range(60)]
            run_ordered(payloads, lambda text: append_log(log, text), max_workers=8)

            lines = log.read_text(encoding="utf-8").splitlines()
            self.assertEqual(sorted(lines), sorted(p.rstrip("\n") for p in payloads))

    def test_unwritable_path_is_survivable(self):
        append_log(Path("/nonexistent-directory-xyz/log.txt"), "text")
        append_log(None, "text")


class VlmClientThreadSafetyTests(unittest.TestCase):
    def _bare_client(self):
        from tools.vlm_client import OnlineVLMClient

        client = OnlineVLMClient.__new__(OnlineVLMClient)
        client._temp_dir_store = threading.local()
        client._load_lock = threading.Lock()
        client._temp_dirs = []
        return client

    @staticmethod
    def _in_fresh_thread(function):
        """Run on a thread of its own: a pool may reuse the caller's thread."""
        captured = {}

        def target():
            try:
                captured["value"] = function()
            except BaseException as exc:  # surfaced by the assertion below
                captured["error"] = exc

        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout=30)
        self_error = captured.get("error")
        if self_error is not None:
            raise self_error
        return captured.get("value")

    def test_temp_dirs_are_per_thread(self):
        client = self._bare_client()
        client._temp_dirs.append("/tmp/main")

        def worker():
            before = list(client._temp_dirs)
            client._temp_dirs.append("/tmp/worker")
            return before, list(client._temp_dirs)

        before, after = self._in_fresh_thread(worker)
        self.assertEqual(before, [], "a new thread must not see another thread's directories")
        self.assertEqual(after, ["/tmp/worker"])
        self.assertEqual(client._temp_dirs, ["/tmp/main"], "the main thread's list is untouched")

    def test_cleanup_only_removes_the_calling_threads_directories(self):
        """The bug this guards: one thread deleting images another is uploading."""
        import tempfile

        client = self._bare_client()
        with tempfile.TemporaryDirectory() as root:
            mine = Path(root) / "mine"
            theirs = Path(root) / "theirs"
            mine.mkdir()
            theirs.mkdir()
            client._temp_dirs.append(str(mine))

            def worker():
                client._temp_dirs.append(str(theirs))
                client.cleanup_temp_files()

            self._in_fresh_thread(worker)
            self.assertFalse(theirs.exists(), "the worker should clean up after itself")
            self.assertTrue(mine.exists(), "another thread's directory must survive")

    def test_scratch_directories_are_removed_even_when_scoring_fails(self):
        """Pool threads are reused, so a skipped cleanup leaks for the whole run."""
        import tempfile

        from tools.vlm_client import _cleans_up_scratch

        client = self._bare_client()
        with tempfile.TemporaryDirectory() as root:
            scratch = Path(root) / "scratch"
            scratch.mkdir()
            (scratch / "resized.png").write_bytes(b"data")

            def failing_scoring(self):
                self._temp_dirs.append(str(scratch))
                raise TimeoutError("request hung")

            with self.assertRaises(TimeoutError):
                _cleans_up_scratch(failing_scoring)(client)

            self.assertFalse(scratch.exists(), "a failed call must still clean up")
            self.assertEqual(client._temp_dirs, [])

    def test_scoring_entry_points_all_clean_up(self):
        from tools.vlm_client import OnlineVLMClient, VLMClient

        for cls in (VLMClient, OnlineVLMClient):
            for name in ("score_feature", "score_features_batch"):
                method = cls.__dict__.get(name)
                self.assertIsNotNone(method, f"{cls.__name__}.{name} missing")
                self.assertTrue(
                    hasattr(method, "__wrapped__"),
                    f"{cls.__name__}.{name} must guarantee scratch cleanup",
                )

    def test_rate_limit_errors_back_off_harder_than_ordinary_ones(self):
        from tools.vlm_client import _is_rate_limit_error, _retry_delay_seconds

        class Response:
            status_code = 429

        class RateLimited(Exception):
            response = Response()

        self.assertTrue(_is_rate_limit_error(RateLimited("429 Too Many Requests")))
        self.assertTrue(_is_rate_limit_error(Exception("rate limit exceeded")))
        self.assertFalse(_is_rate_limit_error(ValueError("bad image")))
        self.assertFalse(_is_rate_limit_error(None))

        for attempt in (1, 2, 3):
            throttled = _retry_delay_seconds(RateLimited("429"), attempt)
            ordinary = _retry_delay_seconds(ValueError("boom"), attempt)
            self.assertGreater(throttled, ordinary)
            self.assertLessEqual(throttled, 75.0)

    def test_backoff_is_jittered_so_threads_do_not_retry_in_lockstep(self):
        from tools.vlm_client import _retry_delay_seconds

        delays = {round(_retry_delay_seconds(ValueError("boom"), 1), 4) for _ in range(20)}
        self.assertGreater(len(delays), 1)

    def test_sigalrm_is_not_armed_off_the_main_thread(self):
        """Arming it from a worker raises ValueError, which would kill the task."""
        from tools.vlm_client import _can_arm_sigalrm

        self.assertFalse(self._in_fresh_thread(_can_arm_sigalrm))
        self.assertTrue(_can_arm_sigalrm())

    def test_singleton_survives_a_concurrent_stampede(self):
        import tools.vlm_client as vlm_client
        from config import settings

        previous_provider = getattr(settings, "vlm_api_provider", "qwen")
        previous_client = vlm_client._global_online_vlm_client
        settings.vlm_api_provider = "online"
        vlm_client._global_online_vlm_client = None
        try:
            clients = run_ordered(range(16), lambda _: vlm_client.get_vlm_client(), max_workers=8)
            self.assertEqual(len({id(client) for client in clients}), 1)
        finally:
            vlm_client._global_online_vlm_client = previous_client
            settings.vlm_api_provider = previous_provider


class DataPathSelectorTests(unittest.TestCase):
    def test_singleton_is_built_once_under_concurrency(self):
        import tools.data_path_selector as module

        previous = module._global_selector
        module._global_selector = None
        built = []
        original_init = module.DataPathSelector.__init__

        def counting_init(self, verbose=False):
            built.append(1)
            self.llm = None
            self.verbose = verbose
            self._first_call_info = {}

        module.DataPathSelector.__init__ = counting_init
        try:
            selectors = run_ordered(
                range(16), lambda _: module.get_data_path_selector(verbose=False), max_workers=8
            )
            self.assertEqual(len({id(selector) for selector in selectors}), 1)
            self.assertEqual(sum(built), 1)
        finally:
            module.DataPathSelector.__init__ = original_init
            module._global_selector = previous


class SandboxInstallMutexTests(unittest.TestCase):
    def test_generated_wrapper_serialises_installs(self):
        from tools.code_executor import _create_wrapper_script

        for has_segmentation in (False, True):
            source = _create_wrapper_script(
                Path("/tmp/feature/extract.py"),
                Path("/tmp/data"),
                has_segmentation=has_segmentation,
                conda_env="morphagent_sandbox",
            )
            compile(source, "<wrapper>", "exec")
            self.assertIn("_acquire_install_lock", source)
            self.assertIn("_release_install_lock()", source)

    def test_wrapper_and_fix_scripts_share_one_lock_path(self):
        """Two mutexes at different paths would not exclude each other at all."""
        from tools.code_executor import _create_wrapper_script
        from tools.concurrency import SANDBOX_INSTALL_LOCK_PATH

        source = _create_wrapper_script(
            Path("/tmp/feature/extract.py"), Path("/tmp/data"), conda_env="sandbox"
        )
        self.assertIn(repr(SANDBOX_INSTALL_LOCK_PATH), source)

    def test_mutex_excludes_a_second_holder_and_releases(self):
        import os

        from tools.concurrency import SANDBOX_INSTALL_LOCK_PATH, sandbox_install_mutex

        if os.path.exists(SANDBOX_INSTALL_LOCK_PATH):
            os.rmdir(SANDBOX_INSTALL_LOCK_PATH)

        with sandbox_install_mutex() as held:
            self.assertTrue(held)
            self.assertTrue(os.path.exists(SANDBOX_INSTALL_LOCK_PATH))
        self.assertFalse(os.path.exists(SANDBOX_INSTALL_LOCK_PATH), "must release on exit")

    def test_mutex_releases_even_when_the_install_fails(self):
        import os

        from tools.concurrency import SANDBOX_INSTALL_LOCK_PATH, sandbox_install_mutex

        with self.assertRaises(RuntimeError):
            with sandbox_install_mutex():
                raise RuntimeError("pip exploded")
        self.assertFalse(os.path.exists(SANDBOX_INSTALL_LOCK_PATH))

    def test_a_stale_lock_cannot_wedge_the_run(self):
        """The owner can die; waiting forever on its lock is not an option."""
        import os

        import tools.concurrency as concurrency

        if os.path.exists(concurrency.SANDBOX_INSTALL_LOCK_PATH):
            os.rmdir(concurrency.SANDBOX_INSTALL_LOCK_PATH)
        os.mkdir(concurrency.SANDBOX_INSTALL_LOCK_PATH)

        previous_timeout = concurrency._INSTALL_LOCK_TIMEOUT
        previous_stale = concurrency._INSTALL_LOCK_STALE_AFTER
        concurrency._INSTALL_LOCK_TIMEOUT = 5.0
        concurrency._INSTALL_LOCK_STALE_AFTER = 0.0  # treat the lock as abandoned
        try:
            started = time.time()
            with concurrency.sandbox_install_mutex() as held:
                self.assertTrue(held, "a stale lock should be reclaimed")
            self.assertLess(time.time() - started, 5.0)
        finally:
            concurrency._INSTALL_LOCK_TIMEOUT = previous_timeout
            concurrency._INSTALL_LOCK_STALE_AFTER = previous_stale
            if os.path.exists(concurrency.SANDBOX_INSTALL_LOCK_PATH):
                os.rmdir(concurrency.SANDBOX_INSTALL_LOCK_PATH)

    def test_mutex_gives_up_rather_than_hanging_forever(self):
        import os

        import tools.concurrency as concurrency

        if os.path.exists(concurrency.SANDBOX_INSTALL_LOCK_PATH):
            os.rmdir(concurrency.SANDBOX_INSTALL_LOCK_PATH)
        os.mkdir(concurrency.SANDBOX_INSTALL_LOCK_PATH)

        previous_timeout = concurrency._INSTALL_LOCK_TIMEOUT
        concurrency._INSTALL_LOCK_TIMEOUT = 1.0
        try:
            started = time.time()
            with concurrency.sandbox_install_mutex() as held:
                self.assertFalse(held, "should report that it proceeded unserialised")
            self.assertLess(time.time() - started, 4.0, "must not block indefinitely")
        finally:
            concurrency._INSTALL_LOCK_TIMEOUT = previous_timeout
            os.rmdir(concurrency.SANDBOX_INSTALL_LOCK_PATH)


class ReproducibilityCacheTests(unittest.TestCase):
    def test_cache_writes_are_atomic_and_leave_no_temp_files(self):
        import tempfile

        from utils_modules.reproducibility import vlm_cache_get, vlm_cache_set

        with tempfile.TemporaryDirectory() as cache_dir:
            def worker(index):
                vlm_cache_set(
                    cache_dir, f"key{index}", score=float(index), response="r", batch_scores={"f": 1.0}
                )
                return vlm_cache_get(cache_dir, f"key{index}")

            results = run_ordered(range(24), worker, max_workers=8)
            for index, result in enumerate(results):
                self.assertIsNotNone(result)
                self.assertEqual(result[0], float(index))
            leftovers = list(Path(cache_dir).rglob("*.tmp"))
            self.assertEqual(leftovers, [])


class ReproduceModeTests(unittest.TestCase):
    """--reproduce yields to an explicit concurrency request, but not a default one."""

    def _flag_given(self, argv, flag):
        import main

        original = sys.argv
        sys.argv = ["main.py"] + argv
        try:
            return main._flag_was_given(flag)
        finally:
            sys.argv = original

    def test_detects_an_explicitly_passed_flag(self):
        self.assertTrue(self._flag_given(["--code-gen-workers", "2"], "--code-gen-workers"))
        self.assertTrue(self._flag_given(["--code-gen-workers=2"], "--code-gen-workers"))

    def test_does_not_confuse_a_similar_flag(self):
        self.assertFalse(self._flag_given(["--code-gen-workers-extra", "2"], "--code-gen-workers"))
        self.assertFalse(self._flag_given(["--vlm-online-concurrency", "8"], "--code-gen-workers"))
        self.assertFalse(self._flag_given([], "--code-gen-workers"))

    def test_ui_command_is_explicit_so_reproduce_keeps_its_concurrency(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from morphagent_ui.models import RunConfig

        config = RunConfig()
        config.data_root = str(Path(__file__).resolve().parent)
        config.question = "q"
        command = config.build_command()
        self.assertIn("--reproduce", command)
        for flag in ("--vlm-online-concurrency", "--code-gen-workers"):
            self.assertTrue(self._flag_given(command, flag))


class RunConfigConcurrencyTests(unittest.TestCase):
    def _config(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from morphagent_ui.models import RunConfig

        return RunConfig()

    def test_defaults_request_the_intended_api_concurrency(self):
        config = self._config()
        self.assertEqual(config.vlm_online_concurrency, 8)
        self.assertEqual(config.code_gen_workers, 2)

    def test_command_passes_concurrency_explicitly(self):
        config = self._config()
        config.data_root = str(Path(__file__).resolve().parent)
        config.question = "q"
        command = config.build_command()
        self.assertIn("--code-gen-workers", command)
        self.assertEqual(command[command.index("--code-gen-workers") + 1], "2")
        self.assertEqual(command[command.index("--vlm-online-concurrency") + 1], "8")


if __name__ == "__main__":
    unittest.main()
