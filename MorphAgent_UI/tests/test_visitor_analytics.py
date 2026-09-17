import importlib
import unittest


class VisitorAnalyticsTests(unittest.TestCase):
    def _module(self):
        try:
            return importlib.import_module("morphagent_ui.visitor_analytics")
        except ModuleNotFoundError:
            self.fail("morphagent_ui.visitor_analytics has not been implemented")

    def test_register_visit_uses_fixed_endpoint_timeout_and_closes_response(self) -> None:
        analytics = self._module()
        captured = {}

        class FakeResponse:
            entered = False
            exited = False

            def __enter__(self):
                self.entered = True
                return self

            def read(self, size: int) -> bytes:
                captured["read_size"] = size
                return b"<"

            def __exit__(self, exc_type, exc, traceback) -> None:
                self.exited = True

        response = FakeResponse()

        def fake_opener(request, *, timeout: float):
            captured["request"] = request
            captured["timeout"] = timeout
            return response

        analytics.register_visit(opener=fake_opener, timeout=1.25)

        request = captured["request"]
        self.assertEqual(request.full_url, analytics.VISITOR_BADGE_URL)
        self.assertIn("hit=true", request.full_url)
        self.assertEqual(request.get_method(), "GET")
        self.assertIsNone(request.data)
        self.assertEqual(captured["timeout"], 1.25)
        self.assertEqual(captured["read_size"], 1)
        self.assertIn("MorphAgent-UI", request.get_header("User-agent"))
        self.assertTrue(response.entered)
        self.assertTrue(response.exited)

    def test_safe_registration_never_propagates_network_errors(self) -> None:
        analytics = self._module()

        def failing_opener(request, *, timeout: float):
            raise OSError("offline")

        analytics._register_visit_safely(opener=failing_opener, timeout=0.1)

    def test_start_registration_uses_a_started_daemon_thread(self) -> None:
        analytics = self._module()
        captured = {}

        class FakeThread:
            def __init__(self, *, target, name: str, daemon: bool) -> None:
                captured.update(target=target, name=name, daemon=daemon)
                self.started = False

            def start(self) -> None:
                self.started = True

        thread = analytics.start_visit_registration(thread_factory=FakeThread)

        self.assertTrue(thread.started)
        self.assertTrue(captured["daemon"])
        self.assertEqual(captured["name"], "morphagent-visitor-analytics")
        self.assertTrue(callable(captured["target"]))


if __name__ == "__main__":
    unittest.main()
