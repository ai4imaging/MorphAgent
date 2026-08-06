"""Tests for quiet VLM temperature compatibility fallback."""
from __future__ import annotations

import contextlib
import io
import unittest
from types import SimpleNamespace


class VLMTemperatureCompatibilityTests(unittest.TestCase):
    def test_retries_without_temperature_without_printing_rejected_error(self) -> None:
        from tools.vlm_client import OnlineVLMClient

        class _Completions:
            def __init__(self) -> None:
                self.calls = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                if len(self.calls) == 1:
                    raise Exception("`temperature` is deprecated for this model.")
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
                )

        completions = _Completions()
        client = OnlineVLMClient(
            base_url="https://api.anthropic.com/v1",
            api_key="test",
            model="claude-sonnet-5",
        )
        client._client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = client._chat_with_retry([{"type": "text", "text": "hello"}])

        self.assertEqual(result, "ok")
        self.assertIn("temperature", completions.calls[0])
        self.assertNotIn("temperature", completions.calls[1])
        self.assertTrue(client._omit_temperature)
        visible_output = output.getvalue()
        self.assertIn("Adjusted sampling settings", visible_output)
        self.assertNotIn("deprecated", visible_output)
        self.assertNotIn("Traceback", visible_output)


if __name__ == "__main__":
    unittest.main()
