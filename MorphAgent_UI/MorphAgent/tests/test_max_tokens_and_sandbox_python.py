"""Tests for Windows sandbox Python discovery and max_tokens clamp."""
from __future__ import annotations

import unittest


class MaxTokensClampTests(unittest.TestCase):
    def test_detects_gpugeek_style_error(self) -> None:
        from config import _is_context_length_error, _is_max_tokens_limit_error

        msg = (
            "Error code: 400 - {'code': 400, 'message': "
            "'max_tokens is too large: 65535. This model supports at most "
            "16384 completion tokens, whereas you provided 65535.'}"
        )
        exc = Exception(msg)
        self.assertTrue(_is_max_tokens_limit_error(exc))
        self.assertFalse(_is_context_length_error(exc))

    def test_clamp_sets_16383(self) -> None:
        from config import RetryableChatLLM

        class _Boom:
            def __init__(self) -> None:
                self.calls = 0

            def invoke(self, *args, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise Exception(
                        "max_tokens is too large: 65535. This model supports at most "
                        "16384 completion tokens, whereas you provided 65535."
                    )
                return "ok"

        class _StubLLM(RetryableChatLLM):
            def _rebuild_llm(self, base_url=None) -> None:  # noqa: ANN001
                # Keep the same stub; only params change.
                return

        boom = _Boom()
        wrapper = _StubLLM(
            llm=boom,
            provider_name="test",
            timeout_seconds=1,
            max_attempts=2,
            base_retry_delay_seconds=1,
            max_retry_delay_seconds=1,
            llm_params={"max_tokens": 65535, "model": "x", "api_key": "y"},
        )
        self.assertEqual(wrapper.invoke("hi"), "ok")
        self.assertEqual(wrapper._llm_params["max_tokens"], 16383)
        self.assertEqual(boom.calls, 2)


class FindCondaPythonWindowsLayoutTests(unittest.TestCase):
    def test_prefers_env_root_python_exe_on_win32(self) -> None:
        import sys
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from tools import code_executor as ce

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "Miniconda3"
            env = base / "envs" / "morphagent_sandbox"
            env.mkdir(parents=True)
            py = env / "python.exe"
            py.write_text("", encoding="utf-8")

            with patch.object(sys, "platform", "win32"), patch.dict(
                "os.environ",
                {"CONDA_BASE": str(base), "CONDA_PREFIX": str(base / "envs" / "morphagent")},
                clear=False,
            ):
                found = ce._find_conda_python("morphagent_sandbox")
            self.assertIsNotNone(found)
            self.assertEqual(found.resolve(), py.resolve())


if __name__ == "__main__":
    unittest.main()
