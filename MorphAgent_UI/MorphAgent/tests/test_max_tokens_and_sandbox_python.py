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

    def test_temperature_rejection_retries_without_parameter(self) -> None:
        from config import (
            RetryableChatLLM,
            _TEMPERATURE_UNSUPPORTED_ENDPOINTS,
            _is_temperature_unsupported_error,
            _llm_endpoint_key,
        )

        error = Exception(
            "Error code: 400 - {'message': '`temperature` is deprecated for this model.'}"
        )
        self.assertTrue(_is_temperature_unsupported_error(error))
        self.assertFalse(_is_temperature_unsupported_error(Exception("context length exceeded")))

        class _Boom:
            def __init__(self) -> None:
                self.calls = 0

            def invoke(self, *args, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise error
                return "ok"

        class _StubLLM(RetryableChatLLM):
            def _rebuild_llm(self, base_url=None) -> None:  # noqa: ANN001
                return

        params = {
            "base_url": "https://api.anthropic.com/v1",
            "model": "claude-sonnet-5",
            "api_key": "test",
            "temperature": 0,
        }
        key = _llm_endpoint_key(params)
        _TEMPERATURE_UNSUPPORTED_ENDPOINTS.discard(key)
        boom = _Boom()
        wrapper = _StubLLM(
            llm=boom,
            provider_name="default",
            timeout_seconds=1,
            max_attempts=1,
            base_retry_delay_seconds=1,
            max_retry_delay_seconds=1,
            llm_params=params,
        )
        try:
            self.assertEqual(wrapper.invoke("hi"), "ok")
            self.assertNotIn("temperature", wrapper._llm_params)
            self.assertIn(key, _TEMPERATURE_UNSUPPORTED_ENDPOINTS)
            self.assertEqual(boom.calls, 2)
        finally:
            _TEMPERATURE_UNSUPPORTED_ENDPOINTS.discard(key)

    def test_cached_temperature_rejection_omits_parameter_on_new_client(self) -> None:
        import sys
        from types import ModuleType
        from unittest.mock import Mock, patch

        import config

        params = {
            "base_url": config.settings.llm_base_url,
            "model": config.settings.llm_model,
        }
        key = config._llm_endpoint_key(params)
        config._TEMPERATURE_UNSUPPORTED_ENDPOINTS.add(key)
        chat_openai = Mock()
        langchain_openai = ModuleType("langchain_openai")
        langchain_openai.ChatOpenAI = chat_openai
        try:
            with patch.dict(sys.modules, {"langchain_openai": langchain_openai}):
                config.make_chat_llm(temperature=0)
            self.assertNotIn("temperature", chat_openai.call_args.kwargs)
        finally:
            config._TEMPERATURE_UNSUPPORTED_ENDPOINTS.discard(key)


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
