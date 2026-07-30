"""Tests for completion max_tokens clamp helpers."""
from __future__ import annotations

import unittest


class MaxTokensClampTests(unittest.TestCase):
    def test_detects_gpugeek_style_error(self) -> None:
        from config import (
            _is_context_length_error,
            _is_max_tokens_limit_error,
            _parse_max_completion_tokens_ceiling,
        )

        msg = (
            "Error code: 400 - {'code': 400, 'message': "
            "'max_tokens is too large: 65535. This model supports at most "
            "16384 completion tokens, whereas you provided 65535.'}"
        )
        exc = Exception(msg)
        self.assertTrue(_is_max_tokens_limit_error(exc))
        self.assertFalse(_is_context_length_error(exc))
        self.assertEqual(_parse_max_completion_tokens_ceiling(exc), 16384)

    def test_defaults_are_gateway_safe(self) -> None:
        from config import MorphAgentConfig

        cfg = MorphAgentConfig()
        self.assertLessEqual(cfg.llm_max_tokens, 16384)
        self.assertLessEqual(int(cfg.merge_max_tokens or 0), 16384)


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
