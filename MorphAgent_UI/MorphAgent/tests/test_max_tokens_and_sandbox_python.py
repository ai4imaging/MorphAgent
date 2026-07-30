"""Tests for Windows sandbox Python discovery."""
from __future__ import annotations

import unittest


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
