import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from launch_ui import launch_standalone, load_repository_environment, main


class RepositoryEnvironmentTests(unittest.TestCase):
    def test_repository_env_is_the_ui_source_of_truth(self) -> None:
        names = (
            "LLM_BASE_URL",
            "LLM_API_KEY",
            "LLM_MODEL",
            "VLM_API_KEY",
        )
        previous = {name: os.environ.get(name) for name in names}
        try:
            for name in names:
                os.environ.pop(name, None)
            # Empty variables inherited from an earlier shell should not block
            # real values from the repository configuration.
            os.environ["LLM_API_KEY"] = ""
            os.environ["VLM_API_KEY"] = ""
            os.environ["LLM_MODEL"] = "explicit-model"
            with tempfile.TemporaryDirectory() as temp_dir:
                env_path = Path(temp_dir) / ".env"
                env_path.write_text(
                    'LLM_BASE_URL="https://gateway.example/v1"\n'
                    'LLM_API_KEY="file-key"\n'
                    'LLM_MODEL="file-model"\n'
                    'VLM_API_KEY="${LLM_API_KEY}"\n',
                    encoding="utf-8",
                )

                loaded = load_repository_environment(env_path)

            self.assertTrue(loaded)
            self.assertEqual(os.environ["LLM_BASE_URL"], "https://gateway.example/v1")
            self.assertEqual(os.environ["LLM_API_KEY"], "file-key")
            self.assertEqual(os.environ["VLM_API_KEY"], "file-key")
            self.assertEqual(os.environ["LLM_MODEL"], "file-model")
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


class WindowLaunchTests(unittest.TestCase):
    def test_standalone_launch_opens_the_window_maximized(self) -> None:
        class FakeApplication:
            def exec_(self) -> int:
                return 17

        class FakeWindow:
            def __init__(self) -> None:
                self.normal_shown = False
                self.maximized_shown = False

            def show(self) -> None:
                self.normal_shown = True

            def showMaximized(self) -> None:
                self.maximized_shown = True

        class FakeWidget:
            def __init__(self) -> None:
                self.focused = False

            def setFocus(self) -> None:
                self.focused = True

        window = FakeWindow()
        widget = FakeWidget()
        with patch("launch_ui.create_standalone_window", return_value=(FakeApplication(), window, widget)):
            result = launch_standalone()

        self.assertEqual(result, 17)
        self.assertTrue(window.maximized_shown)
        self.assertFalse(window.normal_shown)
        self.assertTrue(widget.focused)


class LauncherAnalyticsTests(unittest.TestCase):
    def test_main_registers_one_visit_before_launching_ui(self) -> None:
        with (
            patch("launch_ui.load_repository_environment"),
            patch("launch_ui.start_visit_registration") as register,
            patch("launch_ui.launch_standalone", return_value=23) as launch,
        ):
            result = main([])

        self.assertEqual(result, 23)
        register.assert_called_once_with()
        launch.assert_called_once_with(None)


if __name__ == "__main__":
    unittest.main()
