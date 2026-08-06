import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from tools.code_executor import _create_wrapper_script


class WrapperSerializationTests(unittest.TestCase):
    def test_merged_numpy_scalars_are_json_serializable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            extract_path = root / "extract.py"
            extract_path.write_text(
                "def extract_all(img):\n"
                "    return {'float_feature': np.float32(1.25), "
                "'int_feature': np.int64(3)}\n",
                encoding="utf-8",
            )
            image_path = root / "image.tif"
            tifffile.imwrite(image_path, np.zeros((4, 4), dtype=np.uint8))
            runner_path = root / "runner.py"
            runner_path.write_text(
                _create_wrapper_script(extract_path, root, conda_env="morphagent"),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [sys.executable, str(runner_path), str(image_path)],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
            self.assertTrue(payload["success"])
            self.assertEqual(payload["value"], {"float_feature": 1.25, "int_feature": 3})


if __name__ == "__main__":
    unittest.main()
