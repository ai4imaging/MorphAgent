from __future__ import annotations

import unittest

from tools.sandbox_install_policy import (
    blocked_install_guidance,
    validate_install_script,
    validate_package_install_request,
)


class SandboxInstallPolicyTests(unittest.TestCase):
    def test_blocks_version_pins_and_core_packages(self) -> None:
        ok, reason = validate_package_install_request("scikit-image==0.19.3")
        self.assertFalse(ok)
        self.assertIn("version", reason.lower())

        ok, reason = validate_package_install_request("numpy")
        self.assertFalse(ok)
        self.assertIn("core", reason.lower())

        ok, reason = validate_package_install_request("skimage")
        self.assertFalse(ok)

        ok, reason = validate_package_install_request("some_rare_plugin")
        self.assertTrue(ok)

    def test_blocks_dangerous_install_scripts(self) -> None:
        ok, reason, script = validate_install_script("pip install scikit-image==0.19.3")
        self.assertFalse(ok)
        self.assertEqual(script, "")
        self.assertTrue(reason)

        ok, reason, script = validate_install_script("pip uninstall -y numpy")
        self.assertFalse(ok)

        ok, reason, script = validate_install_script("conda install numpy=1.23.5 scikit-image=0.19.3 -y")
        self.assertFalse(ok)

        ok, reason, script = validate_install_script("pip install rare_extra_lib")
        self.assertTrue(ok)
        self.assertIn("rare_extra_lib", script)

    def test_blocked_guidance_mentions_graycomatrix(self) -> None:
        text = blocked_install_guidance("core science package")
        self.assertIn("graycomatrix", text)


if __name__ == "__main__":
    unittest.main()
