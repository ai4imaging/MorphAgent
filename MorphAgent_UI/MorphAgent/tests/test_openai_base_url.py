"""Tests for OpenAI-compatible base URL helpers."""
import unittest

from utils_modules.openai_base_url import (
    is_http_404_error,
    with_v1_suffix,
)


class TestOpenAIBaseUrl(unittest.TestCase):
    def test_with_v1_suffix(self):
        self.assertEqual(with_v1_suffix("https://api.example.com"), "https://api.example.com/v1")
        self.assertEqual(with_v1_suffix("https://api.example.com/"), "https://api.example.com/v1")
        self.assertIsNone(with_v1_suffix("https://api.example.com/v1"))
        self.assertIsNone(with_v1_suffix("https://api.example.com/v1/"))
        self.assertEqual(
            with_v1_suffix("https://gateway.example/openai"),
            "https://gateway.example/openai/v1",
        )

    def test_is_http_404_error(self):
        class Fake404(Exception):
            status_code = 404

        self.assertTrue(is_http_404_error(Fake404("missing")))
        self.assertTrue(is_http_404_error(Exception("Error code: 404 - Not Found")))
        self.assertFalse(is_http_404_error(Exception("timeout waiting for response")))


if __name__ == "__main__":
    unittest.main()
