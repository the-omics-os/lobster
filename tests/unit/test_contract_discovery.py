"""Root contract discovery must not confuse template assertions with markers."""

import importlib.util
import subprocess  # nosec B404 # Only the local discovery helper runs with fixed Python argv.
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/discover_contract_tests.py"
SPEC = importlib.util.spec_from_file_location("discover_contract_tests", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TestContractDiscovery(unittest.TestCase):
    def test_string_mentions_do_not_mark_tests(self):
        self.assertFalse(
            MODULE.has_contract_marker(
                'def test_template():\n    assert "@pytest.mark.contract" in content\n'
            )
        )

    def test_decorated_test_discovered(self):
        self.assertTrue(
            MODULE.has_contract_marker(
                "@pytest.mark.contract\ndef test_contract():\n    pass\n"
            )
        )

    def test_module_marker_discovered(self):
        self.assertTrue(
            MODULE.has_contract_marker("pytestmark = [pytest.mark.contract]")
        )

    def test_nested_class_marker_discovered(self):
        self.assertTrue(
            MODULE.has_contract_marker(
                "@pytest.mark.contract\nclass TestContract:\n    pass\n"
            )
        )

    def test_empty_discovery_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(SCRIPT), directory],
                capture_output=True,  # nosec B603 # Current Python, repository script and synthetic fixture path; no shell.
            )
            self.assertEqual(result.returncode, 1)


if __name__ == "__main__":
    unittest.main()
