"""
Contract test compliance: generated plugin's tests pass out of the box.

This test scaffolds a plugin, installs it, and verifies its own contract tests pass
without any manual edits.

Mark: slow (runs subprocess pytest)
"""

import subprocess
import sys

import pytest

from lobster.scaffold import scaffold_agent


@pytest.mark.slow
@pytest.mark.integration
class TestScaffoldContractCompliance:

    def test_generated_contract_tests_pass(self, tmp_path):
        """Generated plugin's contract tests must pass out of the box."""
        pkg_dir = scaffold_agent(
            name="contract_test_expert",
            display_name="Contract Test Expert",
            description="Validates contract tests pass",
            tier="free",
            output_dir=tmp_path,
        )

        try:
            # Install the generated package
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", str(pkg_dir)],
                capture_output=True,
                text=True,
                check=True,
            )

            # Run the generated plugin's own contract tests in subprocess
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    str(pkg_dir / "tests"),
                    "-v",
                    "-m",
                    "contract",
                ],
                capture_output=True,
                text=True,
                cwd=str(pkg_dir),
            )
            assert (
                result.returncode == 0
            ), f"Contract tests failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

        finally:
            # Cleanup: uninstall
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "uninstall",
                    "-y",
                    "lobster-contract-test",
                ],
                capture_output=True,
            )
