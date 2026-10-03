"""Fail-control tests for CI evidence validation (no credentials required)."""

import importlib.util
import subprocess  # nosec B404 # Invoke the local validator using fixed Python argv in a fixture.
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/check_test_results.py"
SPEC = importlib.util.spec_from_file_location("check_test_results", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TestCIResults(unittest.TestCase):
    def check_xml(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.xml"
            path.write_text(text)
            return MODULE.check_result(path)

    def test_executed_pass(self):
        self.assertEqual(
            self.check_xml('<testsuite><testcase name="ok"/></testsuite>'), 1
        )

    def test_partial_skip_counts_only_executed(self):
        self.assertEqual(
            self.check_xml(
                "<testsuite><testcase/><testcase><skipped/></testcase></testsuite>"
            ),
            1,
        )

    def test_required_partial_skip_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.xml"
            path.write_text(
                "<testsuite><testcase/><testcase><skipped/></testcase></testsuite>"
            )
            with self.assertRaises(ValueError):
                MODULE.check_result(path, require_no_skips=True)

    def test_empty_placeholder_rejected(self):
        with self.assertRaises(ValueError):
            self.check_xml(
                '<testsuites><testsuite tests="0" failures="0"/></testsuites>'
            )

    def test_all_skipped_rejected(self):
        with self.assertRaises(ValueError):
            self.check_xml("<testsuite><testcase><skipped/></testcase></testsuite>")

    def test_failures_and_errors_rejected(self):
        for tag in ("failure", "error"):
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                self.check_xml(f"<testsuite><testcase><{tag}/></testcase></testsuite>")

    def test_suite_collection_error_rejected(self):
        with self.assertRaises(ValueError):
            self.check_xml('<testsuite errors="1"><testcase/></testsuite>')

    def test_cli_bad_inputs_exit_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.xml"
            for text in (
                None,
                "not XML",
                "<testsuite><testcase><skipped/></testcase></testsuite>",
            ):
                if text is not None:
                    path.write_text(text)
                result = subprocess.run(  # nosec B603 # Current Python executable and fixture paths, no shell interpolation.
                    [sys.executable, str(SCRIPT), str(path)], capture_output=True
                )
                self.assertEqual(result.returncode, 1)

    def test_doctype_rejected_even_without_entities(self):
        with self.assertRaisesRegex(ValueError, "DOCTYPE"):
            self.check_xml("<!DOCTYPE testsuite><testsuite><testcase/></testsuite>")

    def test_internal_entity_dtd_rejected(self):
        with self.assertRaisesRegex(ValueError, "DOCTYPE"):
            self.check_xml(
                '<!DOCTYPE testsuite [<!ENTITY value "expanded">]>'
                '<testsuite><testcase name="&value;"/></testsuite>'
            )

    def test_external_entity_dtd_rejected(self):
        with self.assertRaisesRegex(ValueError, "DOCTYPE"):
            self.check_xml(
                '<!DOCTYPE testsuite [<!ENTITY value SYSTEM "file:///nonexistent-secret">]>'
                '<testsuite><testcase name="&value;"/></testsuite>'
            )

    def test_nested_entity_dtd_rejected(self):
        with self.assertRaisesRegex(ValueError, "DOCTYPE"):
            self.check_xml(
                '<!DOCTYPE testsuite [<!ENTITY a "x"><!ENTITY b "&a;&a;">'
                '<!ENTITY c "&b;&b;">]><testsuite><testcase name="&c;"/></testsuite>'
            )

    def test_utf16_entity_dtd_rejected(self):
        source = (
            '<?xml version="1.0" encoding="UTF-16"?>'
            '<!DOCTYPE testsuite [<!ENTITY value "expanded">]>'
            '<testsuite><testcase name="&value;"/></testsuite>'
        )
        for encoding in ("utf-16", "utf-16-le", "utf-16-be"):
            with (
                self.subTest(encoding=encoding),
                tempfile.TemporaryDirectory() as directory,
            ):
                path = Path(directory) / "result.xml"
                path.write_bytes(source.encode(encoding))
                with self.assertRaisesRegex(ValueError, "DOCTYPE"):
                    MODULE.check_result(path)

    def test_utf16_without_dtd_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.xml"
            path.write_bytes(
                '<testsuite><testcase name="valid"/></testsuite>'.encode("utf-16")
            )
            self.assertEqual(MODULE.check_result(path), 1)

    def test_doctype_mention_inside_comment_is_not_a_declaration(self):
        self.assertEqual(
            self.check_xml(
                "<!-- DOCTYPE is just comment text --><testsuite><testcase/></testsuite>"
            ),
            1,
        )

    def test_hostile_xml_cli_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.xml"
            path.write_bytes(
                '<!DOCTYPE testsuite [<!ENTITY value "expanded">]>'
                '<testsuite><testcase name="&value;"/></testsuite>'.encode("utf-16")
            )
            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(path)],
                capture_output=True,  # nosec B603 # Fixed local validator/fixture argv, no shell interpolation.
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn(b"DOCTYPE", result.stderr)


if __name__ == "__main__":
    unittest.main()
