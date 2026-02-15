"""Smoke tests for OlinkParser.

Basic import and instantiation tests.
"""

from lobster.services.data_access.olink_parser import OlinkParser


def test_parser_import():
    """Verify OlinkParser is importable."""
    assert OlinkParser is not None


def test_parser_instantiation():
    """Verify OlinkParser can be instantiated."""
    parser = OlinkParser()
    assert parser is not None
