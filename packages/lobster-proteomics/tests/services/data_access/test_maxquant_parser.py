"""Smoke tests for MaxQuantParser.

Basic import and instantiation tests.
"""

from lobster.services.data_access.maxquant_parser import MaxQuantParser


def test_parser_import():
    """Verify MaxQuantParser is importable."""
    assert MaxQuantParser is not None


def test_parser_instantiation():
    """Verify MaxQuantParser can be instantiated."""
    parser = MaxQuantParser()
    assert parser is not None
