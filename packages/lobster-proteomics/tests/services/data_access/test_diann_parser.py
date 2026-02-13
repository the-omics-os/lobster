"""Smoke tests for DiaNNParser.

Basic import and instantiation tests.
"""

import pytest
from lobster.services.data_access.diann_parser import DiaNNParser


def test_parser_import():
    """Verify DiaNNParser is importable."""
    assert DiaNNParser is not None


def test_parser_instantiation():
    """Verify DiaNNParser can be instantiated."""
    parser = DiaNNParser()
    assert parser is not None
