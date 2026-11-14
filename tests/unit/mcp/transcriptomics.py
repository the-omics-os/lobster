"""The checkout's transcriptomics package, for the tests that run its QC tools.

An environment that installs the packages from another checkout imports an older
copy of them. Skipping on that would hide whole test modules, so a checkout that
has the package fails loudly instead; only an install with no such package at
all skips.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[3] / "packages" / "lobster-transcriptomics"


def quality_service():
    """``lobster.services.quality.quality_service`` with the QC-flag helpers."""
    try:
        quality = importlib.import_module("lobster.services.quality.quality_service")
    except ImportError:
        quality = None
    if quality is not None and hasattr(quality, "qc_flags_kept"):
        return quality
    if PACKAGE.is_dir():
        found = getattr(quality, "__file__", None)
        pytest.fail(
            f"the imported transcriptomics package ({found}) lacks what these tests "
            f"need, but this checkout has {PACKAGE}. Put the checkout's packages on "
            f"the path, for example PYTHONPATH={PACKAGE.parents[1]}:{PACKAGE}",
            pytrace=False,
        )
    pytest.skip("transcriptomics package not installed", allow_module_level=True)
