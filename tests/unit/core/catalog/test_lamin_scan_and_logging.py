"""The secrets scan's reason strings and the LaminDB log handler swap, without an engine.

These tests need no ``lamindb``: they import ``_scan_for_secret`` and
``_log_to_stderr`` from ``lobster.core.catalog.lamin``, which imports only the
standard library and the catalog's own modules. They run when the ``lamin`` extra is
not installed, unlike ``test_lamin_engine.py``, which skips as a whole.
"""

import io
import logging
from types import SimpleNamespace

import pytest

from lobster.core.catalog.lamin import (
    REPORT_SECRET_PATTERNS,
    LaminCatalog,
    _CurrentStderr,
    _log_to_stderr,
)

pytestmark = pytest.mark.unit

# One fake secret per pattern, made of pieces so that no secret scanner mistakes a test
# for a leak. The scan names the first pattern that matches, so no sample may match an
# earlier one.
SAMPLES = {
    "AWS access key": "x " + "AKIA" + "IOSFODNN7EXAMPLE" + " y",
    "Anthropic key": "key sk-" + "ant-api03-abcdefghijklmnop",
    "API key": "OPENAI sk-" + "proj-abcdefghijklmnopqrstuvwx",
    "GitHub token": "ghp" + "_abcdefghijklmnopqrstuvwxyz0123456789",
    "Slack token": "xox" + "b-1234567890-abcdefghij",
    "private key": "-----BEGIN RSA PRIVATE" + " KEY-----",
    "bearer token": "Authorization: Bearer abcdefghijklmnopqrstuvwxyz",
    "credential assignment": '"password":"hunter2hunter2hunter2"',
}


def test_every_pattern_has_a_sample():
    assert list(SAMPLES) == [label for label, _ in REPORT_SECRET_PATTERNS]


@pytest.mark.parametrize("what", ["notebook", "step log"])
@pytest.mark.parametrize("label", list(SAMPLES))
def test_the_reason_names_the_kind_of_secret_and_nothing_else(label, what):
    """The warning is said once per session, so the reason string is the only place
    a leak of the matched text would show up for every refusal after the first."""
    reason = LaminCatalog._scan_for_secret(None, SAMPLES[label].encode(), what)
    assert reason == f"the {what} looks like it contains a secret ({label})"


def test_text_without_a_secret_has_no_reason():
    plain = b"normalised 3 samples with pyDESeq2 at /data/proc.h5ad"
    assert LaminCatalog._scan_for_secret(None, plain, "step log") is None


def test_only_a_plain_stream_handler_is_pointed_at_stderr(tmp_path, capsys):
    """LaminDB's own handler writes to stdout; a handler someone added for a file
    keeps its file."""
    stdout = io.StringIO()
    path = tmp_path / "lamin.log"
    logger = logging.Logger("fake_lamindb_setup")
    plain = logging.StreamHandler(stdout)
    to_file = logging.FileHandler(path)
    logger.addHandler(plain)
    logger.addHandler(to_file)
    try:
        _log_to_stderr(SimpleNamespace(logger=logger))
        logger.warning("connected")
        to_file.flush()
        assert isinstance(plain.stream, _CurrentStderr)
        assert stdout.getvalue() == ""
        assert "connected" in capsys.readouterr().err
        assert to_file.stream.name == str(path)
        assert "connected" in path.read_text()
    finally:
        to_file.close()
