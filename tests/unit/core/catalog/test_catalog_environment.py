"""The environment file a catalog run records: package names and versions only."""

import importlib.metadata as metadata
import re
import sys

import pytest

from lobster.core.catalog.environment import environment_text

pytestmark = pytest.mark.unit


class _Dist:
    def __init__(self, name, version):
        self.metadata = {"Name": name}
        self.version = version


class _Broken:
    """A half-installed distribution: reading its metadata fails."""

    version = "1"

    @property
    def metadata(self):
        raise TypeError("no METADATA file")


def _installed(monkeypatch, *dists):
    monkeypatch.setattr(metadata, "distributions", lambda: iter(dists))


def _packages(text):
    return [line for line in text.splitlines() if not line.startswith("#")]


def test_lists_name_equals_version_sorted_by_canonical_name(monkeypatch):
    _installed(
        monkeypatch,
        _Dist("Zope.Interface", "6.0"),
        _Dist("numpy", "2.5.4"),
        _Dist("typing_extensions", "4.12"),
    )
    assert _packages(environment_text("1.2.3")) == [
        "numpy==2.5.4",
        "typing-extensions==4.12",
        "zope-interface==6.0",
    ]


def test_the_first_listing_of_a_name_wins(monkeypatch):
    # The same distribution found twice on sys.path: the import system uses the first.
    _installed(monkeypatch, _Dist("numpy", "2.5.4"), _Dist("NumPy", "1.0"))
    assert _packages(environment_text("1.2.3")) == ["numpy==2.5.4"]


def test_a_distribution_without_a_name_is_skipped(monkeypatch):
    _installed(monkeypatch, _Dist(None, "1.0"), _Dist("", "2.0"), _Dist("a", "3"))
    assert _packages(environment_text("1.2.3")) == ["a==3"]


def test_the_header_names_python_platform_and_lobster_and_no_path(monkeypatch):
    _installed(monkeypatch, _Dist("a", "1"))
    text = environment_text("9.8.7")
    header = [line for line in text.splitlines() if line.startswith("#")]
    assert [line.split()[1] for line in header] == ["python", "platform", "lobster"]
    assert "# lobster 9.8.7" in header
    assert sys.prefix not in text
    assert all(re.fullmatch(r"[a-z0-9-]+==\S+", p) for p in _packages(text))


def test_the_same_environment_gives_the_same_text(monkeypatch):
    _installed(monkeypatch, _Dist("b", "2"), _Dist("a", "1"))
    first = environment_text("1.2.3")
    _installed(monkeypatch, _Dist("a", "1"), _Dist("b", "2"))
    assert environment_text("1.2.3") == first
    assert first.endswith("\n")


def test_one_broken_distribution_does_not_cost_the_others(monkeypatch):
    _installed(monkeypatch, _Dist("a", "1"), _Broken(), _Dist("b", "2"))
    assert _packages(environment_text("1.2.3")) == ["a==1", "b==2"]
