"""Tests for scripts/verify_env.py.

These tests pin the script's reporting contract without requiring the scientific stack:

  * a failing check must not be masked by other checks,
  * an absent optional dependency is SKIPPED, never FAILED,
  * exit status is 0 only when nothing FAILED, and --strict also fails on SKIPPED.

An environment check must not report success when a required check fails.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "verify_env.py"


def _load_module():
    """Import verify_env.py by path (scripts/ is not an importable package)."""
    spec = importlib.util.spec_from_file_location("verify_env", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def verify_env():
    return _load_module()


def test_script_exists():
    assert SCRIPT.is_file(), f"missing {SCRIPT}"


class TestRunCheck:
    """run_check must convert every outcome into a (status, detail) pair."""

    def test_passing_check_reports_detail(self, verify_env):
        status, detail = verify_env.run_check("ok", lambda: "all good", False)
        assert status == verify_env.PASSED
        assert detail == "all good"

    def test_absent_dependency_is_skipped_not_failed(self, verify_env):
        def check():
            verify_env._require("a_module_that_does_not_exist")
            return "unreachable"

        status, detail = verify_env.run_check("skip", check, False)
        assert status == verify_env.SKIPPED
        assert "a_module_that_does_not_exist" in detail

    def test_raising_check_is_failed_with_type_and_message(self, verify_env):
        def check():
            raise RuntimeError("boom")

        status, detail = verify_env.run_check("bad", check, False)
        assert status == verify_env.FAILED
        assert "RuntimeError" in detail and "boom" in detail

    def test_check_never_propagates_even_on_baseexception(self, verify_env):
        """A misbehaving check must not abort the whole run."""

        def check():
            raise KeyboardInterrupt("interrupted")

        status, _ = verify_env.run_check("harsh", check, False)
        assert status == verify_env.FAILED

    def test_require_passes_when_module_present(self, verify_env):
        verify_env._require("sys", "os")  # must not raise


class TestExitStatus:
    """Exit status is the contract CI reads."""

    @staticmethod
    def _run(verify_env, monkeypatch, checks, argv):
        monkeypatch.setattr(verify_env, "CHECKS", checks)
        monkeypatch.setattr(sys, "argv", ["verify_env.py", *argv])
        return verify_env.main()

    def test_all_passing_exits_zero(self, verify_env, monkeypatch):
        checks = [("a", lambda: "ok"), ("b", lambda: "ok")]
        assert self._run(verify_env, monkeypatch, checks, []) == 0

    def test_any_failure_exits_nonzero(self, verify_env, monkeypatch):
        def bad():
            raise RuntimeError("nope")

        checks = [("a", lambda: "ok"), ("b", bad)]
        assert self._run(verify_env, monkeypatch, checks, []) == 1

    def test_failure_is_not_masked_by_later_passes(self, verify_env, monkeypatch):
        """Ordering must not matter: a failure anywhere fails the run."""

        def bad():
            raise RuntimeError("nope")

        checks = [("bad", bad), ("good", lambda: "ok"), ("also", lambda: "ok")]
        assert self._run(verify_env, monkeypatch, checks, []) == 1

    def test_skipped_alone_still_exits_zero(self, verify_env, monkeypatch):
        """Core-only installs must be able to run this cleanly."""

        def skipper():
            verify_env._require("definitely_not_installed_xyz")
            return "unreachable"

        checks = [("a", lambda: "ok"), ("b", skipper)]
        assert self._run(verify_env, monkeypatch, checks, []) == 0

    def test_strict_makes_skipped_a_failure(self, verify_env, monkeypatch):
        def skipper():
            verify_env._require("definitely_not_installed_xyz")
            return "unreachable"

        checks = [("a", lambda: "ok"), ("b", skipper)]
        assert self._run(verify_env, monkeypatch, checks, ["--strict"]) == 1

    def test_strict_passes_when_nothing_skipped(self, verify_env, monkeypatch):
        checks = [("a", lambda: "ok")]
        assert self._run(verify_env, monkeypatch, checks, ["--strict"]) == 0


class TestCheckInventory:
    """The registry must cover every path  was opened for."""

    def test_covers_the_documented_failure_paths(self, verify_env):
        names = " ".join(name for name, _ in verify_env.CHECKS).lower()
        for expected in ("numba", "leiden", "umap", "survival", "kaleido"):
            assert expected in names, f"no check covers {expected}"

    def test_every_check_is_callable_and_named(self, verify_env):
        for name, fn in verify_env.CHECKS:
            assert isinstance(name, str) and name
            assert callable(fn)
