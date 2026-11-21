"""Contract of the export guard's refusal (figure contract §1).

``assert_export_allowed`` refuses a tool export into the managed figure folder with ONE
fixed, path-free reason, so every caller that prints the error (tool replies, warnings,
progress lines) shows the same text and no caller path. The error keeps its type and
code; it carries no details (no path, no code); it is raised without a displayed chain.
A managed target and a classifier failure refuse identically (both fail closed).

Real filesystem, real store revision, real classifier (no mock). Tests marked
"regression" were already green before this change; the rest fail on the old guard.
"""

from __future__ import annotations

import hashlib
import os
import stat
import traceback
from pathlib import Path

import pytest

from lobster.core.figures import export_guard as G
from lobster.core.figures import paths as P
from lobster.core.figures.errors import FigureError, ManagedPathError
from tests.unit.export_guard_fixtures import alias, saved_revision, workspace_tree

#: sha256 of ``MANAGED_EXPORT_MESSAGE`` (its bytes move only on purpose: it now also
#: names a refused overwrite of a session input).
BASE_MESSAGE_SHA256 = "81d92ce0afeabe8e619ece54569472f4c786a6b602225c9781d642f21469b75d"

REASON = (
    "this export path is inside the managed figure folder (plots/<figure>/...), "
    "or it is a file this session loaded or imported (or a copy of one). "
    "Exports cannot write there; choose a new file name under exports/ instead."
)

MANAGED_CASES = ["dotdot-existing", "dotdot-new", "absolute", "alias", "new-figure"]


@pytest.fixture
def ws(tmp_path):
    workspace = tmp_path / "ws"
    rev = saved_revision(workspace)
    (workspace / "exports").mkdir()
    alias(workspace / "exports" / "alias", rev.directory)  # an alias INTO the revision
    return workspace, rev


def _managed(workspace: Path, rev) -> dict:
    fig, r = rev.directory.parent.name, rev.directory.name
    return {
        "dotdot-existing": workspace
        / "exports"
        / ".."
        / "plots"
        / fig
        / r
        / "figure.json",
        "dotdot-new": workspace / "exports" / ".." / "plots" / fig / r / "x.csv",
        "absolute": rev.directory / "figure.json",
        "alias": workspace / "exports" / "alias" / "x.csv",
        "new-figure": workspace / "plots" / "fig_new" / "rev_9" / "x.csv",
    }


def _refusal(workspace: Path, target: Path) -> ManagedPathError:
    with pytest.raises(ManagedPathError) as info:
        G.assert_export_allowed(workspace, target)
    return info.value


def _assert_path_free(exc: BaseException, workspace: Path, *needles: str) -> None:
    rendered = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    shown = [str(exc), repr(exc), repr(getattr(exc, "details", None)), rendered]
    for needle in (str(workspace), workspace.name, *needles):
        for text in shown:
            assert needle not in text, (needle, text)
    # Raised without a displayed chain: no cause, and the context is suppressed.
    assert exc.__cause__ is None
    assert exc.__suppress_context__ is True


# ------------------------------------------------------------ the fixed refusal


@pytest.mark.parametrize("case", MANAGED_CASES)
def test_a_managed_refusal_carries_only_the_fixed_reason(ws, case):
    workspace, rev = ws
    target = _managed(workspace, rev)[case]
    before = workspace_tree(workspace)
    exc = _refusal(workspace, target)
    assert str(exc) == REASON
    assert exc.details == {}
    assert type(exc) is ManagedPathError and exc.code == "managed_path"
    _assert_path_free(
        exc, workspace, rev.directory.name, rev.directory.parent.name, target.name
    )
    assert workspace_tree(workspace) == before  # the check writes nothing
    assert rev.unchanged()


def test_a_classifier_failure_refuses_with_the_same_fixed_reason(ws):
    workspace, rev = ws
    target = workspace / "exports" / "a\x00b.csv"
    # The REAL classifier fails on this target (recorded, not mocked) ...
    with pytest.raises(ValueError):
        P.assert_not_managed(workspace, target)
    # ... and the guard turns that into the fixed refusal (fail closed).
    exc = _refusal(workspace, target)
    assert str(exc) == REASON
    assert exc.details == {}
    assert type(exc) is ManagedPathError and exc.code == "managed_path"
    _assert_path_free(exc, workspace, "a\x00b", "embedded null")
    assert G.export_refused(workspace, target) is True


def test_a_case_variant_follows_the_volume(ws):
    workspace, rev = ws
    variant = (
        workspace / "PLOTS" / rev.directory.parent.name / rev.directory.name / "x.csv"
    )
    if (workspace / "PLOTS").exists():  # case-insensitive volume: the managed folder
        exc = _refusal(workspace, variant)
        assert str(exc) == REASON and exc.details == {}
        _assert_path_free(exc, workspace, rev.directory.name)
    else:  # case-sensitive volume: a refusal, if any, is the same fixed one
        try:
            G.assert_export_allowed(workspace, variant)
        except ManagedPathError as exc:
            assert str(exc) == REASON


def test_the_tool_reply_is_the_reason_with_an_error_prefix():
    assert G.MANAGED_EXPORT_REASON == REASON
    assert G.MANAGED_EXPORT_MESSAGE == "Error: " + G.MANAGED_EXPORT_REASON


# ------------------------------------------------- regression: unchanged contracts


def test_regression_the_data_free_reason_is_the_class_code(ws):
    """``safe_reason`` (data-free outcome text) is the class code, never a path."""
    from lobster.core.figures.errors import safe_reason

    workspace, rev = ws
    managed = _refusal(workspace, rev.directory / "figure.json")
    failed = _refusal(workspace, workspace / "exports" / "a\x00b.csv")
    assert safe_reason(managed) == "managed_path"
    assert safe_reason(failed) == "managed_path"


def test_regression_the_tool_reply_bytes_are_unchanged():
    digest = hashlib.sha256(G.MANAGED_EXPORT_MESSAGE.encode("utf-8")).hexdigest()
    assert digest == BASE_MESSAGE_SHA256


def test_regression_ordinary_and_outside_targets_are_allowed(ws, tmp_path):
    workspace, rev = ws
    outside = tmp_path / "outside" / "x.csv"
    for target in (
        workspace / "exports" / "x.csv",
        workspace / "data" / "x.h5ad",
        outside,
    ):
        assert G.assert_export_allowed(workspace, target) is None
        assert G.export_refused(workspace, target) is False


@pytest.mark.parametrize("case", MANAGED_CASES)
def test_regression_export_refused_still_refuses(ws, case):
    workspace, rev = ws
    assert G.export_refused(workspace, _managed(workspace, rev)[case]) is True


def test_regression_the_filesystem_tool_rule_keeps_its_own_text(ws):
    """``paths.assert_not_managed`` (filesystem tools) is not changed by the export guard."""
    workspace, rev = ws
    with pytest.raises(ManagedPathError) as info:
        P.assert_not_managed(workspace, rev.directory / "figure.json")
    assert "managed figure namespace" in str(info.value)
    assert "path" in info.value.details


def test_regression_the_error_class_is_unchanged():
    assert issubclass(ManagedPathError, FigureError)
    assert ManagedPathError.code == "managed_path"


# ------------------------------------------------- fixture: workspace_tree roots


def _write(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_fixture_two_roots_with_one_basename_keep_separate_entries(tmp_path):
    first = tmp_path / "a" / "ws"
    second = tmp_path / "b" / "ws"
    _write(first / "f", b"same")
    _write(second / "f", b"same")
    before = workspace_tree(first, second)
    (first / "f").write_bytes(b"changed")  # only the first root changes
    assert workspace_tree(first, second) != before


def test_fixture_an_empty_root_replaced_by_a_link_is_visible(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    before = workspace_tree(root)
    root.rmdir()
    alias(root, other)  # same (empty) content, now a link
    assert workspace_tree(root) != before


def test_fixture_a_root_link_is_recorded_and_not_followed(tmp_path):
    one = tmp_path / "one"
    two = tmp_path / "two"
    _write(one / "f", b"same")
    _write(two / "f", b"same")
    root = alias(tmp_path / "ws", one)
    snap = workspace_tree(root)
    assert [v[:2] for v in snap.values()] == [("link", str(one))]  # not traversed
    root.unlink()
    alias(root, two)  # identical content behind a new target
    assert workspace_tree(root) != snap


def test_fixture_a_dangling_root_link_is_not_absent(tmp_path):
    root = tmp_path / "ws"
    absent = workspace_tree(root)
    alias(root, tmp_path / "missing")
    dangling = workspace_tree(root)
    assert list(absent.values()) == [("absent",)]
    assert [v[:2] for v in dangling.values()] == [("link", str(tmp_path / "missing"))]
    assert list(dangling) == list(absent)  # one key scheme for both


def test_fixture_an_unreadable_subtree_raises(tmp_path):
    root = tmp_path / "ws"
    locked = root / "locked"
    _write(locked / "f")
    original = stat.S_IMODE(os.lstat(locked).st_mode)
    locked.chmod(0)
    try:
        if os.access(locked, os.R_OK):
            pytest.skip(
                "mode 0 does not block this user (privileged): an unreadable "
                "directory cannot be built here"
            )
        with pytest.raises(PermissionError):
            workspace_tree(root)
    finally:
        locked.chmod(original)
    assert stat.S_IMODE(os.lstat(locked).st_mode) == original


def _shift_mtime(path: Path, seconds: int) -> None:
    st = os.lstat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + seconds * 1_000_000_000))


def test_fixture_an_mtime_only_change_is_visible(tmp_path):
    root = tmp_path / "ws"
    f = _write(root / "f", b"same")
    before = workspace_tree(root)
    _shift_mtime(f, -5)  # same bytes, same size, other modification time
    assert workspace_tree(root) != before


def test_fixture_a_same_byte_overwrite_is_visible(tmp_path):
    root = tmp_path / "ws"
    f = _write(root / "f", b"same")
    _shift_mtime(f, -5)  # a past time, so the rewrite cannot share it
    before = workspace_tree(root)
    f.write_bytes(b"same")
    assert f.read_bytes() == b"same"
    assert workspace_tree(root) != before


def test_fixture_an_access_time_change_is_ignored(tmp_path):
    root = tmp_path / "ws"
    f = _write(root / "f", b"same")
    before = workspace_tree(root)
    st = os.lstat(f)
    os.utime(f, ns=(st.st_atime_ns - 5_000_000_000, st.st_mtime_ns))
    assert f.read_bytes() == b"same"
    assert workspace_tree(root) == before
