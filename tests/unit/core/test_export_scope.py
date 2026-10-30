"""Contract for lobster.core.export_scope: tool exports stay inside the session
workspace, or a folder the person listed in LOBSTER_EXPORT_ROOTS before start.

The helper is checked on real paths under tmp_path: where the OS would write a
spelling (symlinks and '..' resolved), fail-closed errors, fixed refusal text with no path,
the export-roots setting (read once, never widened by a caller) and the
modality-name check.
"""

import inspect
import os
from pathlib import Path

import pytest

import lobster.core.export_scope as scope
from lobster.core.export_scope import (
    EXPORT_ROOTS_ENV,
    MODALITY_NAME_REASON,
    WORKSPACE_EXPORT_MESSAGE,
    WORKSPACE_EXPORT_REASON,
    InvalidModalityNameError,
    WorkspaceExportError,
    assert_inside_workspace,
    check_modality_name,
    export_roots,
    outside_workspace,
    parse_export_roots,
)

REASON = (
    "this export path is outside the session workspace. Use a file name under "
    "exports/ (for example de_results.csv), or list the folder in "
    "LOBSTER_EXPORT_ROOTS before starting Lobster."
)
NAME_REASON = (
    "a modality name must be one plain name: no '/' or '\\', no control characters, "
    "not '.' or '..', at most 200 characters."
)


@pytest.fixture(autouse=True)
def no_roots(monkeypatch):
    """Every test starts with no export roots, whatever the developer's env holds."""
    monkeypatch.setattr(scope, "_EXPORT_ROOTS", ())


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    (root / "exports").mkdir(parents=True)
    return root


def _refusal(exc: BaseException, text: str) -> None:
    assert str(exc) == text
    assert exc.args == (text,)
    assert exc.__cause__ is None
    assert exc.__suppress_context__ is True


class TestFixedText:
    def test_reason_and_message(self):
        assert EXPORT_ROOTS_ENV == "LOBSTER_EXPORT_ROOTS"
        assert WORKSPACE_EXPORT_REASON == REASON
        assert WORKSPACE_EXPORT_MESSAGE == "Error: " + REASON
        assert MODALITY_NAME_REASON == NAME_REASON

    def test_errors_are_value_errors(self):
        assert issubclass(WorkspaceExportError, ValueError)
        assert issubclass(InvalidModalityNameError, ValueError)


class TestInside:
    @pytest.mark.parametrize(
        "spelling",
        [
            "exports/de_results.csv",
            "exports/sub/dir/x.csv",
            "exports/../exports/x.csv",
            "x.csv",
            ".",
        ],
    )
    def test_relative_spellings_under_the_workspace(self, ws, spelling):
        target = ws / spelling
        assert outside_workspace(ws, target) is False
        assert_inside_workspace(ws, target)

    def test_absolute_target_inside(self, ws):
        target = (ws / "exports" / "abs.csv").resolve()
        assert outside_workspace(ws, target) is False

    def test_relative_target_is_taken_against_the_workspace(self, ws):
        assert outside_workspace(ws, Path("exports/x.csv")) is False
        assert outside_workspace(ws, "exports/x.csv") is False

    def test_symlink_staying_inside(self, ws):
        (ws / "real").mkdir()
        (ws / "exports" / "alias").symlink_to(ws / "real", target_is_directory=True)
        assert outside_workspace(ws, ws / "exports" / "alias" / "x.csv") is False

    def test_workspace_spelled_through_a_symlink(self, tmp_path, ws):
        link = tmp_path / "ws_link"
        link.symlink_to(ws, target_is_directory=True)
        assert outside_workspace(link, link / "exports" / "x.csv") is False
        assert outside_workspace(link, ws / "exports" / "x.csv") is False
        assert outside_workspace(ws, link / "exports" / "x.csv") is False

    def test_target_need_not_exist(self, ws):
        assert (
            outside_workspace(ws, ws / "exports" / "new" / "deeper" / "x.csv") is False
        )


class TestOutside:
    def test_absolute_target_elsewhere(self, tmp_path, ws):
        target = tmp_path / "elsewhere" / "x.csv"
        assert outside_workspace(ws, target) is True
        with pytest.raises(WorkspaceExportError) as info:
            assert_inside_workspace(ws, target)
        _refusal(info.value, REASON)
        assert str(tmp_path) not in repr(info.value)

    @pytest.mark.parametrize(
        "spelling",
        [
            "exports/../../x.csv",
            "../x.csv",
            "exports/../../ws_other/x.csv",
            "a/../../..",
        ],
    )
    def test_dotdot_leaving_the_workspace(self, ws, spelling):
        assert outside_workspace(ws, ws / spelling) is True
        assert outside_workspace(ws, spelling) is True

    def test_sibling_with_a_shared_prefix(self, tmp_path, ws):
        sibling = tmp_path / "ws2"
        sibling.mkdir()
        assert outside_workspace(ws, sibling / "x.csv") is True

    def test_symlinked_directory_leading_out(self, tmp_path, ws):
        out = tmp_path / "out"
        out.mkdir()
        (ws / "exports" / "door").symlink_to(out, target_is_directory=True)
        target = ws / "exports" / "door" / "x.csv"
        assert outside_workspace(ws, target) is True
        with pytest.raises(WorkspaceExportError):
            assert_inside_workspace(ws, target)

    def test_symlinked_file_leading_out(self, tmp_path, ws):
        victim = tmp_path / "victim.csv"
        victim.write_text("keep")
        (ws / "exports" / "x.csv").symlink_to(victim)
        assert outside_workspace(ws, ws / "exports" / "x.csv") is True
        assert victim.read_text() == "keep"

    def test_dangling_symlink_leading_out(self, tmp_path, ws):
        (ws / "exports" / "x.csv").symlink_to(tmp_path / "missing" / "x.csv")
        assert outside_workspace(ws, ws / "exports" / "x.csv") is True

    def test_any_target_outside_refuses_all(self, tmp_path, ws):
        assert (
            outside_workspace(ws, ws / "exports" / "a.csv", tmp_path / "b.csv") is True
        )

    def test_no_targets_is_not_a_refusal(self, ws):
        assert outside_workspace(ws) is False


class TestFailClosed:
    @pytest.mark.parametrize("bad", ["exports/x\x00.csv", None, 5])
    def test_unclassifiable_target(self, ws, bad):
        assert outside_workspace(ws, bad) is True
        with pytest.raises(WorkspaceExportError) as info:
            assert_inside_workspace(ws, bad)
        _refusal(info.value, REASON)

    @pytest.mark.parametrize("bad", [None, "", "rel/ws"])
    def test_workspace_must_be_an_absolute_path(self, ws, bad):
        assert outside_workspace(bad, ws / "exports" / "x.csv") is True

    def test_resolution_error_refuses(self, ws, monkeypatch):
        def boom(*_a, **_k):
            raise OSError("loop")

        monkeypatch.setattr(os.path, "realpath", boom)
        assert outside_workspace(ws, ws / "exports" / "x.csv") is True


class TestExportRoots:
    @pytest.fixture
    def root(self, tmp_path):
        r = tmp_path / "desktop"
        r.mkdir()
        return r

    def test_a_listed_root_allows_targets_inside_it(self, monkeypatch, ws, root):
        monkeypatch.setattr(scope, "_EXPORT_ROOTS", (root,))
        assert outside_workspace(ws, root / "x.csv") is False
        assert outside_workspace(ws, root / "sub" / "x.csv") is False
        assert outside_workspace(ws, ws / "exports" / "x.csv") is False

    def test_a_listed_root_does_not_allow_its_neighbours(
        self, monkeypatch, tmp_path, ws, root
    ):
        monkeypatch.setattr(scope, "_EXPORT_ROOTS", (root,))
        (tmp_path / "desktop2").mkdir()
        assert outside_workspace(ws, tmp_path / "desktop2" / "x.csv") is True
        assert outside_workspace(ws, root / ".." / "x.csv") is True
        assert outside_workspace(ws, tmp_path / "x.csv") is True

    def test_a_symlink_inside_a_root_leading_out(self, monkeypatch, tmp_path, ws, root):
        monkeypatch.setattr(scope, "_EXPORT_ROOTS", (root,))
        out = tmp_path / "out"
        out.mkdir()
        (root / "door").symlink_to(out, target_is_directory=True)
        assert outside_workspace(ws, root / "door" / "x.csv") is True

    def test_callers_cannot_pass_roots(self):
        params = list(inspect.signature(outside_workspace).parameters.values())
        assert [p.kind for p in params] == [
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        ]
        assert list(inspect.signature(assert_inside_workspace).parameters) == [
            "workspace",
            "target",
        ]

    def test_the_setting_is_not_read_again_after_load(self, monkeypatch, ws, root):
        monkeypatch.setenv(EXPORT_ROOTS_ENV, str(root))
        assert export_roots() == ()
        assert outside_workspace(ws, root / "x.csv") is True

    def test_load_reads_the_setting(self, monkeypatch, root):
        monkeypatch.setenv(EXPORT_ROOTS_ENV, str(root))
        assert scope._read_export_roots() == (root,)
        monkeypatch.delenv(EXPORT_ROOTS_ENV)
        assert scope._read_export_roots() == ()


class TestParseExportRoots:
    def test_empty(self):
        assert parse_export_roots("") == ()
        assert parse_export_roots(os.pathsep) == ()

    def test_existing_absolute_directories_are_kept_in_order(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b c"
        a.mkdir()
        b.mkdir()
        assert parse_export_roots(os.pathsep.join([str(b), str(a)])) == (b, a)

    def test_home_is_expanded(self, monkeypatch, tmp_path):
        (tmp_path / "Desktop").mkdir()
        monkeypatch.setenv("HOME", str(tmp_path))
        assert parse_export_roots("~/Desktop") == (tmp_path / "Desktop",)

    def test_unusable_entries_are_dropped(self, tmp_path):
        good = tmp_path / "good"
        good.mkdir()
        (tmp_path / "file.txt").write_text("x")
        text = os.pathsep.join(
            [
                "relative/dir",
                str(tmp_path / "missing"),
                str(tmp_path / "file.txt"),
                "",
                str(good),
                "bad\x00",
            ]
        )
        assert parse_export_roots(text) == (good,)

    def test_non_text_is_no_roots(self):
        assert parse_export_roots(None) == ()


class TestModalityName:
    @pytest.mark.parametrize(
        "name",
        [
            "geo_gse12345_filtered",
            "my data",
            "pbmc3k.v2",
            "Zellen_ä",
            "x" * 200,
            "a..b",
            ".hidden",
        ],
    )
    def test_accepted(self, name):
        assert check_modality_name(name) == name

    @pytest.mark.parametrize(
        "name",
        [
            "",
            "a/b",
            "/abs",
            "a\\b",
            ".",
            "..",
            "a\x00b",
            "a\nb",
            "a\tb",
            "\x7f",
            "x" * 201,
            None,
            5,
        ],
    )
    def test_refused(self, name):
        with pytest.raises(InvalidModalityNameError) as info:
            check_modality_name(name)
        _refusal(info.value, NAME_REASON)
