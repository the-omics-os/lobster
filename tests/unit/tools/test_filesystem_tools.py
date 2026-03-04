"""Tests for filesystem_tools factory."""

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def workspace(tmp_path):
    """Create a workspace with test files."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "counts.csv").write_text("gene,sample1\nTP53,100\nBRCA1,200\n")
    (data_dir / "metadata.json").write_text('{"samples": 2}')
    (data_dir / "nested").mkdir()
    (data_dir / "nested" / "deep.txt").write_text("deep content")
    (tmp_path / "exports").mkdir()
    return tmp_path


class TestCreateFilesystemTools:
    """Test that factory returns correct tools with AQUADIF metadata."""

    def test_factory_returns_six_tools(self, workspace):
        from lobster.tools.filesystem_tools import create_filesystem_tools

        tools = create_filesystem_tools(workspace_path=workspace)
        assert len(tools) == 6
        names = {t.name for t in tools}
        assert names == {
            "list_files",
            "read_file",
            "write_file",
            "glob_files",
            "grep_files",
            "shell_execute",
        }

    def test_all_tools_have_aquadif_metadata(self, workspace):
        from lobster.tools.filesystem_tools import create_filesystem_tools

        tools = create_filesystem_tools(workspace_path=workspace)
        for t in tools:
            assert hasattr(t, "metadata"), f"{t.name} missing .metadata"
            assert hasattr(t, "tags"), f"{t.name} missing .tags"
            assert t.metadata["categories"] == ["UTILITY"]
            assert t.metadata["provenance"] is False
            assert t.tags == ["UTILITY"]


class TestListFiles:
    """Test list_files tool."""

    def test_list_root(self, workspace):
        from lobster.tools.filesystem_tools import create_filesystem_tools

        tools = {t.name: t for t in create_filesystem_tools(workspace_path=workspace)}
        result = tools["list_files"].invoke({"path": "."})
        assert "data" in result
        assert "exports" in result

    def test_list_subdirectory(self, workspace):
        from lobster.tools.filesystem_tools import create_filesystem_tools

        tools = {t.name: t for t in create_filesystem_tools(workspace_path=workspace)}
        result = tools["list_files"].invoke({"path": "data"})
        assert "counts.csv" in result
        assert "metadata.json" in result

    def test_list_nonexistent_returns_error(self, workspace):
        from lobster.tools.filesystem_tools import create_filesystem_tools

        tools = {t.name: t for t in create_filesystem_tools(workspace_path=workspace)}
        result = tools["list_files"].invoke({"path": "nonexistent"})
        assert "Error" in result or "not found" in result.lower()

    def test_path_traversal_blocked(self, workspace):
        from lobster.tools.filesystem_tools import create_filesystem_tools

        tools = {t.name: t for t in create_filesystem_tools(workspace_path=workspace)}
        result = tools["list_files"].invoke({"path": "../../../etc"})
        assert "Error" in result or "outside" in result.lower()


class TestReadFile:
    """Test read_file tool."""

    def test_read_full_file(self, workspace):
        from lobster.tools.filesystem_tools import create_filesystem_tools

        tools = {t.name: t for t in create_filesystem_tools(workspace_path=workspace)}
        result = tools["read_file"].invoke({"path": "data/counts.csv"})
        assert "TP53" in result
        assert "BRCA1" in result

    def test_read_with_pagination(self, workspace):
        from lobster.tools.filesystem_tools import create_filesystem_tools

        tools = {t.name: t for t in create_filesystem_tools(workspace_path=workspace)}
        result = tools["read_file"].invoke(
            {"path": "data/counts.csv", "offset": 0, "limit": 1}
        )
        # Should contain first line (header)
        assert "gene" in result
        # Line count should be limited
        lines = [l for l in result.strip().split("\n") if l.strip()]
        # With cat -n format, should have limited lines
        assert len(lines) <= 3  # header + 1 line + possible metadata

    def test_read_nonexistent_returns_error(self, workspace):
        from lobster.tools.filesystem_tools import create_filesystem_tools

        tools = {t.name: t for t in create_filesystem_tools(workspace_path=workspace)}
        result = tools["read_file"].invoke({"path": "data/nonexistent.csv"})
        assert "Error" in result or "not found" in result.lower()

    def test_read_path_traversal_blocked(self, workspace):
        from lobster.tools.filesystem_tools import create_filesystem_tools

        tools = {t.name: t for t in create_filesystem_tools(workspace_path=workspace)}
        result = tools["read_file"].invoke({"path": "../../etc/passwd"})
        assert "Error" in result or "outside" in result.lower()


class TestWriteFile:
    """Test write_file tool."""

    def test_write_new_file(self, workspace):
        from lobster.tools.filesystem_tools import create_filesystem_tools

        tools = {t.name: t for t in create_filesystem_tools(workspace_path=workspace)}
        result = tools["write_file"].invoke(
            {"path": "data/output.csv", "content": "col1,col2\n1,2\n"}
        )
        assert "success" in result.lower() or "wrote" in result.lower()
        assert (workspace / "data" / "output.csv").read_text() == "col1,col2\n1,2\n"

    def test_write_creates_parent_dirs(self, workspace):
        from lobster.tools.filesystem_tools import create_filesystem_tools

        tools = {t.name: t for t in create_filesystem_tools(workspace_path=workspace)}
        result = tools["write_file"].invoke(
            {"path": "data/new_dir/output.txt", "content": "hello"}
        )
        assert (workspace / "data" / "new_dir" / "output.txt").exists()

    def test_write_path_traversal_blocked(self, workspace):
        from lobster.tools.filesystem_tools import create_filesystem_tools

        tools = {t.name: t for t in create_filesystem_tools(workspace_path=workspace)}
        result = tools["write_file"].invoke(
            {"path": "../../evil.sh", "content": "rm -rf /"}
        )
        assert "Error" in result or "outside" in result.lower()


class TestGlobFiles:
    """Test glob_files tool."""

    def test_glob_csv(self, workspace):
        from lobster.tools.filesystem_tools import create_filesystem_tools

        tools = {t.name: t for t in create_filesystem_tools(workspace_path=workspace)}
        result = tools["glob_files"].invoke({"pattern": "**/*.csv"})
        assert "counts.csv" in result

    def test_glob_recursive(self, workspace):
        from lobster.tools.filesystem_tools import create_filesystem_tools

        tools = {t.name: t for t in create_filesystem_tools(workspace_path=workspace)}
        result = tools["glob_files"].invoke({"pattern": "**/*.txt"})
        assert "deep.txt" in result

    def test_glob_no_matches(self, workspace):
        from lobster.tools.filesystem_tools import create_filesystem_tools

        tools = {t.name: t for t in create_filesystem_tools(workspace_path=workspace)}
        result = tools["glob_files"].invoke({"pattern": "**/*.xyz"})
        assert "no matches" in result.lower() or "0 " in result.lower()


class TestGrepFiles:
    """Test grep_files tool."""

    def test_grep_finds_content(self, workspace):
        from lobster.tools.filesystem_tools import create_filesystem_tools

        tools = {t.name: t for t in create_filesystem_tools(workspace_path=workspace)}
        result = tools["grep_files"].invoke({"pattern": "TP53"})
        assert "counts.csv" in result
        assert "TP53" in result

    def test_grep_with_glob_filter(self, workspace):
        from lobster.tools.filesystem_tools import create_filesystem_tools

        tools = {t.name: t for t in create_filesystem_tools(workspace_path=workspace)}
        result = tools["grep_files"].invoke({"pattern": "TP53", "glob": "*.json"})
        # TP53 is in CSV not JSON
        assert "counts.csv" not in result or "no matches" in result.lower()

    def test_grep_no_matches(self, workspace):
        from lobster.tools.filesystem_tools import create_filesystem_tools

        tools = {t.name: t for t in create_filesystem_tools(workspace_path=workspace)}
        result = tools["grep_files"].invoke({"pattern": "NONEXISTENT_GENE_XYZ"})
        assert "no matches" in result.lower() or "0 " in result.lower()


class TestShellExecute:
    """Test shell_execute tool."""

    def test_basic_command(self, workspace):
        from lobster.tools.filesystem_tools import create_filesystem_tools

        tools = {t.name: t for t in create_filesystem_tools(workspace_path=workspace)}
        result = tools["shell_execute"].invoke({"command": "echo hello"})
        assert "hello" in result

    def test_command_runs_in_workspace(self, workspace):
        from lobster.tools.filesystem_tools import create_filesystem_tools

        tools = {t.name: t for t in create_filesystem_tools(workspace_path=workspace)}
        result = tools["shell_execute"].invoke({"command": "ls data/"})
        assert "counts.csv" in result

    def test_timeout_enforced(self, workspace):
        from lobster.tools.filesystem_tools import create_filesystem_tools

        tools = {t.name: t for t in create_filesystem_tools(workspace_path=workspace)}
        result = tools["shell_execute"].invoke({"command": "sleep 120", "timeout": 2})
        assert "timeout" in result.lower() or "timed out" in result.lower()

    def test_failed_command_returns_stderr(self, workspace):
        from lobster.tools.filesystem_tools import create_filesystem_tools

        tools = {t.name: t for t in create_filesystem_tools(workspace_path=workspace)}
        result = tools["shell_execute"].invoke({"command": "ls nonexistent_dir_xyz"})
        assert (
            "exit_code" in result.lower()
            or "error" in result.lower()
            or "no such" in result.lower()
        )

    def test_output_truncation(self, workspace):
        from lobster.tools.filesystem_tools import create_filesystem_tools

        tools = {t.name: t for t in create_filesystem_tools(workspace_path=workspace)}
        # Generate large output
        result = tools["shell_execute"].invoke({"command": "seq 1 100000"})
        # Result should be capped, not 100K lines
        assert len(result) < 50000
