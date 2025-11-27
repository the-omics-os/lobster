"""Tests for workspace path resolution.

This module tests the centralized workspace resolver that ensures consistent
workspace location across all entry points (CLI, AgentClient, DataManagerV2).
"""

import os
from pathlib import Path
from unittest import mock

import pytest

from lobster.core.workspace import (
    WORKSPACE_ENV_VAR,
    WORKSPACE_FOLDER_NAME,
    get_workspace_env_var,
    get_workspace_folder_name,
    resolve_workspace,
)


class TestWorkspaceResolver:
    """Test workspace resolution logic."""

    def test_explicit_path_takes_priority(self, tmp_path):
        """Explicit path parameter should override everything including env var."""
        explicit = tmp_path / "explicit"
        with mock.patch.dict(os.environ, {WORKSPACE_ENV_VAR: "/env/path"}):
            result = resolve_workspace(explicit_path=explicit)
        # Compare using os.path.abspath to match resolver behavior (no symlink following)
        assert result == Path(os.path.abspath(explicit))

    def test_explicit_path_string(self, tmp_path):
        """Explicit path can be provided as string."""
        explicit = tmp_path / "explicit_string"
        result = resolve_workspace(explicit_path=str(explicit))
        assert result == Path(os.path.abspath(explicit))

    def test_env_var_over_default(self, tmp_path, monkeypatch):
        """Environment variable should override cwd default."""
        env_path = tmp_path / "from_env"
        monkeypatch.setenv(WORKSPACE_ENV_VAR, str(env_path))
        result = resolve_workspace()
        assert result == Path(os.path.abspath(env_path))

    def test_cwd_fallback(self, monkeypatch, tmp_path):
        """Should fall back to cwd/.lobster_workspace when no override."""
        monkeypatch.delenv(WORKSPACE_ENV_VAR, raising=False)
        monkeypatch.chdir(tmp_path)
        result = resolve_workspace()
        assert result == Path(os.path.abspath(tmp_path / WORKSPACE_FOLDER_NAME))

    def test_creates_directory_by_default(self, tmp_path):
        """Should create workspace directory by default."""
        workspace = tmp_path / "new_workspace"
        assert not workspace.exists()
        result = resolve_workspace(explicit_path=workspace, create=True)
        assert result.exists()
        assert result.is_dir()

    def test_creates_nested_directories(self, tmp_path):
        """Should create nested directory structure."""
        workspace = tmp_path / "deep" / "nested" / "workspace"
        assert not workspace.exists()
        result = resolve_workspace(explicit_path=workspace, create=True)
        assert result.exists()

    def test_no_create_option(self, tmp_path):
        """Should not create directory when create=False."""
        workspace = tmp_path / "no_create_workspace"
        assert not workspace.exists()
        result = resolve_workspace(explicit_path=workspace, create=False)
        assert not workspace.exists()
        assert result == Path(os.path.abspath(workspace))

    def test_existing_directory_preserved(self, tmp_path):
        """Should work with existing directories without error."""
        workspace = tmp_path / "existing"
        workspace.mkdir()
        (workspace / "test_file.txt").write_text("test")

        result = resolve_workspace(explicit_path=workspace, create=True)

        assert result == Path(os.path.abspath(workspace))
        assert (workspace / "test_file.txt").exists()

    def test_returns_absolute_path(self, tmp_path, monkeypatch):
        """Should always return absolute path."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv(WORKSPACE_ENV_VAR, raising=False)

        # Relative path through cwd fallback
        result = resolve_workspace()
        assert result.is_absolute()

        # Relative explicit path
        result = resolve_workspace(explicit_path="relative/path", create=False)
        assert result.is_absolute()

    def test_env_var_with_relative_path(self, tmp_path, monkeypatch):
        """Environment variable with relative path should be resolved to absolute."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv(WORKSPACE_ENV_VAR, "relative_env_workspace")

        result = resolve_workspace()

        assert result.is_absolute()
        assert result == Path(os.path.abspath(tmp_path / "relative_env_workspace"))


class TestWorkspaceResolverConstants:
    """Test workspace resolver constants and helpers."""

    def test_workspace_env_var_constant(self):
        """WORKSPACE_ENV_VAR should be LOBSTER_WORKSPACE."""
        assert WORKSPACE_ENV_VAR == "LOBSTER_WORKSPACE"

    def test_workspace_folder_name_constant(self):
        """WORKSPACE_FOLDER_NAME should be .lobster_workspace."""
        assert WORKSPACE_FOLDER_NAME == ".lobster_workspace"

    def test_get_workspace_env_var(self):
        """get_workspace_env_var should return the env var name."""
        assert get_workspace_env_var() == "LOBSTER_WORKSPACE"

    def test_get_workspace_folder_name(self):
        """get_workspace_folder_name should return the folder name."""
        assert get_workspace_folder_name() == ".lobster_workspace"


class TestWorkspaceResolverEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_string_explicit_path(self, tmp_path, monkeypatch):
        """Empty string explicit path should use env var or default."""
        # Empty string is falsy but not None, so it's treated as explicit path
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv(WORKSPACE_ENV_VAR, raising=False)

        # Note: Empty string Path resolves to cwd
        result = resolve_workspace(explicit_path="", create=False)
        assert result == Path(os.path.abspath(tmp_path))

    def test_none_explicit_path_uses_env(self, tmp_path, monkeypatch):
        """None explicit path should check env var."""
        env_path = tmp_path / "env_workspace"
        monkeypatch.setenv(WORKSPACE_ENV_VAR, str(env_path))

        result = resolve_workspace(explicit_path=None)
        assert result == Path(os.path.abspath(env_path))

    def test_path_with_symlinks(self, tmp_path):
        """Should preserve symlinks in path (not follow them)."""
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        symlink = tmp_path / "symlink"
        symlink.symlink_to(real_dir)

        result = resolve_workspace(explicit_path=symlink, create=False)

        # os.path.abspath preserves symlinks (doesn't follow them)
        assert result == Path(os.path.abspath(symlink))

    def test_concurrent_directory_creation(self, tmp_path):
        """Multiple calls should handle concurrent creation gracefully."""
        workspace = tmp_path / "concurrent"

        # Simulate concurrent calls
        result1 = resolve_workspace(explicit_path=workspace, create=True)
        result2 = resolve_workspace(explicit_path=workspace, create=True)

        assert result1 == result2
        assert workspace.exists()


class TestWorkspaceResolverIntegration:
    """Integration tests for workspace resolver with other components."""

    def test_resolver_with_data_manager_pattern(self, tmp_path, monkeypatch):
        """Test resolver works with DataManagerV2-style usage pattern."""
        monkeypatch.setenv(WORKSPACE_ENV_VAR, str(tmp_path / "dm_workspace"))

        # Simulate DataManagerV2.__init__ pattern
        workspace_path = resolve_workspace(explicit_path=None, create=True)

        assert workspace_path.exists()
        assert "dm_workspace" in str(workspace_path)

    def test_resolver_with_cli_pattern(self, tmp_path, monkeypatch):
        """Test resolver works with CLI usage pattern (explicit path)."""
        cli_workspace = tmp_path / "cli_workspace"

        # Simulate CLI with --workspace flag
        workspace = resolve_workspace(explicit_path=cli_workspace, create=True)

        assert workspace == cli_workspace.resolve()
        assert workspace.exists()

    def test_resolver_respects_test_isolation(self, tmp_path, monkeypatch):
        """Test that resolver respects test isolation via env var."""
        # This is how tests/conftest.py isolates tests
        test_workspace = tmp_path / "isolated_test"
        monkeypatch.setenv(WORKSPACE_ENV_VAR, str(test_workspace))

        result = resolve_workspace()

        assert result == test_workspace.resolve()
        assert str(tmp_path) in str(result)
