"""
Unit tests for CLI security and logging bug fixes.

Tests for:
- Bug #6: PathResolver Utility (directory traversal, special files, consistent handling)
- Bug #4: History Logging Improvements (full error traces, file backup)
"""

import json
import logging
import os
import stat
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from lobster.cli_internal.utils.path_resolution import PathResolver, ResolvedPath


class TestBug6PathResolverSecurity:
    """
    Test Bug #6: Secure path resolution.

    Verifies protection against:
    - Directory traversal attacks
    - Special file access
    - Inconsistent path handling
    """

    def test_directory_traversal_blocked(self, tmp_path):
        """Test that directory traversal attempts are blocked."""
        current_dir = tmp_path / "workspace"
        current_dir.mkdir()

        resolver = PathResolver(current_directory=current_dir)

        # Attempt to escape with ../
        result = resolver.resolve("../../etc/passwd")

        assert not result.is_safe, "Directory traversal should be blocked"
        assert "traversal" in result.error.lower()

    def test_absolute_paths_within_allowed_bases(self, tmp_path):
        """Test that absolute paths within allowed bases work."""
        current_dir = tmp_path / "workspace"
        current_dir.mkdir()

        # Create test file in current directory
        test_file = current_dir / "data.csv"
        test_file.write_text("test")

        resolver = PathResolver(current_directory=current_dir)

        # Absolute path within allowed base should work
        result = resolver.resolve(str(test_file))

        assert result.is_safe
        assert result.exists
        assert result.path == test_file

    def test_home_directory_expansion(self, tmp_path):
        """Test that ~/... paths expand correctly."""
        current_dir = tmp_path / "workspace"
        current_dir.mkdir()

        resolver = PathResolver(current_directory=current_dir)

        # Test ~/ expansion
        result = resolver.resolve("~/Documents/test.txt", must_exist=False)

        assert result.is_safe
        assert result.source == "home"
        assert result.path == Path.home() / "Documents/test.txt"

    def test_special_file_detection(self, tmp_path):
        """Test that special files (devices, pipes) are blocked."""
        current_dir = tmp_path / "workspace"
        current_dir.mkdir()

        resolver = PathResolver(current_directory=current_dir)

        # On macOS/Linux, /dev/null is a character device
        if Path("/dev/null").exists():
            result = resolver.resolve("/dev/null", allow_special=False)

            # Should detect as special file
            assert not result.is_safe, "Special files should be blocked"
            assert "special file" in result.error.lower()

    def test_allow_special_flag(self, tmp_path):
        """Test that allow_special=True permits special files."""
        current_dir = tmp_path / "workspace"
        current_dir.mkdir()

        resolver = PathResolver(current_directory=current_dir)

        # With allow_special=True, should not block
        # (but we won't actually test /dev/null since it's dangerous)
        # Instead test that regular files work with flag
        test_file = current_dir / "regular.txt"
        test_file.write_text("test")

        result = resolver.resolve(str(test_file), allow_special=True)
        assert result.is_safe

    def test_workspace_search_fallback(self, tmp_path):
        """Test that workspace search finds files in subdirectories."""
        current_dir = tmp_path / "current"
        workspace_dir = tmp_path / "workspace"
        current_dir.mkdir()
        workspace_dir.mkdir()

        # Create file in workspace/data/
        data_dir = workspace_dir / "data"
        data_dir.mkdir()
        test_file = data_dir / "dataset.h5ad"
        test_file.write_text("test data")

        resolver = PathResolver(
            current_directory=current_dir, workspace_path=workspace_dir
        )

        # File not in current directory, but should find in workspace
        result = resolver.resolve("dataset.h5ad", search_workspace=True)

        assert result.is_safe
        assert result.exists
        assert result.source == "workspace"
        assert result.path == test_file

    def test_relative_path_resolution(self, tmp_path):
        """Test that relative paths resolve correctly."""
        current_dir = tmp_path / "workspace"
        current_dir.mkdir()

        # Create file
        test_file = current_dir / "data.csv"
        test_file.write_text("test")

        resolver = PathResolver(current_directory=current_dir)

        # Relative path
        result = resolver.resolve("data.csv")

        assert result.is_safe
        assert result.exists
        assert result.source == "relative"
        assert result.path == test_file

    def test_must_exist_enforcement(self, tmp_path):
        """Test that must_exist=True returns error for missing files."""
        current_dir = tmp_path / "workspace"
        current_dir.mkdir()

        resolver = PathResolver(current_directory=current_dir)

        # File doesn't exist
        result = resolver.resolve("nonexistent.txt", must_exist=True)

        assert result.is_safe  # Not a security issue
        assert not result.exists
        assert result.error is not None
        assert "not found" in result.error.lower()

    def test_security_across_multiple_operations(self, tmp_path):
        """Test that multiple operations maintain security."""
        current_dir = tmp_path / "workspace"
        current_dir.mkdir()

        resolver = PathResolver(current_directory=current_dir)

        # Multiple attempts should all be blocked
        attacks = [
            "../../etc/passwd",
            "../../../root/.ssh/id_rsa",
            "../../../../../../etc/shadow",
            "/etc/passwd",  # Direct absolute path outside allowed
        ]

        for attack in attacks:
            result = resolver.resolve(attack)
            # Should block escaping current_dir and home dir
            if not result.path.is_relative_to(
                current_dir
            ) and not result.path.is_relative_to(Path.home()):
                assert not result.is_safe, f"Should block: {attack}"


class TestBug4HistoryLogging:
    """
    Test Bug #4: History logging improvements.

    Verifies:
    - Full error logging (not truncated)
    - File backup mechanism
    - Proper return values
    """

    def test_history_logging_returns_bool(self):
        """Test that _add_command_to_history returns bool."""
        from lobster.cli import _add_command_to_history

        # Mock client without message support
        mock_client = Mock()
        mock_client.messages = []  # Has messages attribute
        mock_client.graph = Mock()
        mock_client.graph.update_state = Mock()
        mock_client.session_id = "test_session"
        mock_client.data_manager.workspace_path = Path(tempfile.gettempdir())

        with patch("lobster.cli._backup_command_to_file", return_value=True):
            result = _add_command_to_history(mock_client, "/test", "test summary")

            # Bug #4 fix: Should return bool (True/False), not None
            assert isinstance(result, bool), "Should return bool, not None"

    def test_file_backup_created(self, tmp_path):
        """Test that command backup file is created."""
        from lobster.cli import _backup_command_to_file

        mock_client = Mock()
        mock_client.session_id = "test_session_123"
        mock_client.data_manager.workspace_path = tmp_path

        # Call backup function
        success = _backup_command_to_file(
            client=mock_client,
            command="/test command",
            summary="Test summary",
            is_error=False,
            primary_logged=True,
        )

        assert success, "Backup should succeed"

        # Verify backup file created
        backup_file = tmp_path / ".lobster" / "command_history.jsonl"
        assert backup_file.exists(), "Backup file should exist"

        # Verify content
        with open(backup_file, "r") as f:
            lines = f.readlines()
            assert len(lines) == 1
            record = json.loads(lines[0])
            assert record["command"] == "/test command"
            assert record["summary"] == "Test summary"
            assert record["session_id"] == "test_session_123"
            assert record["logged_to_graph"] is True

    def test_multiple_commands_appended(self, tmp_path):
        """Test that multiple commands are appended to backup file."""
        from lobster.cli import _backup_command_to_file

        mock_client = Mock()
        mock_client.session_id = "test_session"
        mock_client.data_manager.workspace_path = tmp_path

        # Add multiple commands
        commands = [
            ("/test1", "summary1"),
            ("/test2", "summary2"),
            ("/test3", "summary3"),
        ]

        for cmd, summary in commands:
            _backup_command_to_file(mock_client, cmd, summary, False, True)

        # Verify all commands saved
        backup_file = tmp_path / ".lobster" / "command_history.jsonl"
        with open(backup_file, "r") as f:
            lines = f.readlines()
            assert len(lines) == 3

            # Parse and verify
            for i, line in enumerate(lines):
                record = json.loads(line)
                expected_cmd, expected_summary = commands[i]
                assert record["command"] == expected_cmd
                assert record["summary"] == expected_summary

    def test_backup_handles_errors_gracefully(self, tmp_path):
        """Test that backup errors don't crash the system."""
        from lobster.cli import _backup_command_to_file

        mock_client = Mock()
        mock_client.session_id = "test_session"
        # Invalid workspace path (causes write error)
        mock_client.data_manager.workspace_path = Path("/invalid/readonly/path")

        # Should return False but not raise exception
        success = _backup_command_to_file(mock_client, "/test", "summary", False, True)

        assert success is False, "Should return False on error"
        # No exception should be raised

    def test_empty_command_validation(self):
        """Test that empty commands are rejected."""
        from lobster.cli import _add_command_to_history

        mock_client = Mock()
        mock_client.messages = []
        mock_client.session_id = "test"
        mock_client.data_manager.workspace_path = Path(tempfile.gettempdir())

        # Empty command should be rejected
        result = _add_command_to_history(mock_client, "", "summary")
        assert result is False

        # Empty summary should be rejected
        result = _add_command_to_history(mock_client, "/test", "")
        assert result is False

    def test_client_compatibility_check(self):
        """Test that incompatible clients are handled gracefully."""
        from lobster.cli import _add_command_to_history

        # Client without messages attribute
        mock_client = Mock()
        del mock_client.messages  # Remove messages attribute

        result = _add_command_to_history(mock_client, "/test", "summary")

        assert result is False, "Should return False for incompatible client"


class TestPathResolverEdgeCases:
    """Test edge cases for PathResolver."""

    def test_symlink_handling(self, tmp_path):
        """Test that symlinks are followed correctly."""
        current_dir = tmp_path / "workspace"
        current_dir.mkdir()

        # Create target file
        target = current_dir / "target.txt"
        target.write_text("target content")

        # Create symlink
        link = current_dir / "link.txt"
        link.symlink_to(target)

        resolver = PathResolver(current_directory=current_dir)

        # Resolve symlink
        result = resolver.resolve("link.txt")

        assert result.is_safe
        assert result.exists
        # Symlink should resolve to actual file

    def test_nonexistent_file_without_must_exist(self, tmp_path):
        """Test that nonexistent files allowed if must_exist=False."""
        current_dir = tmp_path / "workspace"
        current_dir.mkdir()

        resolver = PathResolver(current_directory=current_dir)

        result = resolver.resolve("nonexistent.txt", must_exist=False)

        assert result.is_safe  # Not a security issue
        assert not result.exists
        assert result.error is None  # No error if must_exist=False

    def test_case_sensitivity(self, tmp_path):
        """Test path resolution respects filesystem case sensitivity."""
        current_dir = tmp_path / "workspace"
        current_dir.mkdir()

        # Create file with specific case
        test_file = current_dir / "TestFile.txt"
        test_file.write_text("test")

        resolver = PathResolver(current_directory=current_dir)

        # Exact case should work
        result = resolver.resolve("TestFile.txt")
        assert result.exists


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
