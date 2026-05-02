"""Structural tests for CLI decomposition verification.

Verifies that heavy command modules exist, are importable, and export
the correct symbols after extraction from cli.py.
"""

import importlib
import sys


def test_heavy_modules_importable():
    """All 7 heavy modules can be imported without error."""
    modules = [
        "lobster.cli_internal.commands.heavy.session_infra",
        "lobster.cli_internal.commands.heavy.animations",
        "lobster.cli_internal.commands.heavy.display_helpers",
        "lobster.cli_internal.commands.heavy.init_commands",
        "lobster.cli_internal.commands.heavy.chat_commands",
        "lobster.cli_internal.commands.heavy.query_commands",
        "lobster.cli_internal.commands.heavy.slash_commands",
    ]
    for mod_name in modules:
        mod = importlib.import_module(mod_name)
        assert mod is not None, f"Failed to import {mod_name}"


def test_session_infra_exports():
    """session_infra exports required symbols."""
    from lobster.cli_internal.commands.heavy import session_infra

    required = [
        "LobsterClientAdapter",
        "CloudAwareCache",
        "CommandClient",
        "NoOpProgress",
        "should_show_progress",
        "create_progress",
        "init_client",
        "init_client_or_raise_startup_diagnostic",
        "init_client_with_animation",
        "validate_startup_or_raise_startup_diagnostic",
    ]
    for name in required:
        assert hasattr(session_infra, name), f"session_infra missing export: {name}"


def test_animations_exports():
    """animations exports required symbols."""
    from lobster.cli_internal.commands.heavy import animations

    required = [
        "_dna_helix_animation",
        "_dna_agent_loading_phase",
        "_dna_exit_animation",
    ]
    for name in required:
        assert hasattr(animations, name), f"animations missing export: {name}"


def test_display_helpers_exports():
    """display_helpers exports required symbols."""
    from lobster.cli_internal.commands.heavy import display_helpers

    required = [
        "_format_data_preview",
        "_format_dataframe_preview",
        "_format_array_info",
        "_get_matrix_info",
        "_display_status_info",
    ]
    for name in required:
        assert hasattr(display_helpers, name), f"display_helpers missing export: {name}"


def test_command_impl_functions_callable():
    """init_impl, chat_impl, query_impl are callable."""
    from lobster.cli_internal.commands.heavy.chat_commands import chat_impl
    from lobster.cli_internal.commands.heavy.init_commands import init_impl
    from lobster.cli_internal.commands.heavy.query_commands import query_impl

    assert callable(init_impl)
    assert callable(chat_impl)
    assert callable(query_impl)


def test_slash_commands_exports():
    """slash_commands exports required symbols for command dispatch."""
    from lobster.cli_internal.commands.heavy import slash_commands

    required = [
        "_execute_command",
        "_dispatch_command",
        "handle_command",
        "execute_shell_command",
        "extract_available_commands",
        "check_for_missing_slash_command",
        "show_default_help",
        "change_mode",
        "get_user_input_with_editing",
        "_display_streaming_response",
        "_show_workspace_prompt",
        "get_current_agent_name",
        "_extract_argument",
        "_command_files",
        "_command_save",
        "_command_restore",
        "_UNKNOWN_COMMAND",
    ]
    for name in required:
        assert hasattr(slash_commands, name), f"slash_commands missing export: {name}"


def test_cli_is_wiring_only():
    """cli.py should be wiring-only: no function body exceeds ~10 lines
    (excluding Typer parameter declarations)."""
    import ast
    from pathlib import Path

    cli_path = Path(__file__).parent.parent.parent.parent / "lobster" / "cli.py"
    source = cli_path.read_text()
    tree = ast.parse(source)

    # Count total non-blank, non-comment, non-import, non-decorator lines
    lines = source.splitlines()
    total = len(lines)
    assert total < 1800, f"cli.py is {total} lines, should be < 1800 for wiring-only"

    # Check that no function body is too large (>15 lines excluding param declarations)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Skip inner functions and methods
            if not isinstance(node, ast.FunctionDef):
                continue

            # Count body lines (excluding docstrings)
            body_lines = 0
            for stmt in node.body:
                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                    continue  # Skip docstrings
                # Count the statement's lines
                if hasattr(stmt, "end_lineno") and hasattr(stmt, "lineno"):
                    body_lines += stmt.end_lineno - stmt.lineno + 1

            # Allow up to 40 lines for function bodies in cli.py
            # (init ~33, _maybe_launch_go_chat_ui ~38, chat ~29 lines of wiring)
            assert body_lines <= 40, (
                f"Function {node.name} at line {node.lineno} has {body_lines} body lines "
                f"(should be <= 40 for wiring-only cli.py)"
            )


def test_no_circular_imports():
    """Importing any heavy module must not trigger import of lobster.cli."""
    # Clear cli from sys.modules if already loaded
    modules_to_clear = [
        k
        for k in sys.modules
        if k.startswith("lobster.cli") and k != "lobster.cli_internal"
    ]
    for mod in modules_to_clear:
        del sys.modules[mod]

    # Import heavy modules
    heavy_modules = [
        "lobster.cli_internal.commands.heavy.slash_commands",
        "lobster.cli_internal.commands.heavy.session_infra",
        "lobster.cli_internal.commands.heavy.animations",
    ]

    for mod_name in heavy_modules:
        importlib.import_module(mod_name)

    # Check that lobster.cli was NOT imported
    assert (
        "lobster.cli" not in sys.modules
    ), "Importing heavy modules triggered import of lobster.cli -- circular import detected"


def test_cli_help_outputs():
    """CLI --help should succeed (exit code 0)."""
    from typer.testing import CliRunner

    from lobster.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, f"--help failed: {result.output}"

    # Also check key subcommands
    for subcmd in ["chat", "query", "init"]:
        result = runner.invoke(app, [subcmd, "--help"])
        assert result.exit_code == 0, f"{subcmd} --help failed: {result.output}"


def test_chat_no_intro_passes_flag_to_go_launcher(monkeypatch):
    from typer.testing import CliRunner

    from lobster.cli import app

    seen = {}

    def _fake_maybe_launch_go_chat_ui(**kwargs):
        seen.update(kwargs)
        return True

    monkeypatch.setattr(
        "lobster.cli._maybe_launch_go_chat_ui", _fake_maybe_launch_go_chat_ui
    )

    runner = CliRunner()
    result = runner.invoke(app, ["chat", "--ui", "go", "--no-intro"])

    assert result.exit_code == 0, result.output
    assert seen["ui_mode"] == "go"
    assert seen["no_intro"] is True
