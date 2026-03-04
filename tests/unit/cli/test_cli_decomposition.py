"""Structural tests for CLI decomposition verification.

Verifies that heavy command modules exist, are importable, and export
the correct symbols after extraction from cli.py.
"""

import importlib


def test_heavy_modules_importable():
    """All 6 new heavy modules can be imported without error."""
    modules = [
        "lobster.cli_internal.commands.heavy.session_infra",
        "lobster.cli_internal.commands.heavy.animations",
        "lobster.cli_internal.commands.heavy.display_helpers",
        "lobster.cli_internal.commands.heavy.init_commands",
        "lobster.cli_internal.commands.heavy.chat_commands",
        "lobster.cli_internal.commands.heavy.query_commands",
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
        "init_client_with_animation",
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
    from lobster.cli_internal.commands.heavy.init_commands import init_impl
    from lobster.cli_internal.commands.heavy.chat_commands import chat_impl
    from lobster.cli_internal.commands.heavy.query_commands import query_impl

    assert callable(init_impl)
    assert callable(chat_impl)
    assert callable(query_impl)
