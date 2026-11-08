"""Translate Typer's built-in help screen text for the non-English UI.

Typer draws panel titles, ``Usage:``, ``--help`` and the shell-completion
options from plain English strings. ``localize_typer_help`` routes them through
``lobster.i18n.h`` so ``lobster --lang zh --help`` is fully translated.
"""

from __future__ import annotations

from lobster.i18n import h

_ENGLISH_USAGE = "Usage: "


def _localized_usage(get_usage):
    label = h("Usage:")
    if not label.endswith("："):
        label += " "

    def usage(ctx):
        text = get_usage(ctx)
        if text.startswith(_ENGLISH_USAGE):
            return label + text[len(_ENGLISH_USAGE) :]
        return text

    usage._lobster_localized = True
    return usage


def localize_typer_help() -> None:
    """Patch Typer's help rendering to use the current UI language."""
    import typer.completion as typer_completion
    import typer.rich_utils as rich_utils

    rich_utils.OPTIONS_PANEL_TITLE = h("Options")
    rich_utils.COMMANDS_PANEL_TITLE = h("Commands")
    rich_utils.ARGUMENTS_PANEL_TITLE = h("Arguments")
    rich_utils.ERRORS_PANEL_TITLE = h("Error")
    rich_utils.DEFAULT_STRING = h("[default: {}]")
    rich_utils.ENVVAR_STRING = h("[env var: {}]")

    for placeholder in (
        typer_completion._install_completion_placeholder_function,
        typer_completion._install_completion_no_auto_placeholder_function,
    ):
        for option in placeholder.__defaults__ or ():
            if getattr(option, "help", None):
                option.help = h(option.help)

    original = rich_utils.rich_format_help
    if getattr(original, "_lobster_localized", False):
        return

    def rich_format_help(*args, **kwargs):
        command, context = kwargs.get("obj"), kwargs.get("ctx")
        if command is not None and context is not None:
            for param in command.get_params(context):
                if getattr(param, "help", None):
                    param.help = h(param.help)
            if not getattr(command.get_usage, "_lobster_localized", False):
                command.get_usage = _localized_usage(command.get_usage)
        return original(*args, **kwargs)

    rich_format_help._lobster_localized = True
    rich_utils.rich_format_help = rich_format_help
