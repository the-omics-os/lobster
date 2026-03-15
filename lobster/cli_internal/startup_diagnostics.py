"""Shared startup diagnostics for CLI and alternate UIs.

This module centralizes classification of fatal startup failures so UI layers
can render the same diagnostic content with different presentation systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence


@dataclass(frozen=True)
class StartupDiagnostic:
    """Structured startup failure suitable for multiple UI renderers."""

    code: str
    title: str
    detail_lines: tuple[str, ...] = ()
    fix_lines: tuple[str, ...] = ()
    level: str = "error"
    exit_code: int = 1


class StartupDiagnosticError(Exception):
    """Raised when startup fails and the caller must render a diagnostic."""

    def __init__(self, diagnostic: StartupDiagnostic):
        super().__init__(diagnostic.title)
        self.diagnostic = diagnostic


def _tuple_lines(lines: Optional[Iterable[str]]) -> tuple[str, ...]:
    if not lines:
        return ()
    return tuple("" if line is None else str(line) for line in lines)


def _extract_install_command(message: str) -> Optional[str]:
    marker = "Install with:"
    if marker not in message:
        return None
    command = message.split(marker, 1)[1].strip()
    return command or None


def _format_missing_credentials_fix_lines(
    *,
    workspace: Optional[Path],
    error_message: str,
) -> tuple[str, ...]:
    lines = []
    workspace_path = Path(workspace) if workspace else None
    if workspace_path is not None:
        lines.extend(
            [
                "1. Add the key to the workspace .env file:",
                f"   {workspace_path / '.env'}",
            ]
        )
    lines.extend(
        [
            "2. Or add to global credentials (works everywhere):",
            "   lobster init --global",
            "3. Or export in your shell:",
        ]
    )
    export_hint = (
        error_message.split("Set it with: ", 1)[1].strip()
        if "Set it with: " in error_message
        else "export <KEY>=<value>"
    )
    lines.append(f"   {export_hint}")

    try:
        from lobster.core.config_resolver import _find_existing_configs

        found_configs = _find_existing_configs()
    except Exception:
        found_configs = []

    if found_configs:
        closest = found_configs[0]
        project_root = closest.parent
        lines.extend(
            [
                "",
                f"Found an existing workspace at {closest} --",
                "it may already have credentials configured.",
                f"Try: export LOBSTER_WORKSPACE={project_root}/.lobster_workspace",
                "then re-run your command.",
            ]
        )

    return tuple(lines)


def _build_invalid_provider_diagnostic(message: str) -> StartupDiagnostic:
    valid_prefix = "Valid providers:"
    valid_line = (
        message.split(valid_prefix, 1)[1].strip() if valid_prefix in message else ""
    )
    fix_lines = []
    if valid_line:
        fix_lines.append(f"Use one of: {valid_line}")
    fix_lines.append("Or run: lobster init")
    return StartupDiagnostic(
        code="invalid_provider",
        title="Invalid provider",
        detail_lines=(message,),
        fix_lines=_tuple_lines(fix_lines),
    )


def _build_missing_provider_package_diagnostic(message: str) -> StartupDiagnostic:
    command = _extract_install_command(message)
    fix_lines = (f"Run: {command}",) if command else ()
    return StartupDiagnostic(
        code="missing_provider_package",
        title="Missing provider package",
        detail_lines=(message,),
        fix_lines=fix_lines,
    )


def _build_missing_dependency_diagnostic(message: str) -> StartupDiagnostic:
    command = _extract_install_command(message)
    fix_lines = (f"Run: {command}",) if command else ()
    return StartupDiagnostic(
        code="missing_dependency",
        title="Missing dependency",
        detail_lines=(message,),
        fix_lines=fix_lines,
    )


def _build_missing_credentials_diagnostic(
    message: str,
    *,
    workspace: Optional[Path],
    provider_override: Optional[str],
) -> StartupDiagnostic:
    provider_name = (provider_override or "configured provider").strip()
    return StartupDiagnostic(
        code="missing_credentials",
        title=f"Missing credentials for provider '{provider_name}'",
        detail_lines=(message,),
        fix_lines=_format_missing_credentials_fix_lines(
            workspace=workspace,
            error_message=message,
        ),
    )


def _build_provider_not_configured_fix_lines(
    *,
    workspace: Optional[Path],
) -> tuple[str, ...]:
    lines = [
        "Run: lobster init --global",
        "Or run: lobster init",
        "Or set: export LOBSTER_LLM_PROVIDER=anthropic",
    ]

    try:
        from lobster.config.constants import VALID_PROVIDERS

        lines.append("Valid providers: " + ", ".join(VALID_PROVIDERS))
    except Exception:
        pass

    try:
        from lobster.core.config_resolver import _find_existing_configs

        search_start = workspace.parent if workspace else None
        found_configs = (
            _find_existing_configs(start=search_start)
            if search_start
            else _find_existing_configs()
        )
    except Exception:
        found_configs = []

    if found_configs:
        lines.insert(
            0, f"Reuse existing workspace: export LOBSTER_WORKSPACE={found_configs[0]}"
        )

    return tuple(lines)


def classify_startup_exception(
    exc: BaseException,
    *,
    workspace: Optional[Path] = None,
    provider_override: Optional[str] = None,
) -> StartupDiagnostic:
    """Translate startup failures into a structured, UI-agnostic diagnostic."""

    message = str(exc).strip() or exc.__class__.__name__

    try:
        from lobster.core.config_resolver import ConfigurationError
    except Exception:  # pragma: no cover - fail-open import guard
        ConfigurationError = ()  # type: ignore[assignment]

    if isinstance(exc, ConfigurationError):
        if message.startswith("No provider configured."):
            return StartupDiagnostic(
                code="provider_not_configured",
                title="No provider configured",
                fix_lines=_build_provider_not_configured_fix_lines(
                    workspace=workspace,
                ),
            )
        if message.startswith("Invalid provider "):
            return _build_invalid_provider_diagnostic(message)
        return StartupDiagnostic(
            code="startup_failed",
            title="Startup failed",
            detail_lines=(message,),
            fix_lines=("Run again with --debug for a full traceback.",),
        )

    if isinstance(exc, ImportError):
        has_install_command = _extract_install_command(message) is not None
        if has_install_command and "package not installed" in message:
            return _build_missing_provider_package_diagnostic(message)
        if has_install_command:
            return _build_missing_dependency_diagnostic(message)

    if isinstance(exc, ValueError):
        lowered = message.lower()
        credential_markers = (
            "credential",
            "credentials",
            "api key",
            "not found in environment",
            "not configured",
            "endpoint not found",
            "set it with:",
        )
        if provider_override or any(marker in lowered for marker in credential_markers):
            return _build_missing_credentials_diagnostic(
                message,
                workspace=workspace,
                provider_override=provider_override,
            )

    return StartupDiagnostic(
        code="startup_failed",
        title="Startup failed",
        detail_lines=(message,),
        fix_lines=("Run again with --debug for a full traceback.",),
    )


def raise_startup_diagnostic(
    exc: BaseException,
    *,
    workspace: Optional[Path] = None,
    provider_override: Optional[str] = None,
) -> None:
    """Raise a structured startup diagnostic derived from *exc*."""

    raise StartupDiagnosticError(
        classify_startup_exception(
            exc,
            workspace=workspace,
            provider_override=provider_override,
        )
    ) from exc


def _append_indented(lines: list[str], section_lines: Sequence[str]) -> None:
    for line in section_lines:
        if line:
            lines.append(f"  {line}")
        else:
            lines.append("")


def format_startup_diagnostic_text(diagnostic: StartupDiagnostic) -> str:
    """Return a plain-text representation suitable for stderr or protocol alerts."""

    lines = [diagnostic.title]
    if diagnostic.detail_lines:
        _append_indented(lines, diagnostic.detail_lines)
    if diagnostic.fix_lines:
        if lines and lines[-1] != "":
            lines.append("")
        lines.append("How to fix:")
        _append_indented(lines, diagnostic.fix_lines)
    return "\n".join(lines)


def render_startup_diagnostic_rich(console, diagnostic: StartupDiagnostic) -> None:
    """Render a startup diagnostic using Rich without embedding UI logic upstream."""

    from rich.markup import escape as rich_escape

    def _print_block(lines: Sequence[str], style: Optional[str] = None) -> None:
        for line in lines:
            if line:
                text = f"  {rich_escape(line)}"
                console.print(text if style is None else f"[{style}]{text}[/{style}]")
            else:
                console.print()

    console.print()
    console.print(f"[red bold]{rich_escape(diagnostic.title)}[/red bold]")
    if diagnostic.detail_lines:
        _print_block(diagnostic.detail_lines, style="red")
    if diagnostic.fix_lines:
        console.print("[yellow]How to fix:[/yellow]")
        _print_block(diagnostic.fix_lines, style="dim")
    console.print()
