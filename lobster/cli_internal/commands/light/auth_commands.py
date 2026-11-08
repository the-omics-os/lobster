"""
Provider authentication subcommands for Lobster CLI.

Provides OAuth login/logout for LLM providers that support it.

Commands:
    lobster auth login anthropic   - Login with Claude Pro/Max account
    lobster auth logout anthropic  - Remove stored OAuth credentials
    lobster auth status            - Show auth status for all providers
"""

import logging

import typer
from rich.console import Console
from rich.table import Table

from lobster.i18n import h, t

logger = logging.getLogger(__name__)

auth_app = typer.Typer(
    name="auth",
    help=t("light.auth.help"),
    no_args_is_help=True,
)

console = Console()

# Providers that support OAuth login
OAUTH_PROVIDERS = ["anthropic"]


@auth_app.command()
def login(
    provider: str = typer.Argument(
        ...,
        help=h("Provider to authenticate with (e.g., 'anthropic')."),
    ),
    no_browser: bool = typer.Option(
        False,
        "--no-browser",
        help=h("Don't auto-open browser; print URL instead."),
    ),
    timeout: int = typer.Option(
        120,
        "--timeout",
        help=h("Seconds to wait for browser callback."),
    ),
) -> None:
    """Login to an LLM provider via OAuth.

    Opens your browser for authentication. Currently supports:
    - anthropic: Login with your Claude Pro/Max account.
    """
    provider = provider.lower().strip()

    if provider not in OAUTH_PROVIDERS:
        console.print(
            h(
                "[red]Provider '{name}' does not support OAuth login.[/red]\n"
                "[dim]Supported providers: {providers}[/dim]\n"
                "[dim]For API key auth, use: lobster init[/dim]"
            ).format(name=provider, providers=", ".join(OAUTH_PROVIDERS))
        )
        raise typer.Exit(1)

    if provider == "anthropic":
        _login_anthropic(no_browser=no_browser, timeout=timeout)


@auth_app.command()
def logout(
    provider: str = typer.Argument(
        ...,
        help=h("Provider to log out from (e.g., 'anthropic')."),
    ),
) -> None:
    """Remove stored OAuth credentials for a provider."""
    provider = provider.lower().strip()

    if provider == "anthropic":
        from lobster.config.auth.anthropic_oauth import logout as anth_logout

        if anth_logout():
            console.print(h("[green]Logged out from Anthropic.[/green]"))
        else:
            console.print(h("[dim]No Anthropic OAuth credentials found.[/dim]"))
    else:
        console.print(
            f"[red]{t('light.auth.unknown_provider', provider=provider)}[/red]"
        )
        raise typer.Exit(1)


@auth_app.command()
def status() -> None:
    """Show authentication status for all providers."""
    table = Table(
        title=h("Provider Authentication Status"),
        show_header=True,
        border_style="cyan",
    )
    table.add_column(h("Provider"), style="bold")
    table.add_column(h("Method"))
    table.add_column(h("Status"))

    # Anthropic: check env var, then OAuth
    import os

    anth_key = os.environ.get("ANTHROPIC_API_KEY")
    if anth_key and anth_key.strip():
        table.add_row("anthropic", h("API key (env)"), h("[green]configured[/green]"))
    else:
        try:
            from lobster.config.auth import oauth_store

            creds = oauth_store.load("anthropic")
            if creds:
                if creds.is_expired():
                    table.add_row(
                        "anthropic",
                        "OAuth",
                        h("[yellow]expired (will refresh)[/yellow]"),
                    )
                else:
                    email = f" ({creds.email})" if creds.email else ""
                    table.add_row(
                        "anthropic",
                        "OAuth",
                        h("[green]authenticated{email}[/green]").format(email=email),
                    )
            else:
                table.add_row("anthropic", "—", h("[dim]not configured[/dim]"))
        except Exception:
            table.add_row("anthropic", "—", h("[dim]not configured[/dim]"))

    # Omics-OS Cloud
    try:
        from lobster.config.credentials import load_active_profile

        cloud_creds = load_active_profile()
        if cloud_creds:
            mode = cloud_creds.get("auth_mode", "?")
            email = cloud_creds.get("email", "")
            label = f" ({email})" if email else ""
            table.add_row(
                "omics-os",
                mode,
                h("[green]configured{label}[/green]").format(label=label),
            )
        else:
            table.add_row("omics-os", "—", h("[dim]not configured[/dim]"))
    except Exception:
        table.add_row("omics-os", "—", h("[dim]not configured[/dim]"))

    # Other env-var-based providers
    _env_providers = [
        ("bedrock", "AWS_BEDROCK_ACCESS_KEY"),
        ("openai", "OPENAI_API_KEY"),
        ("gemini", "GOOGLE_API_KEY"),
        ("openrouter", "OPENROUTER_API_KEY"),
        ("nebius", "NEBIUS_API_KEY"),
        ("ollama-cloud", "OLLAMA_API_KEY"),
        ("azure", "AZURE_AI_CREDENTIAL"),
    ]
    for name, env_var in _env_providers:
        val = os.environ.get(env_var)
        if val and val.strip():
            table.add_row(name, h("API key (env)"), h("[green]configured[/green]"))
        else:
            table.add_row(name, "—", h("[dim]not configured[/dim]"))

    console.print(table)


# ---------------------------------------------------------------------------
# Provider-specific login implementations
# ---------------------------------------------------------------------------


def _login_anthropic(no_browser: bool, timeout: int) -> None:
    """Run Anthropic OAuth login flow."""
    from lobster.config.auth.anthropic_oauth import (
        is_authenticated,
        login_interactive,
    )

    if is_authenticated():
        console.print(
            h(
                "[yellow]Already authenticated with Anthropic.[/yellow]\n"
                "[dim]Run 'lobster auth logout anthropic' first to re-authenticate.[/dim]"
            )
        )
        raise typer.Exit(0)

    def _on_url(url: str):
        if no_browser:
            console.print(
                h("\n[bold]Open this URL in your browser:[/bold]\n{url}\n").format(
                    url=url
                )
            )
        else:
            console.print(h("[dim]Opening browser for Anthropic login...[/dim]"))
            console.print(
                h("[dim]If browser doesn't open: {url}[/dim]").format(url=url)
            )

    def _on_progress(msg: str):
        console.print(f"[dim]{msg}[/dim]")

    def _on_manual_input() -> str:
        return typer.prompt(h("Paste the authorization code or full redirect URL"))

    result = login_interactive(
        timeout=timeout,
        open_browser=not no_browser,
        on_url=_on_url,
        on_progress=_on_progress,
        on_manual_input=_on_manual_input,
    )

    if result.success:
        console.print(
            h(
                "[green]Successfully authenticated with Anthropic![/green]\n"
                "[dim]Your Claude Pro/Max account is now linked. "
                "Tokens refresh automatically.[/dim]"
            )
        )
    else:
        console.print(h("[red]Login failed: {error}[/red]").format(error=result.error))
        raise typer.Exit(1)
