"""Supported local-gateway and npm-client setup guidance (no authentication I/O)."""

OMICS_OS_AUTH_GUIDANCE = (
    "Create an API key at https://app.omics-os.com/account.\n"
    "Python local gateway: set OMICS_OS_API_KEY or use "
    "lobster cloud login --api-key <key>.\n"
    "Non-interactive local setup: lobster init --non-interactive --use-omics-os "
    "--skip-extras --skip-ssl-test.\n"
    "Alternative managed Cloud client (npm): npm i -g @omicsos/lobster && "
    "lobster cloud login --device.\n"
    "With both clients installed, lobster-cli cloud login --device selects the "
    "npm client. Device login needs human browser approval; configure the Python "
    "provider separately rather than assuming credential sharing."
)

OMICS_OS_KEY_NOT_VERIFIED = (
    "Omics-OS Cloud API key saved, not checked. "
    "Configuration is not proof of account access."
)


def fail_cloud_setup_write(error: Exception) -> None:
    """Required setup persistence failed; exit before success or UI fallback."""
    import typer
    from rich.console import Console

    Console().print(
        f"Omics-OS Cloud setup failed: {error}. Partial configuration may remain; "
        "setup is not complete.",
        markup=False,
    )
    raise typer.Exit(1) from error
