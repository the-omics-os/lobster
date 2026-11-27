"""
Omics-OS Cloud key and endpoint for Lobster.

Lobster keeps its Omics-OS Cloud key in its own settings: ``OMICS_OS_API_KEY`` in
Lobster's ``credentials.env`` (``lobster init`` and ``lobster cloud login`` write
it), a project's ``.env``, or the shell. It never reads or writes the omics-os
CLI's sign-in (``~/.config/omics-os``): the two tools do not influence each other.
"""

import os

DEFAULT_ENDPOINT = "https://app.omics-os.com"


def get_api_key() -> str | None:
    """The Omics-OS Cloud key: ``OMICS_OS_API_KEY`` as settings resolve it.

    Importing settings loads Lobster's credential files into the environment
    once (a project's ``.env``, then the global ``credentials.env``), so every
    command reads the key the same way.
    """
    import lobster.config.settings  # noqa: F401

    key = os.environ.get("OMICS_OS_API_KEY")
    if key and key.strip():
        return key.strip()
    return None


def get_endpoint() -> str:
    """The gateway endpoint: ``OMICS_OS_ENDPOINT``, else Omics-OS Cloud."""
    import lobster.config.settings  # noqa: F401

    endpoint = os.environ.get("OMICS_OS_ENDPOINT")
    if endpoint and endpoint.strip():
        return endpoint.strip().rstrip("/")
    return DEFAULT_ENDPOINT
