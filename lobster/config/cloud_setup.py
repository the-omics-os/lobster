"""Supported local-gateway and npm-client setup guidance (no authentication I/O)."""

import os
import sys
from typing import Callable

# Where someone without an Omics-OS account asks for one (no query parameters).
ACCESS_REQUEST_URL = "https://www.omics-os.com/early-access"

OMICS_OS_AUTH_GUIDANCE = (
    "Create an API key at https://app.omics-os.com/account.\n"
    "Python local gateway: set OMICS_OS_API_KEY or use "
    "lobster cloud login --api-key <key>.\n"
    "Non-interactive local setup: lobster init --non-interactive --use-omics-os "
    "--skip-extras --skip-ssl-test.\n"
    "Alternative managed Cloud client (npm): npm install -g omics-os && "
    "omics-os login --device.\n"
    "Device login needs human browser approval; configure the Python "
    "provider separately rather than assuming credential sharing."
)

OMICS_OS_KEY_NOT_VERIFIED = (
    "Omics-OS Cloud API key saved, not checked. "
    "Configuration is not proof of account access."
)


def _open_in_browser(url: str) -> bool:
    """Best effort; the wizard always prints the address as well.

    True only when a browser was started; then the wizard says it is opening.
    """
    import subprocess  # nosec B404 # fixed argv, no shell
    import webbrowser

    try:
        if sys.platform != "linux":
            return bool(webbrowser.open(url))
        if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
            # Not webbrowser, nor xdg-open on our terminal: both may fall back to a
            # console browser (lynx, w3m) that takes over the wizard's terminal.
            subprocess.Popen(  # nosec B603 B607 # fixed argv, no shell
                ["xdg-open", url],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return True
    except Exception:  # nosec B110 # the printed address is the way in
        pass
    return False


def request_access(say: Callable[[str], None], wait: Callable[[str], object]) -> None:
    """The wizard's "No, request access": show the request page, then move on.

    Nothing is sent, checked or saved here; the request is made on the web page.
    """
    from lobster.i18n import h

    if _open_in_browser(ACCESS_REQUEST_URL):
        say(h("Opening the access request page in your browser:"))
        say(ACCESS_REQUEST_URL)
        say(h("If it does not open, copy the address into your browser."))
    else:
        say(h("Open this link to request access:"))
        say(ACCESS_REQUEST_URL)
    wait(h("Press Enter when you have sent the request."))
    say(
        h(
            "Your access request is on its way. We'll email you when it's approved. "
            "Until then, choose a different provider to continue."
        )
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
