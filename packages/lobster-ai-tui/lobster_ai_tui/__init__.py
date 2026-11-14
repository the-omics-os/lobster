"""lobster-ai-tui: Platform-specific Go TUI binary for Lobster AI."""

import os
from pathlib import Path


def get_binary_path() -> Path:
    """Return the path to the lobster-tui binary bundled in this package.

    Raises FileNotFoundError if the binary is not present (e.g. unsupported
    platform or packaging error).
    """
    name = "lobster-tui.exe" if os.name == "nt" else "lobster-tui"
    bin_path = Path(__file__).parent / "bin" / name
    if not bin_path.is_file():
        raise FileNotFoundError(
            f"lobster-tui binary not found at {bin_path}. "
            "This may indicate an unsupported platform or a packaging error."
        )
    return bin_path
