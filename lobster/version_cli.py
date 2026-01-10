"""
Ultra-fast version command (bypasses main CLI for <100ms execution).
Installed as standalone 'lobster-version' command.
"""
import sys


def main():
    """Print version and exit immediately (no heavy imports)."""
    # Import only what's needed - fastest possible path
    from lobster.version import __version__
    print(f"lobster version {__version__}")
    sys.exit(0)


if __name__ == "__main__":
    main()
