"""Detection and command building for uv tool environments.

When Lobster is installed via `uv tool install lobster-ai`, the virtualenv
contains a `uv-receipt.toml` at its prefix. This module detects that receipt
and builds `uv tool install` commands that preserve existing --with packages.
"""

import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class UvToolEnvInfo:
    """Information about the current uv tool environment."""

    tool_name: str
    tool_extras: list[str] = field(default_factory=list)
    installed_packages: list[str] = field(default_factory=list)
    prefix: Path = field(default_factory=lambda: Path(sys.prefix))


def detect_uv_tool_env() -> UvToolEnvInfo | None:
    """Detect if running inside a uv tool environment.

    Checks for ``uv-receipt.toml`` at ``sys.prefix``, parses it with
    ``tomllib``, and returns environment info.

    Returns:
        UvToolEnvInfo if inside a uv tool env, None otherwise.
    """
    receipt_path = Path(sys.prefix) / "uv-receipt.toml"
    if not receipt_path.exists():
        return None

    try:
        import tomllib
    except ModuleNotFoundError:
        # Python < 3.11 fallback (shouldn't happen for lobster 3.12+)
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError:
            return None

    try:
        with open(receipt_path, "rb") as f:
            data = tomllib.load(f)
    except Exception:
        return None

    tool_section = data.get("tool", {})
    tool_name = tool_section.get("name", "lobster-ai")

    # Extract installed package names from requirements list
    # Format varies by uv version:
    #   Old: ["lobster-ai==1.0.0", "lobster-transcriptomics>=0.1"]
    #   New: [{ name = "lobster-ai", extras = ["full"] }]
    requirements = tool_section.get("requirements", [])
    packages: list[str] = []
    tool_extras: set[str] = set()
    for req in requirements:
        if isinstance(req, dict):
            # New uv format: {name: "pkg", extras: [...], ...}
            name = req.get("name", "")
            if name == tool_name:
                tool_extras.update(req.get("extras", []) or [])
        else:
            # Old uv format: "pkg==1.0.0" or "pkg>=0.1"
            raw_req = req
            name = raw_req
            if raw_req.startswith(f"{tool_name}[") and "]" in raw_req:
                extras_part = raw_req.split("[", 1)[1].split("]", 1)[0]
                for extra in extras_part.split(","):
                    extra = extra.strip()
                    if extra:
                        tool_extras.add(extra)
            for sep in (">=", "<=", "==", "!=", "~=", ">", "<", "["):
                name = name.split(sep)[0]
        name = name.strip()
        if name:
            packages.append(name)

    return UvToolEnvInfo(
        tool_name=tool_name,
        tool_extras=sorted(tool_extras),
        installed_packages=packages,
        prefix=Path(sys.prefix),
    )


def is_uv_tool_env() -> bool:
    """Check if running inside a uv tool environment."""
    return detect_uv_tool_env() is not None


def build_tool_install_command(
    extras: list[str] | None = None,
    with_packages: list[str] | None = None,
) -> list[str]:
    """Build a ``uv tool install`` command preserving existing packages.

    Reads the current receipt to keep all existing ``--with`` packages,
    then appends any new packages from *with_packages*.

    Args:
        extras: Extras for the main package (e.g. ``["full", "anthropic"]``).
        with_packages: Additional packages to include via ``--with``.

    Returns:
        Command as a list of strings suitable for ``subprocess.run()``.
    """
    uv_path = shutil.which("uv") or "uv"

    info = detect_uv_tool_env()

    merged_extras: set[str] = set(info.tool_extras if info else [])
    if extras:
        merged_extras.update(extras)

    # Start with the base package specifier
    base_pkg = "lobster-ai"
    if merged_extras:
        base_pkg = f"lobster-ai[{','.join(sorted(merged_extras))}]"

    cmd = [uv_path, "tool", "install", base_pkg]

    # Collect existing --with packages from receipt (excluding the main tool)
    existing_with: set[str] = set()
    if info:
        for pkg in info.installed_packages:
            # Normalize for comparison
            norm = pkg.lower().replace("-", "_")
            if norm not in ("lobster_ai", "lobster"):
                existing_with.add(pkg)

    # Add new packages
    if with_packages:
        for pkg in with_packages:
            existing_with.add(pkg)

    # Append --with flags
    for pkg in sorted(existing_with):
        cmd.extend(["--with", pkg])

    return cmd
