#!/usr/bin/env python3
"""Build platform-specific wheels for lobster-ai-tui.

Usage (CI):
    python build_wheel.py --platform linux-amd64
    python build_wheel.py --platform linux-arm64
    python build_wheel.py --platform darwin-arm64
    python build_wheel.py --platform darwin-amd64

This script:
1. Cross-compiles the Go binary from lobster-tui/ source
2. Places it in lobster_ai_tui/bin/
3. Builds a platform-specific wheel with the correct tags
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Map our platform names to Go env + wheel tags
PLATFORMS = {
    "linux-amd64": {
        "goos": "linux",
        "goarch": "amd64",
        "wheel_plat": "manylinux_2_17_x86_64.manylinux2014_x86_64",
    },
    "linux-arm64": {
        "goos": "linux",
        "goarch": "arm64",
        "wheel_plat": "manylinux_2_17_aarch64.manylinux2014_aarch64",
    },
    "darwin-arm64": {
        "goos": "darwin",
        "goarch": "arm64",
        "wheel_plat": "macosx_11_0_arm64",
    },
    "darwin-amd64": {
        "goos": "darwin",
        "goarch": "amd64",
        "wheel_plat": "macosx_10_16_x86_64",
    },
}

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GO_SOURCE = REPO_ROOT / "lobster-tui"
PKG_DIR = Path(__file__).resolve().parent


def get_version() -> str:
    """Read version from lobster/version.py."""
    version_file = REPO_ROOT / "lobster" / "version.py"
    ns: dict = {}
    exec(version_file.read_text(), ns)
    return ns["__version__"]


def build_go_binary(platform: str, version: str) -> Path:
    """Cross-compile the Go binary and return its path."""
    cfg = PLATFORMS[platform]
    bin_dir = PKG_DIR / "lobster_ai_tui" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    output = bin_dir / "lobster-tui"

    env = os.environ.copy()
    env["GOOS"] = cfg["goos"]
    env["GOARCH"] = cfg["goarch"]
    env["CGO_ENABLED"] = "0"

    ldflags = f"-s -w -X main.Version={version}"

    cmd = [
        "go", "build",
        "-ldflags", ldflags,
        "-trimpath",
        "-o", str(output),
        "./cmd/lobster-tui",
    ]

    print(f"Building lobster-tui for {platform} (v{version})...")
    subprocess.run(cmd, cwd=GO_SOURCE, env=env, check=True)

    # Make executable
    output.chmod(0o755)
    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"  Built: {output} ({size_mb:.1f} MB)")
    return output


def build_wheel(platform: str, version: str) -> Path:
    """Build a platform-specific wheel."""
    cfg = PLATFORMS[platform]
    plat_tag = cfg["wheel_plat"]

    dist_dir = PKG_DIR / "dist"
    dist_dir.mkdir(exist_ok=True)

    # Build the wheel with setuptools, then rename with correct platform tag
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(dist_dir)],
        cwd=PKG_DIR,
        check=True,
    )

    # Find the built wheel and rename it with the platform tag
    for whl in dist_dir.glob("lobster_ai_tui-*.whl"):
        # Replace 'any' with the platform-specific tag
        new_name = whl.name.replace("-any.whl", f"-{plat_tag}.whl")
        new_name = new_name.replace("none-any", f"none-{plat_tag}")
        new_path = whl.parent / new_name
        if new_path != whl:
            shutil.move(str(whl), str(new_path))
        print(f"  Wheel: {new_path.name}")
        return new_path

    raise RuntimeError("No wheel found after build")


def main():
    parser = argparse.ArgumentParser(description="Build lobster-ai-tui platform wheel")
    parser.add_argument(
        "--platform",
        required=True,
        choices=list(PLATFORMS.keys()),
        help="Target platform",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Version override (default: read from lobster/version.py)",
    )
    args = parser.parse_args()

    version = args.version or get_version()

    # 1. Cross-compile Go binary
    build_go_binary(args.platform, version)

    # 2. Build platform wheel
    wheel_path = build_wheel(args.platform, version)

    # 3. Clean up binary (CI will have the wheel)
    bin_path = PKG_DIR / "lobster_ai_tui" / "bin" / "lobster-tui"
    if bin_path.exists():
        bin_path.unlink()

    print(f"\nDone: {wheel_path}")


if __name__ == "__main__":
    main()
