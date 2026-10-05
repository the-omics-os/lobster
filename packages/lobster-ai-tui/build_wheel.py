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
import ast
import os

# Build tools use explicit argv, never a shell.
import subprocess  # nosec B404
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
    tree = ast.parse(version_file.read_text())
    values = [
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets)
    ]
    if len(values) != 1 or not isinstance(values[0], str):
        raise ValueError("Expected one literal __version__ assignment")
    return values[0]


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
        "go",
        "build",
        "-ldflags",
        ldflags,
        "-trimpath",
        "-o",
        str(output),
        "./cmd/lobster-tui",
    ]

    print(f"Building lobster-tui for {platform} (v{version})...")
    # Local compiler, fixed source/output paths and allowlisted platform; no shell.
    subprocess.run(cmd, cwd=GO_SOURCE, env=env, check=True)  # nosec B603

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

    # Start with an empty output directory: never retag a stale wheel.
    if any(dist_dir.iterdir()):
        raise RuntimeError(f"Wheel output directory is not empty: {dist_dir}")

    # Invoke the current Python's build module with fixed options; no shell.
    subprocess.run(  # nosec B603
        [
            sys.executable,
            "-m",
            "build",
            "--installer",
            "uv",
            "--wheel",
            "--outdir",
            str(dist_dir),
        ],
        cwd=PKG_DIR,
        check=True,
    )

    wheel = dist_dir / f"lobster_ai_tui-{version}-py3-none-any.whl"
    if not wheel.is_file():
        raise RuntimeError(f"Expected wheel for version {version}: {wheel}")
    # The wheel tool rewrites WHEEL tags and RECORD, not just the filename.
    # Retag our just-built wheel through a fixed module/argv, never a shell.
    subprocess.run(  # nosec B603
        [
            sys.executable,
            "-m",
            "wheel",
            "tags",
            "--remove",
            "--platform-tag",
            plat_tag,
            str(wheel),
        ],
        check=True,
    )
    canonical_tag = ".".join(sorted(plat_tag.split(".")))
    result = dist_dir / f"lobster_ai_tui-{version}-py3-none-{canonical_tag}.whl"
    if not result.is_file():
        raise RuntimeError(f"Retagged wheel missing: {result}")
    print(f"  Wheel: {result.name}")
    return result


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
        help="Version to build (must match lobster/version.py)",
    )
    args = parser.parse_args()

    source_version = get_version()
    version = args.version or source_version
    if version != source_version:
        parser.error("--version must match lobster/version.py")

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
