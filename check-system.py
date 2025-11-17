#!/usr/bin/env python3
"""
Lobster AI - System Dependency Checker

This script checks for required system dependencies before installation.
Run this before attempting to install Lobster AI.
"""

import platform
import shutil
import subprocess
import sys
from pathlib import Path


class Colors:
    """ANSI color codes for terminal output"""

    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def print_header(text: str) -> None:
    """Print a formatted header"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BLUE}{Colors.BOLD}{text.center(60)}{Colors.RESET}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'=' * 60}{Colors.RESET}\n")


def print_success(text: str) -> None:
    """Print success message"""
    print(f"{Colors.GREEN}✅ {text}{Colors.RESET}")


def print_warning(text: str) -> None:
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.RESET}")


def print_error(text: str) -> None:
    """Print error message"""
    print(f"{Colors.RED}❌ {text}{Colors.RESET}")


def print_info(text: str) -> None:
    """Print info message"""
    print(f"{Colors.CYAN}ℹ️  {text}{Colors.RESET}")


def check_python_version() -> bool:
    """Check if Python version is 3.12 or higher"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 12:
        print_success(f"Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print_error(f"Python {version.major}.{version.minor}.{version.micro} (3.12+ required)")
        return False


def check_command(cmd: str) -> bool:
    """Check if a command is available"""
    return shutil.which(cmd) is not None


def check_library(lib: str) -> bool:
    """Check if a library is installed (Linux)"""
    try:
        result = subprocess.run(
            ["pkg-config", "--exists", lib], capture_output=True, check=False
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def check_macos() -> tuple[bool, list[str]]:
    """Check macOS dependencies"""
    print(f"{Colors.BOLD}Checking macOS system...{Colors.RESET}\n")

    issues = []
    all_ok = True

    # Check Xcode Command Line Tools
    if check_command("gcc") and check_command("git"):
        print_success("Xcode Command Line Tools installed")
    else:
        print_error("Xcode Command Line Tools not found")
        issues.append("Install with: xcode-select --install")
        all_ok = False

    # Check Homebrew (optional but recommended)
    if check_command("brew"):
        print_success("Homebrew installed")
    else:
        print_warning("Homebrew not found (optional but recommended)")
        issues.append("Install from: https://brew.sh")

    return all_ok, issues


def check_linux() -> tuple[bool, list[str]]:
    """Check Linux dependencies"""
    print(f"{Colors.BOLD}Checking Linux system...{Colors.RESET}\n")

    issues = []
    all_ok = True

    # Required build tools
    required_commands = ["gcc", "g++", "make", "git"]
    missing_commands = [cmd for cmd in required_commands if not check_command(cmd)]

    if not missing_commands:
        print_success("Build tools installed (gcc, g++, make, git)")
    else:
        print_error(f"Missing commands: {', '.join(missing_commands)}")
        issues.append("Install with: sudo apt install build-essential git")
        all_ok = False

    # Check for pkg-config
    if check_command("pkg-config"):
        print_success("pkg-config installed")

        # Check required libraries
        required_libs = {
            "hdf5": "HDF5 library",
            "libxml-2.0": "libxml2",
            "libxslt": "libxslt",
            "libffi": "libffi",
        }

        missing_libs = []
        for lib, name in required_libs.items():
            if check_library(lib):
                print_success(f"{name} found")
            else:
                print_warning(f"{name} not found")
                missing_libs.append(lib)

        if missing_libs:
            print_error(f"Missing development libraries: {', '.join(missing_libs)}")
            issues.append(
                "Install with: sudo apt install libhdf5-dev libxml2-dev libxslt-dev libffi-dev libssl-dev libblas-dev liblapack-dev"
            )
            all_ok = False
    else:
        print_warning("pkg-config not found (cannot check libraries)")
        issues.append("Install with: sudo apt install pkg-config")

    # Check Python development files
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    python_dev = f"python{python_version}-dev"

    # Try to find python-dev package
    try:
        result = subprocess.run(
            ["dpkg", "-l", python_dev], capture_output=True, check=False, text=True
        )
        if "ii" in result.stdout:
            print_success(f"Python development files ({python_dev}) installed")
        else:
            print_warning(f"Python development files not found")
            issues.append(f"Install with: sudo apt install {python_dev}")
    except FileNotFoundError:
        # Not a Debian-based system
        print_info("Cannot verify Python development files (non-Debian system)")

    return all_ok, issues


def check_windows() -> tuple[bool, list[str]]:
    """Check Windows dependencies"""
    print(f"{Colors.BOLD}Checking Windows system...{Colors.RESET}\n")

    issues = []
    all_ok = True

    # Check Git
    if check_command("git"):
        print_success("Git installed")
    else:
        print_warning("Git not found")
        issues.append("Install from: https://git-scm.com/download/win")

    # Check Docker Desktop
    if check_command("docker"):
        print_success("Docker installed")
        print_info(
            "Docker Desktop is the recommended installation method for Windows"
        )
    else:
        print_warning("Docker not found")
        issues.append(
            "For best experience, install Docker Desktop: https://www.docker.com/products/docker-desktop/"
        )

    # Check for Visual Studio Build Tools (optional for native install)
    if check_command("cl"):  # Microsoft C/C++ Compiler
        print_success("Visual Studio Build Tools found")
    else:
        print_warning("Visual Studio Build Tools not found")
        print_info(
            "  Required for native installation if packages lack pre-built wheels"
        )
        issues.append(
            "Download from: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022"
        )
        issues.append("  (Only needed if you encounter compilation errors)")

    print_warning(
        "Native Windows installation is experimental. Docker is recommended."
    )

    return all_ok, issues


def check_docker() -> bool:
    """Check if Docker is available and running"""
    if not check_command("docker"):
        return False

    try:
        result = subprocess.run(
            ["docker", "info"], capture_output=True, check=False, timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def main():
    """Main checker function"""
    print_header("🦞 Lobster AI - System Dependency Checker")

    # Detect platform
    system = platform.system()
    print_info(f"Platform: {system} ({platform.platform()})")
    print_info(f"Python: {sys.version}")
    print()

    # Check Python version
    print(f"{Colors.BOLD}Checking Python version...{Colors.RESET}\n")
    python_ok = check_python_version()
    print()

    if not python_ok:
        print_error("Python 3.12+ is required!")
        print_info("Download from: https://www.python.org/downloads/")
        print()
        sys.exit(1)

    # Platform-specific checks
    all_ok = True
    issues = []

    if system == "Darwin":
        all_ok, issues = check_macos()
    elif system == "Linux":
        all_ok, issues = check_linux()
    elif system == "Windows":
        all_ok, issues = check_windows()
    else:
        print_warning(f"Unknown platform: {system}")
        print_info("You may encounter installation issues.")
        all_ok = False

    # Check Docker availability
    print(f"\n{Colors.BOLD}Checking Docker...{Colors.RESET}\n")
    if check_docker():
        print_success("Docker is installed and running")
        print_info("You can use: docker-compose run --rm lobster-cli")
    else:
        print_warning("Docker not available")
        if system == "Windows":
            print_info("Docker Desktop is strongly recommended for Windows")

    # Summary
    print_header("📋 Summary")

    if all_ok:
        print_success("All required dependencies found!")
        print()
        print(f"{Colors.GREEN}{Colors.BOLD}✓ Ready to install Lobster AI{Colors.RESET}")
        print()
        print("Next steps:")

        if system == "Windows":
            print(f"  {Colors.CYAN}Option 1 (Recommended):{Colors.RESET}")
            print("    docker-compose run --rm lobster-cli")
            print(f"  {Colors.CYAN}Option 2 (Experimental):{Colors.RESET}")
            print("    .\\install.ps1  or  install.bat")
        elif system == "Linux":
            print(f"  {Colors.CYAN}./install-ubuntu.sh{Colors.RESET}  (for Ubuntu/Debian)")
            print(f"  {Colors.CYAN}make install{Colors.RESET}  (if dependencies installed)")
        else:  # macOS
            print(f"  {Colors.CYAN}make install{Colors.RESET}")

        print()
        return 0
    else:
        print_error("Some dependencies are missing!")
        print()
        print("Issues found:")
        for issue in issues:
            print(f"  • {issue}")
        print()

        if system == "Windows":
            print_info("💡 Tip: Use Docker Desktop for the easiest experience:")
            print(
                f"   {Colors.CYAN}docker-compose run --rm lobster-cli{Colors.RESET}\n"
            )

        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Interrupted by user{Colors.RESET}")
        sys.exit(130)
