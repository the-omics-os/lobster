"""Find Chromium-family browsers without a fixed executable name or installation."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Iterator, Literal

_WINDOWS_BROWSER_PATHS = (
    "Google/Chrome/Application/chrome.exe",
    "Microsoft/Edge/Application/msedge.exe",
    "BraveSoftware/Brave-Browser/Application/brave.exe",
    "Vivaldi/Application/vivaldi.exe",
)
_GRAPHICAL_LINUX_BROWSERS = (
    "firefox",
    "firefox-esr",
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
    "microsoft-edge",
    "microsoft-edge-stable",
    "brave",
    "brave-browser",
    "vivaldi",
    "epiphany",
)


def chromium_candidates() -> list[str]:
    paths = []
    for name in (
        "google-chrome",
        "google-chrome-stable",
        "chromium",
        "chromium-browser",
        "microsoft-edge",
        "microsoft-edge-stable",
        "brave-browser",
        "brave",
        "vivaldi",
        "chrome",
        "chrome.exe",
        "msedge.exe",
        "brave.exe",
    ):
        resolved = shutil.which(name)
        if resolved:
            paths.append(resolved)
    if sys.platform == "darwin":
        for root in (Path("/Applications"), Path.home() / "Applications"):
            for app in (
                "Google Chrome",
                "Chromium",
                "Microsoft Edge",
                "Brave Browser",
                "Vivaldi",
            ):
                paths.append(str(root / (app + ".app") / "Contents/MacOS" / app))
    if sys.platform == "win32":
        import winreg

        for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
            for view in (winreg.KEY_WOW64_64KEY, winreg.KEY_WOW64_32KEY):
                for exe in ("chrome.exe", "msedge.exe", "brave.exe", "vivaldi.exe"):
                    try:
                        with winreg.OpenKey(
                            hive,
                            "Software\\Microsoft\\Windows\\CurrentVersion\\App Paths\\"
                            + exe,
                            0,
                            winreg.KEY_READ | view,
                        ) as key:
                            paths.append(winreg.QueryValue(key, None).strip('"'))
                    except OSError:
                        pass
        for env in ("LOCALAPPDATA", "PROGRAMFILES", "PROGRAMFILES(X86)"):
            if os.environ.get(env):
                for relative in _WINDOWS_BROWSER_PATHS:
                    paths.append(str(Path(os.environ[env]) / relative))
    if sys.platform == "linux" and (
        os.environ.get("WSL_DISTRO_NAME") or "microsoft" in platform.release().lower()
    ):
        # WSL is Linux to Python, so the Windows registry branch above never runs.
        # Windows browsers are commonly installed outside the inherited PATH.
        for root in (Path("/mnt/c/Program Files"), Path("/mnt/c/Program Files (x86)")):
            paths.extend(str(root / relative) for relative in _WINDOWS_BROWSER_PATHS)
    return list(dict.fromkeys(p for p in paths if Path(p).is_file()))


def _graphical_controllers() -> Iterator[webbrowser.BaseBrowser]:
    try:
        controller = webbrowser.get()
        name = Path(str(getattr(controller, "name", ""))).name.lower()
        resolved = shutil.which(str(getattr(controller, "name", "")))
        target = Path(resolved).resolve().name.lower() if resolved else name
        # Generic Linux openers can silently fall back to a text browser. Only
        # invoke a known graphical browser directly, including resolved aliases.
        if sys.platform.startswith("linux"):
            if name in _GRAPHICAL_LINUX_BROWSERS or target in _GRAPHICAL_LINUX_BROWSERS:
                yield controller
        elif target not in ("lynx", "links", "links2", "w3m", "elinks"):
            yield controller
    except (webbrowser.Error, OSError):
        pass
    if sys.platform.startswith("linux"):
        for name in _GRAPHICAL_LINUX_BROWSERS:
            executable = shutil.which(name)
            if executable:
                yield webbrowser.BackgroundBrowser(executable)


def open_browser(url: str) -> Literal["app", "tab"]:
    """Best-effort launch, returning app/tab. Do not kill a shared browser process."""
    for executable in chromium_candidates():
        try:
            process = subprocess.Popen(
                [executable, "--no-first-run", "--app=" + url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                if process.wait(timeout=0.4) != 0:
                    continue
            except subprocess.TimeoutExpired:
                pass
            return "app"
        except OSError:
            continue
    for controller in _graphical_controllers():
        try:
            if controller.open(url, new=2):
                return "tab"
        except (webbrowser.Error, OSError):
            continue
    raise RuntimeError(
        "Editor could not launch: no usable graphical browser was found. "
        "Install Chrome, Chromium, Edge, Brave or Vivaldi, or configure a "
        "modern default browser. A graphical desktop on the Lobster computer is required."
    )
