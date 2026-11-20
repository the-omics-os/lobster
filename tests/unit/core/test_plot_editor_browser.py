"""Graphical editor launch must not accept a terminal browser as a GUI."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from lobster.core.plot_editor import browser


@pytest.mark.parametrize("root", ["/Applications", "/Users/Test User/Applications"])
@pytest.mark.parametrize(
    "app", ["Google Chrome", "Chromium", "Microsoft Edge", "Brave Browser", "Vivaldi"]
)
def test_macos_discovers_application_bundles_and_preserves_spaces(
    monkeypatch, root, app
):
    executable = str(Path(root) / (app + ".app") / "Contents/MacOS" / app)
    monkeypatch.setattr(browser.sys, "platform", "darwin")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path("/Users/Test User")))
    monkeypatch.setattr(browser.shutil, "which", lambda name: None)
    monkeypatch.setattr(Path, "is_file", lambda path: str(path) == executable)
    assert browser.chromium_candidates() == [executable]
    process = Mock()
    process.wait.return_value = 0
    popen = Mock(return_value=process)
    monkeypatch.setattr(browser.subprocess, "Popen", popen)
    url = "http://127.0.0.1:1234/?ref=plot#session"
    assert browser.open_browser(url) == "app"
    assert popen.call_args.args[0] == [executable, "--no-first-run", "--app=" + url]


def test_macos_failed_app_launch_falls_back_to_default_browser(monkeypatch):
    monkeypatch.setattr(browser.sys, "platform", "darwin")
    monkeypatch.setattr(
        browser,
        "chromium_candidates",
        lambda: ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"],
    )
    monkeypatch.setattr(
        browser.subprocess, "Popen", Mock(side_effect=OSError("Launch failed"))
    )
    monkeypatch.setattr(browser.shutil, "which", lambda name: None)
    controller = Mock(name="Safari controller")
    controller.name = "Safari"
    controller.open.return_value = True
    monkeypatch.setattr(browser.webbrowser, "get", lambda: controller)
    assert browser.open_browser("http://127.0.0.1:1234/") == "tab"
    controller.open.assert_called_once_with("http://127.0.0.1:1234/", new=2)


def test_macos_no_browser_returns_explanation(monkeypatch):
    monkeypatch.setattr(browser.sys, "platform", "darwin")
    monkeypatch.setattr(browser, "chromium_candidates", lambda: [])
    monkeypatch.setattr(
        browser.webbrowser,
        "get",
        Mock(side_effect=browser.webbrowser.Error("No browser")),
    )
    with pytest.raises(RuntimeError, match="no usable graphical browser"):
        browser.open_browser("http://127.0.0.1:1234/")


@pytest.mark.parametrize("via_environment", [True, False])
def test_wsl_discovers_windows_browser_outside_path(monkeypatch, via_environment):
    executable = "/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
    monkeypatch.setattr(browser.sys, "platform", "linux")
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    if via_environment:
        monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    monkeypatch.setattr(browser.platform, "release", lambda: "microsoft-standard-WSL2")
    monkeypatch.setattr(browser.shutil, "which", lambda name: None)
    monkeypatch.setattr(Path, "is_file", lambda path: str(path) == executable)
    assert browser.chromium_candidates() == [executable]
    process = Mock()
    process.wait.return_value = 0
    popen = Mock(return_value=process)
    monkeypatch.setattr(browser.subprocess, "Popen", popen)
    assert browser.open_browser("http://127.0.0.1:1234/") == "app"
    assert popen.call_args.args[0] == [
        executable,
        "--no-first-run",
        "--app=http://127.0.0.1:1234/",
    ]


@pytest.mark.parametrize("name", ["xdg-open", "gio", "w3m", "www-browser"])
def test_generic_or_text_browser_is_not_reported_as_gui(monkeypatch, name):
    monkeypatch.setattr(browser.sys, "platform", "linux")
    monkeypatch.setattr(browser, "chromium_candidates", lambda: [])
    monkeypatch.setattr(browser.shutil, "which", lambda name: None)
    controller = Mock()
    controller.name = name
    controller.open.return_value = True
    monkeypatch.setattr(browser.webbrowser, "get", lambda: controller)
    with pytest.raises(RuntimeError, match="no usable graphical browser"):
        browser.open_browser("http://127.0.0.1:1234/")
    controller.open.assert_not_called()


def test_firefox_used_directly_instead_of_xdg_open(monkeypatch):
    monkeypatch.setattr(browser.sys, "platform", "linux")
    monkeypatch.setattr(browser, "chromium_candidates", lambda: [])
    monkeypatch.setattr(
        browser.shutil,
        "which",
        lambda name: "/usr/bin/firefox" if name == "firefox" else None,
    )
    generic = Mock()
    generic.name = "xdg-open"
    monkeypatch.setattr(browser.webbrowser, "get", lambda: generic)
    graphical = Mock()
    graphical.open.return_value = True
    factory = Mock(return_value=graphical)
    monkeypatch.setattr(browser.webbrowser, "BackgroundBrowser", factory)
    assert browser.open_browser("http://127.0.0.1:1234/") == "tab"
    generic.open.assert_not_called()
    factory.assert_called_once_with("/usr/bin/firefox")


def test_failed_app_launch_is_not_success(monkeypatch):
    monkeypatch.setattr(browser.sys, "platform", "linux")
    monkeypatch.setattr(browser, "chromium_candidates", lambda: ["/browser"])
    monkeypatch.setattr(browser.shutil, "which", lambda name: None)
    process = Mock()
    process.wait.return_value = 1
    monkeypatch.setattr(browser.subprocess, "Popen", Mock(return_value=process))
    generic = Mock()
    generic.name = "xdg-open"
    monkeypatch.setattr(browser.webbrowser, "get", lambda: generic)
    with pytest.raises(RuntimeError, match="no usable graphical browser"):
        browser.open_browser("http://127.0.0.1:1234/")
