"""``lobster init [folder]`` makes the folder a project and keeps its identity."""

import json
import os
import re

import pytest
import typer
from typer.testing import CliRunner

LOBSTER = "/opt/lobster/bin/lobster"
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
KEY = "sk-ant-synthetic-test-key"  # nosec B105 - synthetic value


@pytest.fixture
def dirs(monkeypatch, tmp_path):
    from lobster.config import credentials, global_config

    home = tmp_path / "home"
    home.mkdir()
    work = home / "work"
    work.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(work)
    monkeypatch.setattr(global_config, "CONFIG_DIR", tmp_path / "global")
    monkeypatch.setattr(credentials, "CREDENTIALS_DIR", tmp_path / "creds")
    monkeypatch.setattr(
        credentials, "CREDENTIALS_FILE", tmp_path / "creds" / "credentials.json"
    )
    monkeypatch.setattr("lobster.core.uv_tool_env.is_uv_tool_env", lambda: False)
    monkeypatch.setattr(
        "lobster.cli_internal.provider_hint.lobster_path", lambda: LOBSTER
    )
    for name in (
        "OMICS_OS_API_KEY",
        "OMICS_OS_ENDPOINT",
        "LOBSTER_LLM_PROVIDER",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "LOBSTER_WORKSPACE",
    ):
        monkeypatch.delenv(name, raising=False)
    return work


class _Plain:
    """A CliRunner result whose text has no colour codes (Rich styles output when
    FORCE_COLOR or GITHUB_ACTIONS is set), so asserts read the same everywhere."""

    def __init__(self, result):
        self._result = result
        self.output = ANSI.sub("", result.output)
        self.stdout = ANSI.sub("", result.stdout)
        self.stderr = ANSI.sub("", result.stderr) if result.stderr_bytes else ""

    def __getattr__(self, name):
        return getattr(self._result, name)


def invoke(app, args):
    return _Plain(CliRunner().invoke(app, args))


def run_init(*extra):
    from lobster.cli import app

    return invoke(
        app,
        [
            "init",
            "--non-interactive",
            "--skip-extras",
            "--skip-ssl-test",
            "--anthropic-key",
            KEY,
            *extra,
        ],
    )


def identity(folder):
    return json.loads((folder / ".lobster_workspace" / "project.json").read_text())


def drop_backups(folder):
    # two forced runs in one second would write the same backup name
    for path in folder.rglob("*.backup.*"):
        path.unlink()


def test_init_folder_creates_project_without_changing_cwd(dirs):
    before = os.getcwd()
    result = run_init(str(dirs / "study"))
    assert result.exit_code == 0, result.output
    assert os.getcwd() == before
    study = dirs / "study"
    assert identity(study)["name"] == "study"
    assert (study / ".lobster_workspace" / "provider_config.json").is_file()
    assert (study / ".env").is_file()
    assert not (dirs / ".env").exists()
    assert not (dirs / ".lobster_workspace").exists()
    assert "Project study created" in result.output


def test_init_in_current_folder_creates_project(dirs):
    result = run_init()
    assert result.exit_code == 0, result.output
    assert identity(dirs)["name"] == "work"


def test_init_again_keeps_identity_and_renames_on_request(dirs):
    assert run_init().exit_code == 0
    first = identity(dirs)
    drop_backups(dirs)
    again = run_init("--force")
    assert again.exit_code == 0, again.output
    assert identity(dirs) == first
    assert "kept" in again.output
    drop_backups(dirs)
    renamed = run_init("--force", "--name", "Other")
    assert renamed.exit_code == 0, renamed.output
    after = identity(dirs)
    assert after["project_id"] == first["project_id"]
    assert after["name"] == "Other"


def test_init_inside_a_project_is_refused(dirs):
    assert run_init().exit_code == 0
    parent_config = (dirs / ".lobster_workspace" / "provider_config.json").read_text()
    sub = dirs / "sub"
    sub.mkdir()
    for argv in ((str(sub),), (str(dirs / "new" / "deeper"),)):
        result = run_init(*argv)
        assert result.exit_code == 2, result.output
        assert "projects do not nest" in result.output
        assert str(dirs) in result.output
    os.chdir(sub)
    assert run_init().exit_code == 2
    assert not (sub / ".lobster_workspace").exists()
    assert not (sub / ".env").exists()
    assert not (dirs / "new").exists()
    assert (
        dirs / ".lobster_workspace" / "provider_config.json"
    ).read_text() == parent_config


def test_init_in_a_legacy_workspace_under_a_project_is_refused(dirs):
    assert run_init().exit_code == 0
    (dirs / "b" / ".lobster_workspace").mkdir(parents=True)
    assert run_init(str(dirs / "b")).exit_code == 2


def test_init_adopts_existing_configuration_without_force(dirs):
    ws = dirs / ".lobster_workspace"
    ws.mkdir()
    (ws / "provider_config.json").write_text('{"global_provider": "ollama"}')
    (dirs / ".env").write_text("X=1\n")
    result = run_init()
    assert result.exit_code == 0, result.output
    assert identity(dirs)["name"] == "work"
    assert (ws / "provider_config.json").read_text() == '{"global_provider": "ollama"}'


def test_global_init_makes_no_project_and_takes_no_folder(dirs):
    result = run_init("--global")
    assert result.exit_code == 0, result.output
    assert not (dirs / ".lobster_workspace" / "project.json").exists()
    refused = run_init("--global", str(dirs / "x"))
    assert refused.exit_code == 2
    assert not (dirs / "x").exists()
    refused = run_init("--global", "--name", "n")
    assert refused.exit_code == 2


def test_folder_that_is_a_file_is_refused(dirs):
    (dirs / "f").write_text("x")
    result = run_init(str(dirs / "f"))
    assert result.exit_code == 2
    assert (dirs / "f").read_text() == "x"


@pytest.mark.parametrize("name", ["", "x" * 257])
def test_bad_name_is_refused_before_anything_is_written(dirs, name):
    result = run_init(str(dirs / "study"), "--name", name)
    assert result.exit_code == 2
    assert not (dirs / "study").exists()


def test_env_workspace_without_folder_is_kept_and_is_no_project(dirs, monkeypatch):
    elsewhere = dirs.parent / "elsewhere"
    monkeypatch.setenv("LOBSTER_WORKSPACE", str(elsewhere))
    result = run_init()
    assert result.exit_code == 0, result.output
    assert (elsewhere / "provider_config.json").is_file()
    assert not (elsewhere / "project.json").exists()
    assert not (dirs / ".lobster_workspace").exists()


def test_folder_beats_env_workspace(dirs, monkeypatch):
    monkeypatch.setenv("LOBSTER_WORKSPACE", str(dirs.parent / "elsewhere"))
    result = run_init(str(dirs / "study"))
    assert result.exit_code == 0, result.output
    assert identity(dirs / "study")["name"] == "study"
    assert not (dirs.parent / "elsewhere").exists()


def test_failed_init_makes_no_project(dirs):
    from lobster.cli import app

    result = invoke(
        app, ["init", str(dirs / "study"), "--non-interactive", "--skip-extras"]
    )
    assert result.exit_code != 0
    assert not (dirs / "study" / ".lobster_workspace" / "project.json").exists()


def test_help_names_the_folder_and_name(dirs):
    from lobster.cli import app

    out = invoke(app, ["init", "--help"]).output
    assert "FOLDER" in out.upper()
    assert "--name" in out


def _wizard(monkeypatch, result):
    monkeypatch.setattr(
        "sys.stdin", type("FakeTTY", (), {"isatty": lambda self: True})()
    )
    monkeypatch.setattr(
        "lobster.ui.bridge.binary_finder.find_tui_binary", lambda: "/fake/tui"
    )
    monkeypatch.setattr(
        "lobster.ui.bridge.go_tui_bridge.run_init_wizard", lambda *a, **k: result
    )


def _init_impl(folder):
    import typer

    from lobster.cli_internal.commands.heavy import init_commands

    kwargs = dict(
        global_config=False,
        force=True,
        non_interactive=False,
        anthropic_key=None,
        bedrock_access_key=None,
        bedrock_secret_key=None,
        use_ollama=False,
        ollama_model=None,
        gemini_key=None,
        openai_key=None,
        profile=None,
        ncbi_key=None,
        cloud_key=None,
        cloud_endpoint=None,
        skip_ssl_test=False,
        ssl_verify=None,
        ssl_cert_path=None,
        agents=None,
        preset=None,
        auto_agents=False,
        agents_description=None,
        skip_docling=False,
        install_docling=False,
        install_vector_search=False,
        skip_extras=True,
        ui_mode="go",
        folder=folder,
    )
    with pytest.raises(typer.Exit) as exc:
        init_commands.init_impl(**kwargs)
    return exc.value.exit_code


def test_cancelled_wizard_prints_setup_cancelled_exits_0_and_makes_no_project(
    dirs, monkeypatch, capsys
):
    from rich.console import Console

    from lobster.cli_internal.commands.heavy import init_commands

    console = Console(width=200, record=True)
    monkeypatch.setattr(init_commands, "console", console)
    _wizard(monkeypatch, {"cancelled": True})
    assert _init_impl(dirs / "study") == 0
    assert "Setup cancelled." in console.export_text()
    assert not (dirs / "study" / ".lobster_workspace" / "project.json").exists()
    assert not (dirs / "study" / ".env").exists()


def test_finished_wizard_creates_the_project_in_the_named_folder(dirs, monkeypatch):
    _wizard(
        monkeypatch,
        {
            "provider": "anthropic",
            "api_key": KEY,
            "profile": "production",
            "model_id": "claude-test-model",
            "agents": [],
        },
    )
    assert _init_impl(dirs / "study") == 0
    assert identity(dirs / "study")["name"] == "study"
    assert (dirs / "study" / ".env").is_file()
    assert (dirs / "study" / ".lobster_workspace" / "provider_config.json").is_file()
    assert not (dirs / ".env").exists()


def test_project_env_is_loaded_when_the_run_starts_in_a_subfolder(dirs, monkeypatch):
    from lobster.config import settings as settings_module
    from lobster.config.settings import settings
    from lobster.core.workspace import resolve_workspace

    assert run_init().exit_code == 0
    sub = dirs / "sub"
    sub.mkdir()
    monkeypatch.chdir(sub)
    # The loader's path-less fallback (find_dotenv) walks up from cwd while a
    # tracer such as coverage is active and would find <project>/.env by luck;
    # drop it so only the rules the loader names can supply the key.
    real = settings_module.load_dotenv

    def only_with_a_path(*args, **kwargs):
        # A call with neither a positional path nor dotenv_path= is the fallback.
        if args or kwargs.get("dotenv_path"):
            return real(*args, **kwargs)
        return False

    monkeypatch.setattr(settings_module, "load_dotenv", only_with_a_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "placeholder")
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    settings.reload_credentials(resolve_workspace())
    assert os.environ.get("ANTHROPIC_API_KEY") == KEY


def test_path_errors_are_not_wrapped_however_long_the_path(dirs):
    deep = dirs / ("d" * 130)
    deep.mkdir()
    assert run_init(str(deep)).exit_code == 0
    inner = deep / "sub"
    result = run_init(str(inner))
    assert result.exit_code == 2, result.output
    assert f"{inner} is inside the project" in result.output
    assert f"at {deep}; projects" in result.output


# review fixes ------------------------------------------------------------------


def test_init_refuses_the_home_folder_and_its_parents(dirs):
    home = dirs.parent
    for target in (home, home.parent):
        result = run_init(str(target))
        assert result.exit_code == 2, result.output
        assert "make a subfolder" in result.output
        assert "lobster init ~/my-study" in result.output
        assert not (target / ".lobster_workspace").exists()
    os.chdir(home)
    assert run_init().exit_code == 2
    assert not (home / ".lobster_workspace").exists()
    assert not (home / ".env").exists()


def test_the_nested_refusal_names_the_way_out(dirs):
    assert run_init().exit_code == 0
    result = run_init(str(dirs / "sub"))
    assert result.exit_code == 2
    assert f"lobster projects delete {dirs}" in result.output.replace("\n", "")


@pytest.mark.parametrize("bad", ["a\x1b[31mb", "a‮b", "a​b"])
def test_init_refuses_names_with_control_or_invisible_characters(dirs, bad):
    result = run_init(str(dirs / "study"), "--name", bad)
    assert result.exit_code == 2, result.output
    assert not (dirs / "study").exists()


def test_a_cancelled_wizard_removes_what_this_call_made(dirs, monkeypatch):
    _wizard(monkeypatch, {"cancelled": True})
    assert _init_impl(dirs / "new" / "study") == 0
    assert not (dirs / "new" / "study").exists()
    assert (dirs / "new").is_dir() or not (dirs / "new").exists()
    existing = dirs / "kept"
    existing.mkdir()
    (existing / "mine.txt").write_text("x")
    assert _init_impl(existing) == 0
    assert (existing / "mine.txt").read_text() == "x"
    assert not (existing / ".lobster_workspace").exists()


def test_a_cancelled_wizard_leaves_an_existing_workspace_alone(dirs, monkeypatch):
    legacy = dirs / "old"
    (legacy / ".lobster_workspace" / "data").mkdir(parents=True)
    (legacy / ".lobster_workspace" / "data" / "a.csv").write_text("1")
    _wizard(monkeypatch, {"cancelled": True})
    assert _init_impl(legacy) == 0
    assert (legacy / ".lobster_workspace" / "data" / "a.csv").read_text() == "1"


def test_init_refuses_a_linked_workspace_and_writes_nothing_through_it(dirs, tmp_path):
    target = tmp_path / "elsewhere"
    target.mkdir()
    folder = dirs / "sl"
    folder.mkdir()
    os.symlink(target, folder / ".lobster_workspace")
    result = run_init(str(folder))
    assert result.exit_code == 2, result.output
    assert list(target.iterdir()) == []
    assert not (folder / ".env").exists()


def test_the_warning_after_a_refused_project_is_not_wrapped(dirs, monkeypatch):
    from rich.console import Console

    from lobster.cli_internal.commands.heavy import init_commands
    from lobster.core import projects

    folder = dirs / ("w" * 120)
    (folder / ".lobster_workspace").mkdir(parents=True)
    (folder / ".lobster_workspace" / "provider_config.json").write_text("{}")
    console = Console(width=60, record=True)
    monkeypatch.setattr(init_commands, "console", console)

    def refuse(*a, **k):
        raise projects.ProjectError("nested_project", f"{folder} is nested")

    monkeypatch.setattr(projects, "init_project", refuse)
    init_commands._ensure_project(folder, None)
    assert f"{folder} is nested" in console.export_text()


def test_init_records_the_env_file_it_wrote(dirs):
    assert run_init(str(dirs / "study")).exit_code == 0
    record = json.loads(
        (dirs / "study" / ".lobster_workspace" / "created_paths.json").read_text()
    )
    assert record["files"] == [".env"]


def test_an_oserror_from_making_the_project_is_not_swallowed(dirs, monkeypatch):
    from lobster.cli_internal.commands.heavy import init_commands

    folder = dirs / "study"
    assert run_init(str(folder)).exit_code == 0
    (folder / ".lobster_workspace" / "project.json").unlink()

    def boom(*a, **k):
        raise PermissionError("read-only folder")

    monkeypatch.setattr("lobster.core.projects.init_project", boom)
    with pytest.raises(PermissionError):
        init_commands._ensure_project(folder, None, None)


def test_an_oserror_while_recording_the_env_does_not_stop_init(dirs, monkeypatch):
    def boom(*a, **k):
        raise OSError("cannot record")

    monkeypatch.setattr("lobster.core.projects.record_created_file", boom)
    result = run_init(str(dirs / "study"))
    assert result.exit_code == 0
    assert (dirs / "study" / ".lobster_workspace" / "project.json").is_file()


# init without a terminal and without a key flag -------------------------------------


def run_plain(*args):
    """``lobster init <args>`` with no terminal on stdin (CliRunner) and no key flag."""
    from lobster.cli import app

    return invoke(app, ["init", *args])


def global_setup(provider="anthropic"):
    from lobster.config.global_config import GlobalProviderConfig

    config = GlobalProviderConfig.load()
    config.default_provider = provider
    config.save()
    return GlobalProviderConfig.get_config_path()


@pytest.fixture
def installed(monkeypatch):
    names = {"research_agent": object(), "data_expert_agent": object()}
    monkeypatch.setattr(
        "lobster.cli_internal.commands.heavy.init_commands._get_installed_agents",
        lambda: names,
    )

    def no_install(*a, **k):
        raise AssertionError("init without a terminal must not install anything")

    monkeypatch.setattr(
        "lobster.cli_internal.commands.heavy.init_commands._check_and_prompt_install_packages",
        no_install,
    )
    return names


def files_in(folder):
    return sorted(
        p.name for p in (folder / ".lobster_workspace").iterdir() if p.is_file()
    )


def test_no_setup_refuses_with_a_code_and_changes_nothing(dirs):
    result = run_plain(str(dirs / "study"))
    assert result.exit_code == 1
    assert not (dirs / "study").exists()
    text = " ".join(result.output.split())
    assert text.startswith("✗ no_provider_setup:")
    assert f"In a terminal: {LOBSTER} init {dirs / 'study'}" in text
    assert f"{LOBSTER} init --help" in text and "--global" not in text
    assert "--provider" not in text


def test_the_refusal_names_the_providers_whose_key_is_in_the_shell(dirs, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-planted-value-1234")
    result = run_plain(str(dirs / "study"))
    assert result.exit_code == 1
    text = " ".join(result.output.split())
    assert text.startswith("✗ no_provider_setup:")
    assert "OPENAI_API_KEY is set in the environment for openai" in text
    assert f"{LOBSTER} init {dirs / 'study'} --provider openai" in text
    assert "sk-planted-value-1234" not in result.output
    assert not (dirs / "study").exists()


def test_a_global_setup_gives_the_project_identity_only(dirs, monkeypatch):
    path = global_setup()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-never-printed")
    result = run_plain(str(dirs / "study"))
    assert result.exit_code == 0, result.output
    study = dirs / "study"
    assert files_in(study) == ["project.json"]
    assert not (study / ".env").exists()
    text = " ".join(result.output.split())
    assert "Project study created" in text
    assert f"provider: anthropic (not set in this project; from {path})" in text
    assert "provider default" in text
    assert "every installed expert (no list set for this project)" in text
    assert "in a terminal to give this project its own settings" in text
    assert "sk-ant-never-printed" not in result.output


def test_a_provider_in_the_shell_is_written_into_the_project(dirs, monkeypatch):
    monkeypatch.setenv("LOBSTER_LLM_PROVIDER", "openai")
    result = run_plain(str(dirs / "study"))
    assert result.exit_code == 0, result.output
    assert "provider: openai (set for this project, taken from the shell)" in " ".join(
        result.output.split()
    )
    assert files_in(dirs / "study") == [
        "created_paths.json",
        "project.json",
        "provider_config.json",
    ] or files_in(dirs / "study") == ["project.json", "provider_config.json"]
    monkeypatch.delenv("LOBSTER_LLM_PROVIDER")
    from lobster.core import projects

    provider, _ = projects.effective_settings(dirs / "study" / ".lobster_workspace")
    assert (provider.value, provider.source) == ("openai", "project")


def test_provider_flag_names_the_provider_without_any_setup(dirs, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-planted-value-1234")
    result = run_plain(str(dirs / "study"), "--provider", "openai", "--model", "gpt-x")
    assert result.exit_code == 0, result.output
    config = json.loads(
        (dirs / "study" / ".lobster_workspace" / "provider_config.json").read_text()
    )
    assert config["global_provider"] == "openai" and config["openai_model"] == "gpt-x"
    assert "provider: openai (set for this project, taken from --provider)" in " ".join(
        result.output.split()
    )


def test_provider_flag_beats_the_global_file(dirs):
    global_setup("anthropic")
    assert run_plain(str(dirs / "study"), "--provider", "gemini").exit_code == 0
    config = json.loads(
        (dirs / "study" / ".lobster_workspace" / "provider_config.json").read_text()
    )
    assert config["global_provider"] == "gemini"


@pytest.mark.parametrize(
    "args, code",
    [
        (("--provider", "nope"), 2),
        (("--provider", "openai", "--anthropic-key", KEY), 2),
        (("--provider", "openai", "--use-ollama"), 2),
        (("--provider", "openai", "--global"), 2),
        (("--provider", "openai", "--non-interactive"), 2),
    ],
)
def test_provider_flag_refusals_change_nothing(dirs, args, code):
    result = run_plain(str(dirs / "study"), *args)
    assert result.exit_code == code, result.output
    assert not (dirs / "study").exists()
    if "nope" in args:
        assert "anthropic" in result.output and "ollama-cloud" in result.output


def test_provider_flag_needs_a_folder(dirs):
    result = run_plain("--provider", "openai")
    assert result.exit_code == 2


def test_no_secret_lands_under_the_project_or_in_the_output(
    dirs, installed, monkeypatch
):
    value = "sk-planted-never-copy-9f3a"
    monkeypatch.setenv("ANTHROPIC_API_KEY", value)
    monkeypatch.setenv("OPENAI_API_KEY", value)
    from lobster.config.global_config import save_global_credentials

    global_setup("anthropic")
    save_global_credentials({"ANTHROPIC_API_KEY": value})
    study = dirs / "study"
    for extra in (
        (),
        ("--model", "m1", "--agents", "research_agent"),
        ("--provider", "openai", "--model", "m2"),
    ):
        result = run_plain(str(study), *extra)
        assert result.exit_code == 0, result.output
        assert value not in result.output
        assert value not in (result.stderr if result.stderr_bytes else "")
    assert not (study / ".env").exists()
    for path in study.rglob("*"):
        assert value not in str(path)
        if path.is_file():
            assert value not in path.read_text(errors="ignore"), path


def test_an_existing_workspace_init_did_not_make_is_refused(dirs):
    global_setup()
    study = dirs / "study"
    (study / ".lobster_workspace").mkdir(parents=True)
    (study / ".lobster_workspace" / "notes.txt").write_text("mine")
    result = run_plain(str(study), "--model", "m1")
    assert result.exit_code == 1
    assert (
        "bad_argument" in result.output
        and "not a project made by lobster init" in " ".join(result.output.split())
    )
    assert files_in(study) == ["notes.txt"]
    link = dirs / "linked"
    (dirs / "elsewhere").mkdir()
    link.mkdir()
    os.symlink(dirs / "elsewhere", link / ".lobster_workspace")
    assert run_plain(str(link)).exit_code == 2
    assert os.listdir(dirs / "elsewhere") == []


def test_model_is_written_to_the_project_and_not_checked(dirs):
    global_setup()
    result = run_plain(str(dirs / "study"), "--model", "my-model-1")
    assert result.exit_code == 0, result.output
    config = json.loads(
        (dirs / "study" / ".lobster_workspace" / "provider_config.json").read_text()
    )
    assert config["global_provider"] == "anthropic"
    assert config["anthropic_model"] == "my-model-1"
    text = " ".join(result.output.split())
    assert "model: my-model-1 (set for this project; not checked)" in text
    assert "provider: anthropic (set for this project, taken from" in text
    assert "now fixed to that provider and model" in text


def test_agents_are_written_and_uninstalled_ones_are_reported(dirs, installed):
    global_setup()
    result = run_plain(
        str(dirs / "study"),
        "--agents",
        "research_agent, genomics_expert, not_an_expert",
    )
    assert result.exit_code == 0, result.output
    from lobster.config.workspace_agent_config import WorkspaceAgentConfig

    config = WorkspaceAgentConfig.load(dirs / "study" / ".lobster_workspace")
    assert config.enabled_agents == ["research_agent"]
    text = " ".join(result.output.split())
    assert "experts: research_agent (1 of 3 experts set)" in text
    assert (
        f"genomics_expert: not installed; run {LOBSTER} agents install lobster-genomics"
        in text
    )
    assert "not_an_expert: unknown expert" in text


def test_no_installed_expert_in_the_list_changes_nothing(dirs, installed):
    global_setup()
    result = run_plain(str(dirs / "study"), "--agents", "not_an_expert")
    assert result.exit_code == 1
    assert not (dirs / "study").exists()


def test_a_bad_preset_and_auto_agents_change_nothing(dirs, installed):
    global_setup()
    assert run_plain(str(dirs / "a"), "--preset", "nope").exit_code == 1
    assert run_plain(str(dirs / "b"), "--auto-agents").exit_code == 2
    assert not (dirs / "a").exists() and not (dirs / "b").exists()


def test_the_usual_refusals_come_before_anything_is_written(dirs, monkeypatch):
    global_setup()
    assert run_plain(str(dirs.parent)).exit_code == 2  # the home folder
    assert run_plain(str(dirs / "study")).exit_code == 0
    nested = run_plain(str(dirs / "study" / "inner"))
    assert nested.exit_code == 2
    assert not (dirs / "study" / "inner").exists()


def test_a_rerun_updates_only_the_settings_it_was_given(dirs, installed):
    global_setup()
    study = dirs / "study"
    assert run_plain(str(study), "--model", "m1").exit_code == 0
    first = identity(study)
    ws = study / ".lobster_workspace"
    project_bytes = (ws / "project.json").read_bytes()
    result = run_plain(str(study), "--model", "m2", "--agents", "research_agent")
    assert result.exit_code == 0, result.output
    config = json.loads((ws / "provider_config.json").read_text())
    assert (
        config["anthropic_model"] == "m2" and config["global_provider"] == "anthropic"
    )
    assert (ws / "project.json").read_bytes() == project_bytes
    assert identity(study)["project_id"] == first["project_id"]
    assert not (study / ".env").exists()


def test_a_rerun_without_flags_changes_nothing(dirs):
    global_setup()
    study = dirs / "study"
    assert run_plain(str(study), "--model", "m1").exit_code == 0
    snapshot = {p: p.read_bytes() for p in study.rglob("*") if p.is_file()}
    result = run_plain(str(study))
    assert result.exit_code == 0
    assert "Already a project" in " ".join(result.output.split())
    assert {p: p.read_bytes() for p in study.rglob("*") if p.is_file()} == snapshot


def test_running_it_again_keeps_the_project(dirs):
    global_setup()
    first = run_plain(str(dirs / "study"))
    again = run_plain(str(dirs / "study"))
    assert again.exit_code == 0
    assert "Already a project" in " ".join(again.output.split())
    assert identity(dirs / "study")["project_id"] in json.dumps(
        json.loads((dirs / "study" / ".lobster_workspace" / "project.json").read_text())
    )
    assert first.exit_code == 0


def test_a_key_flag_without_non_interactive_and_global_keep_the_old_refusal(dirs):
    glob = run_plain("--global")
    assert glob.exit_code == 1 and "stdin is not a terminal" in glob.output
    keyed = run_plain(str(dirs / "study"), "--anthropic-key", KEY)
    assert keyed.exit_code == 1 and "stdin is not a terminal" in keyed.output
    assert not (dirs / "study").exists()


def test_a_failed_settings_write_leaves_nothing_behind(dirs, installed, monkeypatch):
    global_setup()

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(
        "lobster.config.workspace_agent_config.WorkspaceAgentConfig.save", boom
    )
    result = run_plain(str(dirs / "study"), "--agents", "research_agent")
    assert result.exit_code == 1
    assert not (dirs / "study").exists()


def test_projects_info_says_what_a_run_will_use(dirs):
    from lobster.cli import app

    path = global_setup()
    assert run_plain(str(dirs / "study")).exit_code == 0
    shown = " ".join(
        invoke(app, ["projects", "info", str(dirs / "study")]).output.split()
    )
    assert f"a run uses: provider anthropic (from {path})" in shown
    doc = json.loads(
        invoke(app, ["projects", "info", str(dirs / "study"), "--json"]).output
    )
    assert doc["local"]["provider"] is None and doc["local"]["model"] is None
    assert doc["local"]["experts"] == []


def test_a_project_made_this_way_can_be_deleted(dirs):
    from lobster.cli import app

    global_setup()
    assert run_plain(str(dirs / "study")).exit_code == 0
    dry = invoke(app, ["projects", "delete", str(dirs / "study"), "--dry-run"])
    assert dry.exit_code == 0, dry.output


@pytest.mark.parametrize(
    "args",
    [
        ("--agents", "not_an_expert"),
        ("--preset", "nope"),
        ("--auto-agents",),
        ("--model", " "),
        ("--agents", ","),
    ],
)
def test_every_refusal_leaves_the_folder_as_it_was(dirs, installed, args):
    global_setup()
    study = dirs / "study"
    study.mkdir()
    (study / "keep.txt").write_text("x")
    before = sorted(p.name for p in study.rglob("*"))
    result = run_plain(str(study), *args)
    assert result.exit_code in (1, 2), result.output
    assert sorted(p.name for p in study.rglob("*")) == before


def test_every_printed_command_names_the_running_executable(dirs, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    refused = " ".join(run_plain(str(dirs / "study")).output.split())
    assert f"{LOBSTER} init {dirs / 'study'} --provider openai" in refused
    assert " lobster init" not in refused.replace("by lobster init", "")
    monkeypatch.delenv("OPENAI_API_KEY")
    foreign = dirs / "foreign"
    (foreign / ".lobster_workspace").mkdir(parents=True)
    text = " ".join(run_plain(str(foreign)).output.split())
    assert f"In a terminal: {LOBSTER} init {foreign}" in text


def test_the_refusal_repeats_the_callers_flags_with_the_provider(dirs, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    result = run_plain(
        str(dirs / "study"),
        "--model",
        "gpt-x",
        "--agents",
        "research_agent,data_expert_agent",
    )
    assert result.exit_code == 1
    text = " ".join(result.output.split())
    assert (
        f"{LOBSTER} init {dirs / 'study'} --provider openai --model gpt-x "
        "--agents research_agent,data_expert_agent"
    ) in text
    preset = " ".join(
        run_plain(str(dirs / "study"), "--preset", "scrna-basic").output.split()
    )
    assert "--provider openai --preset scrna-basic" in preset


def test_the_terminal_line_closes_only_an_identity_only_project(dirs, installed):
    global_setup()
    plain = " ".join(run_plain(str(dirs / "a")).output.split())
    assert "in a terminal to give this project its own settings" in plain
    for extra in (("--model", "m1"), ("--agents", "research_agent")):
        out = " ".join(run_plain(str(dirs / "b"), *extra).output.split())
        assert "its own settings" not in out, out


def test_a_rerun_with_only_a_model_reports_the_kept_expert_list(dirs, installed):
    global_setup()
    study = dirs / "study"
    assert run_plain(str(study), "--agents", "research_agent").exit_code == 0
    out = " ".join(run_plain(str(study), "--model", "m2").output.split())
    assert "every installed expert" not in out
    assert "experts: research_agent (set for this project)" in out


def test_a_rerun_does_not_say_a_setting_is_not_set_and_from_this_project(
    dirs, installed
):
    global_setup()
    study = dirs / "study"
    assert run_plain(str(study), "--model", "m1").exit_code == 0
    out = " ".join(run_plain(str(study), "--agents", "research_agent").output.split())
    assert "not set in this project; from this project" not in out
    assert "provider: anthropic (set for this project)" in out
    assert "model: m1 (set for this project)" in out


@pytest.mark.parametrize(
    "extra",
    [
        ("--ncbi-key", "K"),
        ("--profile", "development"),
        ("--ollama-model", "llama"),
        ("--cloud-endpoint", "https://x.example"),
        ("--install-docling",),
        ("--install-vector-search",),
        ("--agents-description", "an agent"),
        ("--no-ssl-verify",),
        ("--ssl-verify",),
        ("--ssl-cert-path", "/tmp/ca.pem"),
        ("--skip-ssl-test",),
    ],
)
def test_an_option_the_key_free_path_cannot_apply_is_refused(dirs, installed, extra):
    global_setup()
    before = sorted(p.name for p in dirs.iterdir())
    result = run_plain(str(dirs / "study"), "--provider", "openai", *extra)
    assert result.exit_code == 2, result.output
    text = " ".join(result.output.split())
    assert extra[0] in text and "needs a terminal; nothing was changed" in text
    assert sorted(p.name for p in dirs.iterdir()) == before


def test_a_provider_without_a_model_setting_refuses_a_model_before_writing(dirs):
    result = run_plain(str(dirs / "study"), "--provider", "omics-os", "--model", "m")
    assert result.exit_code == 2, result.output
    assert "bad_argument" in result.output and "omics-os" in result.output
    assert not (dirs / "study").exists()
    from lobster.core import projects

    assert not any(
        e.root == str(dirs / "study") for e in projects.registered_projects()
    )


def test_any_failure_after_the_first_write_leaves_nothing_behind(dirs, monkeypatch):
    def boom(*a, **k):
        raise typer.Exit(1)

    monkeypatch.setattr(
        "lobster.cli_internal.commands.heavy.init_commands._create_workspace_config",
        boom,
    )
    result = run_plain(str(dirs / "study"), "--provider", "openai", "--model", "m")
    assert result.exit_code == 1
    assert not (dirs / "study").exists()
    from lobster.core import projects

    assert not any(
        e.root == str(dirs / "study") for e in projects.registered_projects()
    )


def test_the_provider_in_a_terminal_error_names_the_running_executable(dirs):
    result = run_plain(str(dirs / "study"), "--provider", "openai", "--non-interactive")
    assert result.exit_code == 2
    assert f"{LOBSTER} init <folder> --provider <name>" in " ".join(
        result.output.split()
    )


def _tree(root):
    """Every file (name, size, sha256) and folder under *root*."""
    import hashlib

    return {
        str(p.relative_to(root)): (
            (p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest())
            if p.is_file()
            else "dir"
        )
        for p in sorted(root.rglob("*"))
    }


def _fail_at(monkeypatch, point, failure):
    """Make the write at *point* raise *failure*; the *after_* points raise once
    the real write is done."""
    from lobster.cli_internal.commands.heavy import init_commands
    from lobster.config.workspace_agent_config import WorkspaceAgentConfig

    def exc():
        return failure(3) if failure is SystemExit else failure()

    def raiser(*a, **k):
        raise exc()

    real_config = init_commands._create_workspace_config
    real_save = WorkspaceAgentConfig.save

    def after_config(*a, **k):
        real_config(*a, **k)
        raise exc()

    def after_save(self, *a, **k):
        real_save(self, *a, **k)
        raise exc()

    if point == "before_provider_file":
        monkeypatch.setattr(init_commands, "_create_workspace_config", raiser)
    elif point == "after_provider_file":
        monkeypatch.setattr(init_commands, "_create_workspace_config", after_config)
    elif point == "before_expert_file":
        monkeypatch.setattr(WorkspaceAgentConfig, "save", raiser)
    else:
        monkeypatch.setattr(WorkspaceAgentConfig, "save", after_save)


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize(
    "point",
    [
        "before_provider_file",
        "after_provider_file",
        "before_expert_file",
        "after_expert_file",
    ],
)
def test_a_failed_rerun_leaves_every_byte_as_it_was(
    dirs, installed, monkeypatch, point, failure
):
    global_setup()
    study = dirs / "study"
    assert (
        run_plain(str(study), "--model", "m1", "--agents", "research_agent").exit_code
        == 0
    )
    ws = study / ".lobster_workspace"
    config = json.loads((ws / "provider_config.json").read_text())
    config["my_own_note"] = "keep me"
    (ws / "provider_config.json").write_text(json.dumps(config))
    (study / ".env").write_text("A=1\n")
    (study / "runs" / "r1").mkdir(parents=True)
    (study / "runs" / "r1" / "out.txt").write_text("x")
    root = dirs.parent.parent  # the home, the global config and the credentials
    before = _tree(root)
    assert any(name.endswith("config.toml") for name in before)
    assert any("projects" in name for name in before)  # the registry is in the compare

    _fail_at(monkeypatch, point, failure)
    try:
        result = run_plain(
            str(study),
            "--provider",
            "openai",
            "--model",
            "m2",
            "--agents",
            "data_expert_agent",
        )
        assert result.exit_code != 0, result.output
    except KeyboardInterrupt:
        pass
    assert _tree(root) == before


def _lamin_setting(folder):
    from dotenv import dotenv_values

    return dotenv_values(folder / ".lobster_workspace" / ".env").get("LOBSTER_LAMIN")


def test_the_catalog_choice_lands_in_the_named_folders_workspace(dirs, monkeypatch):
    from lobster.cli_internal.commands.light import lamin_commands

    monkeypatch.setattr(lamin_commands, "_engine_installed", lambda: True)
    elsewhere = dirs / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    study = dirs / "study"
    assert run_init(str(study), "--lamin").exit_code == 0
    assert _lamin_setting(study) == "1"
    assert run_init(str(study), "--force", "--no-lamin").exit_code == 0
    assert _lamin_setting(study) == "0"
    assert list(elsewhere.iterdir()) == []  # the working folder is untouched


def test_a_wizard_result_puts_the_catalog_choice_in_the_given_workspace(
    dirs, monkeypatch
):
    from lobster.cli_internal.commands.light import lamin_commands
    from lobster.ui.bridge.init_adapter import apply_tui_init_result

    monkeypatch.setattr(lamin_commands, "_engine_installed", lambda: True)
    elsewhere = dirs / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    study = dirs / "study"
    (study / ".lobster_workspace").mkdir(parents=True)
    apply_tui_init_result(
        {"provider": "ollama", "lamin_enabled": True},
        study / ".lobster_workspace",
        study / ".env",
    )
    assert _lamin_setting(study) == "1"
    assert list(elsewhere.iterdir()) == []


def test_the_catalog_flags_are_refused_where_they_cannot_be_applied(dirs, installed):
    global_setup()
    for flag in ("--lamin", "--no-lamin"):
        result = run_plain(str(dirs / "study"), flag)
        assert result.exit_code == 2, result.output
        assert flag in result.output and "needs a terminal" in " ".join(
            result.output.split()
        )
        assert not (dirs / "study").exists()


def _existing_project_with_user_files(dirs):
    study = dirs / "study"
    assert (
        run_plain(str(study), "--model", "m1", "--agents", "research_agent").exit_code
        == 0
    )
    ws = study / ".lobster_workspace"
    (study / ".env").write_text("A=1\n")
    (study / "runs" / "r1").mkdir(parents=True)
    (study / "runs" / "r1" / "out.txt").write_text("x")
    return study, ws


RERUN = ("--provider", "openai", "--model", "m2", "--agents", "data_expert_agent")


def test_a_provider_file_that_cannot_be_written_ends_the_rerun_and_changes_nothing(
    dirs, installed
):
    global_setup()
    study, ws = _existing_project_with_user_files(dirs)
    root = dirs.parent.parent
    (ws / "provider_config.json").chmod(0o444)
    before = _tree(root)
    try:
        result = run_plain(str(study), *RERUN)
    finally:
        (ws / "provider_config.json").chmod(0o644)
    assert result.exit_code == 1, result.output
    text = " ".join(result.output.split())
    assert "provider_config.json" in text and "✓" not in text
    assert _tree(root) == before


def test_a_half_written_provider_file_ends_the_rerun_and_changes_nothing(
    dirs, installed, monkeypatch
):
    global_setup()
    study, ws = _existing_project_with_user_files(dirs)
    root = dirs.parent.parent
    before = _tree(root)

    def half_write(self, workspace_path, *a, **k):
        (workspace_path / "provider_config.json").write_text('{"global_prov')

    monkeypatch.setattr(
        "lobster.config.workspace_config.WorkspaceProviderConfig.save", half_write
    )
    result = run_plain(str(study), *RERUN)
    assert result.exit_code == 1, result.output
    assert "provider_config.json" in " ".join(result.output.split())
    assert _tree(root) == before


def test_a_file_that_cannot_be_given_back_is_named(dirs, installed, monkeypatch):
    global_setup()
    study, ws = _existing_project_with_user_files(dirs)
    real_write = type(ws).write_bytes

    def refuse_to_restore(self, data):
        if self.name == "provider_config.json":
            raise PermissionError("read-only")
        return real_write(self, data)

    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(
        "lobster.config.workspace_agent_config.WorkspaceAgentConfig.save", boom
    )
    monkeypatch.setattr(type(ws), "write_bytes", refuse_to_restore)
    result = run_plain(str(study), *RERUN)
    assert result.exit_code == 1
    text = " ".join(result.output.split())
    assert "could not be given back" in text and "provider_config.json" in text


@pytest.mark.parametrize("preset", ["scrna-basic", "scrna-full"])
def test_a_preset_whose_experts_are_all_installed_is_kept_as_the_preset(
    dirs, monkeypatch, preset
):
    from lobster.config.agent_presets import expand_preset
    from lobster.config.workspace_agent_config import WorkspaceAgentConfig

    names = {name: object() for name in expand_preset(preset)}
    monkeypatch.setattr(
        "lobster.cli_internal.commands.heavy.init_commands._get_installed_agents",
        lambda: names,
    )
    monkeypatch.setattr(
        "lobster.cli_internal.commands.heavy.init_commands._filter_unpublished_preset_agents",
        lambda agents, *a, **k: agents,
    )
    result = run_plain(
        str(dirs / "study"), "--provider", "anthropic", "--preset", preset
    )
    assert result.exit_code == 0, result.output
    config = WorkspaceAgentConfig.load(dirs / "study" / ".lobster_workspace")
    assert config.preset == preset
    assert f"{len(names)} of {len(names)} experts set" in " ".join(
        result.output.split()
    )
