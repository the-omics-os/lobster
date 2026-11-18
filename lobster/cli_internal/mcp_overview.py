"""``lobster_overview``: what the host needs to know before a run, read fresh.

The project that contains the server's start folder, its provider and model,
whether the provider's key is there (by name only), its experts, data files and
past runs, and the command lines for what MCP does not do. Nothing here imports
the engine: an overview is a few file reads.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

LISTED_MAX = 20  # data files and runs shown; the total is given beside them
_SCAN_MAX = 2000  # folder entries looked at for data files, at most

DATA_SUFFIXES = (
    ".h5ad",
    ".h5mu",
    ".h5",
    ".loom",
    ".mtx",
    ".mtx.gz",
    ".csv",
    ".csv.gz",
    ".tsv",
    ".tsv.gz",
    ".txt.gz",
    ".xlsx",
    ".parquet",
    ".rds",
    ".fastq.gz",
    ".fq.gz",
    ".bam",
    ".vcf",
    ".vcf.gz",
    ".mzml",
)
KEY_LIMIT = (
    "Presence only. The server process also holds what was set when it "
    "started (shell exports, a .env its start loaded); that is shared by every "
    "project it serves. A provider that signs in another way (lobster cloud "
    "login) can show no key and still work."
)


# Modules that make a start slow; initialize and the overview import none.
HEAVY = (
    "lobster.agents",
    "lobster.core.client",
    "lobster.core.data_manager_v2",
    "lobster.services",
    "lobster.tools",
    "langchain",
    "langgraph",
    "anndata",
    "scanpy",
)


def start_text(start: Path) -> str:
    """The initialize ``instructions`` for a server started in *start*."""
    from lobster.cli_internal import mcp_contract
    from lobster.cli_internal.mcp_record import EXPERT_NAME
    from lobster.core.projects import project_settings

    try:
        project, broken = find_project_or_reason(start)
        if broken:
            room = mcp_contract.INSTRUCTIONS_MAX - len(mcp_contract.START_FLOW) - 1
            lead = " ".join(broken.split())
            if len(lead) > room:
                lead = lead[: room - 3] + "..."
            return f"{lead} {mcp_contract.START_FLOW}"
        if project is None:
            return mcp_contract.start_text(None, [])
        experts = project_settings(project).experts or sorted(installed_experts())
    except Exception as exc:  # the start text never stops a host from connecting
        print(f"lobster mcp: no project text: {exc!r}", file=sys.stderr)
        return mcp_contract.start_text(None, [])
    # a name from a settings file goes into the host's prompt: names only
    experts = [e for e in experts if isinstance(e, str) and EXPERT_NAME.fullmatch(e)]
    return mcp_contract.start_text(project.name, experts)


TEAM_NOTE = (
    "A list replaces the whole team: only the listed experts and their child "
    "experts load. data_expert_agent is the one that loads data files: a run "
    "given input_path whose list lacks it is refused."
)


def _lobster_path() -> str:
    """The lobster executable of the install serving this process: the script
    that started it, else the one beside its Python; "lobster" when neither is
    found."""
    names = ("lobster", "lobster.exe")
    started = [Path(sys.argv[0])] if sys.argv and sys.argv[0] else []
    beside = [Path(sys.executable).parent / name for name in names]
    for candidate in started + beside:
        if (
            candidate.name in names
            and candidate.is_file()
            and os.access(candidate, os.X_OK)
        ):
            return os.path.abspath(candidate)
    return "lobster"


def _quote(arg: str) -> str:
    if arg.startswith("<"):  # a placeholder the user fills in
        return arg
    return subprocess.list2cmdline([arg]) if os.name == "nt" else shlex.quote(arg)


def lobster_command(*args: str) -> str:
    """The command line that runs this install's lobster with *args*; a host
    shell's PATH can hold another version."""
    return " ".join(_quote(a) for a in (_lobster_path(), *args))


def init_line(folder: str = "<folder>") -> str:
    return lobster_command("init", folder)


def install_line(package: str = "<package>") -> str:
    return lobster_command("agents", "install", package)


def installed_experts() -> Set[str]:
    """The expert names installed now, read from package metadata each call."""
    from importlib.metadata import entry_points

    return {entry.name for entry in entry_points(group="lobster.agents")}


def loaded_experts() -> Optional[Set[str]]:
    """The experts this process's engine has loaded, or None before it loaded
    any (the next run then reads them fresh)."""
    module = sys.modules.get("lobster.core.component_registry")
    registry = getattr(module, "component_registry", None)
    # The registry loads each group once per process; only its own set says so.
    if registry is None or "lobster.agents" not in registry._loaded_groups:
        return None
    return set(registry.list_agents())


def expert_package(name: str) -> Optional[str]:
    """The lobster-* package that provides the expert *name*, if one is known."""
    from lobster.cli_internal.agent_packages import AVAILABLE_AGENT_PACKAGES

    for package, _about, agents, _published, _experimental in AVAILABLE_AGENT_PACKAGES:
        if name in agents:
            return package
    return None


def restart_text(names: List[str]) -> str:
    return (
        f"{', '.join(names)} was installed after this server loaded its "
        "experts; restart the Lobster MCP server (reconnect it in your host) "
        "to use it."
    )


def overview(start: Path) -> Dict[str, Any]:
    from lobster.cli_internal.mcp_record import EXPERT_NAME
    from lobster.core.projects import list_runs, project_settings

    project, broken = find_project_or_reason(start)
    installed = installed_experts()
    seen: Dict[str, Any] = {"project": None, "provider": None, "model": None}
    enabled: List[str] = sorted(installed)
    if project is None:
        seen["key"] = None
        seen["next"] = broken or no_project_text(start)
        if not broken:
            seen["start_lines"] = start_lines(start)
    else:
        settings = project_settings(project)
        seen["project"] = {
            "id": project.project_id,
            "name": project.name,
            "folder": str(project.root),
            "runs_folder": str(project.root / "runs"),
        }
        provider, provider_source, model, model_source = run_will_use(project)
        seen.update(
            provider=provider,
            provider_source=provider_source,
            model=model,
            model_source=model_source,
        )
        seen["key"] = key_presence(provider, Path.cwd() / ".lobster_workspace")
        note = settings_note(project)
        if note:
            seen["project_settings_not_applied"] = note
        unreadable = [
            str(path)
            for path in (
                project.workspace / "provider_config.json",
                project.workspace / "config.toml",
            )
            if not _readable(path)
        ]
        refuse = _missing_key(seen["key"], provider)
        if unreadable:
            seen["unreadable"] = unreadable
            refuse += (
                f"A run refuses until {' and '.join(unreadable)} "
                f"{'is' if len(unreadable) == 1 else 'are'} repaired; ask the "
                "user to repair them. "
            )
        if settings.experts:
            enabled = [e for e in settings.experts if EXPERT_NAME.fullmatch(e)]
        experts_source = "project" if settings.experts else "every installed expert"
        runs = list_runs(project)
        files, total = data_files(project.root)
        seen.update(
            data_files=files,
            data_files_total=total,
            runs=runs[-LISTED_MAX:],
            runs_total=len(runs),
        )
        seen["next"] = refuse + (
            "Ask lobster_run a question; its folder goes in "
            f"{project.root / 'runs'} unless you pass output_dir. Experts the "
            f"run needs and the project lacks: run this line in a shell in "
            f"{project.root}: {install_line()}, then tell the user what you ran."
        )
    seen["experts"] = {
        "enabled": enabled,
        "installed": sorted(installed),
        "installable": installable(installed),
    }
    if project is not None:
        seen["experts"]["source"] = experts_source
        seen["experts"]["note"] = TEAM_NOTE
    loaded = loaded_experts()
    late = sorted(installed - loaded) if loaded is not None else []
    if late:
        seen["restart_needed"] = restart_text(late)
    seen["commands"] = {
        "start_project": init_line(),
        "install_experts": install_line(),
    }
    return seen


_SOURCES = (
    ("workspace", "project"),
    ("global", "global file"),
    ("environment", "environment"),
    ("provider default", "provider default"),
)


def _source(decision: str) -> str:
    for start, label in _SOURCES:
        if decision.startswith(start):
            return label
    return decision


def run_will_use(
    project: Any,
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """What a run resolves. A project provider file that names a provider
    goes into the run's session folder (mcp_stdio._copy_project_provider),
    which the engine reads first. Without one, the fresh session folder takes
    the provider file of the server's folder only when no global file and no
    LOBSTER_LLM_PROVIDER are set (session_infra's seeding), then the usual
    order; a server folder that is not the project is named as such."""
    from lobster.config.global_config import GlobalProviderConfig
    from lobster.config.workspace_config import WorkspaceProviderConfig

    if WorkspaceProviderConfig.load(project.workspace).global_provider:
        return run_settings(project.workspace)
    here = Path.cwd()
    seeded = not (
        os.environ.get("LOBSTER_LLM_PROVIDER") or GlobalProviderConfig.exists()
    )
    found = run_settings(here / ".lobster_workspace", use_workspace=seeded)
    if _in_project(project):
        return found
    label = f"server folder {here}"
    return tuple(label if v == "project" else v for v in found)  # type: ignore[return-value]


def _in_project(project: Any) -> bool:
    return Path.cwd().resolve() == project.root.resolve()


def settings_note(project: Any) -> Optional[str]:
    """Why the project's own .env does not reach the run in progress; None
    when the server works in the project folder. A stdio server is outside
    it only while a run that was active when the project appeared goes on."""
    from lobster.config.workspace_config import WorkspaceProviderConfig

    if _in_project(project):
        return None
    here = Path.cwd()
    if WorkspaceProviderConfig.load(project.workspace).global_provider:
        what = f"its key from the .env of {here}"
    else:  # nothing of the project's own: the server folder's provider file too
        what = f"its key and provider from the .env and settings of {here}"
    return (
        f"This server works in {here}, not in the project folder "
        f"{project.root}, so the run in progress takes {what}, not from the "
        "project folder. The server moves into the project folder once no run "
        "is active, so the next run starts in the project folder."
    )


def run_settings(
    workspace: Path,
    use_workspace: bool = True,
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """``(provider, its source, model, its source)`` as a run with this
    project workspace resolves them; all None when no provider is set."""
    from lobster.core.config_resolver import ConfigResolver, ConfigurationError

    resolver = ConfigResolver(workspace)
    if not use_workspace:
        resolver.workspace_config = None
    try:
        provider, decided = resolver.resolve_provider()
    except ConfigurationError:
        return None, None, None, None
    model, model_decided = resolver.resolve_model(provider=provider)
    return provider, _source(decided), model, _source(model_decided) if model else None


def find_project_or_reason(start: Path) -> Tuple[Any, Optional[str]]:
    """``(the project that contains start, None)``, or ``(None, why)`` when a
    project file is there and cannot be read, or ``(None, None)``."""
    from lobster.core.projects import ProjectError, find_project, load_project
    from lobster.core.runtime.workspace import find_project_root

    project = find_project(start)
    if project is not None:
        return project, None
    root = find_project_root(start)
    if root is None:
        return None, None
    try:
        load_project(root)
    except ProjectError as exc:
        return None, (
            f"project.json in {root} cannot be read ({exc}). Ask the user to "
            "repair or remove it; this server does not change it."
        )
    return None, None


def start_lines(start: Path) -> List[str]:
    """The provider hint's init lines for *start*, naming this install's
    lobster by its absolute path."""
    from lobster.cli_internal.provider_hint import (
        Placeholder,
        command_line,
        providers_with_keys,
        suggested_lines,
    )

    command = _quote(_lobster_path())
    if _no_project_here(start):
        return suggested_lines(Placeholder("~/<name>"), command=command)
    if not providers_with_keys() and _setup_exists(start):
        return [command_line("init", str(start), command=command)]
    return suggested_lines(str(start), command=command)


def _no_project_here(start: Path) -> bool:
    """init refuses *start*: the home folder, or a folder above it such as the
    / a desktop host starts a server in when its config sets no folder."""
    from lobster.core.projects import is_home_or_above

    return is_home_or_above(start)


def _setup_exists(start: Path) -> bool:
    """A provider is set where a bare init in *start* finds it."""
    return run_settings(start / ".lobster_workspace")[0] is not None


_CHOOSE = (
    "Add --model <model> and --agents <a,b> or --preset <name> to choose the "
    "model and the experts."
)


def no_project_text(start: Path) -> str:
    from lobster.cli_internal.provider_hint import (
        NO_SETUP_CODE,
        providers_with_keys,
        variables_found,
    )

    lead = f"No Lobster project contains {start}, where this server was started."
    if _no_project_here(start):
        lines = [
            line.removeprefix("In a terminal: ")
            for line in start_lines(start)
            if not line.startswith("Options: ")
        ]
        return (
            f"{lead} A project cannot be made there: it is the home folder or "
            "contains it (a desktop host starts the server in / when its MCP "
            "config sets no folder). Ask the user to make a project in a "
            "terminal, choosing <name>: " + " or ".join(lines) + ". Then add "
            "--workspace <project folder> to this server's args in the host's "
            "MCP config (for Codex, set cwd to the project folder) and restart "
            "the server."
        )
    found = providers_with_keys()
    if found:
        names = ", ".join(
            f"{', '.join(variables_found(p))} is set for {p}" for p in found
        )
        how = (
            "To start one, run one of the start_lines in a shell: "
            + "; ".join(start_lines(start))
            + f" ({names}; never pass a key on the line). {_CHOOSE}"
        )
    elif _setup_exists(start):
        # a global or shell provider setup: the bare line uses it
        how = (
            f"To start one, run this line in a shell: {init_line(str(start))}. "
            f"It uses the user's provider setup. {_CHOOSE}"
        )
    else:
        how = (
            f"To start one, run this line in a shell: {init_line(str(start))}. "
            f"No provider is set up, so it changes nothing and says "
            f"{NO_SETUP_CODE}; ask the user to run it in a terminal to set one "
            f"up. {_CHOOSE}"
        )
    return (
        f"{lead} {how} Tell the user what you ran, then call lobster_overview "
        "again (no restart needed)."
    )


def _readable(path: Path) -> bool:
    """False when *path* is there and a run would refuse it."""
    if not os.path.lexists(path):
        return True
    try:
        if path.suffix == ".json":
            load_provider_config(path)
        else:
            load_agent_config(path)
    except Exception:  # OSError, JSON, TOML or pydantic
        return False
    return True


def load_provider_config(path: Path) -> Any:
    """WorkspaceProviderConfig.load without its fallback: that one turns a
    file it cannot read into "no provider"; this raises."""
    import json

    from lobster.config.workspace_config import WorkspaceProviderConfig

    return WorkspaceProviderConfig(**json.loads(path.read_text(encoding="utf-8")))


def load_agent_config(path: Path) -> Any:
    """WorkspaceAgentConfig.load without its fallback: that one turns a file
    it cannot read into "every installed expert"; this raises."""
    import tomllib

    from lobster.config.workspace_agent_config import (
        AgentSettings,
        ProviderSettings,
        WorkspaceAgentConfig,
    )

    data = tomllib.loads(path.read_text(encoding="utf-8"))
    provider = data.pop("provider", None)
    if provider:
        data["provider_settings"] = ProviderSettings(**provider)
    if "agent_settings" in data:
        data["agent_settings"] = {
            name: AgentSettings(**settings)
            for name, settings in data["agent_settings"].items()
        }
    return WorkspaceAgentConfig(**data)


def installable(installed: Set[str]) -> List[Dict[str, Any]]:
    """The known packages with experts that are not installed, each with the
    command line that installs them."""
    from lobster.cli_internal.agent_packages import AVAILABLE_AGENT_PACKAGES

    found = []
    for package, about, agents, published, experimental in AVAILABLE_AGENT_PACKAGES:
        missing = [a for a in agents if a not in installed]
        if missing and published:
            found.append(
                {
                    "package": package,
                    "about": about,
                    "experts": missing,
                    "experimental": experimental,
                    "install": install_line(package),
                }
            )
    return found


def _missing_key(key: Optional[Dict[str, Any]], provider: Optional[str]) -> str:
    """The sentence ``next`` leads with when the run's key is not found."""
    if not key or key.get("present") is not False:
        return ""
    one = len(key["names"]) == 1
    return (
        f"No {' and '.join(key['names'])} {'was' if one else 'were'} found for "
        f"{provider}, and a run fails at startup without {'it' if one else 'them'} "
        "(unless the provider signs in another way; see key.limit). key.fix "
        f"says how to set {'it' if one else 'them'}. "
    )


def key_presence(provider: Optional[str], workspace: Path) -> Optional[Dict[str, Any]]:
    """Whether the provider's key variables are set, and where the value that
    a run would use comes from: the project's ``.env`` files (``<ws>/.env``,
    then ``<ws>/../.env``), then the server's environment, then the global
    credentials file. Never any part of a value."""
    if provider is None:
        return None
    names = key_names(provider)
    if not names:
        return {"needed": False}
    from dotenv import dotenv_values

    from lobster.config import global_config

    sources = []
    unreadable = []

    def read(path: Path) -> Optional[Dict[str, Any]]:
        try:
            return dotenv_values(path)
        except (OSError, ValueError):  # UnicodeDecodeError is a ValueError
            unreadable.append(str(path))
            return None

    # The run reads both (settings: the workspace .env wins a conflict).
    for env in (workspace / ".env", workspace.parent / ".env"):
        if env.is_file():
            values = read(env)
            if values is not None:
                sources.append((str(env), values))
    # the server's environment: a shell export or a host config's env block
    sources.append(("environment", os.environ))
    path = global_config.get_global_credentials_path()
    if path.is_file():
        values = read(path)
        if values is not None:
            sources.append(("global config", values))
    found_in = None
    for where, values in sources:
        if all(values.get(name) for name in names):
            found_in = where
            break
    login = None
    if found_in is None and provider == "omics-os":
        login = _saved_login()
        if login == "usable":
            found_in = "omics-os login"
    presence: Dict[str, Any] = {
        "names": names,
        "present": found_in is not None,
        "found_in": found_in,
        "limit": KEY_LIMIT,
    }
    if unreadable:
        presence["unreadable"] = unreadable
    if found_in is None:
        presence["fix"] = key_fix(names)
    if login in ("expired", "unusable"):
        presence["login"] = login
        presence["fix"] = f"{login_line()} Or: {presence['fix']}"
    return presence


def key_fix(names: Optional[List[str]] = None) -> str:
    """What to do when no key is found; names only, never a value."""
    which = (
        ", ".join(names)
        if names
        else "the provider's key variable (lobster_overview names it)"
    )
    return (
        f"Set {which} in the env block of this server's entry in your host's "
        "MCP config (a desktop host passes no shell exports), or ask the user "
        f"to run {lobster_command('init', '--global')} in a terminal to save "
        "the key for every project; then restart the server."
    )


def provider_fix() -> str:
    return (
        "Set LOBSTER_LLM_PROVIDER and the provider's key in the env block of "
        "this server's entry in your host's MCP config, or ask the user to run "
        f"{lobster_command('init', '--global')} in a terminal; then restart "
        "the server."
    )


def login_line() -> str:
    return (
        "The saved login has expired or cannot be used: ask the user to run "
        f"{lobster_command('cloud', 'login')} in a terminal."
    )


def saved_login_present() -> bool:
    """Whether ``lobster cloud login`` saved a key or token; never read for
    use, never refreshed."""
    from lobster.config import credentials

    try:
        saved = credentials.load_active_profile() or {}
    except Exception:  # unreadable: the provider would not read it either
        return False
    return any(str(saved.get(k) or "").strip() for k in ("api_key", "access_token"))


def _saved_login() -> Optional[str]:
    """None without a key or token that ``lobster cloud login`` saved; else
    "usable" when the omics-os provider gets a token from it the way a run
    does (``get_api_key``, which first refreshes an expired login, over the
    network); else the run would fail: "expired", or "unusable" (a token
    without its OAuth mode, which the provider does not read)."""
    from lobster.config import credentials

    if not saved_login_present():
        return None
    try:
        if credentials.get_api_key():
            return "usable"
        return "expired" if credentials.is_token_expired() else "unusable"
    except Exception:  # unreadable: the provider would not read it either
        return None


def key_names(provider: str) -> List[str]:
    """The environment variables that hold *provider*'s credentials."""
    from lobster.ui.wizard.manifest import _PROVIDER_CREDENTIALS

    return [f.env_var or f.key for f in _PROVIDER_CREDENTIALS.get(provider, [])]


def data_files(root: Path) -> tuple:
    """``(the first LISTED_MAX data files, how many)`` in *root* and the folders
    directly in it, as paths relative to *root*; hidden folders and the runs
    folder are skipped (a run's files are listed with the run)."""
    found: List[str] = []
    looked = 0
    folders = [root]
    for depth in range(2):
        nested = []
        for folder in folders:
            try:
                entries = sorted(os.scandir(folder), key=lambda e: e.name)
            except OSError:
                continue
            for entry in entries:
                looked += 1
                if looked > _SCAN_MAX:
                    break
                if entry.name.startswith("."):
                    continue
                if entry.is_dir(follow_symlinks=False):
                    if depth == 0 and entry.name != "runs":
                        nested.append(Path(entry.path))
                elif entry.name.lower().endswith(DATA_SUFFIXES):
                    found.append(os.path.relpath(entry.path, root))
        folders = nested
    return found[:LISTED_MAX], len(found)
