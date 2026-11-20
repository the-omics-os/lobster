"""
Per-agent switch for the shared ``execute_custom_code`` tool.

Every agent factory that offers the code tool adds it through :func:`attach_code_tool`,
so each such agent can be switched on or off without a code change. An agent whose
factory never calls the helper never gets the tool, whatever the switch says.

Resolution for one agent (first match wins):

1. env ``LOBSTER_CODE_TOOL_<AGENT_NAME_UPPER>`` = ``1`` | ``0``
2. env ``LOBSTER_CODE_TOOL`` = ``default`` | ``all`` | ``none``
   (``default`` defers to the layers below)
3. the workspace ``config.toml``:
   ``[agent_settings.<agent_name>] code_tool = true | false``
4. the agent's declared default (``AgentRegistryConfig.code_tool_default``)
5. ``False``

An invalid value logs a warning and only that layer is ignored. A ``config.toml`` that
exists but cannot be read may hold the user's ``code_tool = false``: the tool is then off
for every agent (the env layers still apply) and one warning names the file. So is an
agent's setting that is there but is not true or false (``code_tool = "false"``): the
tool is off for that agent. The inputs are only the
process environment, the agent's own workspace ``config.toml`` and the declared default;
no user-level or global configuration is read. The workspace setting is read on its own
from exactly ``<workspace_path>/config.toml``, so other settings in that file never
change it. ``none`` is not a hard stop: a per-agent ``1`` still turns the tool on.
"""

import logging
import os
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional

logger = logging.getLogger(__name__)

GLOBAL_ENV_VAR = "LOBSTER_CODE_TOOL"
GLOBAL_MODES = ("default", "all", "none")

#: The supervisor has no AGENT_CONFIG; its code tool is on unless switched off.
SUPERVISOR_CODE_TOOL_DEFAULT = True

# file -> (what it held, the warnings given for that): every graph build asks for
# every agent, so each warning is given once. A file that changes is told again.
_told: dict = {}


def agent_env_var(agent_name: str) -> str:
    """Name of the per-agent override variable, e.g. ``LOBSTER_CODE_TOOL_DATA_EXPERT_AGENT``."""
    return f"{GLOBAL_ENV_VAR}_{agent_name.upper()}"


def _agent_env_layer(agent_name: str) -> Optional[bool]:
    var = agent_env_var(agent_name)
    raw = os.environ.get(var)
    if raw is None or not raw.strip():
        return None
    value = raw.strip()
    if value == "1":
        return True
    if value == "0":
        return False
    logger.warning("Ignoring %s: expected 1 or 0", var)
    return None


def _global_env_layer() -> Optional[str]:
    raw = os.environ.get(GLOBAL_ENV_VAR)
    if raw is None or not raw.strip():
        return None
    mode = raw.strip().lower()
    if mode in GLOBAL_MODES:
        return mode
    logger.warning(
        "Ignoring %s: expected one of %s", GLOBAL_ENV_VAR, ", ".join(GLOBAL_MODES)
    )
    return None


def _workspace_layer(agent_name: str, workspace_path: Any) -> Optional[bool]:
    """``agent_settings.<agent_name>.code_tool`` from the workspace config.toml only.

    Reads that one key; the rest of the file is not validated here. A missing file or
    key is silent. A file, an ``agent_settings`` or an entry of this agent's that is
    there but unusable may hold the user's off-switch: it warns and turns the tool off.
    """
    if workspace_path is None or not isinstance(workspace_path, (str, os.PathLike)):
        return None
    import tomllib

    from lobster.config.workspace_agent_config import CONFIG_FILE_NAME
    from lobster.i18n import h

    path = Path(os.fspath(workspace_path)) / CONFIG_FILE_NAME
    try:
        raw = path.read_bytes()
        data = tomllib.loads(raw.decode("utf-8-sig"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as error:
        _tell_once(
            _told_for(path, str(error)),
            h(
                "{file} could not be read, so the code tool is off for every agent until it is fixed: {error}"
            ).format(file=path, error=error),
        )
        return False
    said = _told_for(path, raw)
    all_settings = data.get("agent_settings")
    if all_settings is not None and not isinstance(all_settings, dict):
        _tell_once(
            said,
            h(
                "{file}: agent_settings must be a table (found {kind}); the code tool is off for every agent until it is fixed."
            ).format(file=path, kind=type(all_settings).__name__),
        )
        return False
    if all_settings is None:
        return None
    settings = all_settings.get(agent_name)
    if settings is None:
        return None
    if not isinstance(settings, dict):
        _tell_once(
            said,
            _not_a_bool(path, f"agent_settings.{agent_name}", settings, agent_name),
        )
        return False
    value = settings.get("code_tool")
    if value is None or isinstance(value, bool):
        return value
    _tell_once(
        said,
        _not_a_bool(path, f"agent_settings.{agent_name}.code_tool", value, agent_name),
    )
    return False


def _told_for(path: Path, held: Any) -> set:
    """The warnings already given for ``path`` while it holds ``held``."""
    before, said = _told.get(str(path), (None, set()))
    if before != held:
        said = set()
        _told[str(path)] = (held, said)
    return said


def _tell_once(said: set, message: str) -> None:
    if message not in said:
        said.add(message)
        logger.warning(message)


def _not_a_bool(path: Path, key: str, value: Any, agent_name: str) -> str:
    from lobster.i18n import h

    return h(
        "{file}: {key} must be true or false (found {kind}); the code tool is off for {agent} until it is fixed."
    ).format(file=path, key=key, kind=type(value).__name__, agent=agent_name)


def code_tool_enabled(
    agent_name: str, *, default: bool = False, workspace_path: Any = None
) -> bool:
    """Whether ``agent_name`` gets ``execute_custom_code`` (see the module docstring).

    Args:
        agent_name: Canonical registry name of the agent (``"supervisor"`` for the
            supervisor). It keys the env override and ``[agent_settings.<name>]``.
        default: The agent's declared default (``AgentRegistryConfig.code_tool_default``).
        workspace_path: The agent's own workspace directory (holds ``config.toml``).
    """
    value = _agent_env_layer(agent_name)
    if value is not None:
        return value
    mode = _global_env_layer()
    if mode == "all":
        return True
    if mode == "none":
        return False
    value = _workspace_layer(agent_name, workspace_path)
    if value is not None:
        return value
    if isinstance(default, bool):
        return default
    logger.warning("Ignoring a non-boolean code tool default for %s", agent_name)
    return False


def attach_code_tool(
    tools: Iterable[Any],
    data_manager: Any,
    switch_name: str,
    *,
    default: bool = False,
    tool_agent_name: Optional[str] = None,
    post_processor: Optional[Callable] = None,
) -> List[Any]:
    """Return ``tools`` plus ``execute_custom_code`` when the switch is on.

    Off: the same tools in a new list, and no code execution service is built.
    On: the tool from ``create_execute_custom_code_tool`` is appended, with the
    ``CODE_EXEC`` metadata and tags.

    Args:
        tools: The tools that come before the code tool.
        data_manager: The agent's data manager; its ``workspace_path`` locates the
            workspace ``config.toml``.
        switch_name: Canonical registry name the switch is keyed on.
        default: The agent's declared default.
        tool_agent_name: Agent name the tool records (defaults to ``switch_name``).
        post_processor: Optional post-processor passed to the tool factory.
    """
    tools = list(tools)
    enabled = code_tool_enabled(
        switch_name,
        default=default,
        workspace_path=getattr(data_manager, "workspace_path", None),
    )
    if not enabled:
        logger.debug("execute_custom_code is off for %s", switch_name)
        return tools

    from lobster.services.execution.custom_code_execution_service import (
        CustomCodeExecutionService,
    )
    from lobster.tools.custom_code_tool import create_execute_custom_code_tool

    execute_custom_code = create_execute_custom_code_tool(
        data_manager,
        CustomCodeExecutionService(data_manager),
        switch_name if tool_agent_name is None else tool_agent_name,
        post_processor,
    )
    execute_custom_code.metadata = {"categories": ["CODE_EXEC"], "provenance": False}
    execute_custom_code.tags = ["CODE_EXEC"]
    return [*tools, execute_custom_code]
