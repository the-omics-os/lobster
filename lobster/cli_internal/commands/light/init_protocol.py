"""Init protocol commands for npm CLI ↔ Python handoff.

Hidden commands (not shown in --help) that expose a structured interface
for external consumers (lobster-cli npm package) to drive the init wizard.

Protocol version 1:
  - init-capabilities: version + accepted schemas
  - init-manifest: full manifest JSON (providers, agents, existing state, env)
  - apply-init-result: apply wizard result, return summary
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer

init_protocol_app = typer.Typer(hidden=True)


@init_protocol_app.command(name="init-capabilities", hidden=True)
def init_capabilities(
    json_output: bool = typer.Option(True, "--json/--no-json", hidden=True),
) -> None:
    """Report init protocol version and capabilities."""
    from lobster.version import __version__

    caps = {
        "init_protocol_version": 1,
        "lobster_ai_version": __version__,
        "accepted_result_schemas": [1],
    }
    sys.stdout.write(json.dumps(caps) + "\n")


@init_protocol_app.command(name="init-manifest", hidden=True)
def init_manifest(
    schema_version: int = typer.Option(1, "--schema-version"),
    detect_ollama: bool = typer.Option(True, "--detect-ollama/--no-detect-ollama"),
) -> None:
    """Output full wizard manifest as JSON to stdout."""
    if schema_version != 1:
        sys.stderr.write(f"Unsupported schema version: {schema_version}\n")
        raise typer.Exit(1)

    from lobster.ui.wizard.manifest import build_init_manifest

    manifest = build_init_manifest(detect_ollama=detect_ollama)

    envelope = json.loads(manifest.to_json())
    envelope["schema_version"] = 1
    envelope["existing_state"] = _detect_existing_state()
    envelope["environment"] = _detect_environment()

    sys.stdout.write(json.dumps(envelope, indent=2) + "\n")


@init_protocol_app.command(name="apply-init-result", hidden=True)
def apply_init_result(
    result_file: Path = typer.Option(..., "--result-file"),
    schema_version: int = typer.Option(1, "--schema-version"),
    global_config: bool = typer.Option(False, "--global/--local"),
) -> None:
    """Apply wizard result JSON and return summary."""
    if schema_version != 1:
        sys.stderr.write(f"Unsupported schema version: {schema_version}\n")
        raise typer.Exit(1)

    if str(result_file) == "-":
        raw = sys.stdin.read()
    else:
        if not result_file.exists():
            sys.stderr.write(f"Result file not found: {result_file}\n")
            raise typer.Exit(1)
        raw = result_file.read_text()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as e:
        sys.stderr.write(f"Invalid JSON in result file: {e}\n")
        raise typer.Exit(1)

    if result.get("cancelled"):
        sys.stdout.write(json.dumps({"applied": False, "reason": "cancelled"}) + "\n")
        raise typer.Exit(0)

    from lobster.core.workspace import resolve_workspace
    from lobster.ui.bridge.init_adapter import apply_tui_init_result

    workspace_path = resolve_workspace(None, create=True)
    ws_dir = workspace_path / ".lobster_workspace"
    ws_dir.mkdir(parents=True, exist_ok=True)
    env_path = workspace_path / ".env"

    apply_tui_init_result(
        result=result,
        workspace_path=ws_dir,
        env_path=env_path,
        global_config=global_config,
    )

    summary = {
        "applied": True,
        "provider": result.get("provider"),
        "profile": result.get("profile") or None,
        "agents_configured": len(result.get("agents") or []),
        "workspace_path": str(workspace_path),
        "global_config": global_config,
    }
    sys.stdout.write(json.dumps(summary) + "\n")


def _detect_existing_state() -> dict:
    """Detect existing credentials and configuration."""
    state = {
        "has_cloud_credentials": False,
        "cloud_user_email": None,
        "cloud_tier": None,
        "cloud_token_expired": None,
        "has_local_env": False,
        "current_provider": None,
        "current_profile": None,
        "installed_agents": [],
        "npm_cli_available": False,
    }

    try:
        from lobster.config.credentials import load_credentials

        creds = load_credentials()
        if creds and creds.get("access_token"):
            state["has_cloud_credentials"] = True
            state["cloud_user_email"] = creds.get("email")
            state["cloud_tier"] = creds.get("tier")
            from lobster.config.credentials import is_token_expired

            state["cloud_token_expired"] = is_token_expired()
    except Exception:
        pass

    try:
        from lobster.core.workspace import resolve_workspace

        ws = resolve_workspace(None, create=False)
        env_file = ws / ".env"
        if env_file.exists():
            state["has_local_env"] = True
            content = env_file.read_text()
            for line in content.splitlines():
                if line.startswith("LOBSTER_PROVIDER="):
                    state["current_provider"] = line.split("=", 1)[1].strip()
                elif line.startswith("LOBSTER_PROFILE="):
                    state["current_profile"] = line.split("=", 1)[1].strip()
    except Exception:
        pass

    try:
        from lobster.core.component_registry import component_registry

        agents = component_registry.list_agents()
        state["installed_agents"] = [a.name for a in agents]
    except Exception:
        pass

    try:
        from lobster.cli_internal.npm_launcher import find_npm_binary

        state["npm_cli_available"] = find_npm_binary() is not None
    except Exception:
        pass

    return state


def _detect_environment() -> dict:
    """Detect runtime environment."""
    import platform

    env = {
        "is_venv": sys.prefix != sys.base_prefix,
        "is_uv_tool_env": False,
        "python_version": platform.python_version(),
        "workspace_path": None,
        "global_config_dir": str(Path.home() / ".config" / "omics-os"),
        "platform": sys.platform,
    }

    try:
        from lobster.core.uv_tool_env import is_uv_tool_env

        env["is_uv_tool_env"] = is_uv_tool_env()
    except Exception:
        pass

    try:
        from lobster.core.workspace import resolve_workspace

        env["workspace_path"] = str(resolve_workspace(None, create=False))
    except Exception:
        pass

    return env
