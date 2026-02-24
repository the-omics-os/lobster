"""LobsterBackend — adapter from Lobster AI clients to langchain-tui-template.

Wraps AgentClient (local) and CloudLobsterClient (cloud) through the
BackendProtocol, exposing full Tier 2 capabilities including modalities,
provenance, pipelines, and notebook export.

Registered via entry point:
    [project.entry-points."langchain_tui_template.backends"]
    lobster = "lobster.integrations.langchain_tui_backend:LobsterBackend"

This module has NO hard dependency on langchain-tui-template at import time.
It only depends on lobster-ai internals.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)


class LobsterBackend:
    """Backend adapter wrapping Lobster AI's AgentClient and CloudLobsterClient.

    Implements the full langchain-tui-template BackendProtocol with Tier 0/1/2
    capabilities. Auto-detects whether to use local or cloud mode based on
    the LOBSTER_CLOUD_KEY environment variable.
    """

    def __init__(
        self,
        workspace: Optional[Path] = None,
        session_id: Optional[str] = None,
    ):
        self._client = self._create_client(workspace, session_id)
        self._is_cloud = self._detect_cloud()

    def _create_client(self, workspace: Optional[Path], session_id: Optional[str]):
        """Create the appropriate Lobster client."""
        import os

        cloud_key = os.environ.get("LOBSTER_CLOUD_KEY")
        if cloud_key:
            return self._create_cloud_client(cloud_key, session_id)
        else:
            return self._create_local_client(workspace, session_id)

    def _create_local_client(self, workspace: Optional[Path], session_id: Optional[str]):
        """Create a local AgentClient."""
        from lobster.core.client import AgentClient
        from lobster.core.workspace import resolve_workspace

        workspace_path = resolve_workspace(explicit_path=workspace, create=True)

        client = AgentClient(
            workspace_path=workspace_path,
            session_id=session_id,
        )
        return client

    def _create_cloud_client(self, cloud_key: str, session_id: Optional[str]):
        """Create a CloudLobsterClient."""
        try:
            from lobster.core.api_client import CloudLobsterClient

            return CloudLobsterClient(api_key=cloud_key, session_id=session_id)
        except ImportError:
            logger.warning("CloudLobsterClient not available, falling back to local")
            return self._create_local_client(None, session_id)

    def _detect_cloud(self) -> bool:
        """Detect if using cloud client."""
        return type(self._client).__name__ == "CloudLobsterClient"

    # --- Tier 0: Required ---

    @property
    def capabilities(self) -> set[str]:
        caps = {
            "streaming",
            "workspace",
            "data_management",
            "session",
            "token_usage",
            "modalities",
            "provenance",
            "notebook_export",
            "custom_commands",
        }
        # Cloud may not support all features
        if self._is_cloud:
            caps.discard("notebook_export")
        return caps

    def query(
        self, user_input: str, stream: bool = False
    ) -> Dict[str, Any] | Generator[Dict[str, Any], None, None]:
        """Route query to Lobster client."""
        return self._client.query(user_input, stream=stream)

    def get_status(self) -> Dict[str, Any]:
        """Get Lobster session status."""
        return self._client.get_status()

    # --- Tier 1: Common ---

    def list_workspace_files(self, pattern: str = "*") -> List[Dict[str, Any]]:
        return self._client.list_workspace_files(pattern)

    def read_file(self, filename: str) -> Optional[str]:
        return self._client.read_file(filename)

    def write_file(self, filename: str, content: str) -> bool:
        return self._client.write_file(filename, content)

    def get_conversation_history(self) -> List[Dict[str, str]]:
        return self._client.get_conversation_history()

    def reset(self) -> None:
        self._client.reset()

    def export_session(self, export_path: Optional[Path] = None) -> Path:
        return self._client.export_session(export_path)

    def get_token_usage(self) -> Dict[str, Any]:
        return self._client.get_token_usage()

    # --- Tier 2: Rich (Lobster-specific) ---

    def list_modalities(self) -> List[Dict[str, Any]]:
        """List loaded data modalities via DataManagerV2."""
        if self._is_cloud:
            # Cloud client may have different API
            try:
                return self._client.list_modalities()
            except AttributeError:
                return []

        if not hasattr(self._client, "data_manager"):
            return []

        dm = self._client.data_manager
        modality_names = dm.list_modalities()
        result = []
        for name in modality_names:
            adata = dm.get_modality(name)
            info = {"name": name}
            if adata is not None:
                info["shape"] = f"{adata.n_obs} x {adata.n_vars}"
                info["type"] = type(adata).__name__
            result.append(info)
        return result

    def get_modality_info(self, name: str) -> Dict[str, Any]:
        """Get detailed modality info."""
        if self._is_cloud:
            try:
                return self._client.get_modality_info(name)
            except AttributeError:
                return {}

        if not hasattr(self._client, "data_manager"):
            return {}

        dm = self._client.data_manager
        adata = dm.get_modality(name)
        if adata is None:
            return {}

        return {
            "name": name,
            "n_obs": adata.n_obs,
            "n_vars": adata.n_vars,
            "obs_columns": list(adata.obs.columns) if hasattr(adata.obs, "columns") else [],
            "var_columns": list(adata.var.columns) if hasattr(adata.var, "columns") else [],
            "obsm_keys": list(adata.obsm.keys()) if adata.obsm else [],
            "uns_keys": list(adata.uns.keys()) if adata.uns else [],
        }

    def export_notebook(self) -> Path:
        """Export analysis as Jupyter notebook."""
        if hasattr(self._client, "export_pipeline_notebook"):
            return self._client.export_pipeline_notebook()
        raise NotImplementedError("Notebook export requires local AgentClient")

    def register_commands(self, registry: Any) -> None:
        """Register Lobster-specific slash commands.

        Maps the 60+ Lobster commands to the langchain-tui command registry.
        Only registers a subset of the most important ones to keep
        the langchain-tui experience clean.
        """
        # /pipeline command
        registry.register(
            name="pipeline",
            handler=self._handle_pipeline,
            description="Export or list analysis pipelines",
            usage="/pipeline [export|list|run]",
            required_capability="provenance",
            category="lobster",
        )

        # /workspace command
        registry.register(
            name="workspace",
            handler=self._handle_workspace,
            description="Workspace management (load, list)",
            usage="/workspace [list|load <name>]",
            required_capability="workspace",
            category="lobster",
        )

        # /plots command
        registry.register(
            name="plots",
            handler=self._handle_plots,
            description="List generated plots",
            required_capability="workspace",
            category="lobster",
        )

    def _handle_pipeline(self, args: str, backend: Any, output: Any) -> None:
        """Handle /pipeline command."""
        subcommand = args.strip().lower() if args else "list"

        if subcommand == "export":
            try:
                path = self.export_notebook()
                output.print(f"[status.success]Pipeline exported to:[/status.success] {path}")
            except Exception as e:
                output.print(f"[status.error]Export failed: {e}[/status.error]")
        elif subcommand == "list":
            status = self.get_status()
            steps = status.get("pipeline_steps", [])
            if not steps:
                output.print("[text.muted]No pipeline steps recorded yet.[/text.muted]")
                return
            for i, step in enumerate(steps, 1):
                output.print(f"  {i}. {step}")
        else:
            output.print(f"[status.warning]Unknown subcommand: {subcommand}[/status.warning]")

    def _handle_workspace(self, args: str, backend: Any, output: Any) -> None:
        """Handle /workspace command."""
        output.print(f"[data.key]Workspace:[/data.key] {self.get_status().get('workspace', 'N/A')}")

    def _handle_plots(self, args: str, backend: Any, output: Any) -> None:
        """Handle /plots command."""
        files = self.list_workspace_files("*.html")
        plots = [f for f in files if "plot" in f.get("name", "").lower() or f.get("name", "").endswith(".html")]
        if not plots:
            output.print("[text.muted]No plots generated yet.[/text.muted]")
            return
        for p in plots:
            output.print(f"  [accent]{p['name']}[/accent] ({p.get('size', '?')} bytes)")
