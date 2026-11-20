"""
Helpers for testing that a tool's provenance IR reproduces what the tool did.

A tool that declares ``provenance=True`` must log an ``AnalysisStep`` whose
``code_template`` re-creates the tool's effect in an exported notebook. These
helpers let a test run a real tool against an in-memory data manager, capture
the IR it logged, render that IR and execute it on a copy of the input data,
so the replayed result can be compared with the tool's own result.

Example:
    >>> dm = RecordingDataManager({"cells": adata})
    >>> tools = factory_tools(my_agent_factory, "my_pkg.my_agent", dm)
    >>> tools["annotate"].invoke({"modality_name": "cells", ...})
    >>> ir = dm.last_ir("annotate")
    >>> replayed = replay_ir(ir, adata=adata.copy())["adata"]
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import MagicMock, patch

__all__ = ["RecordingDataManager", "factory_tools", "replay_ir"]


class RecordingDataManager:
    """In-memory stand-in for DataManagerV2 that records every provenance call.

    Implements the subset of DataManagerV2 that agent tools use: modality
    access and storage, the metadata store, a mocked plot manager and
    ``log_tool_usage``. Stored modalities replace earlier ones by name; saving
    is a no-op so tests never write into the working directory.
    """

    def __init__(
        self,
        modalities: Optional[Dict[str, Any]] = None,
        workspace_path: Optional[Path] = None,
    ):
        self.modalities: Dict[str, Any] = dict(modalities or {})
        self.metadata_store: Dict[str, Any] = {}
        self.workspace_path = Path(workspace_path or tempfile.mkdtemp())
        self.current_dataset = None
        self.calls: List[SimpleNamespace] = []
        self.plot_manager = MagicMock()
        self.plot_manager.save_plots_to_workspace.return_value = ([], None, None)

    def list_modalities(self) -> List[str]:
        return list(self.modalities)

    def get_modality(self, name: str) -> Any:
        if name not in self.modalities:
            raise KeyError(name)
        return self.modalities[name]

    def store_modality(self, name: str, adata: Any, **_: Any) -> str:
        self.modalities[name] = adata
        return name

    def save_modality(self, name: str, path: Any = None, **_: Any) -> Any:
        return path

    def unloaded_name(self, name: str) -> str:
        return name  # nothing was loaded from a file here

    def assert_save_target(self, name: str, path: Any = None) -> None:
        return None  # nothing is saved here

    def anchored_import(self, name: str) -> None:
        return None  # nothing is saved here, so no import is anchored

    def log_tool_usage(
        self,
        tool_name: str,
        parameters: Optional[Dict[str, Any]] = None,
        description: Any = None,
        ir: Any = None,
        **kwargs: Any,
    ) -> None:
        self.calls.append(
            SimpleNamespace(
                tool_name=tool_name,
                parameters=parameters or {},
                description=description,
                ir=ir,
                extra=kwargs,
            )
        )

    def calls_for(self, tool_name: str) -> List[SimpleNamespace]:
        """All recorded provenance calls made under ``tool_name``."""
        return [c for c in self.calls if c.tool_name == tool_name]

    def last_ir(self, tool_name: str) -> Any:
        """The IR of the most recent call under ``tool_name`` (AssertionError if none)."""
        calls = self.calls_for(tool_name)
        if not calls:
            logged = [c.tool_name for c in self.calls]
            raise AssertionError(f"{tool_name} logged no provenance; logged: {logged}")
        return calls[-1].ir


def factory_tools(
    factory: Callable[..., Any], module_path: str, data_manager: Any, **kwargs: Any
) -> Dict[str, Any]:
    """Build an agent with ``factory`` and return its tools by name.

    The LLM is replaced by a mock (patched in ``lobster.config.llm_factory`` and
    in ``module_path``), so no provider configuration or network is needed.
    """
    with (
        patch("lobster.config.llm_factory.create_llm", return_value=MagicMock()),
        patch(f"{module_path}.create_llm", return_value=MagicMock()),
    ):
        graph = factory(
            data_manager=data_manager,
            callback_handler=None,
            delegation_tools=[],
            workspace_path=None,
            **kwargs,
        )
    node = graph.nodes["tools"]
    tools_by_name = getattr(node, "tools_by_name", None) or node.bound.tools_by_name
    return dict(tools_by_name)


def replay_ir(ir: Any, **namespace: Any) -> Dict[str, Any]:
    """Run the IR's imports, render ``ir`` and execute it with ``namespace`` as globals; return them.

    Only for tests: the template is trusted code produced by the tool under test.
    """
    code = "\n".join([*ir.imports, ir.render()])
    scope: Dict[str, Any] = dict(namespace)
    exec(compile(code, f"<ir:{ir.tool_name}>", "exec"), scope)  # nosec B102
    return scope
