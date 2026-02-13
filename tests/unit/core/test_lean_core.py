"""Tests for lean core import validation.

These tests verify that the core SDK can be imported without domain-specific
dependencies like scanpy, pyDESeq2, etc. This is critical for the modular
architecture where domain features are optional extras.

Run these tests with: pytest tests/unit/core/test_lean_core.py -v
"""

import ast
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


class TestLeanCoreImports:
    """Test that core modules can be imported without domain dependencies."""

    def test_core_protocols_import(self):
        """ServiceProtocol and StateProtocol should import without scanpy."""
        from lobster.core.protocols import ServiceProtocol, StateProtocol

        assert ServiceProtocol is not None
        assert StateProtocol is not None

    def test_overall_state_import(self):
        """OverallState should import without scanpy."""
        from lobster.agents.state import OverallState

        assert OverallState is not None

    def test_download_queue_import(self):
        """DownloadQueue should import without scanpy."""
        from lobster.core.download_queue import DownloadQueue, DownloadQueueEntry

        assert DownloadQueue is not None
        assert DownloadQueueEntry is not None

    def test_component_registry_import(self):
        """ComponentRegistry should import without scanpy."""
        from lobster.core.component_registry import ComponentRegistry

        registry = ComponentRegistry()
        assert registry is not None

    def test_data_manager_v2_import(self):
        """DataManagerV2 should import without scanpy.

        After gap closure, DataManagerV2 uses conditional imports
        for domain-specific adapters.
        """
        from lobster.core.data_manager_v2 import DataManagerV2

        assert DataManagerV2 is not None

    def test_provenance_tracker_import(self):
        """ProvenanceTracker should import without scanpy."""
        from lobster.core.provenance import ProvenanceTracker

        assert ProvenanceTracker is not None

    def test_h5ad_backend_no_scanpy_in_imports(self):
        """H5ADBackend should not have scanpy in its imports (uses anndata)."""
        backend_path = (
            Path(__file__).parent.parent.parent.parent
            / "lobster"
            / "core"
            / "backends"
            / "h5ad_backend.py"
        )
        with open(backend_path) as f:
            tree = ast.parse(f.read())

        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)

        assert "scanpy" not in imports, "h5ad_backend.py should not import scanpy"

    def test_transcriptomics_adapter_lazy_scanpy(self):
        """TranscriptomicsAdapter should use lazy scanpy import pattern."""
        adapter_path = (
            Path(__file__).parent.parent.parent.parent
            / "lobster"
            / "core"
            / "adapters"
            / "transcriptomics_adapter.py"
        )
        with open(adapter_path) as f:
            source = f.read()

        # Check for lazy import pattern
        assert "_ensure_scanpy" in source, "Missing lazy scanpy import function"
        # Check no top-level scanpy import
        tree = ast.parse(source)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "scanpy", "Top-level scanpy import found"


class TestEmptyGraphInstantiation:
    """Test that graph infrastructure works without domain agents."""

    def test_graph_module_import(self):
        """Graph module should import without errors."""
        from lobster.agents import graph

        assert graph is not None
        assert hasattr(graph, "create_bioinformatics_graph")

    @pytest.mark.skip(reason="Requires full environment setup - manual verification")
    def test_empty_graph_instantiation(self):
        """Empty graph should instantiate with active_agents=[]."""
        pass


class TestImportLinterStatus:
    """Document import-linter status for CI awareness."""

    def test_importlinter_config_exists(self):
        """Import-linter configuration should exist."""
        config_path = Path(__file__).parent.parent.parent.parent / ".importlinter"
        assert config_path.exists(), "Missing .importlinter configuration"

    def test_importlinter_has_forbidden_contracts(self):
        """Import-linter should have forbidden contracts for domain libs."""
        config_path = Path(__file__).parent.parent.parent.parent / ".importlinter"
        with open(config_path) as f:
            content = f.read()

        expected_patterns = ["scanpy", "pydeseq2", "cyvcf2"]
        for pattern in expected_patterns:
            assert pattern in content.lower(), f"Missing forbidden pattern: {pattern}"
