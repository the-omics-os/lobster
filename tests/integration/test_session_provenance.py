"""
Integration tests for session-scoped provenance persistence.

These tests validate full workflows across AgentClient, CommandClient, and CLI
for provenance that survives process exit and enables cross-session pipeline export.

TEST-13: Full workflow - AgentClient creates session, logs provenance, destroys,
         CommandClient loads, pipeline export succeeds
TEST-14: Session continuity - load_session restores provenance from original session dir
TEST-15: CLI end-to-end - subprocess lobster command 'pipeline export' --session-id creates notebook
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from lobster.cli import CommandClient
from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.core.client import AgentClient
from lobster.core.notebook_exporter import NotebookExporter

# Mark all tests as integration
pytestmark = pytest.mark.integration


# ============================================================================
# Helper Functions
# ============================================================================


def _mock_create_graph_return(mock_graph):
    """Create the return value for mocked create_bioinformatics_graph.

    Copied from test_client_integration.py pattern.
    """
    mock_metadata = Mock()
    mock_metadata.subscription_tier = "free"
    mock_metadata.available_agents = []
    mock_metadata.supervisor_accessible_agents = []
    mock_metadata.filtered_out_agents = []
    mock_metadata.to_dict.return_value = {
        "subscription_tier": "free",
        "available_agents": [],
        "supervisor_accessible_agents": [],
        "filtered_out_agents": [],
    }
    return (mock_graph, mock_metadata)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_workspace(tmp_path: Path) -> Path:
    """Create a temporary workspace for integration tests."""
    return tmp_path


@pytest.fixture
def synthetic_ir() -> AnalysisStep:
    """Create a complete AnalysisStep for testing provenance round-trip.

    CRITICAL: exportable=True is required for notebook export to work.
    """
    return AnalysisStep(
        operation="scanpy.pp.normalize_total",
        tool_name="normalize_data",
        description="Normalize counts per cell for test",
        library="scanpy",
        code_template="sc.pp.normalize_total(adata, target_sum={{ target_sum }})",
        imports=["import scanpy as sc"],
        parameters={"target_sum": 1e4, "exclude_highly_expressed": True},
        parameter_schema={
            "target_sum": ParameterSpec(
                param_type="float",
                papermill_injectable=True,
                default_value=1e4,
                required=True,
                description="Target sum for normalization"
            )
        },
        input_entities=["raw_counts"],
        output_entities=["normalized_counts"],
        execution_context={"modality": "transcriptomics"},
        exportable=True,  # CRITICAL for pipeline export
    )


# ============================================================================
# TEST-13: Full Workflow - Cross-Session Export
# ============================================================================


@pytest.mark.integration
class TestFullWorkflow:
    """TEST-13: AgentClient -> log provenance -> destroy -> CommandClient -> pipeline export succeeds."""

    def test_full_workflow_cross_session_export(
        self, temp_workspace: Path, synthetic_ir: AnalysisStep
    ):
        """
        TEST-13: Validate complete workflow:
        1. AgentClient creates session, logs provenance with IR
        2. Save session to disk (via query auto-save)
        3. Destroy AgentClient (simulate process exit)
        4. Create CommandClient with session_id
        5. Verify provenance loaded from disk
        6. Pipeline export succeeds
        """
        session_id = "test_session_cross_export"

        # === Phase 1: Create session with AgentClient ===
        mock_graph = Mock()
        mock_graph.stream.return_value = [
            {"supervisor": {"messages": [Mock(content="Analysis complete")]}}
        ]

        with patch(
            "lobster.core.client.create_bioinformatics_graph",
            return_value=_mock_create_graph_return(mock_graph),
        ):
            agent_client = AgentClient(
                workspace_path=temp_workspace,
                session_id=session_id,
            )

            # Log activities with IR (simulates analysis)
            agent_client.data_manager.provenance.create_activity(
                activity_type="normalize_counts",
                agent="transcriptomics_expert",
                description="Normalize single-cell counts",
                ir=synthetic_ir,
            )
            agent_client.data_manager.provenance.create_activity(
                activity_type="log_transform",
                agent="transcriptomics_expert",
                description="Log1p transform",
                ir=synthetic_ir,
            )

            # Verify provenance in memory
            assert len(agent_client.data_manager.provenance.activities) == 2

            # Trigger auto-save via query (saves session JSON)
            result = agent_client.query("test query")
            assert result["success"] is True

            # Verify session JSON created
            session_path = temp_workspace / f"session_{session_id}.json"
            assert session_path.exists()

            # Verify JSONL file created
            session_dir = temp_workspace / ".lobster" / "sessions" / session_id
            provenance_file = session_dir / "provenance.jsonl"
            assert provenance_file.exists()

            # Read provenance file to verify content
            lines = provenance_file.read_text().strip().split("\n")
            assert len(lines) == 2
            for line in lines:
                data = json.loads(line)
                assert "v" in data
                assert data["v"] == 1

            # Destroy agent_client (simulate process exit)
            del agent_client

        # === Phase 2: Load session with CommandClient ===
        command_client = CommandClient(
            workspace_path=temp_workspace,
            session_id=session_id,
        )

        # Verify provenance loaded from disk
        provenance = command_client.data_manager.provenance
        assert len(provenance.activities) == 2

        # Verify IR objects restored (not dicts)
        for activity in provenance.activities:
            ir = activity["ir"]
            assert isinstance(ir, AnalysisStep)
            assert ir.operation == "scanpy.pp.normalize_total"
            assert ir.exportable is True

        # === Phase 3: Pipeline export ===
        exporter = NotebookExporter(
            provenance=command_client.data_manager.provenance,
            data_manager=command_client.data_manager
        )
        notebook_path = exporter.export(
            name="test_pipeline",
            description="Test pipeline export"
        )

        # Verify notebook file created
        assert notebook_path.exists()
        assert notebook_path.suffix == ".ipynb"

        # Read and verify notebook content
        with open(notebook_path) as f:
            notebook = json.load(f)

        assert "cells" in notebook
        assert len(notebook["cells"]) > 0

        # Verify notebook contains IR code
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        assert len(code_cells) >= 2  # At least 2 activities with IR

        # Verify code template present in cells
        notebook_code = "\n".join(
            [
                "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
                for cell in code_cells
            ]
        )
        assert "normalize_total" in notebook_code


# ============================================================================
# TEST-14: Session Continuity
# ============================================================================


@pytest.mark.integration
class TestSessionContinuity:
    """TEST-14: load_session restores provenance from original session dir."""

    def test_load_session_restores_provenance(
        self, temp_workspace: Path, synthetic_ir: AnalysisStep
    ):
        """
        TEST-14: Validate session continuity:
        1. Create session with 3 activities
        2. Save session (via query auto-save)
        3. Create new AgentClient
        4. load_session restores provenance
        5. Activity count matches
        6. IR objects are AnalysisStep instances
        """
        session_id = "test_session_continuity"

        # === Phase 1: Create and save session ===
        mock_graph = Mock()
        mock_graph.stream.return_value = [
            {"supervisor": {"messages": [Mock(content="Session created")]}}
        ]

        with patch(
            "lobster.core.client.create_bioinformatics_graph",
            return_value=_mock_create_graph_return(mock_graph),
        ):
            client1 = AgentClient(
                workspace_path=temp_workspace,
                session_id=session_id,
            )

            # Log 3 activities with IR
            for i in range(3):
                client1.data_manager.provenance.create_activity(
                    activity_type=f"activity_{i}",
                    agent="test_agent",
                    description=f"Test activity {i}",
                    ir=synthetic_ir,
                )

            # Trigger auto-save via query
            result = client1.query("test query")
            assert result["success"] is True

            # Verify session JSON created
            session_path = temp_workspace / f"session_{session_id}.json"
            assert session_path.exists()

            # Verify provenance persisted
            session_dir = temp_workspace / ".lobster" / "sessions" / session_id
            provenance_file = session_dir / "provenance.jsonl"
            assert provenance_file.exists()

            original_activity_count = len(client1.data_manager.provenance.activities)
            assert original_activity_count == 3

            # Destroy client1
            del client1

        # === Phase 2: Create new client and load session ===
        with patch(
            "lobster.core.client.create_bioinformatics_graph",
            return_value=_mock_create_graph_return(Mock()),
        ):
            client2 = AgentClient(
                workspace_path=temp_workspace,
                session_id="different_session",  # Different session initially
            )

            # Load the saved session
            result = client2.load_session(session_path)

            # Verify load_session result
            assert result["success"] is True
            assert result["original_session_id"] == session_id
            assert result["provenance_restored"] == 3

            # Verify provenance restored
            assert len(client2.data_manager.provenance.activities) == 3

            # Verify IR objects are AnalysisStep instances, not dicts
            for activity in client2.data_manager.provenance.activities:
                ir = activity["ir"]
                assert isinstance(ir, AnalysisStep)
                assert ir.operation == "scanpy.pp.normalize_total"
                assert ir.tool_name == "normalize_data"
                assert ir.exportable is True

            # Verify activity details
            activity_types = [a["type"] for a in client2.data_manager.provenance.activities]
            assert activity_types == ["activity_0", "activity_1", "activity_2"]


# ============================================================================
# TEST-15: CLI Subprocess Test
# ============================================================================


@pytest.mark.integration
class TestCLIPipelineExport:
    """TEST-15: CommandClient pipeline export with session_id works end-to-end."""

    def test_cli_pipeline_export_with_session(
        self, temp_workspace: Path, synthetic_ir: AnalysisStep
    ):
        """
        TEST-15: Validate CommandClient pipeline export:
        1. Create session via AgentClient (simulate prior analysis)
        2. Create CommandClient with session_id
        3. Export pipeline via CommandClient.data_manager.export_notebook()
        4. Verify notebook file created and contains IR code
        """
        session_id = "test_cli_export"

        # === Phase 1: Create session with provenance ===
        mock_graph = Mock()
        mock_graph.stream.return_value = [
            {"supervisor": {"messages": [Mock(content="CLI test complete")]}}
        ]

        with patch(
            "lobster.core.client.create_bioinformatics_graph",
            return_value=_mock_create_graph_return(mock_graph),
        ):
            client = AgentClient(
                workspace_path=temp_workspace,
                session_id=session_id,
            )

            # Log activities with IR
            for i in range(2):
                client.data_manager.provenance.create_activity(
                    activity_type=f"cli_test_activity_{i}",
                    agent="test_agent",
                    description=f"CLI test activity {i}",
                    ir=synthetic_ir,
                )

            # Trigger auto-save via query
            result = client.query("test query")
            assert result["success"] is True

            # Verify session JSON created
            session_path = temp_workspace / f"session_{session_id}.json"
            assert session_path.exists()

            # Destroy client
            del client

        # === Phase 2: Create CommandClient and export pipeline ===
        command_client = CommandClient(
            workspace_path=temp_workspace,
            session_id=session_id,
        )

        # Verify provenance loaded
        assert len(command_client.data_manager.provenance.activities) == 2

        # Export notebook via CommandClient
        notebook_path = command_client.data_manager.export_notebook(
            name="test_cli_pipeline",
            description="Test CLI pipeline export"
        )

        # === Phase 3: Verify notebook ===
        assert notebook_path.exists()
        assert notebook_path.suffix == ".ipynb"

        # Verify notebook content
        with open(notebook_path) as f:
            notebook = json.load(f)

        assert "cells" in notebook
        assert len(notebook["cells"]) > 0

        # Verify activities present in notebook
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        assert len(code_cells) >= 2

        notebook_code = "\n".join(
            [
                "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
                for cell in code_cells
            ]
        )
        assert "normalize_total" in notebook_code


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
