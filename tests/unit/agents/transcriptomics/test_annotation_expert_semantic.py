"""
Unit tests for annotate_cell_types_semantic tool in annotation_expert.

Tests the semantic cell type annotation tool that queries Cell Ontology
via VectorSearchService using marker gene signatures as text queries.

Uses MockEmbedder/MockVectorBackend pattern from test_vector_search_service.py
and skipif guard for optional vector-search dependencies (TEST-08).
"""

import hashlib
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest

# Skip entire module if vector-search deps not installed
try:
    from lobster.services.vector.service import VectorSearchService
    from lobster.services.vector.backends.base import BaseVectorBackend
    from lobster.services.vector.embeddings.base import BaseEmbedder
    from lobster.core.schemas.search import OntologyMatch

    HAS_VECTOR_SEARCH = True
except ImportError:
    HAS_VECTOR_SEARCH = False

pytestmark = pytest.mark.skipif(
    not HAS_VECTOR_SEARCH, reason="vector-search deps not installed"
)


# ---------------------------------------------------------------------------
# Mock implementations
# ---------------------------------------------------------------------------


class MockEmbedder(BaseEmbedder):
    """Deterministic mock embedder for unit tests -- no torch required."""

    DIMENSIONS = 768

    def embed_text(self, text: str) -> list[float]:
        h = hashlib.md5(text.encode()).hexdigest()
        return [int(c, 16) / 15.0 for c in h] * 48  # 768 dims

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]

    @property
    def dimensions(self) -> int:
        return self.DIMENSIONS


class MockVectorBackend(BaseVectorBackend):
    """In-memory mock backend returning predetermined Cell Ontology results."""

    def __init__(self) -> None:
        self._collections: dict[str, list[dict]] = {}
        self._search_results: dict[str, Any] | None = None
        self._last_query_collection: str | None = None

    def set_results(self, results: dict[str, Any]) -> None:
        """Configure what search() returns."""
        self._search_results = results

    def add_documents(
        self,
        collection_name: str,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        pass

    def search(
        self,
        collection_name: str,
        query_embedding: list[float],
        n_results: int = 5,
    ) -> dict[str, Any]:
        self._last_query_collection = collection_name
        if self._search_results is not None:
            return self._search_results

        # Default: return Cell Ontology matches
        return {
            "ids": [["CL:0000084", "CL:0000625", "CL:0000624"]],
            "distances": [[0.1, 0.3, 0.5]],
            "documents": [["T cell", "CD8-positive T cell", "CD4-positive T cell"]],
            "metadatas": [
                [
                    {"ontology_id": "CL:0000084", "source": "CL"},
                    {"ontology_id": "CL:0000625", "source": "CL"},
                    {"ontology_id": "CL:0000624", "source": "CL"},
                ]
            ],
        }

    def delete(self, collection_name: str, ids: list[str]) -> None:
        pass

    def count(self, collection_name: str) -> int:
        return 0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_clustered_adata(n_cells=300, n_clusters=3):
    """Create a minimal clustered AnnData for testing."""
    np.random.seed(42)
    # Create expression matrix with some realistic gene names
    gene_names = [
        "CD3D", "CD3E", "CD8A", "CD4",  # T cell markers
        "CD19", "MS4A1", "CD79A", "IGHM",  # B cell markers
        "CD14", "FCGR3A", "LYZ", "CSF1R",  # Monocyte markers
        "GNLY", "NKG7", "KLRD1",  # NK markers
    ] + [f"Gene_{i}" for i in range(85)]  # padding to 100 genes

    X = np.random.negative_binomial(n=5, p=0.3, size=(n_cells, len(gene_names))).astype(
        np.float32
    )
    cell_names = [f"Cell_{i}" for i in range(n_cells)]

    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=cell_names),
        var=pd.DataFrame(index=gene_names),
    )

    # Add leiden clustering with n_clusters
    cluster_labels = np.array(
        [str(i % n_clusters) for i in range(n_cells)]
    )
    adata.obs["leiden"] = pd.Categorical(cluster_labels)

    return adata


@pytest.fixture
def mock_data_manager(tmp_path):
    """Create mock DataManagerV2 with clustered single-cell data."""
    from lobster.core.data_manager_v2 import DataManagerV2

    mock_dm = Mock(spec=DataManagerV2)
    mock_dm.workspace_path = Path(tmp_path / "workspace")

    adata = _make_clustered_adata(n_cells=300, n_clusters=3)

    mock_dm.get_modality.return_value = adata
    mock_dm.modalities = {"test_modality": adata}
    mock_dm.list_modalities.return_value = ["test_modality"]
    mock_dm.log_tool_usage.return_value = None

    return mock_dm


@pytest.fixture
def mock_singlecell_service():
    """Mock EnhancedSingleCellService with predetermined marker scores."""
    from lobster.services.analysis.enhanced_singlecell_service import (
        EnhancedSingleCellService,
    )

    mock_service = Mock(spec=EnhancedSingleCellService)

    # cell_type_markers dict (used by the tool to get marker gene lists)
    mock_service.cell_type_markers = {
        "T cells": ["CD3D", "CD3E", "CD8A", "CD4"],
        "B cells": ["CD19", "MS4A1", "CD79A", "IGHM"],
        "Monocytes": ["CD14", "FCGR3A", "LYZ", "CSF1R"],
    }

    # _calculate_marker_scores_from_adata returns per-cluster scores
    mock_service._calculate_marker_scores_from_adata.return_value = {
        "0": {"T cells": 0.8, "B cells": 0.1, "Monocytes": 0.05},
        "1": {"B cells": 0.9, "T cells": 0.05, "Monocytes": 0.03},
        "2": {"Monocytes": 0.7, "T cells": 0.1, "B cells": 0.15},
    }

    return mock_service


@pytest.fixture
def mock_vector_service():
    """Create VectorSearchService with mock backend/embedder."""
    from lobster.services.vector.config import VectorSearchConfig

    config = VectorSearchConfig()
    backend = MockVectorBackend()
    embedder = MockEmbedder()
    service = VectorSearchService(config=config, backend=backend, embedder=embedder)
    return service, backend


def _get_semantic_tool(mock_data_manager, mock_singlecell_service, mock_vector_service):
    """Extract the annotate_cell_types_semantic tool function from the agent.

    Patches the factory's service construction and VectorSearchService to use mocks.
    Returns the tool's invoke function.
    """
    vs, _ = mock_vector_service

    with patch(
        "lobster.agents.transcriptomics.annotation_expert.EnhancedSingleCellService",
        return_value=mock_singlecell_service,
    ), patch(
        "lobster.agents.transcriptomics.annotation_expert.ManualAnnotationService",
    ), patch(
        "lobster.agents.transcriptomics.annotation_expert.AnnotationTemplateService",
    ):
        from lobster.agents.transcriptomics.annotation_expert import annotation_expert

        # We need to patch VectorSearchService inside the factory closure
        with patch(
            "lobster.services.vector.service.VectorSearchService",
            return_value=vs,
        ):
            agent = annotation_expert(mock_data_manager)

    # Extract the semantic tool from the agent's tools
    tools = agent.get_graph().nodes.get("agent", {}).get("metadata", {})

    # Search for the tool in the agent's tool list
    tool_func = None
    for node_name, node_data in agent.get_graph().nodes.items():
        if "annotate_cell_types_semantic" in node_name:
            tool_func = node_name
            break

    # Alternative: find the tool directly from the factory's base_tools
    # by re-calling with inspection
    return agent, vs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSemanticAnnotationBasic:
    """Basic semantic annotation functionality."""

    def test_semantic_annotation_basic(
        self, mock_data_manager, mock_singlecell_service, mock_vector_service, mock_provider_config
    ):
        """Call tool with valid modality, verify annotations, storage, and IR logging."""
        from lobster.core.analysis_ir import AnalysisStep

        vs, backend = mock_vector_service

        # Capture tools list by patching create_react_agent
        with patch(
            "lobster.agents.transcriptomics.annotation_expert.EnhancedSingleCellService",
            return_value=mock_singlecell_service,
        ), patch(
            "lobster.agents.transcriptomics.annotation_expert.ManualAnnotationService",
        ), patch(
            "lobster.agents.transcriptomics.annotation_expert.AnnotationTemplateService",
        ), patch(
            "lobster.agents.transcriptomics.annotation_expert.create_react_agent",
        ) as mock_create:
            from lobster.agents.transcriptomics.annotation_expert import (
                annotation_expert as ae_factory,
            )

            # HAS_VECTOR_SEARCH was refactored from a module-level constant into a
            # factory-local variable (Hard Rule #10). Use the test module's own
            # HAS_VECTOR_SEARCH (defined at the top of this file) to verify deps.
            assert HAS_VECTOR_SEARCH is True

            mock_create.return_value = Mock()
            ae_factory(mock_data_manager)

            call_kwargs = mock_create.call_args
            tools = call_kwargs.kwargs.get("tools", []) or (
                call_kwargs.args[1] if len(call_kwargs.args) > 1 else []
            )

        # Find the semantic tool
        semantic_tool = None
        for t in tools:
            if hasattr(t, "name") and t.name == "annotate_cell_types_semantic":
                semantic_tool = t
                break

        assert semantic_tool is not None, (
            f"annotate_cell_types_semantic not found in tools. "
            f"Available: {[getattr(t, 'name', str(t)) for t in tools]}"
        )

        # Invoke the tool
        with patch(
            "lobster.services.vector.service.VectorSearchService",
            return_value=vs,
        ):
            result = semantic_tool.invoke({
                "modality_name": "test_modality",
                "cluster_key": "leiden",
            })

        # Verify it annotated clusters
        assert isinstance(result, str)
        assert "Successfully annotated" in result

        # Verify modality stored via store_modality()
        mock_data_manager.store_modality.assert_called_once()
        store_call = mock_data_manager.store_modality.call_args
        stored_name = store_call.kwargs.get("name") or store_call.args[0]
        assert stored_name == "test_modality_semantic_annotated"

        # Verify IR logged
        mock_data_manager.log_tool_usage.assert_called_once()
        ir = mock_data_manager.log_tool_usage.call_args.kwargs.get("ir")
        assert ir is not None
        assert isinstance(ir, AnalysisStep)
        assert ir.operation == "semantic_cell_type_annotation"


class TestSemanticToolDirect:
    """Tests that directly invoke the semantic annotation tool function.

    Uses a helper to reconstruct the tool from the factory and call it.
    """

    def _build_and_call_tool(
        self,
        mock_data_manager,
        mock_singlecell_service,
        mock_vector_service,
        tool_kwargs=None,
    ):
        """Build the annotation_expert factory, extract and invoke the semantic tool."""
        vs, backend = mock_vector_service

        with patch(
            "lobster.agents.transcriptomics.annotation_expert.EnhancedSingleCellService",
            return_value=mock_singlecell_service,
        ), patch(
            "lobster.agents.transcriptomics.annotation_expert.ManualAnnotationService",
        ), patch(
            "lobster.agents.transcriptomics.annotation_expert.AnnotationTemplateService",
        ):
            from lobster.agents.transcriptomics.annotation_expert import (
                annotation_expert as ae_factory,
            )

            agent = ae_factory(mock_data_manager)

        # Extract the tool function from the agent's tool list
        # In LangGraph react agents, tools are accessible via the graph
        tools = []
        for node_name, node_data in agent.get_graph().nodes.items():
            if node_name == "tools":
                # The tools node has the tool executors
                pass

        # More direct approach: inspect the agent's bound tools
        # LangGraph react agents store tools in the agent config
        bound_tools = None
        try:
            # The agent stores tools internally
            bound_tools = agent.get_graph().nodes["tools"]
        except (KeyError, AttributeError):
            pass

        # Use the simplest approach: reconstruct the factory, capture tools list
        # by patching create_react_agent to capture the tools argument
        captured_tools = []

        original_create = None
        with patch(
            "lobster.agents.transcriptomics.annotation_expert.EnhancedSingleCellService",
            return_value=mock_singlecell_service,
        ), patch(
            "lobster.agents.transcriptomics.annotation_expert.ManualAnnotationService",
        ), patch(
            "lobster.agents.transcriptomics.annotation_expert.AnnotationTemplateService",
        ), patch(
            "lobster.agents.transcriptomics.annotation_expert.create_react_agent",
        ) as mock_create:
            from lobster.agents.transcriptomics.annotation_expert import (
                annotation_expert as ae_factory2,
            )

            mock_create.return_value = Mock()
            ae_factory2(mock_data_manager)

            # Capture the tools argument
            call_kwargs = mock_create.call_args
            captured_tools = call_kwargs.kwargs.get("tools", []) or (
                call_kwargs.args[1] if len(call_kwargs.args) > 1 else []
            )

        # Find the semantic tool
        semantic_tool = None
        for t in captured_tools:
            if hasattr(t, "name") and t.name == "annotate_cell_types_semantic":
                semantic_tool = t
                break

        assert semantic_tool is not None, (
            f"annotate_cell_types_semantic not found. "
            f"Available tools: {[getattr(t, 'name', str(t)) for t in captured_tools]}"
        )

        # Invoke the tool with patched VectorSearchService
        kwargs = {
            "modality_name": "test_modality",
            "cluster_key": "leiden",
            "k": 5,
            "min_confidence": 0.5,
            "save_result": True,
        }
        if tool_kwargs:
            kwargs.update(tool_kwargs)

        with patch(
            "lobster.services.vector.service.VectorSearchService",
            return_value=vs,
        ):
            result = semantic_tool.invoke(kwargs)

        return result, mock_data_manager, captured_tools

    def test_semantic_annotation_returns_string(
        self, mock_data_manager, mock_singlecell_service, mock_vector_service, mock_provider_config
    ):
        """Semantic tool returns a formatted string result."""
        result, _, _ = self._build_and_call_tool(
            mock_data_manager, mock_singlecell_service, mock_vector_service
        )
        assert isinstance(result, str)
        assert "Successfully annotated" in result or "semantic" in result.lower()

    def test_semantic_annotation_stores_modality(
        self, mock_data_manager, mock_singlecell_service, mock_vector_service, mock_provider_config
    ):
        """Semantic tool stores annotated modality in data_manager via store_modality()."""
        result, dm, _ = self._build_and_call_tool(
            mock_data_manager, mock_singlecell_service, mock_vector_service
        )
        # Source code calls data_manager.store_modality(name=..., adata=...)
        # instead of directly assigning to data_manager.modalities dict.
        dm.store_modality.assert_called_once()
        call_kwargs = dm.store_modality.call_args
        stored_name = call_kwargs.kwargs.get("name") or call_kwargs.args[0]
        assert stored_name == "test_modality_semantic_annotated"

    def test_ir_logged(
        self, mock_data_manager, mock_singlecell_service, mock_vector_service, mock_provider_config
    ):
        """Verify data_manager.log_tool_usage called with ir= kwarg containing AnalysisStep."""
        from lobster.core.analysis_ir import AnalysisStep

        result, dm, _ = self._build_and_call_tool(
            mock_data_manager, mock_singlecell_service, mock_vector_service
        )
        dm.log_tool_usage.assert_called_once()
        call_kwargs = dm.log_tool_usage.call_args
        # Check ir= keyword argument
        ir = call_kwargs.kwargs.get("ir") or (
            call_kwargs[1].get("ir") if len(call_kwargs) > 1 and isinstance(call_kwargs[1], dict) else None
        )
        assert ir is not None, "log_tool_usage must be called with ir= keyword argument"
        assert isinstance(ir, AnalysisStep)
        assert ir.operation == "semantic_cell_type_annotation"
        assert ir.tool_name == "annotate_cell_types_semantic"

    def test_stats_dict_content(
        self, mock_data_manager, mock_singlecell_service, mock_vector_service, mock_provider_config
    ):
        """Verify stats dict has required keys: n_clusters_annotated, n_unknown, mean_confidence, cluster_annotations."""
        result, dm, _ = self._build_and_call_tool(
            mock_data_manager, mock_singlecell_service, mock_vector_service
        )
        dm.log_tool_usage.assert_called_once()
        call_args = dm.log_tool_usage.call_args

        # Stats dict is the third positional argument (after tool_name and params)
        stats = call_args.args[2] if len(call_args.args) > 2 else call_args[0][2]

        assert "n_clusters_annotated" in stats
        assert "n_unknown" in stats
        assert "mean_confidence" in stats
        assert "cluster_annotations" in stats
        assert isinstance(stats["n_clusters_annotated"], int)
        assert isinstance(stats["n_unknown"], int)
        assert isinstance(stats["mean_confidence"], float)
        assert isinstance(stats["cluster_annotations"], dict)

    def test_confidence_filtering(
        self, mock_data_manager, mock_singlecell_service, mock_vector_service, mock_provider_config
    ):
        """Set min_confidence=0.8, mock low-score results, verify cluster gets 'Unknown'."""
        vs, backend = mock_vector_service

        # Set backend results with scores below 0.8 (distance 0.4 = score 0.6)
        backend.set_results(
            {
                "ids": [["CL:0000084"]],
                "distances": [[0.4]],  # score = 1 - 0.4 = 0.6 (below 0.8 threshold)
                "documents": [["T cell"]],
                "metadatas": [[{"ontology_id": "CL:0000084", "source": "CL"}]],
            }
        )

        result, dm, _ = self._build_and_call_tool(
            mock_data_manager,
            mock_singlecell_service,
            mock_vector_service,
            tool_kwargs={"min_confidence": 0.8},
        )

        # All clusters should be "Unknown" since all scores are below 0.8
        assert "Unknown" in result
        stats = dm.log_tool_usage.call_args.args[2]
        assert stats["n_unknown"] > 0

    def test_modality_not_found(
        self, mock_data_manager, mock_singlecell_service, mock_vector_service, mock_provider_config
    ):
        """Call with nonexistent modality, expect error message."""
        mock_data_manager.list_modalities.return_value = []

        result, _, _ = self._build_and_call_tool(
            mock_data_manager,
            mock_singlecell_service,
            mock_vector_service,
            tool_kwargs={"modality_name": "nonexistent"},
        )
        assert "not found" in result.lower() or "error" in result.lower()

    def test_invalid_cluster_key(
        self, mock_data_manager, mock_singlecell_service, mock_vector_service, mock_provider_config
    ):
        """Call with cluster_key not in obs columns, verify helpful error."""
        result, _, _ = self._build_and_call_tool(
            mock_data_manager,
            mock_singlecell_service,
            mock_vector_service,
            tool_kwargs={"cluster_key": "nonexistent_clustering"},
        )
        assert "not found" in result.lower()
        assert "Available columns" in result

    def test_save_result_false(
        self, mock_data_manager, mock_singlecell_service, mock_vector_service, mock_provider_config
    ):
        """Call with save_result=False, verify modality NOT stored."""
        result, dm, _ = self._build_and_call_tool(
            mock_data_manager,
            mock_singlecell_service,
            mock_vector_service,
            tool_kwargs={"save_result": False},
        )

        # store_modality should NOT have been called when save_result=False
        dm.store_modality.assert_not_called()

    def test_marker_query_format(
        self, mock_data_manager, mock_singlecell_service, mock_vector_service, mock_provider_config
    ):
        """Verify query text format matches 'Cluster N: high GENE1, GENE2, ...' pattern."""
        vs, backend = mock_vector_service

        # Track queries by intercepting embed_text calls
        queries_seen = []
        original_embed = vs._get_embedder().embed_text

        def tracking_embed(text):
            queries_seen.append(text)
            return original_embed(text)

        vs._get_embedder().embed_text = tracking_embed

        result, _, _ = self._build_and_call_tool(
            mock_data_manager, mock_singlecell_service, mock_vector_service
        )

        # Check that at least one query matches the expected format
        assert len(queries_seen) >= 3, f"Expected >= 3 queries, got {len(queries_seen)}"
        for q in queries_seen:
            assert q.startswith("Cluster "), f"Query should start with 'Cluster': {q}"
            assert ": high " in q or ": unknown markers" in q, (
                f"Query should contain ': high ' or ': unknown markers': {q}"
            )


class TestConditionalRegistration:
    """Test that semantic tool is conditionally registered."""

    def test_conditional_registration_when_available(self, mock_data_manager, mock_provider_config):
        """Verify annotate_cell_types_semantic is in tools list when vector search deps are available."""
        # HAS_VECTOR_SEARCH was refactored from a module-level constant into a
        # factory-local variable (Hard Rule #10). Use the test module's own check.
        assert HAS_VECTOR_SEARCH is True

        # Capture tools list by patching create_react_agent
        with patch(
            "lobster.agents.transcriptomics.annotation_expert.EnhancedSingleCellService",
        ), patch(
            "lobster.agents.transcriptomics.annotation_expert.ManualAnnotationService",
        ), patch(
            "lobster.agents.transcriptomics.annotation_expert.AnnotationTemplateService",
        ), patch(
            "lobster.agents.transcriptomics.annotation_expert.create_react_agent",
        ) as mock_create:
            from lobster.agents.transcriptomics.annotation_expert import (
                annotation_expert as ae_factory,
            )

            mock_create.return_value = Mock()
            ae_factory(mock_data_manager)

            call_kwargs = mock_create.call_args
            tools = call_kwargs.kwargs.get("tools", []) or (
                call_kwargs.args[1] if len(call_kwargs.args) > 1 else []
            )

        tool_names = [getattr(t, "name", str(t)) for t in tools]
        assert "annotate_cell_types_semantic" in tool_names, (
            f"annotate_cell_types_semantic should be in tool list when HAS_VECTOR_SEARCH=True. "
            f"Got: {tool_names}"
        )

        # Also verify existing tools are still present
        assert "annotate_cell_types_auto" in tool_names
        assert "manually_annotate_clusters" in tool_names


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
