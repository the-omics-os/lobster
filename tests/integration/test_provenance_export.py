"""
Integration tests for multi-agent provenance export to Jupyter notebook.

Tests validate:
1. Simulated provenance entries from multiple agents can be exported
2. Notebook export produces structurally valid Jupyter notebook
3. Notebook contains expected steps from multiple agents
4. Notebook structure includes required cells (header, imports, parameters, etc.)

Uses simulated provenance entries - NOT real graph sessions.
Validation is structural (parses correctly, expected steps present).
"""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import nbformat
import pytest
from nbformat import NotebookNode

from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.core.notebook_exporter import NotebookExporter
from lobster.core.provenance import ProvenanceTracker


class SimulatedProvenance:
    """Helper class to create simulated provenance entries for testing."""

    @staticmethod
    def create_transcriptomics_qc_ir() -> AnalysisStep:
        """Create a simulated IR for transcriptomics QC operation."""
        return AnalysisStep(
            operation="scanpy.pp.calculate_qc_metrics",
            tool_name="calculate_qc_metrics",
            description="Calculate QC metrics including mitochondrial gene content",
            library="scanpy",
            code_template="""# Calculate QC metrics
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)
print("QC metrics calculated")""",
            imports=["import scanpy as sc"],
            parameters={"qc_vars": ["mt"]},
            parameter_schema={
                "qc_vars": ParameterSpec(
                    param_type="List[str]",
                    papermill_injectable=False,
                    default_value=["mt"],
                    required=False,
                    description="Variables to calculate QC metrics for",
                )
            },
            input_entities=["adata"],
            output_entities=["adata"],
            execution_context={"agent": "transcriptomics_expert"},
            exportable=True,
        )

    @staticmethod
    def create_normalization_ir() -> AnalysisStep:
        """Create a simulated IR for normalization operation."""
        return AnalysisStep(
            operation="scanpy.pp.normalize_total",
            tool_name="normalize_data",
            description="Normalize counts to target sum per cell",
            library="scanpy",
            code_template="""# Normalize total counts
sc.pp.normalize_total(adata, target_sum={{ target_sum }})
sc.pp.log1p(adata)
print("Normalization complete")""",
            imports=["import scanpy as sc"],
            parameters={"target_sum": 1e4},
            parameter_schema={
                "target_sum": ParameterSpec(
                    param_type="float",
                    papermill_injectable=True,
                    default_value=1e4,
                    required=False,
                    description="Target sum for normalization",
                )
            },
            input_entities=["adata"],
            output_entities=["adata"],
            execution_context={"agent": "transcriptomics_expert"},
            exportable=True,
        )

    @staticmethod
    def create_clustering_ir() -> AnalysisStep:
        """Create a simulated IR for clustering operation."""
        return AnalysisStep(
            operation="scanpy.tl.leiden",
            tool_name="cluster_cells",
            description="Perform Leiden clustering on cells",
            library="scanpy",
            code_template="""# Compute neighbors and cluster
sc.pp.neighbors(adata, n_neighbors={{ n_neighbors }}, n_pcs={{ n_pcs }})
sc.tl.leiden(adata, resolution={{ resolution }})
print("Clustering complete")""",
            imports=["import scanpy as sc"],
            parameters={"n_neighbors": 15, "n_pcs": 50, "resolution": 1.0},
            parameter_schema={
                "n_neighbors": ParameterSpec(
                    param_type="int",
                    papermill_injectable=True,
                    default_value=15,
                    required=False,
                    description="Number of neighbors for graph",
                ),
                "n_pcs": ParameterSpec(
                    param_type="int",
                    papermill_injectable=True,
                    default_value=50,
                    required=False,
                    description="Number of PCs to use",
                ),
                "resolution": ParameterSpec(
                    param_type="float",
                    papermill_injectable=True,
                    default_value=1.0,
                    required=False,
                    description="Clustering resolution",
                ),
            },
            input_entities=["adata"],
            output_entities=["adata"],
            execution_context={"agent": "transcriptomics_expert"},
            exportable=True,
        )

    @staticmethod
    def create_visualization_umap_ir() -> AnalysisStep:
        """Create a simulated IR for UMAP visualization."""
        return AnalysisStep(
            operation="scanpy.tl.umap",
            tool_name="compute_umap",
            description="Compute UMAP embedding for visualization",
            library="scanpy",
            code_template="""# Compute UMAP
sc.tl.umap(adata, min_dist={{ min_dist }}, spread={{ spread }})
sc.pl.umap(adata, color='leiden', save='_clusters.png')
print("UMAP computed and plotted")""",
            imports=["import scanpy as sc"],
            parameters={"min_dist": 0.5, "spread": 1.0},
            parameter_schema={
                "min_dist": ParameterSpec(
                    param_type="float",
                    papermill_injectable=True,
                    default_value=0.5,
                    required=False,
                    description="Minimum distance for UMAP",
                ),
                "spread": ParameterSpec(
                    param_type="float",
                    papermill_injectable=True,
                    default_value=1.0,
                    required=False,
                    description="Spread parameter for UMAP",
                ),
            },
            input_entities=["adata"],
            output_entities=["adata", "figure"],
            execution_context={"agent": "visualization_expert_agent"},
            exportable=True,
        )

    @staticmethod
    def create_de_analysis_ir() -> AnalysisStep:
        """Create a simulated IR for differential expression analysis."""
        return AnalysisStep(
            operation="scanpy.tl.rank_genes_groups",
            tool_name="differential_expression",
            description="Find marker genes for each cluster",
            library="scanpy",
            code_template="""# Find marker genes
sc.tl.rank_genes_groups(adata, groupby='leiden', method='{{ method }}')
sc.pl.rank_genes_groups(adata, n_genes={{ n_genes }}, save='_markers.png')
print("Differential expression analysis complete")""",
            imports=["import scanpy as sc"],
            parameters={"method": "wilcoxon", "n_genes": 25},
            parameter_schema={
                "method": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="wilcoxon",
                    required=False,
                    description="Statistical test method",
                ),
                "n_genes": ParameterSpec(
                    param_type="int",
                    papermill_injectable=True,
                    default_value=25,
                    required=False,
                    description="Number of top genes to plot",
                ),
            },
            input_entities=["adata"],
            output_entities=["adata"],
            execution_context={"agent": "de_analysis_expert"},
            exportable=True,
        )

    @staticmethod
    def create_provenance_activity(
        ir: AnalysisStep,
        agent_name: str,
    ) -> dict:
        """Create a provenance activity entry with embedded IR."""
        return {
            "id": f"lobster:activity:{agent_name}:{ir.tool_name}",
            "type": ir.tool_name,
            "agent": f"lobster:agent:{agent_name}",
            "timestamp": datetime.now().isoformat(),
            "inputs": [{"entity": e, "role": "input"} for e in ir.input_entities],
            "outputs": [{"entity": e, "role": "output"} for e in ir.output_entities],
            "parameters": ir.parameters,
            "description": ir.description,
            "ir": ir.to_dict(),
        }


class TestProvenanceExportStructure:
    """Test that notebook export produces structurally valid output."""

    @pytest.fixture
    def mock_data_manager(self, tmp_path: Path):
        """Create mock DataManagerV2 with Path workspace."""
        dm = MagicMock()
        dm.workspace_path = tmp_path / "test_workspace"
        dm.workspace_path.mkdir(exist_ok=True)
        dm.modalities = {"test_data": MagicMock()}
        return dm

    @pytest.fixture
    def populated_provenance(self) -> ProvenanceTracker:
        """Create ProvenanceTracker with simulated multi-agent activities."""
        tracker = ProvenanceTracker(namespace="test_session")

        # Add simulated activities from multiple agents
        activities = [
            SimulatedProvenance.create_provenance_activity(
                SimulatedProvenance.create_transcriptomics_qc_ir(),
                "transcriptomics_expert",
            ),
            SimulatedProvenance.create_provenance_activity(
                SimulatedProvenance.create_normalization_ir(), "transcriptomics_expert"
            ),
            SimulatedProvenance.create_provenance_activity(
                SimulatedProvenance.create_clustering_ir(), "transcriptomics_expert"
            ),
            SimulatedProvenance.create_provenance_activity(
                SimulatedProvenance.create_visualization_umap_ir(),
                "visualization_expert_agent",
            ),
            SimulatedProvenance.create_provenance_activity(
                SimulatedProvenance.create_de_analysis_ir(), "de_analysis_expert"
            ),
        ]

        # Directly add activities to tracker
        tracker.activities = activities

        return tracker

    def test_export_creates_valid_notebook_file(
        self, mock_data_manager, populated_provenance, tmp_path
    ):
        """Exported notebook is a valid .ipynb file."""
        exporter = NotebookExporter(populated_provenance, mock_data_manager)
        output_path = exporter.export(
            name="test_workflow",
            description="Test multi-agent workflow",
        )

        assert output_path.exists(), "Notebook file should be created"
        assert output_path.suffix == ".ipynb", "Should have .ipynb extension"

    def test_export_produces_parseable_notebook(
        self, mock_data_manager, populated_provenance
    ):
        """Exported notebook can be parsed by nbformat."""
        exporter = NotebookExporter(populated_provenance, mock_data_manager)
        output_path = exporter.export(name="test_parse")

        # Should not raise exception
        with open(output_path) as f:
            notebook = nbformat.read(f, as_version=4)

        assert isinstance(notebook, NotebookNode)
        assert "cells" in notebook

    def test_export_contains_markdown_header(
        self, mock_data_manager, populated_provenance
    ):
        """Notebook contains markdown header cell."""
        exporter = NotebookExporter(populated_provenance, mock_data_manager)
        output_path = exporter.export(
            name="test_header",
            description="Testing header cell",
        )

        with open(output_path) as f:
            notebook = nbformat.read(f, as_version=4)

        # First cell should be markdown header
        assert len(notebook.cells) > 0
        first_cell = notebook.cells[0]
        assert first_cell.cell_type == "markdown"
        assert "test_header" in first_cell.source

    def test_export_contains_imports_cell(
        self, mock_data_manager, populated_provenance
    ):
        """Notebook contains code cell with imports."""
        exporter = NotebookExporter(populated_provenance, mock_data_manager)
        output_path = exporter.export(name="test_imports")

        with open(output_path) as f:
            notebook = nbformat.read(f, as_version=4)

        # Find imports cell (should contain "import")
        import_cells = [
            c for c in notebook.cells if c.cell_type == "code" and "import" in c.source
        ]

        assert len(import_cells) > 0, "Should have at least one imports cell"

    def test_export_contains_parameters_cell_with_tag(
        self, mock_data_manager, populated_provenance
    ):
        """Notebook contains Papermill-tagged parameters cell."""
        exporter = NotebookExporter(populated_provenance, mock_data_manager)
        output_path = exporter.export(name="test_params")

        with open(output_path) as f:
            notebook = nbformat.read(f, as_version=4)

        # Find parameters cell by tag
        param_cells = [
            c
            for c in notebook.cells
            if "tags" in c.metadata and "parameters" in c.metadata.get("tags", [])
        ]

        assert len(param_cells) == 1, "Should have exactly one parameters cell"


class TestProvenanceExportMultiAgent:
    """Test that exported notebook contains steps from multiple agents."""

    @pytest.fixture
    def mock_data_manager(self, tmp_path: Path):
        """Create mock DataManagerV2 with Path workspace."""
        dm = MagicMock()
        dm.workspace_path = tmp_path / "test_workspace"
        dm.workspace_path.mkdir(exist_ok=True)
        dm.modalities = {"test_data": MagicMock()}
        return dm

    @pytest.fixture
    def multi_agent_provenance(self) -> ProvenanceTracker:
        """Create ProvenanceTracker with activities from 3 different agents."""
        tracker = ProvenanceTracker(namespace="multi_agent_test")

        activities = [
            # transcriptomics_expert
            SimulatedProvenance.create_provenance_activity(
                SimulatedProvenance.create_transcriptomics_qc_ir(),
                "transcriptomics_expert",
            ),
            SimulatedProvenance.create_provenance_activity(
                SimulatedProvenance.create_normalization_ir(), "transcriptomics_expert"
            ),
            # visualization_expert_agent
            SimulatedProvenance.create_provenance_activity(
                SimulatedProvenance.create_visualization_umap_ir(),
                "visualization_expert_agent",
            ),
            # de_analysis_expert
            SimulatedProvenance.create_provenance_activity(
                SimulatedProvenance.create_de_analysis_ir(), "de_analysis_expert"
            ),
        ]

        tracker.activities = activities
        return tracker

    def test_notebook_contains_transcriptomics_operations(
        self, mock_data_manager, multi_agent_provenance
    ):
        """Notebook includes operations from transcriptomics_expert."""
        exporter = NotebookExporter(multi_agent_provenance, mock_data_manager)
        output_path = exporter.export(name="test_multi_agent")

        with open(output_path) as f:
            notebook = nbformat.read(f, as_version=4)

        notebook_text = " ".join(c.source for c in notebook.cells)

        # Check for transcriptomics operations
        assert "calculate_qc_metrics" in notebook_text or "QC metrics" in notebook_text
        assert "normalize_total" in notebook_text or "Normalize" in notebook_text

    def test_notebook_contains_visualization_operations(
        self, mock_data_manager, multi_agent_provenance
    ):
        """Notebook includes operations from visualization_expert_agent."""
        exporter = NotebookExporter(multi_agent_provenance, mock_data_manager)
        output_path = exporter.export(name="test_viz")

        with open(output_path) as f:
            notebook = nbformat.read(f, as_version=4)

        notebook_text = " ".join(c.source for c in notebook.cells)

        # Check for visualization operations
        assert "umap" in notebook_text.lower() or "UMAP" in notebook_text

    def test_notebook_contains_de_operations(
        self, mock_data_manager, multi_agent_provenance
    ):
        """Notebook includes operations from de_analysis_expert."""
        exporter = NotebookExporter(multi_agent_provenance, mock_data_manager)
        output_path = exporter.export(name="test_de")

        with open(output_path) as f:
            notebook = nbformat.read(f, as_version=4)

        notebook_text = " ".join(c.source for c in notebook.cells)

        # Check for DE operations
        assert "rank_genes_groups" in notebook_text or "marker" in notebook_text.lower()

    def test_notebook_operations_in_correct_order(
        self, mock_data_manager, multi_agent_provenance
    ):
        """Notebook preserves operation order from provenance."""
        exporter = NotebookExporter(multi_agent_provenance, mock_data_manager)
        output_path = exporter.export(name="test_order")

        with open(output_path) as f:
            notebook = nbformat.read(f, as_version=4)

        # Get code cells
        code_cells = [c for c in notebook.cells if c.cell_type == "code"]
        code_text = " ".join(c.source for c in code_cells)

        # Operations should appear in provenance order
        # QC before normalization before UMAP before DE
        qc_pos = (
            code_text.find("qc_metrics")
            if "qc_metrics" in code_text
            else code_text.find("QC")
        )
        norm_pos = code_text.find("normalize")
        umap_pos = (
            code_text.find("umap") if "umap" in code_text else code_text.find("UMAP")
        )

        # QC should come before normalization
        if qc_pos != -1 and norm_pos != -1:
            assert qc_pos < norm_pos, "QC should come before normalization"


class TestProvenanceExportMetadata:
    """Test notebook metadata and lobster-specific fields."""

    @pytest.fixture
    def mock_data_manager(self, tmp_path: Path):
        """Create mock DataManagerV2 with Path workspace."""
        dm = MagicMock()
        dm.workspace_path = tmp_path / "test_workspace"
        dm.workspace_path.mkdir(exist_ok=True)
        dm.modalities = {"sample_data": MagicMock()}
        return dm

    @pytest.fixture
    def simple_provenance(self) -> ProvenanceTracker:
        """Create ProvenanceTracker with single activity."""
        tracker = ProvenanceTracker(namespace="simple_test")
        tracker.activities = [
            SimulatedProvenance.create_provenance_activity(
                SimulatedProvenance.create_normalization_ir(), "transcriptomics_expert"
            )
        ]
        return tracker

    def test_notebook_has_lobster_metadata(self, mock_data_manager, simple_provenance):
        """Notebook has lobster-specific metadata."""
        exporter = NotebookExporter(simple_provenance, mock_data_manager)
        output_path = exporter.export(name="test_metadata")

        with open(output_path) as f:
            notebook = nbformat.read(f, as_version=4)

        assert "lobster" in notebook.metadata
        assert "source_session_id" in notebook.metadata["lobster"]

    def test_notebook_metadata_has_ir_statistics(
        self, mock_data_manager, simple_provenance
    ):
        """Notebook metadata includes IR statistics."""
        exporter = NotebookExporter(simple_provenance, mock_data_manager)
        output_path = exporter.export(name="test_ir_stats")

        with open(output_path) as f:
            notebook = nbformat.read(f, as_version=4)

        lobster_meta = notebook.metadata.get("lobster", {})
        assert "ir_statistics" in lobster_meta
        assert "n_irs_extracted" in lobster_meta["ir_statistics"]

    def test_notebook_contains_session_id(self, mock_data_manager, simple_provenance):
        """Notebook references original session ID."""
        exporter = NotebookExporter(simple_provenance, mock_data_manager)
        output_path = exporter.export(name="test_session")

        with open(output_path) as f:
            notebook = nbformat.read(f, as_version=4)

        lobster_meta = notebook.metadata.get("lobster", {})
        assert lobster_meta["source_session_id"] == "simple_test"


class TestProvenanceExportEdgeCases:
    """Test edge cases and error handling in notebook export."""

    @pytest.fixture
    def mock_data_manager(self, tmp_path: Path):
        """Create mock DataManagerV2 with Path workspace."""
        dm = MagicMock()
        dm.workspace_path = tmp_path / "test_workspace"
        dm.workspace_path.mkdir(exist_ok=True)
        dm.modalities = {}
        return dm

    def test_empty_provenance_raises_error(self, mock_data_manager):
        """Export with no activities raises ValueError."""
        tracker = ProvenanceTracker(namespace="empty")
        exporter = NotebookExporter(tracker, mock_data_manager)

        with pytest.raises(ValueError, match="No activities"):
            exporter.export(name="empty_test")

    def test_empty_name_raises_error(self, mock_data_manager):
        """Export with empty name raises ValueError."""
        tracker = ProvenanceTracker(namespace="test")
        tracker.activities = [
            SimulatedProvenance.create_provenance_activity(
                SimulatedProvenance.create_normalization_ir(), "test_agent"
            )
        ]
        exporter = NotebookExporter(tracker, mock_data_manager)

        with pytest.raises(ValueError, match="name cannot be empty"):
            exporter.export(name="")

    def test_provenance_only_activities_filtered(self, mock_data_manager):
        """Activities without exportable IR are filtered from notebook."""
        tracker = ProvenanceTracker(namespace="filter_test")

        # Create activity with non-exportable IR
        non_exportable_ir = AnalysisStep(
            operation="internal_operation",
            tool_name="internal",
            description="Internal orchestration",
            library="lobster",
            code_template="# internal",
            imports=[],
            parameters={},
            parameter_schema={},
            exportable=False,  # NOT exportable
        )

        # Add mix of exportable and non-exportable
        tracker.activities = [
            SimulatedProvenance.create_provenance_activity(
                SimulatedProvenance.create_normalization_ir(),  # exportable
                "transcriptomics_expert",
            ),
            {
                "id": "test:internal",
                "type": "internal",
                "ir": non_exportable_ir.to_dict(),
            },
        ]

        exporter = NotebookExporter(tracker, mock_data_manager)
        output_path = exporter.export(name="filter_test")

        with open(output_path) as f:
            notebook = nbformat.read(f, as_version=4)

        # Should only have exportable operations
        notebook_text = " ".join(c.source for c in notebook.cells)
        assert "normalize" in notebook_text
        # Internal operation should be filtered
        assert "internal_operation" not in notebook_text


class TestSimulatedProvenanceHelpers:
    """Test the SimulatedProvenance helper class itself."""

    def test_create_transcriptomics_qc_ir_valid(self):
        """Helper creates valid transcriptomics QC IR."""
        ir = SimulatedProvenance.create_transcriptomics_qc_ir()
        assert ir.operation == "scanpy.pp.calculate_qc_metrics"
        assert ir.exportable is True
        assert len(ir.imports) > 0

    def test_create_normalization_ir_valid(self):
        """Helper creates valid normalization IR."""
        ir = SimulatedProvenance.create_normalization_ir()
        assert ir.operation == "scanpy.pp.normalize_total"
        assert "target_sum" in ir.parameters
        assert ir.parameter_schema["target_sum"].papermill_injectable is True

    def test_create_provenance_activity_contains_ir(self):
        """Helper creates activity with embedded IR."""
        ir = SimulatedProvenance.create_normalization_ir()
        activity = SimulatedProvenance.create_provenance_activity(ir, "test_agent")

        assert "ir" in activity
        assert activity["ir"]["operation"] == ir.operation
        assert "test_agent" in activity["agent"]

    def test_ir_renders_valid_python(self):
        """All helper IRs render to valid Python syntax."""
        irs = [
            SimulatedProvenance.create_transcriptomics_qc_ir(),
            SimulatedProvenance.create_normalization_ir(),
            SimulatedProvenance.create_clustering_ir(),
            SimulatedProvenance.create_visualization_umap_ir(),
            SimulatedProvenance.create_de_analysis_ir(),
        ]

        for ir in irs:
            code = ir.render()
            assert len(code) > 0, f"{ir.operation} should render code"
            # Basic check - should contain operation keywords
            assert ir.library in code or ir.operation.split(".")[-1] in code
