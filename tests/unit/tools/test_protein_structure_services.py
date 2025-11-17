"""
Comprehensive unit tests for protein structure visualization services.

This module tests:
- ProteinStructureFetchService: Fetching structures from PDB
- ChimeraXVisualizationService: Creating ChimeraX visualizations
- StructureAnalysisService: Analyzing structure properties and RMSD

Test coverage target: 85%+ with meaningful tests for structure operations.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, mock_open

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from Bio import PDB

from lobster.core.analysis_ir import AnalysisStep
from lobster.tools.chimerax_visualization_service_ALPHA import (
    ChimeraXVisualizationError,
    ChimeraXVisualizationService,
)
from lobster.tools.pymol_visualization_service import (
    PyMOLVisualizationService,
    PyMOLVisualizationError,
)
from lobster.tools.protein_structure_fetch_service import (
    ProteinStructureFetchError,
    ProteinStructureFetchService,
)
from lobster.tools.providers.pdb_provider import PDBStructureMetadata
from lobster.tools.structure_analysis_service import (
    StructureAnalysisError,
    StructureAnalysisService,
)

# ===============================================================================
# Fixtures
# ===============================================================================


@pytest.fixture
def fetch_service():
    """Create ProteinStructureFetchService instance."""
    return ProteinStructureFetchService()


@pytest.fixture
def viz_service():
    """Create ChimeraXVisualizationService instance."""
    return ChimeraXVisualizationService()


@pytest.fixture
def pymol_viz_service():
    """Create PyMOLVisualizationService instance."""
    return PyMOLVisualizationService()


@pytest.fixture
def analysis_service():
    """Create StructureAnalysisService instance."""
    return StructureAnalysisService()


@pytest.fixture
def mock_pdb_metadata():
    """Create mock PDB structure metadata."""
    return PDBStructureMetadata(
        pdb_id="1AKE",
        title="Adenylate Kinase from E. coli",
        experiment_method="X-RAY DIFFRACTION",
        resolution=2.0,
        organism="Escherichia coli",
        chains=["A"],
        ligands=["AP5"],
        deposition_date="1990-05-15",
        release_date="1991-01-15",
        authors=["Mueller, C.W.", "Schulz, G.E."],
        publication_doi="10.1016/0022-2836(92)90693-E",
        citation="J. Mol. Biol. (1992)",
    )


@pytest.fixture
def mock_structure_file(tmp_path):
    """Create mock PDB structure file."""
    pdb_content = """HEADER    TRANSFERASE                             15-MAY-90   1AKE
TITLE     ADENYLATE KINASE FROM E. COLI
ATOM      1  N   MET A   1      27.340  24.430   2.614  1.00  9.67           N
ATOM      2  CA  MET A   1      26.266  25.413   2.842  1.00 10.38           C
ATOM      3  C   MET A   1      26.913  26.639   3.531  1.00  9.62           C
ATOM      4  O   MET A   1      27.886  26.463   4.263  1.00  9.62           O
END
"""
    structure_file = tmp_path / "1AKE.pdb"
    structure_file.write_text(pdb_content)
    return structure_file


@pytest.fixture
def sample_adata():
    """Create sample AnnData with gene symbols."""
    n_obs, n_vars = 100, 50
    X = np.random.negative_binomial(5, 0.3, size=(n_obs, n_vars))
    adata = ad.AnnData(X=X)
    adata.obs_names = [f"Cell_{i}" for i in range(n_obs)]
    adata.var_names = [f"Gene_{i}" for i in range(n_vars)]
    adata.var["gene_symbol"] = [f"GENE{i}" for i in range(n_vars)]
    return adata


# ===============================================================================
# ProteinStructureFetchService Tests
# ===============================================================================


class TestProteinStructureFetchService:
    """Test suite for ProteinStructureFetchService."""

    def test_service_initialization(self, fetch_service):
        """Test that service initializes correctly."""
        assert fetch_service is not None
        assert fetch_service.config == {}

    def test_validate_pdb_id_format_valid(self, fetch_service):
        """Test PDB ID format validation with valid IDs."""
        assert fetch_service._validate_pdb_id_format("1AKE") is True
        assert fetch_service._validate_pdb_id_format("4HHB") is True
        assert fetch_service._validate_pdb_id_format("2ABC") is True

    def test_validate_pdb_id_format_invalid(self, fetch_service):
        """Test PDB ID format validation with invalid IDs."""
        assert fetch_service._validate_pdb_id_format("") is False
        assert fetch_service._validate_pdb_id_format("1AK") is False  # Too short
        assert fetch_service._validate_pdb_id_format("1AKEE") is False  # Too long
        assert fetch_service._validate_pdb_id_format("1AK-") is False  # Invalid char

    @patch("lobster.tools.protein_structure_fetch_service.PDBProvider")
    def test_fetch_structure_success(
        self, mock_provider_class, fetch_service, mock_pdb_metadata, mock_structure_file, tmp_path
    ):
        """Test successful structure fetch."""
        # Setup mock
        mock_provider = Mock()
        mock_provider.download_structure.return_value = mock_structure_file
        mock_provider.get_structure_metadata.return_value = mock_pdb_metadata
        mock_provider_class.return_value = mock_provider

        # Test fetch
        structure_data, stats, ir = fetch_service.fetch_structure(
            pdb_id="1AKE", format="pdb", cache_dir=tmp_path
        )

        # Assertions
        assert structure_data["pdb_id"] == "1AKE"
        assert structure_data["file_format"] == "pdb"
        assert stats["pdb_id"] == "1AKE"
        assert stats["resolution"] == 2.0
        assert isinstance(ir, AnalysisStep)
        assert ir.operation == "pdb.fetch_structure"

    def test_fetch_structure_invalid_pdb_id(self, fetch_service, tmp_path):
        """Test fetch with invalid PDB ID."""
        with pytest.raises(ProteinStructureFetchError, match="Invalid PDB ID format"):
            fetch_service.fetch_structure(pdb_id="INVALID", cache_dir=tmp_path)

    @patch("lobster.tools.protein_structure_fetch_service.PDBProvider")
    def test_fetch_structure_cached(
        self, mock_provider_class, fetch_service, mock_pdb_metadata, mock_structure_file, tmp_path
    ):
        """Test fetch with cached structure."""
        # Create cached file
        cached_file = tmp_path / "1AKE.pdb"
        mock_structure_file.rename(cached_file)

        # Setup mock
        mock_provider = Mock()
        mock_provider.get_structure_metadata.return_value = mock_pdb_metadata
        mock_provider_class.return_value = mock_provider

        # Test fetch (should use cache)
        structure_data, stats, ir = fetch_service.fetch_structure(
            pdb_id="1AKE", format="pdb", cache_dir=tmp_path
        )

        # Should not call download_structure
        mock_provider.download_structure.assert_not_called()
        assert stats["cached"] is True

    @patch("lobster.tools.protein_structure_fetch_service.PDBProvider")
    def test_link_structures_to_genes(
        self, mock_provider_class, fetch_service, sample_adata
    ):
        """Test linking genes to structures."""
        # Setup mock
        mock_provider = Mock()
        mock_search_result = Mock()
        mock_search_result.uid = "1AKE"
        mock_provider.search_publications.return_value = [mock_search_result]
        mock_provider_class.return_value = mock_provider

        # Test linking
        adata_linked, stats, ir = fetch_service.link_structures_to_genes(
            adata=sample_adata, gene_column="gene_symbol", organism="Homo sapiens"
        )

        # Assertions
        assert "pdb_structures" in adata_linked.var.columns
        assert "has_structure" in adata_linked.var.columns
        assert stats["genes_searched"] > 0
        assert isinstance(ir, AnalysisStep)
        assert ir.operation == "pdb.link_genes_to_structures"

    def test_link_structures_invalid_gene_column(self, fetch_service, sample_adata):
        """Test linking with invalid gene column."""
        with pytest.raises(ProteinStructureFetchError, match="Gene column.*not found"):
            fetch_service.link_structures_to_genes(
                adata=sample_adata, gene_column="invalid_column"
            )

    def test_metadata_to_dict(self, fetch_service, mock_pdb_metadata):
        """Test metadata conversion to dictionary."""
        metadata_dict = fetch_service._metadata_to_dict(mock_pdb_metadata)

        assert metadata_dict["pdb_id"] == "1AKE"
        assert metadata_dict["title"] == "Adenylate Kinase from E. coli"
        assert metadata_dict["resolution"] == 2.0
        assert metadata_dict["organism"] == "Escherichia coli"


# ===============================================================================
# ChimeraXVisualizationService Tests
# ===============================================================================


class TestChimeraXVisualizationService:
    """Test suite for ChimeraXVisualizationService."""

    def test_service_initialization(self, viz_service):
        """Test that service initializes correctly."""
        assert viz_service is not None
        assert viz_service._chimerax_available is None

    def test_generate_chimerax_commands_cartoon(
        self, viz_service, mock_structure_file, tmp_path
    ):
        """Test ChimeraX command generation for cartoon style."""
        output_image = tmp_path / "output.png"
        commands = viz_service._generate_chimerax_commands(
            structure_file=mock_structure_file,
            style="cartoon",
            color_by="chain",
            output_image=output_image,
            width=1920,
            height=1080,
            background="white",
        )

        # Check commands
        assert any("open" in cmd for cmd in commands)
        assert any("cartoon" in cmd for cmd in commands)
        assert any("color bychain" in cmd for cmd in commands)
        assert any("save" in cmd for cmd in commands)
        assert any("exit" in cmd for cmd in commands)

    def test_generate_chimerax_commands_surface(
        self, viz_service, mock_structure_file, tmp_path
    ):
        """Test ChimeraX command generation for surface style."""
        output_image = tmp_path / "output.png"
        commands = viz_service._generate_chimerax_commands(
            structure_file=mock_structure_file,
            style="surface",
            color_by="hydrophobicity",
            output_image=output_image,
            width=1920,
            height=1080,
            background="black",
        )

        assert any("surface" in cmd for cmd in commands)
        assert any("bykdhydrophobicity" in cmd for cmd in commands)

    def test_visualize_structure_file_not_found(self, viz_service, tmp_path):
        """Test visualization with non-existent structure file."""
        fake_file = tmp_path / "nonexistent.pdb"

        with pytest.raises(ChimeraXVisualizationError, match="Structure file not found"):
            viz_service.visualize_structure(
                structure_file=fake_file, execute_commands=False
            )

    def test_visualize_structure_success(
        self, viz_service, mock_structure_file, tmp_path
    ):
        """Test successful visualization (script generation, no execution)."""
        viz_data, stats, ir = viz_service.visualize_structure(
            structure_file=mock_structure_file,
            style="cartoon",
            color_by="chain",
            execute_commands=False,
        )

        # Assertions
        assert viz_data["style"] == "cartoon"
        assert viz_data["color_by"] == "chain"
        assert viz_data["executed"] is False
        assert stats["style"] == "cartoon"
        assert isinstance(ir, AnalysisStep)
        assert ir.operation == "chimerax.visualize_structure"
        assert Path(viz_data["script_file"]).exists()

    @patch("subprocess.run")
    def test_check_chimerax_installation_in_path(self, mock_run, viz_service):
        """Test ChimeraX installation check when in PATH."""
        mock_run.return_value = Mock(returncode=0, stdout="ChimeraX 1.5")

        result = viz_service.check_chimerax_installation()

        assert result["installed"] is True
        assert result["path"] == "chimerax"
        assert "1.5" in result["version"]

    @patch("subprocess.run")
    @patch("pathlib.Path.exists")
    def test_check_chimerax_installation_not_found(
        self, mock_exists, mock_run, viz_service
    ):
        """Test ChimeraX installation check when not found."""
        mock_run.side_effect = FileNotFoundError()
        mock_exists.return_value = False

        result = viz_service.check_chimerax_installation()

        assert result["installed"] is False
        assert result["path"] is None
        assert "not found" in result["message"].lower()


# ===============================================================================
# PyMOLVisualizationService Tests
# ===============================================================================


class TestPyMOLVisualizationService:
    """Test suite for PyMOLVisualizationService with residue highlighting."""

    def test_service_initialization(self, pymol_viz_service):
        """Test that PyMOL service initializes correctly."""
        assert pymol_viz_service is not None
        assert pymol_viz_service._pymol_available is None

    def test_parse_highlight_groups_single(self, pymol_viz_service):
        """Test parsing single highlight group."""
        groups = pymol_viz_service._parse_highlight_groups(
            highlight_residues="15,42,89",
            highlight_color="red",
            highlight_style="sticks",
            highlight_groups=None,
        )

        assert len(groups) == 1
        assert groups[0]["residues"] == "15,42,89"
        assert groups[0]["color"] == "red"
        assert groups[0]["style"] == "sticks"
        assert groups[0]["label"] == "highlight_residues"
        assert groups[0]["pymol_selection"] == "resi 15+42+89"

    def test_parse_highlight_groups_multiple(self, pymol_viz_service):
        """Test parsing multiple highlight groups."""
        groups = pymol_viz_service._parse_highlight_groups(
            highlight_residues=None,
            highlight_color="red",
            highlight_style="sticks",
            highlight_groups="15,42|red|sticks;100-120|blue|surface;200,215|green|spheres",
        )

        assert len(groups) == 3

        # First group
        assert groups[0]["residues"] == "15,42"
        assert groups[0]["color"] == "red"
        assert groups[0]["style"] == "sticks"
        assert groups[0]["label"] == "highlight_group_1"
        assert groups[0]["pymol_selection"] == "resi 15+42"

        # Second group
        assert groups[1]["residues"] == "100-120"
        assert groups[1]["color"] == "blue"
        assert groups[1]["style"] == "surface"
        assert groups[1]["label"] == "highlight_group_2"
        assert groups[1]["pymol_selection"] == "resi 100-120"

        # Third group
        assert groups[2]["residues"] == "200,215"
        assert groups[2]["color"] == "green"
        assert groups[2]["style"] == "spheres"
        assert groups[2]["label"] == "highlight_group_3"
        assert groups[2]["pymol_selection"] == "resi 200+215"

    def test_parse_highlight_groups_invalid_format(self, pymol_viz_service):
        """Test parsing with invalid format (should skip malformed groups)."""
        groups = pymol_viz_service._parse_highlight_groups(
            highlight_residues=None,
            highlight_color="red",
            highlight_style="sticks",
            highlight_groups="15,42|red;100-120|blue|surface",  # First group missing style
        )

        # Should only parse the valid second group
        assert len(groups) == 1
        assert groups[0]["residues"] == "100-120"
        assert groups[0]["color"] == "blue"

    def test_convert_to_pymol_selection_simple(self, pymol_viz_service):
        """Test conversion of simple residue list."""
        selection = pymol_viz_service._convert_to_pymol_selection("15,42,89")
        assert selection == "resi 15+42+89"

    def test_convert_to_pymol_selection_ranges(self, pymol_viz_service):
        """Test conversion of residue ranges."""
        selection = pymol_viz_service._convert_to_pymol_selection("15-20,42-50")
        assert selection == "resi 15-20+42-50"

    def test_convert_to_pymol_selection_chain_specific(self, pymol_viz_service):
        """Test conversion of chain-specific residues."""
        selection = pymol_viz_service._convert_to_pymol_selection("A:15,B:42")
        assert selection == "(chain A and resi 15) or (chain B and resi 42)"

    def test_convert_to_pymol_selection_chain_ranges(self, pymol_viz_service):
        """Test conversion of chain-specific ranges."""
        selection = pymol_viz_service._convert_to_pymol_selection("A:15-20,B:30-35")
        assert selection == "(chain A and resi 15-20) or (chain B and resi 30-35)"

    def test_generate_pymol_commands_with_single_highlight(
        self, pymol_viz_service, mock_structure_file, tmp_path
    ):
        """Test PyMOL command generation with single highlight group."""
        output_image = tmp_path / "output.png"

        # Parse highlight groups first
        highlight_groups_parsed = pymol_viz_service._parse_highlight_groups(
            highlight_residues="15,42,89",
            highlight_color="red",
            highlight_style="sticks",
            highlight_groups=None,
        )

        commands = pymol_viz_service._generate_pymol_commands(
            structure_file=mock_structure_file,
            mode="batch",
            style="cartoon",
            color_by="chain",
            output_image=output_image,
            width=1920,
            height=1080,
            background="white",
            highlight_groups_parsed=highlight_groups_parsed,
        )

        # Check highlight commands are present
        assert any("select highlight_residues, resi 15+42+89" in cmd for cmd in commands)
        assert any("show sticks, highlight_residues" in cmd for cmd in commands)
        assert any("color red, highlight_residues" in cmd for cmd in commands)

    def test_generate_pymol_commands_with_multiple_highlights(
        self, pymol_viz_service, mock_structure_file, tmp_path
    ):
        """Test PyMOL command generation with multiple highlight groups."""
        output_image = tmp_path / "output.png"

        # Parse highlight groups
        highlight_groups_parsed = pymol_viz_service._parse_highlight_groups(
            highlight_residues=None,
            highlight_color="red",
            highlight_style="sticks",
            highlight_groups="15,42|red|sticks;100-120|blue|surface",
        )

        commands = pymol_viz_service._generate_pymol_commands(
            structure_file=mock_structure_file,
            mode="batch",
            style="cartoon",
            color_by="chain",
            output_image=output_image,
            width=1920,
            height=1080,
            background="white",
            highlight_groups_parsed=highlight_groups_parsed,
        )

        # Check first group
        assert any("select highlight_group_1, resi 15+42" in cmd for cmd in commands)
        assert any("show sticks, highlight_group_1" in cmd for cmd in commands)
        assert any("color red, highlight_group_1" in cmd for cmd in commands)

        # Check second group
        assert any("select highlight_group_2, resi 100-120" in cmd for cmd in commands)
        assert any("show surface, highlight_group_2" in cmd for cmd in commands)
        assert any("color blue, highlight_group_2" in cmd for cmd in commands)

    def test_visualize_structure_with_highlights(
        self, pymol_viz_service, mock_structure_file, tmp_path
    ):
        """Test full visualization workflow with residue highlighting."""
        viz_data, stats, ir = pymol_viz_service.visualize_structure(
            structure_file=mock_structure_file,
            mode="batch",
            style="cartoon",
            color_by="chain",
            execute_commands=False,
            highlight_residues="15,42,89",
            highlight_color="red",
            highlight_style="sticks",
        )

        # Assertions
        assert viz_data["style"] == "cartoon"
        assert viz_data["color_by"] == "chain"
        assert viz_data["executed"] is False
        assert stats["style"] == "cartoon"
        assert isinstance(ir, AnalysisStep)
        assert ir.operation == "pymol.visualize_structure"
        assert Path(viz_data["script_file"]).exists()

        # Check that highlight parameters are in IR
        assert "highlight_residues" in ir.parameters
        assert ir.parameters["highlight_residues"] == "15,42,89"
        assert ir.parameters["highlight_color"] == "red"
        assert ir.parameters["highlight_style"] == "sticks"

    def test_visualize_structure_with_multiple_highlight_groups(
        self, pymol_viz_service, mock_structure_file, tmp_path
    ):
        """Test visualization with multiple highlight groups."""
        viz_data, stats, ir = pymol_viz_service.visualize_structure(
            structure_file=mock_structure_file,
            mode="batch",
            style="cartoon",
            color_by="chain",
            execute_commands=False,
            highlight_groups="15,42|red|sticks;100-120|blue|surface",
        )

        # Check IR includes highlight_groups parameter
        assert "highlight_groups" in ir.parameters
        assert ir.parameters["highlight_groups"] == "15,42|red|sticks;100-120|blue|surface"

        # Read script file and verify commands
        script_file = Path(viz_data["script_file"])
        script_content = script_file.read_text()

        assert "select highlight_group_1, resi 15+42" in script_content
        assert "show sticks, highlight_group_1" in script_content
        assert "color red, highlight_group_1" in script_content
        assert "select highlight_group_2, resi 100-120" in script_content
        assert "show surface, highlight_group_2" in script_content
        assert "color blue, highlight_group_2" in script_content

    def test_visualize_structure_no_highlights(
        self, pymol_viz_service, mock_structure_file, tmp_path
    ):
        """Test visualization without any highlights (baseline)."""
        viz_data, stats, ir = pymol_viz_service.visualize_structure(
            structure_file=mock_structure_file,
            mode="batch",
            style="cartoon",
            color_by="chain",
            execute_commands=False,
        )

        # Check that highlight parameters are None in IR
        assert ir.parameters["highlight_residues"] is None
        assert ir.parameters["highlight_groups"] is None

        # Read script file and verify no highlight commands
        script_file = Path(viz_data["script_file"])
        script_content = script_file.read_text()

        assert "select highlight" not in script_content

    def test_create_ir_with_highlights(self, pymol_viz_service, mock_structure_file):
        """Test IR creation includes highlight parameters."""
        ir = pymol_viz_service._create_visualization_ir(
            structure_file=mock_structure_file,
            style="cartoon",
            color_by="chain",
            width=1920,
            height=1080,
            background="white",
            highlight_residues="15,42,89",
            highlight_color="red",
            highlight_style="sticks",
            highlight_groups=None,
        )

        # Check IR structure
        assert ir.operation == "pymol.visualize_structure"
        assert ir.tool_name == "visualize_with_pymol"
        assert "with residue highlights" in ir.description

        # Check parameters
        assert ir.parameters["highlight_residues"] == "15,42,89"
        assert ir.parameters["highlight_color"] == "red"
        assert ir.parameters["highlight_style"] == "sticks"
        assert ir.parameters["highlight_groups"] is None

        # Check parameter schema
        assert "highlight_residues" in ir.parameter_schema
        assert "highlight_color" in ir.parameter_schema
        assert "highlight_style" in ir.parameter_schema
        assert "highlight_groups" in ir.parameter_schema

        # Check code template includes highlight logic
        assert "highlight_residues" in ir.code_template
        assert "highlight_groups" in ir.code_template


# ===============================================================================
# StructureAnalysisService Tests
# ===============================================================================


class TestStructureAnalysisService:
    """Test suite for StructureAnalysisService."""

    def test_service_initialization(self, analysis_service):
        """Test that service initializes correctly."""
        assert analysis_service is not None
        assert analysis_service.config == {}

    def test_parse_structure_pdb(self, analysis_service, mock_structure_file):
        """Test structure parsing from PDB file."""
        structure = analysis_service._parse_structure(mock_structure_file)

        assert structure is not None
        assert len(structure) > 0  # At least one model

    def test_parse_structure_nonexistent(self, analysis_service, tmp_path):
        """Test parsing with non-existent file."""
        fake_file = tmp_path / "nonexistent.pdb"

        with pytest.raises(StructureAnalysisError, match="Failed to parse"):
            analysis_service._parse_structure(fake_file)

    def test_get_chain_specific(self, analysis_service, mock_structure_file):
        """Test getting specific chain from structure."""
        structure = analysis_service._parse_structure(mock_structure_file)
        chain = analysis_service._get_chain(structure, "A")

        assert chain is not None
        assert chain.get_id() == "A"

    def test_get_chain_invalid(self, analysis_service, mock_structure_file):
        """Test getting invalid chain from structure."""
        structure = analysis_service._parse_structure(mock_structure_file)

        with pytest.raises(StructureAnalysisError, match="Chain.*not found"):
            analysis_service._get_chain(structure, "Z")

    def test_get_chain_default(self, analysis_service, mock_structure_file):
        """Test getting default (first) chain from structure."""
        structure = analysis_service._parse_structure(mock_structure_file)
        chain = analysis_service._get_chain(structure, None)

        assert chain is not None

    def test_analyze_structure_geometry(self, analysis_service, mock_structure_file):
        """Test geometric analysis of structure."""
        analysis_results, stats, ir = analysis_service.analyze_structure(
            structure_file=mock_structure_file, analysis_type="geometry"
        )

        # Assertions
        assert "chain_properties" in analysis_results
        assert "overall_radius_of_gyration" in analysis_results
        assert stats["analysis_type"] == "geometry"
        assert isinstance(ir, AnalysisStep)

    def test_analyze_structure_secondary_structure(
        self, analysis_service, mock_structure_file
    ):
        """Test secondary structure analysis (without DSSP)."""
        analysis_results, stats, ir = analysis_service.analyze_structure(
            structure_file=mock_structure_file, analysis_type="secondary_structure"
        )

        # Will use simplified method without DSSP
        assert "method" in analysis_results
        assert stats["analysis_type"] == "secondary_structure"
        assert isinstance(ir, AnalysisStep)

    def test_analyze_structure_residue_contacts(
        self, analysis_service, mock_structure_file
    ):
        """Test residue contact analysis."""
        analysis_results, stats, ir = analysis_service.analyze_structure(
            structure_file=mock_structure_file, analysis_type="residue_contacts"
        )

        # Assertions
        assert "n_contacts" in analysis_results
        assert "distance_cutoff" in analysis_results
        assert stats["analysis_type"] == "residue_contacts"
        assert isinstance(ir, AnalysisStep)

    def test_analyze_structure_invalid_type(self, analysis_service, mock_structure_file):
        """Test analysis with invalid analysis type."""
        with pytest.raises(StructureAnalysisError, match="Unknown analysis type"):
            analysis_service.analyze_structure(
                structure_file=mock_structure_file, analysis_type="invalid"
            )

    def test_calculate_rmsd_aligned(self, analysis_service, mock_structure_file, tmp_path):
        """Test RMSD calculation with alignment."""
        # Create second structure file (copy of first for testing)
        structure_file2 = tmp_path / "1AKE_copy.pdb"
        structure_file2.write_text(mock_structure_file.read_text())

        rmsd_results, stats, ir = analysis_service.calculate_rmsd(
            structure_file1=mock_structure_file,
            structure_file2=structure_file2,
            align=True,
        )

        # Should have very low RMSD (same structure)
        assert rmsd_results["rmsd"] < 0.1  # Nearly identical
        assert rmsd_results["aligned"] is True
        assert "rotation_matrix" in rmsd_results
        assert stats["aligned"] is True
        assert isinstance(ir, AnalysisStep)

    def test_calculate_rmsd_unaligned(
        self, analysis_service, mock_structure_file, tmp_path
    ):
        """Test RMSD calculation without alignment."""
        structure_file2 = tmp_path / "1AKE_copy.pdb"
        structure_file2.write_text(mock_structure_file.read_text())

        rmsd_results, stats, ir = analysis_service.calculate_rmsd(
            structure_file1=mock_structure_file,
            structure_file2=structure_file2,
            align=False,
        )

        # Should still have low RMSD (same structure)
        assert rmsd_results["aligned"] is False
        assert "rotation_matrix" not in rmsd_results
        assert stats["aligned"] is False


# ===============================================================================
# Integration Tests
# ===============================================================================


class TestProteinStructureServicesIntegration:
    """Integration tests for all protein structure services."""

    @patch("lobster.tools.protein_structure_fetch_service.PDBProvider")
    def test_full_workflow_fetch_visualize_analyze(
        self,
        mock_provider_class,
        fetch_service,
        viz_service,
        analysis_service,
        mock_pdb_metadata,
        mock_structure_file,
        tmp_path,
    ):
        """Test complete workflow: fetch → visualize → analyze."""
        # Setup mock
        mock_provider = Mock()
        mock_provider.download_structure.return_value = mock_structure_file
        mock_provider.get_structure_metadata.return_value = mock_pdb_metadata
        mock_provider_class.return_value = mock_provider

        # Step 1: Fetch structure
        structure_data, fetch_stats, fetch_ir = fetch_service.fetch_structure(
            pdb_id="1AKE", format="pdb", cache_dir=tmp_path
        )
        assert structure_data["pdb_id"] == "1AKE"

        # Step 2: Visualize structure
        structure_file = Path(structure_data["file_path"])
        viz_data, viz_stats, viz_ir = viz_service.visualize_structure(
            structure_file=structure_file, execute_commands=False
        )
        assert viz_data["executed"] is False
        assert Path(viz_data["script_file"]).exists()

        # Step 3: Analyze structure
        analysis_results, analysis_stats, analysis_ir = (
            analysis_service.analyze_structure(
                structure_file=structure_file, analysis_type="geometry"
            )
        )
        assert "overall_radius_of_gyration" in analysis_results

        # Verify all IRs were created
        assert isinstance(fetch_ir, AnalysisStep)
        assert isinstance(viz_ir, AnalysisStep)
        assert isinstance(analysis_ir, AnalysisStep)


# ===============================================================================
# Edge Cases and Error Handling
# ===============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_fetch_service_metadata_fallback(self, fetch_service):
        """Test metadata creation when API fetch fails."""
        metadata = fetch_service._create_minimal_metadata("1AKE")

        assert metadata.pdb_id == "1AKE"
        assert metadata.title == "Unknown"
        assert metadata.experiment_method == "UNKNOWN"

    def test_viz_service_output_path_auto_generation(
        self, viz_service, mock_structure_file
    ):
        """Test automatic output path generation."""
        viz_data, stats, ir = viz_service.visualize_structure(
            structure_file=mock_structure_file,
            output_image=None,  # Should auto-generate
            execute_commands=False,
        )

        # Output image path should be auto-generated
        assert viz_data["output_image"] is not None
        assert "visualizations" in viz_data["output_image"]

    def test_analysis_service_empty_structure(self, analysis_service, tmp_path):
        """Test analysis with empty/corrupt structure file."""
        corrupt_file = tmp_path / "corrupt.pdb"
        corrupt_file.write_text("INVALID PDB CONTENT")

        with pytest.raises(StructureAnalysisError):
            analysis_service.analyze_structure(
                structure_file=corrupt_file, analysis_type="geometry"
            )
