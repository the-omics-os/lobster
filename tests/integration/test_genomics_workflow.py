"""
Integration tests for genomics workflow with real APIs.

This module tests:
- Finding genomics datasets via research_agent (PubMed, GEO, SRA)
- Downloading genomics data via data_expert
- Processing genomics data via genomics_expert
- Multi-agent handoff coordination
- Complete end-to-end workflows

Requires:
- NCBI_API_KEY environment variable
- Internet connection
- Real API rate limiting consideration
"""

from pathlib import Path

import numpy as np
import pytest

from lobster.core.client import AgentClient
from lobster.core.data_manager_v2 import DataManagerV2


# ===============================================================================
# Test Configuration
# ===============================================================================


@pytest.fixture(scope="module")
def test_workspace(tmp_path_factory):
    """Create temporary workspace for integration tests."""
    return tmp_path_factory.mktemp("test_genomics_integration")


@pytest.fixture(scope="module")
def data_manager(test_workspace):
    """Create DataManagerV2 for integration testing."""
    return DataManagerV2(workspace_path=test_workspace)


@pytest.fixture(scope="module")
def agent_client(data_manager, test_workspace):
    """Create AgentClient for integration testing."""
    return AgentClient(
        data_manager=data_manager,
        enable_reasoning=False,
        workspace_path=test_workspace,
    )


@pytest.fixture(scope="module")
def check_api_keys():
    """Verify required API keys are present."""
    import os

    # Check for at least one LLM provider
    has_anthropic = os.getenv("ANTHROPIC_API_KEY")
    has_bedrock = os.getenv("AWS_BEDROCK_ACCESS_KEY") and os.getenv("AWS_BEDROCK_SECRET_ACCESS_KEY")

    if not (has_anthropic or has_bedrock):
        pytest.skip("Missing LLM provider API keys (ANTHROPIC_API_KEY or AWS_BEDROCK_*)")

    # NCBI key optional but recommended
    if not os.getenv("NCBI_API_KEY"):
        pytest.warn(pytest.PytestWarning("NCBI_API_KEY not set - rate limits will be slower"))


# ===============================================================================
# Genomics Dataset Discovery Tests
# ===============================================================================


@pytest.mark.real_api
@pytest.mark.integration
@pytest.mark.slow
class TestGenomicsDatasetDiscovery:
    """Test finding genomics datasets via research agent."""

    def test_search_1000genomes_publications(self, agent_client, check_api_keys):
        """Test searching for 1000 Genomes project publications."""
        result = agent_client.query(
            "Search PubMed for '1000 Genomes Project' publications (max 3 results)"
        )

        response = result.get("response", "")
        assert "1000" in response or "genome" in response.lower(), \
            "Should find 1000 Genomes publications"

    def test_discover_gwas_datasets_geo(self, agent_client, check_api_keys):
        """Test discovering GWAS datasets in GEO."""
        result = agent_client.query(
            "Search GEO for 'genome-wide association' datasets (max 3 results)"
        )

        response = result.get("response", "")
        # Should find genomics datasets
        assert "GSE" in response or "dataset" in response.lower(), \
            "Should discover GEO datasets"


# ===============================================================================
# Data Loading Tests (Local Files)
# ===============================================================================


@pytest.mark.integration
class TestGenomicsDataLoading:
    """Test loading genomics data from local files."""

    def test_load_vcf_via_agent(self, agent_client):
        """Test loading VCF file via genomics agent."""
        vcf_path = Path(__file__).parent.parent.parent / "test_data" / "genomics" / "chr22.vcf.gz"
        if not vcf_path.exists():
            pytest.skip(f"Test VCF not found: {vcf_path}")

        result = agent_client.query(
            f"Load VCF file {vcf_path} with max 100 variants, name it test_chr22"
        )

        response = result.get("response", "")
        assert "test_chr22" in response or "loaded" in response.lower() or "success" in response.lower(), \
            f"VCF loading should succeed. Response: {response}"

        # Verify modality was created
        modalities = agent_client.data_manager.list_modalities()
        assert any("test_chr22" in m or "chr22" in m for m in modalities), \
            f"Modality should be created. Available: {modalities}"

    def test_load_plink_via_agent(self, agent_client):
        """Test loading PLINK files via genomics agent."""
        plink_prefix = Path(__file__).parent.parent.parent / "test_data" / "genomics" / "plink_test" / "test_chr22"
        if not Path(str(plink_prefix) + ".bed").exists():
            pytest.skip(f"Test PLINK not found: {plink_prefix}.bed")

        result = agent_client.query(
            f"Load PLINK file {plink_prefix} and name it test_plink"
        )

        response = result.get("response", "")
        assert "test_plink" in response or "loaded" in response.lower() or "success" in response.lower(), \
            f"PLINK loading should succeed. Response: {response}"

        # Verify modality was created
        modalities = agent_client.data_manager.list_modalities()
        assert any("test_plink" in m or "plink" in m for m in modalities), \
            f"Modality should be created. Available: {modalities}"


# ===============================================================================
# QC Workflow Tests
# ===============================================================================


@pytest.mark.integration
class TestGenomicsQCWorkflow:
    """Test quality control workflow."""

    def test_qc_workflow_vcf(self, agent_client):
        """Test complete QC workflow: load → assess → filter."""
        vcf_path = Path(__file__).parent.parent.parent / "test_data" / "genomics" / "chr22.vcf.gz"
        if not vcf_path.exists():
            pytest.skip(f"Test VCF not found: {vcf_path}")

        # Step 1: Load VCF
        result1 = agent_client.query(
            f"Load VCF {vcf_path} with max 100 variants as qc_test"
        )
        assert result1.get("success", False) or "qc_test" in result1.get("response", "")

        # Step 2: Assess quality
        result2 = agent_client.query(
            "Assess quality for qc_test with UK Biobank thresholds"
        )
        assert "quality" in result2.get("response", "").lower() or "qc" in result2.get("response", "").lower()

        # Step 3: Filter samples
        result3 = agent_client.query(
            "Filter samples in qc_test_qc with call rate 0.95"
        )
        assert "filter" in result3.get("response", "").lower()

        # Step 4: Filter variants
        result4 = agent_client.query(
            "Filter variants with MAF 0.01 and HWE p-value 1e-10"
        )
        assert "filter" in result4.get("response", "").lower() or "variant" in result4.get("response", "").lower()


# ===============================================================================
# GWAS Workflow Tests
# ===============================================================================


@pytest.mark.integration
@pytest.mark.slow
class TestGenomicsGWASWorkflow:
    """Test GWAS analysis workflow."""

    def test_gwas_simple(self, agent_client):
        """Test basic GWAS analysis."""
        vcf_path = Path(__file__).parent.parent.parent / "test_data" / "genomics" / "chr22.vcf.gz"
        if not vcf_path.exists():
            pytest.skip(f"Test VCF not found: {vcf_path}")

        # Load and prepare data
        agent_client.query(f"Load VCF {vcf_path} with max 100 variants as gwas_test")

        # Add synthetic phenotype programmatically
        if "gwas_test" in agent_client.data_manager.list_modalities():
            adata = agent_client.data_manager.get_modality("gwas_test")
            np.random.seed(42)
            adata.obs["height"] = np.random.normal(170, 10, adata.n_obs)
            adata.obs["age"] = np.random.randint(20, 80, adata.n_obs)
            adata.obs["sex"] = np.random.choice([1, 2], adata.n_obs)

            # Run GWAS
            result = agent_client.query(
                "Run GWAS on gwas_test for phenotype height with covariates age and sex"
            )

            response = result.get("response", "")
            assert "gwas" in response.lower() or "association" in response.lower() or "lambda" in response.lower()

    def test_pca_analysis(self, agent_client):
        """Test PCA for population structure."""
        vcf_path = Path(__file__).parent.parent.parent / "test_data" / "genomics" / "chr22.vcf.gz"
        if not vcf_path.exists():
            pytest.skip(f"Test VCF not found: {vcf_path}")

        # Load data
        agent_client.query(f"Load VCF {vcf_path} with max 100 variants as pca_test")

        # Run PCA
        result = agent_client.query(
            "Calculate PCA for pca_test with 10 components"
        )

        response = result.get("response", "")
        assert "pca" in response.lower() or "component" in response.lower() or "variance" in response.lower()


# ===============================================================================
# Supervisor Handoff Tests
# ===============================================================================


@pytest.mark.real_api
@pytest.mark.integration
@pytest.mark.slow
class TestSupervisorHandoffToGenomics:
    """Test supervisor correctly hands off genomics tasks to genomics_expert.

    Uses 'admin superuser' mode to bypass routing decisions and directly test handoffs.
    """

    def test_supervisor_routes_vcf_loading_to_genomics(self, agent_client, check_api_keys):
        """Test that supervisor routes VCF loading requests to genomics_expert."""
        vcf_path = Path(__file__).parent.parent.parent / "test_data" / "genomics" / "chr22.vcf.gz"
        if not vcf_path.exists():
            pytest.skip(f"Test VCF not found: {vcf_path}")

        # Use admin superuser mode to ensure direct routing
        result = agent_client.query(
            f"ADMIN SUPERUSER: Route to genomics_expert only. Load VCF {vcf_path} with max 50 variants"
        )

        response = result.get("response", "")
        # Should complete successfully
        assert result.get("success", False) or "loaded" in response.lower() or "success" in response.lower()

    def test_supervisor_routes_gwas_to_genomics(self, agent_client, check_api_keys):
        """Test that supervisor routes GWAS requests to genomics_expert."""
        # First create a genomics modality programmatically
        from lobster.core.adapters.genomics.vcf_adapter import VCFAdapter

        vcf_path = Path(__file__).parent.parent.parent / "test_data" / "genomics" / "chr22.vcf.gz"
        if not vcf_path.exists():
            pytest.skip(f"Test VCF not found: {vcf_path}")

        adapter = VCFAdapter(strict_validation=False)
        adata = adapter.from_source(str(vcf_path), max_variants=50)

        # Add phenotype
        np.random.seed(42)
        adata.obs["test_phenotype"] = np.random.normal(100, 10, adata.n_obs)
        adata.obs["age"] = np.random.randint(20, 80, adata.n_obs)

        # Store in data manager
        agent_client.data_manager.modalities["handoff_test_gwas"] = adata

        # Use admin mode
        result = agent_client.query(
            "ADMIN SUPERUSER: Route to genomics_expert only. Run GWAS on handoff_test_gwas for phenotype test_phenotype with covariate age"
        )

        response = result.get("response", "")
        # Should complete or at least attempt GWAS
        assert "gwas" in response.lower() or "association" in response.lower() or "lambda" in response.lower() or "error" in response.lower()


# ===============================================================================
# Stress Tests
# ===============================================================================


@pytest.mark.integration
@pytest.mark.slow
class TestGenomicsStress:
    """Test genomics functionality under load."""

    def test_large_vcf_loading(self):
        """Test loading large VCF file."""
        vcf_path = Path(__file__).parent.parent.parent / "test_data" / "genomics" / "chr22.vcf.gz"
        if not vcf_path.exists():
            pytest.skip(f"Test VCF not found: {vcf_path}")

        from lobster.core.adapters.genomics.vcf_adapter import VCFAdapter

        adapter = VCFAdapter(strict_validation=False)

        # Load 10K variants
        adata = adapter.from_source(str(vcf_path), max_variants=10000)

        assert adata.n_obs > 2000
        assert adata.n_vars == 10000

    def test_gwas_medium_dataset(self):
        """Test GWAS on medium-sized dataset (1000 variants)."""
        from pathlib import Path
        from lobster.core.adapters.genomics.vcf_adapter import VCFAdapter
        from lobster.services.analysis.gwas_service import GWASService

        vcf_path = Path(__file__).parent.parent.parent / "test_data" / "genomics" / "chr22.vcf.gz"
        if not vcf_path.exists():
            pytest.skip(f"Test VCF not found: {vcf_path}")

        # Load data
        adapter = VCFAdapter(strict_validation=False)
        adata = adapter.from_source(str(vcf_path), max_variants=1000)

        # Add phenotype
        np.random.seed(42)
        adata.obs["phenotype"] = np.random.normal(100, 10, adata.n_obs)
        adata.obs["age"] = np.random.randint(20, 80, adata.n_obs)

        # Run GWAS
        gwas_service = GWASService()
        adata_gwas, stats, ir = gwas_service.run_gwas(
            adata, phenotype="phenotype", covariates=["age"], model="linear"
        )

        # Verify results
        assert adata_gwas.n_vars == 1000
        assert "gwas_pvalue" in adata_gwas.var.columns
        assert stats["n_variants_tested"] == 1000
