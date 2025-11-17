"""
Comprehensive pytest configuration and fixtures for Lobster AI testing framework.

This module provides all core fixtures, mock configurations, and test utilities
needed for testing the multi-agent bioinformatics analysis platform.

Fixture Dependency Map:
=======================

Core Environment Fixtures:
├── test_config (session-scoped base configuration)
├── temp_workspace (function-scoped isolated workspace)
│   ├── isolated_environment (complete environment isolation)
│   │   └── mock_agent_environment (agent testing with mocked LLMs)
│   ├── mock_data_manager_v2 (mocked data manager)
│   └── mock_agent_client (mocked agent client)

Data Generation Fixtures:
├── synthetic_single_cell_data (scRNA-seq test data)
├── synthetic_bulk_rnaseq_data (bulk RNA-seq test data)
├── synthetic_proteomics_data (proteomics test data)
└── mock_geo_response (GEO API response data)

Dataset Management Fixtures:
├── dataset_manager (session-scoped real dataset access)
└── datasets_dir (root directory for test datasets)

Mock Service Fixtures:
├── mock_llm_responses (mocked LLM API responses)
├── mock_geo_service (mocked GEO data service)
└── mock_external_apis (mocked external HTTP APIs)

Performance & Utilities:
├── benchmark_config (performance testing configuration)
└── test_data_registry (registry of test data descriptions)

Cleanup Fixtures:
└── cleanup_test_artifacts (session-scoped automatic cleanup)
"""

import json
import logging
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional
from unittest.mock import MagicMock, Mock, patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import responses
from faker import Faker
from pytest_mock import MockerFixture

# Suppress warnings during testing
logging.getLogger("scanpy").setLevel(logging.ERROR)
logging.getLogger("anndata").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("docling_core.transforms").setLevel(logging.ERROR)

# Initialize faker for generating test data
fake = Faker()
Faker.seed(42)  # For reproducible test data

# Test constants
TEST_WORKSPACE_PREFIX = "lobster_test_"
DEFAULT_TEST_TIMEOUT = 300
MOCK_API_BASE_URL = "https://api.mock-lobster.test"

# Configure pytest plugins
pytest_plugins = [
    "pytest_mock",
    "pytest_benchmark",
    "pytest_html",
]


# ==============================================================================
# Pytest Configuration Hooks
# ==============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "system: mark test as a system test")
    config.addinivalue_line(
        "markers", "performance: mark test as a performance benchmark"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location."""
    for item in items:
        # Auto-mark based on test path
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "system" in str(item.fspath):
            item.add_marker(pytest.mark.system)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)


# ==============================================================================
# Core Infrastructure Fixtures
# ==============================================================================


@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """Global test configuration shared across all tests.

    Provides consistent configuration values for test execution including
    workspace prefixes, timeouts, and synthetic data generation parameters.

    Returns:
        Dict[str, Any]: Configuration dictionary with the following keys:
            - workspace_prefix: Prefix for temporary test workspaces
            - timeout: Default test timeout in seconds (300s)
            - mock_api_url: Base URL for mocked API endpoints
            - enable_logging: Whether to enable logging during tests
            - cleanup_workspaces: Whether to cleanup workspaces after tests
            - synthetic_data_seed: Random seed for reproducible synthetic data (42)
            - default_cell_count: Default number of cells for scRNA-seq data (1000)
            - default_gene_count: Default number of genes/features (2000)

    Example:
        >>> def test_workspace_setup(test_config):
        ...     assert test_config["synthetic_data_seed"] == 42
        ...     assert test_config["default_cell_count"] == 1000
    """
    return {
        "workspace_prefix": TEST_WORKSPACE_PREFIX,
        "timeout": DEFAULT_TEST_TIMEOUT,
        "mock_api_url": MOCK_API_BASE_URL,
        "enable_logging": False,
        "cleanup_workspaces": True,
        "synthetic_data_seed": 42,
        "default_cell_count": 1000,
        "default_gene_count": 2000,
    }


@pytest.fixture(scope="function")
def temp_workspace(test_config: Dict[str, Any]) -> Generator[Path, None, None]:
    """Create isolated temporary workspace for each test.

    Creates a temporary directory with standard Lobster workspace structure
    (data/, exports/, cache/) and automatically cleans it up after the test.

    Args:
        test_config: Global test configuration fixture

    Yields:
        Path: Path to temporary workspace directory

    Example:
        >>> def test_file_operations(temp_workspace):
        ...     data_file = temp_workspace / "data" / "test.h5ad"
        ...     data_file.write_text("test data")
        ...     assert data_file.exists()
    """
    workspace_path = Path(tempfile.mkdtemp(prefix=test_config["workspace_prefix"]))

    # Create standard workspace structure
    (workspace_path / "data").mkdir(exist_ok=True)
    (workspace_path / "exports").mkdir(exist_ok=True)
    (workspace_path / "cache").mkdir(exist_ok=True)

    try:
        yield workspace_path
    finally:
        # Cleanup workspace after test
        if test_config["cleanup_workspaces"] and workspace_path.exists():
            shutil.rmtree(workspace_path, ignore_errors=True)


@pytest.fixture(scope="function")
def isolated_environment(temp_workspace: Path, monkeypatch) -> Generator[Path, None, None]:
    """Create completely isolated environment for testing.

    Provides full environment isolation by:
    - Setting temp_workspace as current working directory
    - Mocking all required API keys (OpenAI, AWS Bedrock, NCBI, Anthropic)
    - Ensuring no real API calls or file system pollution

    Args:
        temp_workspace: Temporary workspace path fixture
        monkeypatch: Pytest monkeypatch fixture for environment modification

    Yields:
        Path: Path to isolated workspace (same as temp_workspace)

    Example:
        >>> def test_with_isolation(isolated_environment):
        ...     # All API keys are mocked, safe to test API code
        ...     assert os.getenv("ANTHROPIC_API_KEY") == "test-anthropic-key"
        ...     # Working directory is isolated
        ...     assert Path.cwd() == isolated_environment
    """
    # Set temporary workspace as working directory
    original_cwd = os.getcwd()
    monkeypatch.chdir(temp_workspace)

    # Mock environment variables
    test_env = {
        "LOBSTER_WORKSPACE": str(temp_workspace),
        "OPENAI_API_KEY": "test-openai-key",
        "AWS_BEDROCK_ACCESS_KEY": "test-aws-access-key",
        "AWS_BEDROCK_SECRET_ACCESS_KEY": "test-aws-secret-key",
        "NCBI_API_KEY": "test-ncbi-key",
        "ANTHROPIC_API_KEY": "test-anthropic-key",
    }

    for key, value in test_env.items():
        monkeypatch.setenv(key, value)

    yield temp_workspace

    # Restore original working directory
    os.chdir(original_cwd)


@pytest.fixture(scope="function")
def mock_agent_environment(isolated_environment: Path, mocker: MockerFixture) -> Dict[str, Any]:
    """Complete agent testing environment with mocked LLMs and settings.

    This fixture provides a unified environment for testing all agents with:
    - Mocked settings that return test API keys
    - Mocked LLM creation to prevent real API calls
    - Isolated workspace and environment variables
    - Mocked agent configurator for consistent agent behavior

    Args:
        isolated_environment: Isolated environment fixture
        mocker: Pytest-mock fixture for patching

    Returns:
        Dict[str, Any]: Environment dictionary containing:
            - workspace: Path to isolated workspace
            - settings: Mocked settings instance
            - llm: Mocked LLM instance
            - agent_config: Mocked agent configurator

    Example:
        >>> def test_agent(mock_agent_environment):
        ...     workspace = mock_agent_environment["workspace"]
        ...     llm = mock_agent_environment["llm"]
        ...     # Test agent logic without real API calls
        ...     assert workspace.exists()
    """
    # Mock the settings to ensure they use our test environment
    mock_settings = mocker.patch("lobster.config.settings.get_settings")
    mock_settings_instance = Mock()
    mock_settings_instance.OPENAI_API_KEY = "test-openai-key"
    mock_settings_instance.AWS_BEDROCK_ACCESS_KEY = "test-aws-access-key"
    mock_settings_instance.AWS_BEDROCK_SECRET_ACCESS_KEY = "test-aws-secret-key"
    mock_settings_instance.ANTHROPIC_API_KEY = "test-anthropic-key"
    mock_settings_instance.NCBI_API_KEY = "test-ncbi-key"
    mock_settings_instance.llm_provider = "anthropic"
    mock_settings.return_value = mock_settings_instance

    # Mock agent configurator
    mock_agent_config = mocker.patch(
        "lobster.config.agent_config.initialize_configurator"
    )
    mock_agent_config.return_value.get_agent_llm_params.return_value = {
        "model": "claude-3-sonnet-20240229",
        "temperature": 0.1,
        "max_tokens": 4096,
    }

    # Mock LLM creation to prevent any real API calls
    mock_llm = Mock()
    mock_llm.with_config.return_value = mock_llm
    mock_create_llm = mocker.patch("lobster.config.llm_factory.create_llm")
    mock_create_llm.return_value = mock_llm

    return {
        "workspace": isolated_environment,
        "settings": mock_settings_instance,
        "llm": mock_llm,
        "agent_config": mock_agent_config.return_value,
    }


@pytest.fixture(scope="session")
def dataset_manager():
    """
    Provide DatasetManager instance for all tests.

    This fixture gives tests access to all configured datasets via
    hierarchical paths and tag-based queries.

    Usage:
        def test_example(dataset_manager):
            # Path-based access
            path = dataset_manager.get('single_cell/clustering/10x_chromium/GSE132044')

            # Metadata retrieval
            metadata = dataset_manager.get_metadata('single_cell/clustering/10x_chromium/GSE132044')

            # Tag-based queries
            transpose_tests = dataset_manager.list_by_tag('transpose_correctness')

            # Type-based queries
            bulk_datasets = dataset_manager.list_by_type('bulk_rnaseq')
    """
    from tests.fixtures.datasets.dataset_manager import get_dataset_manager

    return get_dataset_manager()


@pytest.fixture
def datasets_dir(dataset_manager) -> Path:
    """Root directory containing all test datasets.

    Provided for backward compatibility with tests that need direct path access.
    Use dataset_manager fixture for hierarchical and tag-based dataset access.

    Args:
        dataset_manager: Dataset manager fixture

    Returns:
        Path: Root directory path to test datasets

    Example:
        >>> def test_dataset_path(datasets_dir):
        ...     assert datasets_dir.exists()
        ...     assert (datasets_dir / "single_cell").exists()
    """
    return dataset_manager.datasets_dir


# ==============================================================================
# Mock Data Generation Fixtures
# ==============================================================================


@pytest.fixture(scope="function")
def synthetic_single_cell_data(test_config: Dict[str, Any]) -> ad.AnnData:
    """Generate realistic synthetic single-cell RNA-seq data.

    Creates an AnnData object with realistic single-cell RNA-seq characteristics:
    - Negative binomial count distribution (simulates biological variation)
    - 70% sparsity (typical for scRNA-seq)
    - Cell type annotations and batch metadata
    - QC metrics (total_counts, n_genes_by_counts, pct_counts_mt)

    Args:
        test_config: Global test configuration with data generation parameters

    Returns:
        ad.AnnData: Synthetic scRNA-seq dataset with:
            - X: Sparse count matrix (default: 1000 cells x 2000 genes)
            - obs: Cell metadata (cell_type, batch, QC metrics)
            - var: Gene metadata (gene_ids, chromosome, feature_types)

    Example:
        >>> def test_clustering(synthetic_single_cell_data):
        ...     adata = synthetic_single_cell_data
        ...     assert adata.shape == (1000, 2000)
        ...     assert "cell_type" in adata.obs.columns
        ...     assert adata.X.min() >= 0  # Non-negative counts
    """
    n_obs = test_config["default_cell_count"]
    n_vars = test_config["default_gene_count"]

    # Set random seed for reproducibility
    np.random.seed(test_config["synthetic_data_seed"])

    # Generate count matrix with negative binomial distribution
    # Simulate realistic single-cell count distributions
    X = np.random.negative_binomial(n=5, p=0.3, size=(n_obs, n_vars)).astype(np.float32)

    # Add some zeros to make it realistic (sparse)
    zero_mask = np.random.random((n_obs, n_vars)) < 0.7
    X[zero_mask] = 0

    # Create gene names
    var_names = [f"Gene_{i:04d}" for i in range(n_vars)]

    # Create cell barcodes
    obs_names = [f"Cell_{fake.uuid4()[:8]}" for _ in range(n_obs)]

    # Create AnnData object
    adata = ad.AnnData(
        X=X, var=pd.DataFrame(index=var_names), obs=pd.DataFrame(index=obs_names)
    )

    # Add realistic metadata
    adata.var["gene_ids"] = [f"ENSG{i:011d}" for i in range(n_vars)]
    adata.var["feature_types"] = ["Gene Expression"] * n_vars
    adata.var["chromosome"] = np.random.choice(
        [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"], size=n_vars
    )

    # Add cell metadata
    adata.obs["total_counts"] = np.array(X.sum(axis=1))
    adata.obs["n_genes_by_counts"] = np.array((X > 0).sum(axis=1))
    adata.obs["pct_counts_mt"] = np.random.uniform(
        0, 30, n_obs
    )  # Mitochondrial gene percentage
    adata.obs["pct_counts_ribo"] = np.random.uniform(
        0, 50, n_obs
    )  # Ribosomal gene percentage

    # Add simulated cell types
    cell_types = ["T_cell", "B_cell", "NK_cell", "Monocyte", "Dendritic_cell"]
    adata.obs["cell_type"] = np.random.choice(cell_types, size=n_obs)

    # Add batch information
    adata.obs["batch"] = np.random.choice(["Batch1", "Batch2", "Batch3"], size=n_obs)

    return adata


@pytest.fixture(scope="function")
def synthetic_bulk_rnaseq_data(test_config: Dict[str, Any]) -> ad.AnnData:
    """Generate realistic synthetic bulk RNA-seq data.

    Creates an AnnData object with typical bulk RNA-seq experiment structure:
    - Higher counts than single-cell (negative binomial n=20)
    - Balanced experimental design (12 treatment, 12 control)
    - Batch effects (2 batches)
    - Biological covariates (sex, age)

    Args:
        test_config: Global test configuration with data generation parameters

    Returns:
        ad.AnnData: Synthetic bulk RNA-seq dataset with:
            - X: Count matrix (24 samples x 2000 genes)
            - obs: Sample metadata (condition, batch, sex, age)
            - var: Gene metadata (gene_ids, gene_name, biotype)

    Example:
        >>> def test_differential_expression(synthetic_bulk_rnaseq_data):
        ...     adata = synthetic_bulk_rnaseq_data
        ...     assert adata.shape == (24, 2000)
        ...     assert adata.obs["condition"].value_counts()["Treatment"] == 12
        ...     assert "protein_coding" in adata.var["biotype"].values
    """
    n_obs = 24  # Typical sample count for bulk RNA-seq
    n_vars = test_config["default_gene_count"]

    np.random.seed(test_config["synthetic_data_seed"])

    # Generate count matrix with higher counts than single-cell
    X = np.random.negative_binomial(n=20, p=0.1, size=(n_obs, n_vars)).astype(
        np.float32
    )

    # Create sample and gene names
    obs_names = [f"Sample_{i:02d}" for i in range(n_obs)]
    var_names = [f"Gene_{i:04d}" for i in range(n_vars)]

    adata = ad.AnnData(
        X=X, obs=pd.DataFrame(index=obs_names), var=pd.DataFrame(index=var_names)
    )

    # Add realistic bulk RNA-seq metadata
    adata.var["gene_ids"] = [f"ENSG{i:011d}" for i in range(n_vars)]
    adata.var["gene_name"] = [f"GENE{i}" for i in range(n_vars)]
    adata.var["biotype"] = np.random.choice(
        ["protein_coding", "lncRNA", "miRNA", "pseudogene"],
        size=n_vars,
        p=[0.7, 0.15, 0.05, 0.1],
    )

    # Add sample metadata
    adata.obs["condition"] = ["Treatment"] * 12 + ["Control"] * 12
    adata.obs["batch"] = (["Batch1"] * 6 + ["Batch2"] * 6) * 2
    adata.obs["sex"] = np.random.choice(["M", "F"], size=n_obs)
    adata.obs["age"] = np.random.randint(20, 80, size=n_obs)

    return adata


@pytest.fixture(scope="function")
def synthetic_proteomics_data(test_config: Dict[str, Any]) -> ad.AnnData:
    """Generate realistic synthetic proteomics data.

    Creates an AnnData object with typical proteomics characteristics:
    - Log-normal intensity distribution (mass spec output)
    - 20% missing values (common in proteomics)
    - Three-condition experimental design (Disease/Healthy/Control)
    - Multiple tissue types and batch structure

    Args:
        test_config: Global test configuration with data generation parameters

    Returns:
        ad.AnnData: Synthetic proteomics dataset with:
            - X: Intensity matrix with NaN values (48 samples x 500 proteins)
            - obs: Sample metadata (condition, tissue, batch)
            - var: Protein metadata (protein_ids, protein_names, molecular_weight)

    Example:
        >>> def test_proteomics_normalization(synthetic_proteomics_data):
        ...     adata = synthetic_proteomics_data
        ...     assert adata.shape == (48, 500)
        ...     assert adata.obs["condition"].nunique() == 3
        ...     # Check for missing values (realistic for proteomics)
        ...     assert np.isnan(adata.X).sum() > 0
    """
    n_obs = 48  # Typical proteomics sample count
    n_vars = 500  # Typical protein count

    np.random.seed(test_config["synthetic_data_seed"])

    # Generate intensity matrix with log-normal distribution
    X = np.random.lognormal(mean=10, sigma=2, size=(n_obs, n_vars)).astype(np.float32)

    # Add missing values (common in proteomics)
    missing_mask = np.random.random((n_obs, n_vars)) < 0.2
    X[missing_mask] = np.nan

    obs_names = [f"Sample_{i:03d}" for i in range(n_obs)]
    var_names = [f"Protein_{i:03d}" for i in range(n_vars)]

    adata = ad.AnnData(
        X=X, obs=pd.DataFrame(index=obs_names), var=pd.DataFrame(index=var_names)
    )

    # Add protein metadata
    adata.var["protein_ids"] = [f"P{i:05d}" for i in range(n_vars)]
    adata.var["protein_names"] = [f"PROT{i}" for i in range(n_vars)]
    adata.var["molecular_weight"] = np.random.uniform(10, 200, n_vars)

    # Add sample metadata
    adata.obs["condition"] = ["Disease"] * 16 + ["Healthy"] * 16 + ["Control"] * 16
    adata.obs["tissue"] = np.random.choice(["Brain", "Liver", "Kidney"], size=n_obs)
    adata.obs["batch"] = np.random.choice(
        ["Batch1", "Batch2", "Batch3", "Batch4"], size=n_obs
    )

    return adata


@pytest.fixture(scope="function")
def mock_geo_response() -> Dict[str, Any]:
    """Generate mock GEO dataset response.

    Provides realistic GEO dataset metadata structure for testing GEO
    data provider and download functionality without making real API calls.

    Returns:
        Dict[str, Any]: Mock GEO response with structure:
            - gse_id: GEO series ID
            - title: Dataset title
            - summary: Dataset description
            - organism: Source organism
            - platform: Sequencing/array platform
            - samples: List of sample metadata dictionaries
            - supplementary_files: List of downloadable file names

    Example:
        >>> def test_geo_parser(mock_geo_response):
        ...     assert mock_geo_response["gse_id"] == "GSE123456"
        ...     assert len(mock_geo_response["samples"]) == 2
        ...     assert "matrix.mtx.gz" in mock_geo_response["supplementary_files"][0]
    """
    return {
        "gse_id": "GSE123456",
        "title": "Test Single-Cell RNA-seq Dataset",
        "summary": "This is a synthetic dataset for testing purposes",
        "organism": "Homo sapiens",
        "platform": "GPL24676",
        "samples": [
            {
                "gsm_id": "GSM1234567",
                "title": "Sample 1",
                "characteristics": {
                    "cell type": "T cell",
                    "tissue": "PBMC",
                    "treatment": "Control",
                },
            },
            {
                "gsm_id": "GSM1234568",
                "title": "Sample 2",
                "characteristics": {
                    "cell type": "B cell",
                    "tissue": "PBMC",
                    "treatment": "Treatment",
                },
            },
        ],
        "supplementary_files": [
            "GSE123456_matrix.mtx.gz",
            "GSE123456_features.tsv.gz",
            "GSE123456_barcodes.tsv.gz",
        ],
    }


# ==============================================================================
# Core Component Mocks
# ==============================================================================


@pytest.fixture(scope="function")
def mock_data_manager_v2(temp_workspace: Path) -> Mock:
    """Mock DataManagerV2 with realistic behavior.

    Provides a fully mocked DataManagerV2 instance with working methods
    for testing agents and services without real data persistence.

    Args:
        temp_workspace: Temporary workspace path fixture

    Returns:
        Mock: Mocked DataManagerV2 with functional methods:
            - modalities: Dict of loaded datasets
            - metadata_store: Dict of dataset metadata
            - latest_plots: List of generated plot data
            - tool_usage_history: List of tool usage records
            - list_modalities(): Returns available dataset names
            - get_modality(name): Returns dataset by name
            - add_modality(name, data): Adds new dataset
            - remove_modality(name): Removes dataset
            - save_modality(): Persists dataset (mocked)
            - load_modality(): Loads dataset (mocked)
            - export_workspace(): Exports workspace (mocked)

    Example:
        >>> def test_data_operations(mock_data_manager_v2, synthetic_single_cell_data):
        ...     mock_data_manager_v2.add_modality("test_data", synthetic_single_cell_data)
        ...     assert "test_data" in mock_data_manager_v2.list_modalities()
        ...     data = mock_data_manager_v2.get_modality("test_data")
        ...     assert data is not None
    """
    mock_dm = Mock()

    # Mock basic properties
    mock_dm.workspace_path = temp_workspace
    mock_dm.modalities = {}
    mock_dm.metadata_store = {}
    mock_dm.latest_plots = []
    mock_dm.tool_usage_history = []

    # Mock methods
    mock_dm.list_modalities.return_value = list(mock_dm.modalities.keys())
    mock_dm.get_modality.side_effect = lambda name: mock_dm.modalities.get(name)
    mock_dm.add_modality.side_effect = lambda name, data: mock_dm.modalities.update(
        {name: data}
    )
    mock_dm.remove_modality.side_effect = lambda name: mock_dm.modalities.pop(
        name, None
    )

    # Mock file operations
    mock_dm.save_modality.return_value = True
    mock_dm.load_modality.return_value = True
    mock_dm.export_workspace.return_value = temp_workspace / "export.zip"

    return mock_dm


@pytest.fixture(scope="function")
def mock_agent_client(temp_workspace: Path) -> Mock:
    """Mock AgentClient for testing agent interactions.

    Provides a mocked AgentClient for testing CLI commands and client
    behavior without running the full LangGraph agent system.

    Args:
        temp_workspace: Temporary workspace path fixture

    Returns:
        Mock: Mocked AgentClient with methods:
            - session_id: Unique test session identifier
            - workspace_path: Path to workspace
            - query(user_input, stream): Execute query (returns mock response)
            - get_status(): Get session status (returns mock status)

    Example:
        >>> def test_client_query(mock_agent_client):
        ...     response = mock_agent_client.query("analyze my data")
        ...     assert response["success"] is True
        ...     assert "Mock response" in response["response"]
    """
    mock_client = Mock()

    # Mock basic properties
    mock_client.session_id = f"test_session_{fake.uuid4()[:8]}"
    mock_client.workspace_path = temp_workspace

    # Mock query method with realistic responses
    def mock_query(user_input: str, stream: bool = False):
        return {
            "success": True,
            "response": f"Mock response to: {user_input[:50]}...",
            "agent_used": "supervisor_agent",
            "execution_time": 1.23,
            "tools_used": ["list_available_modalities"],
        }

    mock_client.query.side_effect = mock_query

    # Mock status method
    mock_client.get_status.return_value = {
        "session_id": mock_client.session_id,
        "workspace_path": str(temp_workspace),
        "active_modalities": 0,
        "total_interactions": 0,
        "last_activity": datetime.now().isoformat(),
    }

    return mock_client


@pytest.fixture(scope="function")
def mock_llm_responses(mocker: MockerFixture) -> Mock:
    """Mock LLM API responses for consistent agent testing.

    Mocks both OpenAI and AWS Bedrock API calls to prevent real API usage
    during testing. Returns consistent, predictable responses for each agent.

    Args:
        mocker: Pytest-mock fixture for patching

    Returns:
        Mock: Mocked OpenAI API response object with predefined agent responses

    Example:
        >>> def test_llm_integration(mock_llm_responses):
        ...     # LLM calls are automatically mocked
        ...     # Test code that would normally call OpenAI/Bedrock
        ...     assert mock_llm_responses.called
    """
    mock_responses = {
        "supervisor": "I understand your request. Let me delegate this to the appropriate expert agent.",
        "data_expert": "I can help you load and analyze your dataset. Let me check the data format.",
        "singlecell_expert": "I'll perform single-cell RNA-seq analysis including QC, normalization, and clustering.",
        "research_agent": "I can search for relevant datasets and literature for your research question.",
    }

    # Mock OpenAI API calls
    mock_openai = mocker.patch("openai.resources.chat.completions.Completions.create")
    mock_openai.return_value.choices = [
        Mock(message=Mock(content=mock_responses["supervisor"]))
    ]

    # Mock AWS Bedrock calls
    mock_bedrock = mocker.patch("boto3.client")
    mock_bedrock.return_value.invoke_model.return_value = {
        "body": Mock(
            read=lambda: json.dumps(
                {"content": [{"text": mock_responses["supervisor"]}]}
            ).encode()
        )
    }

    return mock_openai


# ==============================================================================
# External Service Mocks
# ==============================================================================


@pytest.fixture(scope="function")
def mock_geo_service(mocker: MockerFixture) -> Mock:
    """Mock GEO service for testing data download.

    Provides a fully mocked GEO data service to test download workflows
    without making real network requests to NCBI GEO servers.

    Args:
        mocker: Pytest-mock fixture for patching

    Returns:
        Mock: Mocked GEOService with methods:
            - download_gse(): Returns success status and mock download info
            - get_gse_metadata(): Returns mock GEO metadata

    Example:
        >>> def test_geo_download(mock_geo_service):
        ...     result = mock_geo_service.download_gse("GSE123456")
        ...     assert result["success"] is True
        ...     assert result["gse_id"] == "GSE123456"
    """
    mock_service = Mock()

    # Mock successful download
    mock_service.download_gse.return_value = {
        "success": True,
        "gse_id": "GSE123456",
        "files_downloaded": 3,
        "local_path": "/mock/path/GSE123456",
    }

    # Mock GEO metadata fetch
    mock_service.get_gse_metadata.return_value = {
        "gse_id": "GSE123456",
        "title": "Test Dataset",
        "organism": "Homo sapiens",
        "sample_count": 24,
        "platform": "GPL24676",
    }

    return mock_service


@pytest.fixture(scope="function")
def mock_external_apis() -> Generator[responses.RequestsMock, None, None]:
    """Mock external API calls using responses library.

    Provides mocked HTTP responses for external APIs including NCBI, PubMed,
    and GEO services. Uses the `responses` library to intercept HTTP requests.

    Yields:
        RequestsMock: Active responses mock context with pre-configured endpoints:
            - NCBI E-utilities search endpoint
            - PubMed fetch endpoint

    Example:
        >>> def test_pubmed_search(mock_external_apis):
        ...     # HTTP calls to NCBI are automatically mocked
        ...     import requests
        ...     response = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi")
        ...     assert response.status_code == 200
    """
    with responses.RequestsMock() as rsps:
        # Mock NCBI/GEO API
        rsps.add(
            responses.GET,
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            json={"esearchresult": {"idlist": ["123456"], "count": "1"}},
            status=200,
        )

        # Mock PubMed API
        rsps.add(
            responses.GET,
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            xml='<?xml version="1.0"?><PubmedArticle></PubmedArticle>',
            status=200,
        )

        yield rsps


# ==============================================================================
# Performance Testing Fixtures
# ==============================================================================


@pytest.fixture(scope="function")
def benchmark_config() -> Dict[str, Any]:
    """Configuration for performance benchmarking.

    Provides standardized configuration for pytest-benchmark performance tests
    to ensure consistent and reproducible benchmark measurements.

    Returns:
        Dict[str, Any]: Benchmark configuration with:
            - min_rounds: Minimum number of test rounds (3)
            - max_time: Maximum benchmark time in seconds (10.0)
            - timer: Timer function to use ("time.perf_counter")
            - disable_gc: Whether to disable GC during benchmarks (True)
            - warmup: Whether to run warmup rounds (True)

    Example:
        >>> def test_performance(benchmark, benchmark_config):
        ...     result = benchmark.pedantic(
        ...         my_function,
        ...         rounds=benchmark_config["min_rounds"],
        ...         warmup_rounds=1 if benchmark_config["warmup"] else 0
        ...     )
    """
    return {
        "min_rounds": 3,
        "max_time": 10.0,
        "timer": "time.perf_counter",
        "disable_gc": True,
        "warmup": True,
    }


# ==============================================================================
# Test Utilities
# ==============================================================================


@pytest.fixture(scope="session")
def test_data_registry() -> Dict[str, str]:
    """Registry of test data files and their descriptions.

    Provides a catalog of available test datasets with human-readable
    descriptions for documentation and test discovery.

    Returns:
        Dict[str, str]: Registry mapping dataset keys to descriptions

    Example:
        >>> def test_registry_access(test_data_registry):
        ...     assert "small_single_cell" in test_data_registry
        ...     description = test_data_registry["small_single_cell"]
        ...     assert "100 cells" in description
    """
    return {
        "small_single_cell": "Small single-cell dataset (100 cells, 500 genes)",
        "medium_single_cell": "Medium single-cell dataset (1000 cells, 2000 genes)",
        "large_single_cell": "Large single-cell dataset (10000 cells, 5000 genes)",
        "bulk_rnaseq": "Bulk RNA-seq dataset (24 samples, 2000 genes)",
        "proteomics": "Proteomics dataset (48 samples, 500 proteins)",
        "multimodal": "Multi-modal dataset (single-cell + proteomics)",
    }


def create_mock_file(file_path: Path, content: str = "") -> Path:
    """Utility function to create mock files for testing."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content)
    return file_path


def assert_adata_equal(
    adata1: ad.AnnData, adata2: ad.AnnData, check_dtype: bool = True
) -> None:
    """Assert that two AnnData objects are equal."""
    assert adata1.shape == adata2.shape, "Shape mismatch"

    # Check data matrix
    if hasattr(adata1.X, "toarray"):
        assert np.allclose(adata1.X.toarray(), adata2.X.toarray(), equal_nan=True)
    else:
        assert np.allclose(adata1.X, adata2.X, equal_nan=True)

    # Check obs and var
    pd.testing.assert_frame_equal(adata1.obs, adata2.obs)
    pd.testing.assert_frame_equal(adata1.var, adata2.var)


# ==============================================================================
# Cleanup and Finalization
# ==============================================================================


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_artifacts() -> Generator[None, None, None]:
    """Cleanup test artifacts at the end of test session.

    Automatically removes all temporary test workspaces created during
    the test session. This fixture runs automatically (autouse=True) and
    ensures no test artifacts are left in the system temp directory.

    Yields:
        None: Allows tests to run, then performs cleanup

    Example:
        This fixture runs automatically, no explicit usage needed:
        >>> def test_something(temp_workspace):
        ...     # temp_workspace will be cleaned up after session
        ...     pass
    """
    yield

    # Cleanup any remaining temporary files
    temp_dir = Path(tempfile.gettempdir())
    for path in temp_dir.glob(f"{TEST_WORKSPACE_PREFIX}*"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
