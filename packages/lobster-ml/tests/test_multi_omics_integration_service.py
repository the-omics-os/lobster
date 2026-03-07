"""
Unit tests for MultiOmicsIntegrationService.

Tests cover:
- Sample overlap computation
- Memory estimation
- Overlap validation
- ZeroOverlapError handling
- IR template generation
- (Integration tests skipped if mofapy2 not installed)
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from lobster.services.ml.multi_omics_integration_service import (
    MultiOmicsIntegrationService,
    ZeroOverlapError,
)


@pytest.fixture
def service():
    """Create a MultiOmicsIntegrationService instance."""
    return MultiOmicsIntegrationService()


@pytest.fixture
def mock_modalities():
    """
    Create two mock modalities with known overlap.

    Modality rna: samples A, B, C (3 samples, 10 features)
    Modality protein: samples B, C, D (3 samples, 5 features)
    Complete cases: B, C (2 samples)
    """
    rna = AnnData(
        X=np.random.rand(3, 10),
        obs=pd.DataFrame(index=["A", "B", "C"]),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(10)]),
    )

    protein = AnnData(
        X=np.random.rand(3, 5),
        obs=pd.DataFrame(index=["B", "C", "D"]),
        var=pd.DataFrame(index=[f"protein_{i}" for i in range(5)]),
    )

    return {"rna": rna, "protein": protein}


@pytest.fixture
def complete_overlap_modalities():
    """Create modalities with 100% overlap."""
    samples = ["S1", "S2", "S3"]

    rna = AnnData(
        X=np.random.rand(3, 10),
        obs=pd.DataFrame(index=samples),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(10)]),
    )

    protein = AnnData(
        X=np.random.rand(3, 5),
        obs=pd.DataFrame(index=samples),
        var=pd.DataFrame(index=[f"protein_{i}" for i in range(5)]),
    )

    return {"rna": rna, "protein": protein}


@pytest.fixture
def zero_overlap_modalities():
    """Create modalities with zero overlap."""
    rna = AnnData(
        X=np.random.rand(3, 10),
        obs=pd.DataFrame(index=["A", "B", "C"]),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(10)]),
    )

    protein = AnnData(
        X=np.random.rand(3, 5),
        obs=pd.DataFrame(index=["X", "Y", "Z"]),
        var=pd.DataFrame(index=[f"protein_{i}" for i in range(5)]),
    )

    return {"rna": rna, "protein": protein}


class TestSampleOverlap:
    """Tests for compute_sample_overlap method."""

    def test_complete_overlap(self, service, complete_overlap_modalities):
        """All samples present in all modalities returns 100% overlap."""
        result = service.compute_sample_overlap(complete_overlap_modalities)

        # All 3 samples should be complete cases
        assert len(result["complete_cases"]) == 3
        assert set(result["complete_cases"]) == {"S1", "S2", "S3"}

        # 100% overlap at threshold 1.0
        assert result["overlap_fractions"][1.0] == 1.0

        # Overlap matrix should be all True
        overlap_matrix = result["overlap_matrix"]
        assert overlap_matrix.shape == (3, 2)  # 3 samples, 2 modalities
        assert overlap_matrix.all().all()

    def test_partial_overlap(self, service, mock_modalities):
        """Some samples missing from some modalities."""
        result = service.compute_sample_overlap(mock_modalities)

        # Only B, C are in both modalities
        assert len(result["complete_cases"]) == 2
        assert set(result["complete_cases"]) == {"B", "C"}

        # Total unique samples: A, B, C, D (4 samples)
        # Complete cases: B, C (2 samples)
        # Overlap fraction: 2/4 = 0.5
        assert result["overlap_fractions"][1.0] == 0.5

        # Check overlap matrix
        overlap_matrix = result["overlap_matrix"]
        assert overlap_matrix.shape == (4, 2)  # 4 samples, 2 modalities

        # Sample A only in rna
        assert overlap_matrix.loc["A", "rna"] == True
        assert overlap_matrix.loc["A", "protein"] == False

        # Sample D only in protein
        assert overlap_matrix.loc["D", "rna"] == False
        assert overlap_matrix.loc["D", "protein"] == True

        # Samples B, C in both
        assert overlap_matrix.loc["B", "rna"] == True
        assert overlap_matrix.loc["B", "protein"] == True
        assert overlap_matrix.loc["C", "rna"] == True
        assert overlap_matrix.loc["C", "protein"] == True

    def test_zero_overlap(self, service, zero_overlap_modalities):
        """No common samples across modalities."""
        result = service.compute_sample_overlap(zero_overlap_modalities)

        # No complete cases
        assert len(result["complete_cases"]) == 0

        # Zero overlap at threshold 1.0
        assert result["overlap_fractions"][1.0] == 0.0

    def test_single_modality(self, service):
        """Single modality returns itself as complete cases."""
        single = AnnData(
            X=np.random.rand(5, 10),
            obs=pd.DataFrame(index=[f"S{i}" for i in range(5)]),
            var=pd.DataFrame(index=[f"gene_{i}" for i in range(10)]),
        )

        result = service.compute_sample_overlap({"rna": single})

        # All samples should be complete cases
        assert len(result["complete_cases"]) == 5
        assert result["overlap_fractions"][1.0] == 1.0

    def test_per_modality_counts(self, service, mock_modalities):
        """Per-modality counts are correct."""
        result = service.compute_sample_overlap(mock_modalities)

        assert result["per_modality_counts"]["rna"] == 3
        assert result["per_modality_counts"]["protein"] == 3

    def test_overlap_fractions_at_thresholds(self, service):
        """Overlap fractions computed correctly at each threshold."""
        # Create 3 modalities with varying overlap
        # All 3: samples A, B
        # 2 of 3: samples C, D (missing from metabolite)
        # 1 of 3: samples E, F, G (each in one modality only)

        rna = AnnData(
            X=np.random.rand(5, 10),
            obs=pd.DataFrame(index=["A", "B", "C", "D", "E"]),
            var=pd.DataFrame(index=[f"gene_{i}" for i in range(10)]),
        )

        protein = AnnData(
            X=np.random.rand(5, 5),
            obs=pd.DataFrame(index=["A", "B", "C", "D", "F"]),
            var=pd.DataFrame(index=[f"protein_{i}" for i in range(5)]),
        )

        metabolite = AnnData(
            X=np.random.rand(3, 8),
            obs=pd.DataFrame(index=["A", "B", "G"]),
            var=pd.DataFrame(index=[f"metabolite_{i}" for i in range(8)]),
        )

        modalities = {"rna": rna, "protein": protein, "metabolite": metabolite}
        result = service.compute_sample_overlap(modalities)

        # Total samples: A, B, C, D, E, F, G (7 samples)
        # In all 3 (100%): A, B (2 samples) → 2/7
        # In 2+ (67%): A, B, C, D, G (5 samples) → 5/7
        # Note: threshold 0.7 → ceil(0.7*3) = 3 modalities = all 3
        # So 0.7 threshold also requires all 3 modalities
        assert result["overlap_fractions"][1.0] == pytest.approx(2 / 7)
        assert result["overlap_fractions"][0.5] >= 4 / 7  # At least 2 modalities

    def test_empty_modalities_raises(self, service):
        """Empty modalities dict raises ValueError."""
        with pytest.raises(ValueError, match="At least one modality required"):
            service.compute_sample_overlap({})


class TestMemoryEstimation:
    """Tests for _estimate_mofa_memory method."""

    def test_small_dataset(self, service, mock_modalities):
        """Small dataset returns reasonable memory estimate."""
        n_factors = 10
        result = service._estimate_mofa_memory(mock_modalities, n_factors)

        # Check all components present
        assert "data_gb" in result
        assert "factors_gb" in result
        assert "weights_gb" in result
        assert "base_gb" in result
        assert "total_gb" in result
        assert "overhead_multiplier" in result

        # Memory should be positive
        assert result["data_gb"] > 0
        assert result["factors_gb"] > 0
        assert result["weights_gb"] > 0
        assert result["base_gb"] > 0
        assert result["total_gb"] > 0

        # Total should be base * multiplier
        assert result["total_gb"] == pytest.approx(
            result["base_gb"] * result["overhead_multiplier"]
        )

    def test_memory_components(self, service, complete_overlap_modalities):
        """Memory breakdown includes data, factors, weights."""
        n_factors = 15
        result = service._estimate_mofa_memory(complete_overlap_modalities, n_factors)

        # Data memory: n_samples * total_features * 8 bytes
        # n_samples = 3, total_features = 10 + 5 = 15
        expected_data_bytes = 3 * 15 * 8
        expected_data_gb = expected_data_bytes / (1024**3)
        assert result["data_gb"] == pytest.approx(expected_data_gb)

        # Factors memory: n_samples * n_factors * 8 bytes
        expected_factors_bytes = 3 * 15 * 8
        expected_factors_gb = expected_factors_bytes / (1024**3)
        assert result["factors_gb"] == pytest.approx(expected_factors_gb)

        # Weights memory: total_features * n_factors * 8 bytes
        expected_weights_bytes = 15 * 15 * 8
        expected_weights_gb = expected_weights_bytes / (1024**3)
        assert result["weights_gb"] == pytest.approx(expected_weights_gb)

        # Base = sum of components
        assert result["base_gb"] == pytest.approx(
            result["data_gb"] + result["factors_gb"] + result["weights_gb"]
        )

    def test_overhead_multiplier(self, service, complete_overlap_modalities):
        """Custom overhead multiplier affects total."""
        n_factors = 10

        # Test with multiplier 2.0
        result_2x = service._estimate_mofa_memory(
            complete_overlap_modalities, n_factors, overhead_multiplier=2.0
        )
        assert result_2x["overhead_multiplier"] == 2.0
        assert result_2x["total_gb"] == pytest.approx(result_2x["base_gb"] * 2.0)

        # Test with multiplier 5.0
        result_5x = service._estimate_mofa_memory(
            complete_overlap_modalities, n_factors, overhead_multiplier=5.0
        )
        assert result_5x["overhead_multiplier"] == 5.0
        assert result_5x["total_gb"] == pytest.approx(result_5x["base_gb"] * 5.0)

        # Total should scale with multiplier (base stays same)
        assert result_2x["base_gb"] == pytest.approx(result_5x["base_gb"])
        assert result_5x["total_gb"] == pytest.approx(result_2x["total_gb"] * 2.5)


class TestOverlapValidation:
    """Tests for _validate_overlap method."""

    def test_zero_overlap_raises(self, service, zero_overlap_modalities):
        """ZeroOverlapError raised when no common samples."""
        overlap_stats = service.compute_sample_overlap(zero_overlap_modalities)

        with pytest.raises(ZeroOverlapError) as exc_info:
            service._validate_overlap(overlap_stats, min_overlap_fraction=1.0)

        # Error message should be actionable
        error_msg = str(exc_info.value)
        assert "No samples found in all modalities" in error_msg
        assert "sample ids" in error_msg.lower()  # Case-insensitive check
        assert "obs_names" in error_msg

    def test_below_threshold_raises(self, service, mock_modalities):
        """ValueError raised when overlap below min_overlap_fraction."""
        overlap_stats = service.compute_sample_overlap(mock_modalities)

        # Overlap is 0.5 (2 complete cases out of 4 total samples)
        # Require 0.8 overlap
        with pytest.raises(ValueError) as exc_info:
            service._validate_overlap(overlap_stats, min_overlap_fraction=0.8)

        error_msg = str(exc_info.value)
        assert "overlap" in error_msg.lower()
        assert "below minimum threshold" in error_msg.lower()
        assert "50.00%" in error_msg or "50%" in error_msg  # Actual overlap
        assert "80.00%" in error_msg or "80%" in error_msg  # Threshold

        # Error message should suggest solutions
        assert "Relaxing" in error_msg or "relaxing" in error_msg
        assert "min_overlap_fraction" in error_msg

    def test_at_threshold_passes(self, service, mock_modalities):
        """Exactly at threshold passes validation."""
        overlap_stats = service.compute_sample_overlap(mock_modalities)

        # Overlap is 0.5, require 0.5 → should pass
        try:
            service._validate_overlap(overlap_stats, min_overlap_fraction=0.5)
        except Exception as e:
            pytest.fail(f"Validation at exact threshold should pass: {e}")

    def test_below_threshold_passes(self, service, mock_modalities):
        """Below threshold (lower requirement) passes validation."""
        overlap_stats = service.compute_sample_overlap(mock_modalities)

        # Overlap is 0.5, require 0.3 → should pass
        try:
            service._validate_overlap(overlap_stats, min_overlap_fraction=0.3)
        except Exception as e:
            pytest.fail(f"Validation below threshold should pass: {e}")

    def test_complete_overlap_passes(self, service, complete_overlap_modalities):
        """Complete overlap passes any threshold."""
        overlap_stats = service.compute_sample_overlap(complete_overlap_modalities)

        # Test with strict threshold
        try:
            service._validate_overlap(overlap_stats, min_overlap_fraction=1.0)
        except Exception as e:
            pytest.fail(f"Complete overlap should pass 100% threshold: {e}")


class TestIRTemplate:
    """Tests for IR template generation."""

    def test_template_minimum_lines(self, service):
        """Template has at least 50 lines."""
        ir = service._create_integration_ir(
            modality_names=["rna", "protein"],
            n_factors=15,
            min_overlap_fraction=1.0,
            memory_threshold_gb=2.0,
            gpu_mode=False,
            convergence_mode="fast",
            n_iterations=1000,
            random_state=42,
        )

        # Count lines in code template
        template_lines = ir.code_template.strip().split("\n")
        assert len(template_lines) >= 50, (
            f"IR template should have at least 50 lines, got {len(template_lines)}"
        )

    def test_parameter_schema_complete(self, service):
        """All parameters have ParameterSpec."""
        ir = service._create_integration_ir(
            modality_names=["rna", "protein"],
            n_factors=15,
            min_overlap_fraction=1.0,
            memory_threshold_gb=2.0,
            gpu_mode=False,
            convergence_mode="fast",
            n_iterations=1000,
            random_state=42,
        )

        # All expected parameters should have schema entries
        expected_params = [
            "modality_names",
            "n_factors",
            "min_overlap_fraction",
            "memory_threshold_gb",
            "gpu_mode",
            "convergence_mode",
            "n_iterations",
            "random_state",
        ]

        for param in expected_params:
            assert param in ir.parameter_schema, f"Missing parameter schema: {param}"

            spec = ir.parameter_schema[param]
            assert spec.param_type is not None, f"{param} missing param_type"
            assert spec.description, f"{param} missing description"

    def test_papermill_injectable_params(self, service):
        """Key parameters have papermill_injectable=True."""
        ir = service._create_integration_ir(
            modality_names=["rna", "protein"],
            n_factors=15,
            min_overlap_fraction=1.0,
            memory_threshold_gb=2.0,
            gpu_mode=False,
            convergence_mode="fast",
            n_iterations=1000,
            random_state=42,
        )

        # These parameters should be papermill injectable
        injectable_params = [
            "modality_names",
            "n_factors",
            "min_overlap_fraction",
            "memory_threshold_gb",
            "gpu_mode",
            "convergence_mode",
            "n_iterations",
            "random_state",
        ]

        for param in injectable_params:
            spec = ir.parameter_schema[param]
            assert spec.papermill_injectable == True, (
                f"{param} should be papermill_injectable"
            )

    def test_imports_complete(self, service):
        """All required imports listed."""
        ir = service._create_integration_ir(
            modality_names=["rna", "protein"],
            n_factors=15,
            min_overlap_fraction=1.0,
            memory_threshold_gb=2.0,
            gpu_mode=False,
            convergence_mode="fast",
            n_iterations=1000,
            random_state=42,
        )

        # Check essential imports are present
        imports_str = "\n".join(ir.imports)
        assert "numpy" in imports_str
        assert "pandas" in imports_str
        assert "anndata" in imports_str or "AnnData" in imports_str
        assert "mofapy2" in imports_str

    def test_template_parameters_match_schema(self, service):
        """Template uses Jinja2 syntax for all schema parameters."""
        ir = service._create_integration_ir(
            modality_names=["rna", "protein"],
            n_factors=15,
            min_overlap_fraction=1.0,
            memory_threshold_gb=2.0,
            gpu_mode=False,
            convergence_mode="fast",
            n_iterations=1000,
            random_state=42,
        )

        template = ir.code_template

        # Check that parameters appear in template with Jinja2 syntax
        for param in ir.parameter_schema.keys():
            # Allow for {{ param }}, {{ param|filter }}, etc.
            assert f"{{{{ {param}" in template or f"{param} =" in template, (
                f"Parameter {param} not found in template"
            )

    def test_ir_metadata_fields(self, service):
        """IR has all required metadata fields."""
        ir = service._create_integration_ir(
            modality_names=["rna", "protein"],
            n_factors=15,
            min_overlap_fraction=1.0,
            memory_threshold_gb=2.0,
            gpu_mode=False,
            convergence_mode="fast",
            n_iterations=1000,
            random_state=42,
        )

        assert ir.operation == "mofapy2.multi_omics_integration"
        assert ir.tool_name == "integrate_modalities"
        assert ir.library == "mofapy2"
        assert "Multi-omics" in ir.description or "multi-omics" in ir.description
        assert len(ir.input_entities) == 2  # rna, protein
        assert "X_mofa" in ir.output_entities
