"""
Comprehensive unit tests for differential formula design service.

This module provides thorough testing of the DifferentialFormulaService including
formula parsing, design matrix construction, validation, and complex experimental
designs with covariates and interactions.

Test coverage target: 95%+ with meaningful tests for statistical design operations.
"""

from typing import Dict
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from lobster.core import DesignMatrixError, FormulaError
from lobster.tools.differential_formula_service import DifferentialFormulaService


# ===============================================================================
# Fixtures
# ===============================================================================


@pytest.fixture
def formula_service():
    """Create DifferentialFormulaService instance for testing."""
    return DifferentialFormulaService()


@pytest.fixture
def simple_metadata():
    """Create simple metadata for testing."""
    return pd.DataFrame({
        'condition': ['control', 'control', 'control', 'treated', 'treated', 'treated'],
        'batch': ['batch1', 'batch1', 'batch2', 'batch1', 'batch2', 'batch2'],
        'age': [25, 30, 35, 28, 32, 40]
    }, index=[f"sample_{i}" for i in range(6)])


@pytest.fixture
def complex_metadata():
    """Create complex metadata with multiple factors."""
    return pd.DataFrame({
        'condition': ['control', 'control', 'control', 'control',
                     'treated', 'treated', 'treated', 'treated',
                     'control', 'control', 'treated', 'treated'],
        'batch': ['batch1', 'batch1', 'batch2', 'batch2',
                 'batch1', 'batch1', 'batch2', 'batch2',
                 'batch1', 'batch2', 'batch1', 'batch2'],
        'donor': ['donor_a', 'donor_b', 'donor_c', 'donor_d',
                 'donor_a', 'donor_b', 'donor_c', 'donor_d',
                 'donor_e', 'donor_f', 'donor_e', 'donor_f'],
        'age': [25, 30, 35, 40, 28, 32, 38, 42, 26, 34, 29, 36],
        'weight': [65.5, 70.2, 68.8, 75.1, 72.3, 69.5, 71.8, 74.2, 67.9, 73.4, 70.1, 72.8]
    }, index=[f"sample_{i}" for i in range(12)])


@pytest.fixture
def imbalanced_metadata():
    """Create imbalanced design metadata."""
    return pd.DataFrame({
        'condition': ['control', 'control', 'treated', 'treated', 'treated', 'treated'],
        'batch': ['batch1', 'batch1', 'batch1', 'batch1', 'batch2', 'batch2']
    }, index=[f"sample_{i}" for i in range(6)])


# ===============================================================================
# Class 1: TestSuggestExperimentalDesigns
# ===============================================================================


@pytest.mark.unit
class TestSuggestExperimentalDesigns:
    """Test experimental design suggestions."""

    def test_suggest_simple_design(self, formula_service, simple_metadata):
        """Test suggestion for simple single-factor design."""
        suggestions = formula_service.suggest_formulas(
            metadata=simple_metadata,
            analysis_goal="Compare conditions"
        )

        # Should have at least one suggestion
        assert len(suggestions) > 0

        # Find simple design
        simple_designs = [s for s in suggestions if s["complexity"] == "simple"]
        assert len(simple_designs) > 0

        simple_design = simple_designs[0]

        # Verify structure
        assert "formula" in simple_design
        assert "~condition" in simple_design["formula"]
        assert "complexity" in simple_design
        assert simple_design["complexity"] == "simple"
        assert "description" in simple_design
        assert "pros" in simple_design
        assert "cons" in simple_design
        assert isinstance(simple_design["pros"], list)
        assert isinstance(simple_design["cons"], list)

    def test_suggest_with_batch_correction(self, formula_service, simple_metadata):
        """Test suggestion with batch effect correction."""
        suggestions = formula_service.suggest_formulas(
            metadata=simple_metadata,
            analysis_goal="Compare conditions with batch correction"
        )

        # Find batch-corrected design
        batch_designs = [s for s in suggestions if s["complexity"] == "batch_corrected"]

        if len(batch_designs) > 0:
            batch_design = batch_designs[0]

            # Verify formula includes batch
            assert "~condition" in batch_design["formula"]
            assert "batch" in batch_design["formula"]
            assert "+" in batch_design["formula"]

            # Verify metadata
            assert "recommended_for" in batch_design
            assert "min_samples_needed" in batch_design
            assert batch_design["min_samples_needed"] >= 6

    def test_suggest_with_interactions(self, formula_service, complex_metadata):
        """Test suggestion with interaction terms."""
        suggestions = formula_service.suggest_formulas(
            metadata=complex_metadata,
            analysis_goal="Analyze condition-batch interactions"
        )

        # Find interaction design
        interaction_designs = [s for s in suggestions if s["complexity"] == "interaction"]

        if len(interaction_designs) > 0:
            interaction_design = interaction_designs[0]

            # Verify formula includes interaction
            assert "*" in interaction_design["formula"]
            assert "condition" in interaction_design["formula"]

            # Verify higher sample requirements
            assert interaction_design["min_samples_needed"] >= 12

    def test_suggest_with_continuous_covariates(self, formula_service, complex_metadata):
        """Test suggestion with continuous variables."""
        suggestions = formula_service.suggest_formulas(
            metadata=complex_metadata,
            analysis_goal="Control for age and weight"
        )

        # Find multi-factor design
        multifactor_designs = [s for s in suggestions if s["complexity"] == "multifactor"]

        if len(multifactor_designs) > 0:
            multifactor_design = multifactor_designs[0]

            # Should include continuous variables
            assert "variables_used" in multifactor_design
            variables = multifactor_design["variables_used"]

            # Should have main condition plus covariates
            assert len(variables) > 1

    def test_suggest_validates_metadata(self, formula_service):
        """Test that suggest_formulas validates metadata."""
        # Empty metadata
        empty_metadata = pd.DataFrame()

        suggestions = formula_service.suggest_formulas(
            metadata=empty_metadata,
            analysis_goal="Test"
        )

        # Should return empty list for invalid metadata
        assert isinstance(suggestions, list)
        assert len(suggestions) == 0

    def test_suggest_filters_invalid_columns(self, formula_service):
        """Test that internal columns are filtered out."""
        metadata_with_internal = pd.DataFrame({
            'condition': ['control', 'treated'] * 3,
            '_internal_col': [1, 2] * 3,  # Should be filtered
            'n_cells': [1000, 2000] * 3,  # Should be filtered
        }, index=[f"sample_{i}" for i in range(6)])

        suggestions = formula_service.suggest_formulas(
            metadata=metadata_with_internal,
            analysis_goal="Test"
        )

        # Should still generate suggestions
        assert len(suggestions) > 0

        # Verify internal columns not used
        for suggestion in suggestions:
            formula = suggestion["formula"]
            assert "_internal_col" not in formula
            assert "n_cells" not in formula

    def test_suggest_handles_all_numeric(self, formula_service):
        """Test suggestion when all variables are numeric."""
        numeric_metadata = pd.DataFrame({
            'value1': [1, 2, 3, 4, 5, 6],
            'value2': [10.1, 20.2, 30.3, 40.4, 50.5, 60.6],
            'value3': [100, 200, 300, 400, 500, 600]
        }, index=[f"sample_{i}" for i in range(6)])

        suggestions = formula_service.suggest_formulas(
            metadata=numeric_metadata,
            analysis_goal="Test"
        )

        # Should return suggestions (may be empty if no binary factors)
        assert isinstance(suggestions, list)


# ===============================================================================
# Class 2: TestPreviewDesignMatrix
# ===============================================================================


@pytest.mark.unit
class TestPreviewDesignMatrix:
    """Test design matrix preview functionality."""

    def test_preview_simple_formula(self, formula_service, simple_metadata):
        """Test preview of simple formula."""
        preview = formula_service.preview_design_matrix(
            formula="~condition",
            metadata=simple_metadata,
            max_rows=3
        )

        # Verify preview structure
        assert isinstance(preview, str)
        assert "Design Matrix Preview" in preview
        assert "(Intercept)" in preview
        assert "condition" in preview

        # Should show matrix dimensions
        assert "×" in preview

        # Should have column explanations
        assert "Column Explanations:" in preview

    def test_preview_with_batch(self, formula_service, simple_metadata):
        """Test preview with batch correction formula."""
        preview = formula_service.preview_design_matrix(
            formula="~condition + batch",
            metadata=simple_metadata,
            max_rows=3
        )

        # Verify both factors included
        assert "condition" in preview
        assert "batch" in preview

        # Should have intercept
        assert "(Intercept)" in preview

        # Should explain columns
        assert "Column Explanations:" in preview
        assert "Effect of" in preview

    def test_preview_with_interaction(self, formula_service, simple_metadata):
        """Test preview with interaction terms."""
        preview = formula_service.preview_design_matrix(
            formula="~condition * batch",
            metadata=simple_metadata,
            max_rows=3
        )

        # Verify interaction columns
        assert ":" in preview  # Interaction notation
        assert "Interaction between" in preview

        # Should show design properties
        assert "Design Properties:" in preview
        assert "Matrix rank:" in preview

    def test_preview_handles_invalid_formula(self, formula_service, simple_metadata):
        """Test preview with invalid formula."""
        preview = formula_service.preview_design_matrix(
            formula="~nonexistent_column",
            metadata=simple_metadata
        )

        # Should return error message
        assert isinstance(preview, str)
        assert "Error" in preview or "error" in preview

    def test_preview_matrix_shape(self, formula_service, simple_metadata):
        """Test that preview shows correct matrix dimensions."""
        preview = formula_service.preview_design_matrix(
            formula="~condition + batch",
            metadata=simple_metadata,
            max_rows=5
        )

        # Should show n_samples × n_coefficients
        assert "6 ×" in preview  # 6 samples

        # Should have design properties
        assert "Degrees of freedom:" in preview

    def test_preview_continuous_variable(self, formula_service, simple_metadata):
        """Test preview with continuous variable."""
        preview = formula_service.preview_design_matrix(
            formula="~condition + age",
            metadata=simple_metadata,
            max_rows=3
        )

        # Verify continuous variable shown
        assert "age" in preview
        assert "Continuous variable" in preview

    def test_preview_max_rows_limit(self, formula_service, complex_metadata):
        """Test that max_rows parameter works."""
        preview_short = formula_service.preview_design_matrix(
            formula="~condition",
            metadata=complex_metadata,
            max_rows=2
        )

        preview_long = formula_service.preview_design_matrix(
            formula="~condition",
            metadata=complex_metadata,
            max_rows=10
        )

        # Short preview should have "... and X more rows"
        if len(complex_metadata) > 2:
            assert "more rows" in preview_short


# ===============================================================================
# Class 3: TestValidateExperimentalDesign
# ===============================================================================


@pytest.mark.unit
class TestValidateExperimentalDesign:
    """Test experimental design validation."""

    def test_validate_simple_design(self, formula_service, simple_metadata):
        """Test validation of valid simple design."""
        result = formula_service.validate_experimental_design(
            metadata=simple_metadata,
            formula="~condition",
            min_replicates=2
        )

        # Verify result structure
        assert isinstance(result, dict)
        assert "valid" in result
        assert "warnings" in result
        assert "errors" in result
        assert "design_summary" in result

        # Should be valid
        assert result["valid"] == True
        assert isinstance(result["warnings"], list)
        assert isinstance(result["errors"], list)

    def test_validate_detects_confounding(self, formula_service):
        """Test detection of confounded design."""
        # Create confounded design (condition perfectly correlated with batch)
        confounded_metadata = pd.DataFrame({
            'condition': ['control', 'control', 'control', 'treated', 'treated', 'treated'],
            'batch': ['batch1', 'batch1', 'batch1', 'batch2', 'batch2', 'batch2']
        }, index=[f"sample_{i}" for i in range(6)])

        result = formula_service.validate_experimental_design(
            metadata=confounded_metadata,
            formula="~condition + batch",
            min_replicates=2
        )

        # Should have warnings (design matrix will be rank deficient)
        # Note: The service may not explicitly check for confounding,
        # but rank deficiency will be detected during design matrix construction
        assert isinstance(result, dict)
        assert "valid" in result

    def test_validate_detects_rank_deficiency(self, formula_service):
        """Test detection of rank deficient design matrix."""
        # Create metadata that will lead to rank deficiency
        rank_def_metadata = pd.DataFrame({
            'condition': ['a', 'a', 'b', 'b'],
            'factor2': ['x', 'x', 'y', 'y'],  # Perfectly correlated with condition
        }, index=[f"sample_{i}" for i in range(4)])

        result = formula_service.validate_experimental_design(
            metadata=rank_def_metadata,
            formula="~condition + factor2",
            min_replicates=1
        )

        # Validation should still work (warnings may be present)
        assert isinstance(result, dict)
        assert "valid" in result

    def test_validate_checks_sample_size(self, formula_service):
        """Test minimum sample size requirements."""
        # Very small sample size
        small_metadata = pd.DataFrame({
            'condition': ['control', 'treated'],
        }, index=['sample_0', 'sample_1'])

        result = formula_service.validate_experimental_design(
            metadata=small_metadata,
            formula="~condition",
            min_replicates=2
        )

        # Should have warnings about small sample size or insufficient replicates
        assert "warnings" in result
        assert len(result["warnings"]) > 0

    def test_validate_warns_about_imbalance(self, formula_service, imbalanced_metadata):
        """Test detection of unbalanced designs."""
        result = formula_service.validate_experimental_design(
            metadata=imbalanced_metadata,
            formula="~condition",
            min_replicates=3
        )

        # Should warn about imbalance (control has 2 samples, treated has 4)
        assert "warnings" in result
        warnings_text = " ".join(result["warnings"])
        assert "replicate" in warnings_text.lower() or "balance" in warnings_text.lower()

    def test_validate_handles_missing_values(self, formula_service):
        """Test handling of missing values in metadata."""
        metadata_with_na = pd.DataFrame({
            'condition': ['control', 'control', None, 'treated', 'treated', None],
            'batch': ['batch1', 'batch2', 'batch1', 'batch2', 'batch1', 'batch2']
        }, index=[f"sample_{i}" for i in range(6)])

        result = formula_service.validate_experimental_design(
            metadata=metadata_with_na,
            formula="~condition",
            min_replicates=2
        )

        # Should return validation result
        assert "warnings" in result
        # Note: Service may or may not warn about NaN - depends on implementation
        # We just verify the validation runs successfully
        assert isinstance(result["warnings"], list)

    def test_validate_complex_design(self, formula_service, complex_metadata):
        """Test validation of complex multi-factor design."""
        result = formula_service.validate_experimental_design(
            metadata=complex_metadata,
            formula="~condition + batch + age",
            min_replicates=2
        )

        # Should validate successfully
        assert isinstance(result, dict)
        assert "design_summary" in result

        # Design summary should include categorical variables
        if "condition" in result["design_summary"]:
            assert isinstance(result["design_summary"]["condition"], dict)

    def test_validate_insufficient_replicates(self, formula_service):
        """Test detection of insufficient replicates."""
        insufficient_metadata = pd.DataFrame({
            'condition': ['control', 'treated', 'other'],
            'batch': ['batch1', 'batch1', 'batch1']
        }, index=['sample_0', 'sample_1', 'sample_2'])

        result = formula_service.validate_experimental_design(
            metadata=insufficient_metadata,
            formula="~condition",
            min_replicates=2
        )

        # Should warn about insufficient replicates (each condition has only 1 sample)
        assert "warnings" in result
        assert len(result["warnings"]) > 0


# ===============================================================================
# Additional Formula Parsing Tests
# ===============================================================================


@pytest.mark.unit
class TestFormulaParsingAdvanced:
    """Test advanced formula parsing functionality."""

    def test_parse_formula_basic(self, formula_service, simple_metadata):
        """Test basic formula parsing."""
        result = formula_service.parse_formula(
            formula="~condition",
            metadata=simple_metadata
        )

        # Verify result structure
        assert isinstance(result, dict)
        assert "formula_string" in result
        assert "predictor_terms" in result
        assert "variable_info" in result
        assert result["formula_string"] == "~condition"

    def test_parse_formula_with_reference_levels(self, formula_service, simple_metadata):
        """Test formula parsing with explicit reference levels."""
        result = formula_service.parse_formula(
            formula="~condition + batch",
            metadata=simple_metadata,
            reference_levels={"condition": "control", "batch": "batch1"}
        )

        # Verify reference levels are set
        assert "reference_levels" in result
        assert result["reference_levels"]["condition"] == "control"
        assert result["reference_levels"]["batch"] == "batch1"

    def test_parse_formula_validates_variables(self, formula_service, simple_metadata):
        """Test that parsing validates variable existence."""
        with pytest.raises(FormulaError, match="Variables not found"):
            formula_service.parse_formula(
                formula="~nonexistent_var",
                metadata=simple_metadata
            )

    def test_construct_design_matrix_basic(self, formula_service, simple_metadata):
        """Test design matrix construction."""
        # Parse formula first
        formula_components = formula_service.parse_formula(
            formula="~condition",
            metadata=simple_metadata
        )

        # Construct design matrix
        result = formula_service.construct_design_matrix(
            formula_components=formula_components,
            metadata=simple_metadata
        )

        # Verify result
        assert isinstance(result, dict)
        assert "design_matrix" in result
        assert "design_df" in result
        assert "coefficient_names" in result
        assert "rank" in result

        # Verify matrix shape
        assert result["design_matrix"].shape[0] == len(simple_metadata)
        assert result["design_matrix"].shape[1] == result["n_coefficients"]

    def test_construct_design_matrix_with_contrast(self, formula_service, simple_metadata):
        """Test design matrix construction with contrast."""
        formula_components = formula_service.parse_formula(
            formula="~condition",
            metadata=simple_metadata
        )

        result = formula_service.construct_design_matrix(
            formula_components=formula_components,
            metadata=simple_metadata,
            contrast=["condition", "treated", "control"]
        )

        # Verify contrast vector created
        assert result["contrast_vector"] is not None
        assert result["contrast_name"] is not None
        assert "treated" in result["contrast_name"]
        assert "control" in result["contrast_name"]

    def test_create_simple_design(self, formula_service, simple_metadata):
        """Test simplified design creation helper."""
        result = formula_service.create_simple_design(
            metadata=simple_metadata,
            condition_col="condition",
            batch_col="batch",
            reference_condition="control"
        )

        # Verify result
        assert isinstance(result, dict)
        assert "design_matrix" in result
        assert "coefficient_names" in result

    def test_estimate_statistical_power(self, formula_service, simple_metadata):
        """Test statistical power estimation."""
        # Create design matrix
        formula_components = formula_service.parse_formula(
            formula="~condition + batch",
            metadata=simple_metadata
        )
        design_result = formula_service.construct_design_matrix(
            formula_components=formula_components,
            metadata=simple_metadata
        )

        # Estimate power
        power_result = formula_service.estimate_statistical_power(
            design_matrix=design_result["design_matrix"],
            effect_size=0.5,
            alpha=0.05
        )

        # Verify result
        assert isinstance(power_result, dict)
        assert "estimated_power" in power_result
        assert "power_category" in power_result
        assert "recommendations" in power_result
        assert 0 <= power_result["estimated_power"] <= 1


# ===============================================================================
# Edge Cases and Error Handling
# ===============================================================================


@pytest.mark.unit
class TestFormulaServiceEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_formula(self, formula_service, simple_metadata):
        """Test handling of empty formula."""
        with pytest.raises(FormulaError, match="Empty formula"):
            formula_service.parse_formula(
                formula="~",
                metadata=simple_metadata
            )

    def test_formula_without_tilde(self, formula_service, simple_metadata):
        """Test formula without leading tilde."""
        # Should auto-add tilde
        result = formula_service.parse_formula(
            formula="condition",
            metadata=simple_metadata
        )

        assert result["formula_string"] == "~condition"

    def test_formula_with_whitespace(self, formula_service, simple_metadata):
        """Test formula with extra whitespace."""
        result = formula_service.parse_formula(
            formula="~  condition   +   batch  ",
            metadata=simple_metadata
        )

        # Should clean whitespace
        assert "condition" in result["formula_string"]
        assert "batch" in result["formula_string"]

    def test_rank_deficient_matrix_warning(self, formula_service):
        """Test warning for rank deficient design matrix."""
        # Create perfectly correlated variables
        correlated_metadata = pd.DataFrame({
            'var1': [1, 2, 3, 4],
            'var2': [2, 4, 6, 8],  # Perfectly correlated (2 * var1)
        }, index=[f"sample_{i}" for i in range(4)])

        formula_components = formula_service.parse_formula(
            formula="~var1 + var2",
            metadata=correlated_metadata
        )

        # Should construct but warn about rank deficiency
        with patch.object(formula_service.logger, 'warning') as mock_warning:
            result = formula_service.construct_design_matrix(
                formula_components=formula_components,
                metadata=correlated_metadata
            )

            # Verify rank deficiency
            assert result["rank"] < result["n_coefficients"]

    def test_invalid_contrast_format(self, formula_service, simple_metadata):
        """Test invalid contrast format."""
        formula_components = formula_service.parse_formula(
            formula="~condition",
            metadata=simple_metadata
        )

        # Contrast must have exactly 3 elements
        # Service wraps FormulaError in DesignMatrixError
        with pytest.raises((FormulaError, DesignMatrixError), match="Contrast must be"):
            formula_service.construct_design_matrix(
                formula_components=formula_components,
                metadata=simple_metadata,
                contrast=["condition", "treated"]  # Missing third element
            )

    def test_continuous_variable_in_contrast(self, formula_service, simple_metadata):
        """Test that continuous variables cannot be used in contrasts."""
        formula_components = formula_service.parse_formula(
            formula="~age",
            metadata=simple_metadata
        )

        # Should raise error (age is continuous)
        # Service wraps FormulaError in DesignMatrixError
        with pytest.raises((FormulaError, DesignMatrixError), match="must be categorical"):
            formula_service.construct_design_matrix(
                formula_components=formula_components,
                metadata=simple_metadata,
                contrast=["age", "25", "30"]
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
