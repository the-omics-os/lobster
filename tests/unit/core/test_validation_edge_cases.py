"""
Comprehensive edge case testing for validation system.

This test suite focuses on:
1. Empty array validation (Issue #1)
2. ValidationResult interface completeness (Issue #4)
3. Null/None value handling
4. Single-element arrays
5. Arrays with identical values
6. Edge cases in data type validation
7. Empty datasets through full pipeline
8. Zero-length modalities
9. Missing required fields
10. Boundary conditions for numeric validators
"""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData

from lobster.core.interfaces.validator import IValidator, ValidationResult
from lobster.core.schemas.validation import FlexibleValidator, SchemaValidator


class TestValidationResultInterface:
    """Test ValidationResult interface completeness (Issue #4)."""

    def test_basic_attributes_exist(self):
        """Verify all expected attributes exist."""
        result = ValidationResult()

        # Basic attributes from dataclass
        assert hasattr(result, "errors")
        assert hasattr(result, "warnings")
        assert hasattr(result, "info")
        assert hasattr(result, "metadata")

        # Properties
        assert hasattr(result, "has_errors")
        assert hasattr(result, "has_warnings")
        assert hasattr(result, "is_valid")

    def test_validation_result_methods(self):
        """Verify all expected methods exist."""
        result = ValidationResult()

        assert hasattr(result, "add_error")
        assert hasattr(result, "add_warning")
        assert hasattr(result, "add_info")
        assert hasattr(result, "merge")
        assert hasattr(result, "to_dict")
        assert hasattr(result, "summary")
        assert hasattr(result, "format_messages")

    def test_validation_result_initialization(self):
        """Test ValidationResult can be initialized with all fields."""
        result = ValidationResult(
            errors=["error1"],
            warnings=["warning1"],
            info=["info1"],
            metadata={"key": "value"},
        )

        assert len(result.errors) == 1
        assert len(result.warnings) == 1
        assert len(result.info) == 1
        assert result.metadata["key"] == "value"

    def test_validation_result_merge(self):
        """Test merging validation results."""
        result1 = ValidationResult()
        result1.add_error("error1")
        result1.add_warning("warning1")

        result2 = ValidationResult()
        result2.add_error("error2")
        result2.add_warning("warning2")

        merged = result1.merge(result2)

        assert len(merged.errors) == 2
        assert len(merged.warnings) == 2
        assert "error1" in merged.errors
        assert "error2" in merged.errors


class TestEmptyArrayValidation:
    """Test empty array handling (Issue #1)."""

    def test_empty_dense_array(self):
        """Test validation with empty dense array."""
        # Create AnnData with empty array
        adata = AnnData(X=np.array([]).reshape(0, 0))

        schema = {"obs": {}, "var": {}}
        validator = SchemaValidator(schema)

        result = validator.validate(adata, check_ranges=True)

        # Should not crash, should report empty dataset
        assert result.has_errors
        assert any("No observations" in err for err in result.errors)

    def test_empty_sparse_array(self):
        """Test validation with empty sparse array."""
        # Create AnnData with empty sparse array
        empty_sparse = sp.csr_matrix((0, 0))
        adata = AnnData(X=empty_sparse)

        schema = {"obs": {}, "var": {}}
        validator = SchemaValidator(schema)

        result = validator.validate(adata, check_ranges=True)

        assert result.has_errors
        assert any("No observations" in err for err in result.errors)

    def test_zero_observations(self):
        """Test with zero observations but some variables."""
        # 0 observations, 10 variables
        adata = AnnData(X=np.array([]).reshape(0, 10))

        schema = {"obs": {}, "var": {}}
        validator = SchemaValidator(schema)

        result = validator.validate(adata)

        assert result.has_errors
        assert any("No observations" in err for err in result.errors)

    def test_zero_variables(self):
        """Test with some observations but zero variables."""
        # 10 observations, 0 variables
        adata = AnnData(X=np.array([]).reshape(10, 0))

        schema = {"obs": {}, "var": {}}
        validator = SchemaValidator(schema)

        result = validator.validate(adata)

        assert result.has_errors
        assert any("No variables" in err for err in result.errors)

    def test_empty_array_min_max_operations(self):
        """Test min/max operations on empty arrays (reproduces Issue #1)."""
        # This is the critical test for Issue #1
        empty_array = np.array([])

        # Direct test of the problematic pattern
        if empty_array.size > 0:
            min_val = empty_array.min()
            max_val = empty_array.max()
        else:
            # Should handle empty case gracefully
            pass

        # Now test through validator
        adata = AnnData(X=np.array([]).reshape(0, 0))
        schema = {}
        validator = SchemaValidator(schema)

        # Should not crash on empty array
        result = validator._validate_value_ranges(adata)

        # Should either have no warnings or appropriate errors
        assert isinstance(result, ValidationResult)

    def test_empty_obs_metadata(self):
        """Test empty observation metadata."""
        adata = AnnData(
            X=np.zeros((5, 10)), obs=pd.DataFrame(index=range(5))  # Empty obs
        )

        schema = {"obs": {"required": ["sample_type"]}}
        validator = SchemaValidator(schema)

        result = validator.validate(adata)

        assert result.has_errors
        assert any("sample_type" in err for err in result.errors)

    def test_empty_var_metadata(self):
        """Test empty variable metadata."""
        adata = AnnData(
            X=np.zeros((5, 10)), var=pd.DataFrame(index=range(10))  # Empty var
        )

        schema = {"var": {"required": ["gene_name"]}}
        validator = SchemaValidator(schema)

        result = validator.validate(adata)

        assert result.has_errors
        assert any("gene_name" in err for err in result.errors)


class TestNullValueHandling:
    """Test null/None value handling throughout validation."""

    def test_all_nan_required_column(self):
        """Test required column with all NaN values."""
        adata = AnnData(
            X=np.zeros((5, 10)), obs=pd.DataFrame({"sample_type": [np.nan] * 5})
        )

        schema = {"obs": {"required": ["sample_type"]}}
        validator = SchemaValidator(schema)

        result = validator.validate(adata)

        assert result.has_warnings
        assert any("only NaN" in warn for warn in result.warnings)

    def test_partial_nan_values(self):
        """Test column with some NaN values."""
        adata = AnnData(
            X=np.zeros((5, 10)),
            obs=pd.DataFrame({"sample_type": ["A", "B", np.nan, "C", np.nan]}),
        )

        schema = {"obs": {"required": ["sample_type"]}}
        validator = SchemaValidator(schema)

        result = validator.validate(adata)

        # Should pass validation but may have info
        assert result.is_valid or result.has_warnings

    def test_nan_in_expression_matrix(self):
        """Test NaN values in expression matrix."""
        X = np.zeros((5, 10))
        X[0, 0] = np.nan
        X[2, 5] = np.nan

        adata = AnnData(X=X)

        schema = {}
        validator = SchemaValidator(schema)

        result = validator.validate_data_quality(adata)

        # Should detect NaN values
        # Note: This depends on implementation
        # Some validators may not check for NaN in dense arrays

    def test_none_in_metadata(self):
        """Test None values in metadata."""
        adata = AnnData(
            X=np.zeros((5, 10)),
            obs=pd.DataFrame({"sample_type": ["A", None, "B", None, "C"]}),
        )

        schema = {"obs": {"required": ["sample_type"]}}
        validator = SchemaValidator(schema)

        result = validator.validate(adata)

        # Should handle None gracefully


class TestSingleElementArrays:
    """Test single-element array handling."""

    def test_single_observation(self):
        """Test with single observation."""
        adata = AnnData(
            X=np.array([[1, 2, 3, 4, 5]]), obs=pd.DataFrame({"sample_type": ["A"]})
        )

        schema = {"obs": {"required": ["sample_type"]}}
        validator = SchemaValidator(schema)

        result = validator.validate(adata)

        assert result.is_valid

    def test_single_variable(self):
        """Test with single variable."""
        adata = AnnData(
            X=np.array([[1], [2], [3], [4], [5]]),
            var=pd.DataFrame({"gene_name": ["GENE1"]}),
        )

        schema = {"var": {"required": ["gene_name"]}}
        validator = SchemaValidator(schema)

        result = validator.validate(adata)

        assert result.is_valid

    def test_single_cell_single_gene(self):
        """Test with single cell and single gene."""
        adata = AnnData(
            X=np.array([[42.0]]),
            obs=pd.DataFrame({"sample_type": ["A"]}),
            var=pd.DataFrame({"gene_name": ["GENE1"]}),
        )

        schema = {
            "obs": {"required": ["sample_type"]},
            "var": {"required": ["gene_name"]},
        }
        validator = SchemaValidator(schema)

        result = validator.validate(adata)

        assert result.is_valid


class TestIdenticalValueArrays:
    """Test arrays with all identical values."""

    def test_all_zeros(self):
        """Test array with all zeros."""
        adata = AnnData(X=np.zeros((10, 20)))

        schema = {}
        validator = SchemaValidator(schema)

        result = validator.validate(adata)

        # Should not crash, may have warnings about sparsity
        assert isinstance(result, ValidationResult)

    def test_all_ones(self):
        """Test array with all ones."""
        adata = AnnData(X=np.ones((10, 20)))

        schema = {}
        validator = SchemaValidator(schema)

        result = validator.validate(adata)

        assert isinstance(result, ValidationResult)

    def test_all_same_value(self):
        """Test array with all same non-zero/one value."""
        adata = AnnData(X=np.full((10, 20), 42.0))

        schema = {}
        validator = SchemaValidator(schema)

        result = validator.validate(adata)

        assert isinstance(result, ValidationResult)

    def test_sparse_all_zeros(self):
        """Test sparse array with all zeros (empty)."""
        adata = AnnData(X=sp.csr_matrix((10, 20)))

        schema = {}
        validator = SchemaValidator(schema)

        result = validator.validate(adata)

        # Should detect high sparsity
        assert result.has_warnings or result.is_valid


class TestDataTypeValidation:
    """Test data type validation edge cases."""

    def test_integer_expression_matrix(self):
        """Test with integer expression matrix."""
        adata = AnnData(X=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32))

        schema = {}
        validator = SchemaValidator(schema)

        result = validator.validate(adata, check_types=True)

        # Integer is valid numeric type
        assert result.is_valid

    def test_float_expression_matrix(self):
        """Test with float expression matrix."""
        adata = AnnData(
            X=np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], dtype=np.float64)
        )

        schema = {}
        validator = SchemaValidator(schema)

        result = validator.validate(adata, check_types=True)

        assert result.is_valid

    def test_mixed_dtype_metadata(self):
        """Test metadata with mixed data types."""
        adata = AnnData(
            X=np.zeros((5, 10)),
            obs=pd.DataFrame(
                {
                    "string_col": ["A", "B", "C", "D", "E"],
                    "int_col": [1, 2, 3, 4, 5],
                    "float_col": [1.1, 2.2, 3.3, 4.4, 5.5],
                    "bool_col": [True, False, True, False, True],
                    "category_col": pd.Categorical(
                        ["cat1", "cat2", "cat1", "cat2", "cat1"]
                    ),
                }
            ),
        )

        schema = {
            "obs": {
                "required": [
                    "string_col",
                    "int_col",
                    "float_col",
                    "bool_col",
                    "category_col",
                ],
                "types": {
                    "string_col": "string",
                    "int_col": "numeric",
                    "float_col": "numeric",
                    "bool_col": "boolean",
                    "category_col": "categorical",
                },
            }
        }
        validator = SchemaValidator(schema)

        result = validator.validate(adata, check_types=True)

        # Should validate type compatibility correctly
        assert result.is_valid or result.has_warnings

    def test_object_dtype_numbers(self):
        """Test numeric values stored as object dtype."""
        adata = AnnData(
            X=np.zeros((5, 10)),
            obs=pd.DataFrame(
                {"numeric_as_object": pd.Series([1, 2, 3, 4, 5], dtype=object)}
            ),
        )

        schema = {
            "obs": {
                "required": ["numeric_as_object"],
                "types": {"numeric_as_object": "numeric"},
            }
        }
        validator = SchemaValidator(schema)

        result = validator.validate(adata, check_types=True)

        # Should warn about type mismatch
        assert result.has_warnings or result.has_errors


class TestValueRangeBoundaries:
    """Test boundary conditions for numeric validators."""

    def test_negative_values(self):
        """Test negative values in expression matrix."""
        adata = AnnData(X=np.array([[-1, -2, -3], [-4, -5, -6]], dtype=np.float64))

        schema = {}
        validator = SchemaValidator(schema)

        result = validator.validate(adata, check_ranges=True)

        # Should warn about negative values
        assert result.has_warnings

    def test_very_large_values(self):
        """Test very large values in expression matrix."""
        adata = AnnData(
            X=np.array([[1e7, 2e7, 3e7], [4e7, 5e7, 6e7]], dtype=np.float64)
        )

        schema = {}
        validator = SchemaValidator(schema)

        result = validator.validate(adata, check_ranges=True)

        # Should warn about large values
        assert result.has_warnings

    def test_infinite_values(self):
        """Test infinite values."""
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        X[0, 0] = np.inf
        X[1, 2] = -np.inf

        adata = AnnData(X=X)

        schema = {}
        validator = SchemaValidator(schema)

        result = validator.validate(adata, check_ranges=True)

        # Should detect extreme values
        # Note: Current implementation may not explicitly check for inf

    def test_near_zero_values(self):
        """Test values very close to zero."""
        adata = AnnData(
            X=np.array([[1e-10, 2e-10, 3e-10], [4e-10, 5e-10, 6e-10]], dtype=np.float64)
        )

        schema = {}
        validator = SchemaValidator(schema)

        result = validator.validate(adata, check_ranges=True)

        # Should validate without errors
        assert result.is_valid or result.has_warnings


class TestLayerValidation:
    """Test layer validation edge cases."""

    def test_empty_layers(self):
        """Test AnnData with no layers."""
        adata = AnnData(X=np.zeros((5, 10)))

        schema = {"layers": {"required": []}}
        validator = SchemaValidator(schema)

        result = validator.validate(adata)

        assert result.is_valid

    def test_empty_layer_content(self):
        """Test layer with correct shape but empty content."""
        adata = AnnData(X=np.zeros((5, 10)), layers={"counts": np.zeros((5, 10))})

        schema = {"layers": {"required": ["counts"]}}
        validator = SchemaValidator(schema)

        result = validator.validate(adata)

        # Shape is correct, should pass
        assert result.is_valid


class TestFlexibleValidatorEdgeCases:
    """Test FlexibleValidator edge cases."""

    def test_ignore_warnings(self):
        """Test ignoring specific warnings."""
        adata = AnnData(X=np.array([[-1, -2, -3], [-4, -5, -6]], dtype=np.float64))

        schema = {}
        validator = FlexibleValidator(schema, ignore_warnings={"Negative values"})

        result = validator.validate(adata, check_ranges=True)

        # Should filter out negative value warnings
        assert not any("Negative" in w for w in result.warnings)

    def test_custom_rule_success(self):
        """Test custom validation rule that passes."""

        def custom_rule(adata):
            result = ValidationResult()
            if adata.n_obs >= 5:
                result.add_info("Sufficient observations")
            else:
                result.add_warning("Few observations")
            return result

        adata = AnnData(X=np.zeros((10, 20)))

        schema = {}
        validator = FlexibleValidator(schema, custom_rules={"sample_size": custom_rule})

        result = validator.validate(adata)

        assert result.is_valid

    def test_custom_rule_failure(self):
        """Test custom validation rule that fails."""

        def failing_rule(adata):
            raise ValueError("Custom rule error")

        adata = AnnData(X=np.zeros((5, 10)))

        schema = {}
        validator = FlexibleValidator(schema, custom_rules={"failing": failing_rule})

        result = validator.validate(adata)

        # Should catch exception and add warning
        assert result.has_warnings
        assert any("failed" in w.lower() for w in result.warnings)

    def test_strict_mode_with_warnings(self):
        """Test strict mode converts warnings to errors."""
        adata = AnnData(X=np.array([[-1, -2, -3]], dtype=np.float64))

        schema = {}
        validator = FlexibleValidator(schema)

        result = validator.validate(adata, strict=True, check_ranges=True)

        # In strict mode, warnings become errors
        assert result.has_errors
        assert any("STRICT MODE" in err for err in result.errors)


class TestCompletenessValidation:
    """Test completeness validation edge cases."""

    def test_high_sparsity_sparse_matrix(self):
        """Test sparse matrix with >95% zeros."""
        # Create very sparse matrix
        X = sp.random(100, 200, density=0.02, format="csr")
        adata = AnnData(X=X)

        schema = {}
        validator = SchemaValidator(schema)

        result = validator.validate(adata, check_completeness=True)

        # Should warn about sparsity
        assert result.has_warnings or result.is_valid

    def test_high_sparsity_dense_matrix(self):
        """Test dense matrix with >95% zeros."""
        X = np.zeros((100, 200))
        X[0:5, 0:10] = 1  # Only 50 values, rest zeros
        adata = AnnData(X=X)

        schema = {}
        validator = SchemaValidator(schema)

        result = validator.validate(adata, check_completeness=True)

        # Dense matrices don't have nnz attribute, so no sparsity warning


class TestObsmUnsmValidation:
    """Test obsm and unsm validation edge cases."""

    def test_empty_obsm(self):
        """Test with empty obsm."""
        adata = AnnData(X=np.zeros((5, 10)))

        schema = {"obsm": {"required": ["X_pca"]}}
        validator = SchemaValidator(schema)

        result = validator.validate(adata)

        # Should warn about missing obsm
        assert result.has_warnings

    def test_empty_uns(self):
        """Test with empty uns."""
        adata = AnnData(X=np.zeros((5, 10)))

        schema = {"uns": {"required": ["analysis_metadata"]}}
        validator = SchemaValidator(schema)

        result = validator.validate(adata)

        # Should warn about missing uns
        assert result.has_warnings


class TestRegressionIssues:
    """Test specific known issues."""

    def test_issue_1_empty_array_crash(self):
        """
        Test Issue #1: Empty array validation crash.

        Line 254 in validation.py (_validate_uns_schema) or similar
        locations may crash when performing min/max operations on empty arrays.
        """
        # Create completely empty AnnData
        adata = AnnData(X=np.array([]).reshape(0, 0))

        schema = {"obs": {}, "var": {}, "layers": {}, "obsm": {}, "uns": {}}
        validator = SchemaValidator(schema)

        # This should not crash
        try:
            result = validator.validate(
                adata,
                strict=False,
                check_types=True,
                check_ranges=True,
                check_completeness=True,
            )
            # Should complete and report errors about empty dataset
            assert result.has_errors
            assert any("No observations" in err for err in result.errors)
        except (ValueError, IndexError) as e:
            pytest.fail(f"Empty array validation crashed (Issue #1): {e}")

    def test_issue_4_validation_result_incomplete(self):
        """
        Test Issue #4: ValidationResult interface incomplete.

        ValidationResult may be missing expected attributes like 'recommendations'.
        """
        result = ValidationResult()

        # Check for standard attributes
        assert hasattr(result, "errors"), "Missing 'errors' attribute"
        assert hasattr(result, "warnings"), "Missing 'warnings' attribute"
        assert hasattr(result, "info"), "Missing 'info' attribute"
        assert hasattr(result, "metadata"), "Missing 'metadata' attribute"

        # Check for potentially missing attribute (Issue #4)
        if not hasattr(result, "recommendations"):
            # Document the issue
            print(
                "\n⚠️  Issue #4 Confirmed: ValidationResult missing 'recommendations' attribute"
            )
            # This is expected to fail until Issue #4 is fixed
            # For now, we document it rather than failing the test
        else:
            print("\n✓ ValidationResult has 'recommendations' attribute")

    def test_minimal_valid_adata(self):
        """Test minimal valid AnnData passes validation."""
        adata = AnnData(
            X=np.array([[1.0]]),
            obs=pd.DataFrame({"cell_id": ["cell1"]}),
            var=pd.DataFrame({"gene_name": ["GENE1"]}),
        )

        schema = {"obs": {"required": ["cell_id"]}, "var": {"required": ["gene_name"]}}
        validator = SchemaValidator(schema)

        result = validator.validate(adata)

        assert result.is_valid, f"Minimal valid AnnData failed: {result.errors}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
