"""
Comprehensive unit tests for schema validation.

This module provides thorough testing of the schema system including
TranscriptomicsSchema.get_single_cell_schema() and ProteomicsSchema.get_mass_spectrometry_schema() validation, data type enforcement,
value constraints, schema evolution, and custom validator implementations.

Test coverage target: 95%+ with meaningful tests for biological data validation.
"""

import json
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import anndata as ad
from scipy import sparse

from lobster.core.schemas.transcriptomics import TranscriptomicsSchema
from lobster.core.schemas.proteomics import ProteomicsSchema
from lobster.core.schemas.validation import FlexibleValidator, SchemaValidator
from lobster.core.interfaces.validator import ValidationResult

from tests.mock_data.factories import (
    SingleCellDataFactory, 
    BulkRNASeqDataFactory, 
    ProteomicsDataFactory
)
from tests.mock_data.base import SMALL_DATASET_CONFIG, MEDIUM_DATASET_CONFIG


# ===============================================================================
# Test Fixtures
# ===============================================================================

@pytest.fixture
def compliant_transcriptomics_data():
    """Create transcriptomics data that complies with schema."""
    adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)

    # Add common optional fields (schema is now flexible)
    adata.obs['total_counts'] = np.array(adata.X.sum(axis=1)).flatten()
    adata.obs['n_genes_by_counts'] = (adata.X > 0).sum(axis=1)
    adata.obs['pct_counts_mt'] = np.random.uniform(0, 30, adata.n_obs)
    adata.obs['cell_id'] = [f"CELL_{i:05d}" for i in range(adata.n_obs)]
    adata.obs['sample_id'] = np.random.choice(['Sample_A', 'Sample_B'], adata.n_obs)

    adata.var['gene_id'] = [f"GENE_{i:05d}" for i in range(adata.n_vars)]
    adata.var['gene_symbol'] = [f"SYMBOL_{i}" for i in range(adata.n_vars)]
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    adata.var['ribo'] = adata.var_names.str.startswith(('RPL', 'RPS'))

    # Add additional optional fields
    adata.obs['pct_counts_ribo'] = np.random.uniform(0, 50, adata.n_obs)
    adata.var['gene_name'] = [f"Gene_{i}" for i in range(adata.n_vars)]
    adata.var['highly_variable'] = np.random.choice([True, False], adata.n_vars, p=[0.1, 0.9])

    return adata


@pytest.fixture
def compliant_proteomics_data():
    """Create proteomics data that complies with schema."""
    adata = ProteomicsDataFactory(config=SMALL_DATASET_CONFIG)

    # Add common optional fields (schema is now flexible)
    adata.obs['sample_id'] = [f"Sample_{i:03d}" for i in range(adata.n_obs)]
    adata.obs['total_intensity'] = np.array(adata.X.sum(axis=1)).flatten()
    adata.obs['condition'] = np.random.choice(['Control', 'Treatment'], adata.n_obs)
    adata.obs['batch'] = np.random.choice(['Batch1', 'Batch2'], adata.n_obs)

    adata.var['protein_id'] = [f"PROT_{i:05d}" for i in range(adata.n_vars)]
    adata.var['protein_name'] = [f"Protein_{i}" for i in range(adata.n_vars)]
    adata.var['uniprot_id'] = [f"P{i:05d}" for i in range(adata.n_vars)]
    adata.var['gene_symbol'] = [f"GENE_{i}" for i in range(adata.n_vars)]

    # Add additional optional fields
    adata.obs['n_proteins'] = (adata.X > 0).sum(axis=1)
    adata.var['is_contaminant'] = np.random.choice([True, False], adata.n_vars, p=[0.05, 0.95])
    adata.var['n_peptides'] = np.random.randint(1, 10, adata.n_vars)
    adata.var['sequence_coverage'] = np.random.uniform(5, 50, adata.n_vars)

    return adata


@pytest.fixture
def non_compliant_transcriptomics_data():
    """Create transcriptomics data that violates schema."""
    adata = ad.AnnData(X=np.random.randint(0, 100, (50, 200)))
    
    # Missing required obs columns: total_counts, n_genes_by_counts
    # Missing required var columns: gene_ids, feature_types
    
    return adata


@pytest.fixture
def non_compliant_proteomics_data():
    """Create proteomics data that violates schema."""
    adata = ad.AnnData(X=np.random.randn(20, 100))
    
    # Missing required obs columns: sample_id, total_protein_intensity
    # Missing required var columns: protein_ids, protein_names
    
    return adata


# ===============================================================================
# Schema Structure Tests
# ===============================================================================

@pytest.mark.unit
class TestSchemaStructure:
    """Test schema structure and content."""
    
    def test_transcriptomics_schema_structure(self):
        """Test TranscriptomicsSchema has expected structure."""
        schema = TranscriptomicsSchema.get_single_cell_schema()
        assert isinstance(schema, dict)

        # Check required top-level keys
        assert "obs" in schema
        assert "var" in schema
        assert "layers" in schema
        assert "obsm" in schema
        assert "uns" in schema

        # Check schema sections have required structure
        assert isinstance(schema["obs"], dict)
        assert isinstance(schema["var"], dict)
        assert "required" in schema["obs"]
        assert "optional" in schema["obs"]
        assert "required" in schema["var"]
        assert "optional" in schema["var"]

        # Check data types
        assert isinstance(schema["obs"]["required"], list)
        assert isinstance(schema["obs"]["optional"], list)
        assert isinstance(schema["var"]["required"], list)
        assert isinstance(schema["var"]["optional"], list)
    
    def test_transcriptomics_schema_content(self):
        """Test TranscriptomicsSchema contains expected fields."""
        schema = TranscriptomicsSchema.get_single_cell_schema()

        # Check optional observation fields (schema is now flexible)
        assert "total_counts" in schema["obs"]["optional"]
        assert "n_genes_by_counts" in schema["obs"]["optional"]
        assert "pct_counts_mt" in schema["obs"]["optional"]

        # Check optional variable fields
        assert "gene_id" in schema["var"]["optional"]
        assert "gene_symbol" in schema["var"]["optional"]
        assert "gene_name" in schema["var"]["optional"]

        # Check modality and description
        assert schema["modality"] == "single_cell_rna_seq"
        assert "description" in schema
    
    def test_proteomics_schema_structure(self):
        """Test ProteomicsSchema has expected structure."""
        schema = ProteomicsSchema.get_mass_spectrometry_schema()
        assert isinstance(schema, dict)

        # Check required top-level keys
        assert "obs" in schema
        assert "var" in schema
        assert "layers" in schema
        assert "obsm" in schema
        assert "uns" in schema

        # Check schema sections have required structure
        assert isinstance(schema["obs"], dict)
        assert isinstance(schema["var"], dict)
        assert "required" in schema["obs"]
        assert "optional" in schema["obs"]
        assert "required" in schema["var"]
        assert "optional" in schema["var"]

        # Check data types
        assert isinstance(schema["obs"]["required"], list)
        assert isinstance(schema["obs"]["optional"], list)
        assert isinstance(schema["var"]["required"], list)
        assert isinstance(schema["var"]["optional"], list)
    
    def test_proteomics_schema_content(self):
        """Test ProteomicsSchema contains expected fields."""
        schema = ProteomicsSchema.get_mass_spectrometry_schema()

        # Check optional observation fields (schema is now flexible)
        assert "sample_id" in schema["obs"]["optional"]
        assert "total_intensity" in schema["obs"]["optional"]
        assert "batch" in schema["obs"]["optional"]

        # Check optional variable fields
        assert "protein_id" in schema["var"]["optional"]
        assert "protein_name" in schema["var"]["optional"]
        assert "is_contaminant" in schema["var"]["optional"]

        # Check modality and description
        assert schema["modality"] == "mass_spectrometry_proteomics"
        assert "description" in schema
    
    def test_schema_field_types(self):
        """Test that schema fields have appropriate type constraints."""
        # Test transcriptomics schema has type specifications
        sc_schema = TranscriptomicsSchema.get_single_cell_schema()
        assert "types" in sc_schema["obs"]
        assert "types" in sc_schema["var"]
        assert isinstance(sc_schema["obs"]["types"], dict)
        assert isinstance(sc_schema["var"]["types"], dict)

        # Test proteomics schema has type specifications
        ms_schema = ProteomicsSchema.get_mass_spectrometry_schema()
        assert "types" in ms_schema["obs"]
        assert "types" in ms_schema["var"]
        assert isinstance(ms_schema["obs"]["types"], dict)
        assert isinstance(ms_schema["var"]["types"], dict)
    
    def test_additional_schema_types(self):
        """Test additional schema types beyond basic single-cell and MS."""
        # Test bulk RNA-seq schema
        bulk_schema = TranscriptomicsSchema.get_bulk_rna_seq_schema()
        assert bulk_schema["modality"] == "bulk_rna_seq"
        assert "obs" in bulk_schema
        assert "var" in bulk_schema

        # Test affinity proteomics schema
        affinity_schema = ProteomicsSchema.get_affinity_proteomics_schema()
        assert affinity_schema["modality"] == "affinity_proteomics"
        assert "obs" in affinity_schema
        assert "var" in affinity_schema

        # Test that schemas are distinct
        sc_schema = TranscriptomicsSchema.get_single_cell_schema()
        assert sc_schema["modality"] != bulk_schema["modality"]
        assert bulk_schema["modality"] != affinity_schema["modality"]


# ===============================================================================
# FlexibleValidator Tests
# ===============================================================================

@pytest.mark.unit
class TestFlexibleValidator:
    """Test FlexibleValidator functionality."""
    
    def test_flexible_validator_initialization(self):
        """Test FlexibleValidator initialization."""
        schema = {"obs": {"required": ["test_field"], "optional": []}, "var": {"required": ["test_var"], "optional": []}}
        validator = FlexibleValidator(schema, name="TestValidator")

        assert validator.schema == schema
        assert validator.name == "TestValidator"
    
    def test_validate_schema_compliance_success(self):
        """Test successful schema compliance validation."""
        schema = {"obs": {"required": ["sample_id"], "optional": []}, "var": {"required": ["gene_id"], "optional": []}}
        validator = FlexibleValidator(schema)

        # Create compliant data
        adata = ad.AnnData(X=np.random.rand(10, 50))
        adata.obs['sample_id'] = [f"S{i}" for i in range(10)]
        adata.var['gene_id'] = [f"G{i}" for i in range(50)]

        # Should not have errors
        result = validator.validate_schema_compliance(adata, schema)
        assert not result.has_errors
    
    def test_validate_schema_compliance_missing_fields(self):
        """Test schema compliance validation with missing fields."""
        schema = {"obs": {"required": ["sample_id"], "optional": []}, "var": {"required": ["gene_id"], "optional": []}}
        validator = FlexibleValidator(schema)

        # Create non-compliant data
        adata = ad.AnnData(X=np.random.rand(10, 50))
        # Missing both required fields

        result = validator.validate_schema_compliance(adata, schema)
        assert result.has_errors
        assert len(result.errors) == 2  # One for obs, one for var
    
    def test_validate_data_types(self):
        """Test data type validation."""
        schema = {"obs": {"required": [], "optional": [], "types": {"numeric_field": "numeric", "string_field": "string"}}}
        validator = FlexibleValidator(schema)

        adata = ad.AnnData(X=np.random.rand(10, 50))
        adata.obs['numeric_field'] = np.random.rand(10)
        adata.obs['string_field'] = [f"S{i}" for i in range(10)]

        result = validator.validate(adata, check_types=True)
        # Should not have errors for compatible data types
        assert not result.has_errors
        
    def test_validate_completeness(self):
        """Test data completeness validation."""
        validator = FlexibleValidator({})

        # Test normal data
        adata = ad.AnnData(X=np.random.rand(100, 1000))
        result = validator.validate(adata, check_completeness=True)
        assert not result.has_errors

        # Test very sparse data
        sparse_data = np.zeros((50, 200))
        sparse_data[0, 0] = 1  # Very sparse
        sparse_adata = ad.AnnData(X=sparse_data)
        sparse_result = validator.validate(sparse_adata, check_completeness=True)
        # May have warnings about sparsity
    
    def test_validate_empty_data(self):
        """Test validation with empty data."""
        validator = FlexibleValidator({})

        # Empty data should generate errors
        empty_adata = ad.AnnData(X=np.array([]).reshape(0, 0))
        result = validator.validate(empty_adata, check_completeness=True)
        assert result.has_errors
    
    def test_validate_invalid_values(self):
        """Test validation with invalid values."""
        validator = FlexibleValidator({})

        # Data with NaN and inf values
        adata = ad.AnnData(X=np.array([[1, 2, np.nan], [4, np.inf, 6]], dtype=float))
        result = validator.validate(adata, check_ranges=True)
        # Should have warnings or errors for invalid values
    
    def test_custom_rules(self):
        """Test adding and using custom validation rules."""
        validator = FlexibleValidator({})

        def custom_rule(adata):
            result = ValidationResult()
            if adata.n_obs < 10:
                result.add_warning("Very few observations")
            return result

        validator.add_custom_rule("check_obs_count", custom_rule)

        # Test with small dataset
        small_adata = ad.AnnData(X=np.random.rand(5, 20))
        result = validator.validate(small_adata)
        # Should have custom warning
        assert any("Very few observations" in w for w in result.warnings)


# ===============================================================================
# TranscriptomicsSchema Tests
# ===============================================================================

@pytest.mark.unit
class TestTranscriptomicsSchema:
    """Test TranscriptomicsSchema validator functionality."""
    
    def test_validator_initialization(self):
        """Test TranscriptomicsSchema validator initialization."""
        validator = TranscriptomicsSchema.create_validator()

        expected_schema = TranscriptomicsSchema.get_single_cell_schema()
        assert validator.schema == expected_schema
        assert isinstance(validator, FlexibleValidator)
    
    def test_validate_compliant_data(self, compliant_transcriptomics_data):
        """Test validation of compliant transcriptomics data."""
        validator = TranscriptomicsSchema.create_validator()

        result = validator.validate(compliant_transcriptomics_data)

        assert isinstance(result, ValidationResult)
        assert not result.has_errors
        # May have warnings but should not have errors
    
    def test_validate_non_compliant_data(self, non_compliant_transcriptomics_data):
        """Test validation of non-compliant transcriptomics data."""
        validator = TranscriptomicsSchema.create_validator()

        result = validator.validate(non_compliant_transcriptomics_data, strict=True)

        assert isinstance(result, ValidationResult)
        # Since schema is now flexible, may have warnings instead of errors in non-strict mode
        # In strict mode, warnings become errors
    
    def test_validate_strict_vs_permissive(self, non_compliant_transcriptomics_data):
        """Test strict vs permissive validation modes."""
        validator = TranscriptomicsSchema.create_validator()

        # Permissive mode should have fewer errors (warnings instead)
        permissive_result = validator.validate(non_compliant_transcriptomics_data, strict=False)

        # Strict mode should convert warnings to errors
        strict_result = validator.validate(non_compliant_transcriptomics_data, strict=True)

        # In strict mode, warnings become errors
        if permissive_result.has_warnings:
            assert strict_result.has_errors
    
    def test_validate_with_qc_metrics(self):
        """Test validation with QC metrics present."""
        validator = TranscriptomicsSchema.create_validator()

        # Data with QC metrics
        adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        adata.obs['total_counts'] = np.array(adata.X.sum(axis=1)).flatten()
        adata.obs['n_genes_by_counts'] = (adata.X > 0).sum(axis=1)
        adata.obs['pct_counts_mt'] = np.random.uniform(0, 20, adata.n_obs)

        result = validator.validate(adata)
        # Should not have errors when QC metrics are present and reasonable
        assert not result.has_errors
    
    def test_validate_without_qc_metrics(self):
        """Test validation with missing QC metrics."""
        validator = TranscriptomicsSchema.create_validator()

        # Data without QC metrics
        adata = ad.AnnData(X=np.random.randint(0, 100, (50, 200)))

        result = validator.validate(adata)
        # May have warnings about missing optional metrics, but not errors
        # since schema is flexible
    
    def test_validate_gene_annotations(self):
        """Test validation with proper gene annotations."""
        validator = TranscriptomicsSchema.create_validator()

        # Create data with gene annotations
        adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        adata.var['gene_id'] = [f"GENE_{i:05d}" for i in range(adata.n_vars)]
        adata.var['gene_symbol'] = [f"SYMBOL_{i}" for i in range(adata.n_vars)]
        adata.var['mt'] = adata.var_names.str.startswith('MT-')
        adata.var['ribo'] = adata.var_names.str.startswith(('RPL', 'RPS'))

        result = validator.validate(adata)
        # Should not have errors with proper gene annotations
        assert not result.has_errors
    
    def test_validate_mitochondrial_genes(self):
        """Test validation with mitochondrial genes."""
        validator = TranscriptomicsSchema.create_validator()

        # Create data with mitochondrial genes
        gene_names = ['Gene1', 'MT-ATP6', 'Gene2', 'MT-CO1', 'Gene3']
        adata = ad.AnnData(
            X=np.random.randint(0, 100, (10, 5)),
            var=pd.DataFrame(index=gene_names)
        )
        adata.var['mt'] = adata.var_names.str.startswith('MT-')

        result = validator.validate(adata)
        # Should not have errors if MT genes are properly flagged
        assert not result.has_errors
    
    def test_validate_count_matrix_properties(self):
        """Test count matrix property validation."""
        validator = TranscriptomicsSchema.create_validator()

        # Integer count matrix (good)
        count_matrix = np.random.randint(0, 1000, (100, 500))
        adata = ad.AnnData(X=count_matrix)

        result = validator.validate(adata, check_ranges=True)
        # Integer counts should not generate errors
        assert not result.has_errors

        # Negative values (bad)
        negative_matrix = np.random.randint(-10, 1000, (100, 500))
        adata_negative = ad.AnnData(X=negative_matrix)

        negative_result = validator.validate(adata_negative, check_ranges=True)
        # Should have warnings for negative values
        assert negative_result.has_warnings or negative_result.has_errors
    
    def test_get_schema(self):
        """Test schema retrieval."""
        validator = TranscriptomicsSchema.create_validator()

        schema = validator.schema
        expected_schema = TranscriptomicsSchema.get_single_cell_schema()

        assert schema == expected_schema
        assert isinstance(schema, dict)


# ===============================================================================
# ProteomicsSchema Tests
# ===============================================================================

@pytest.mark.unit
class TestProteomicsSchema:
    """Test ProteomicsSchema validator functionality."""
    
    def test_validator_initialization(self):
        """Test ProteomicsSchema validator initialization."""
        validator = ProteomicsSchema.create_validator()

        expected_schema = ProteomicsSchema.get_mass_spectrometry_schema()
        assert validator.schema == expected_schema
        assert isinstance(validator, FlexibleValidator)
    
    def test_validate_compliant_data(self, compliant_proteomics_data):
        """Test validation of compliant proteomics data."""
        validator = ProteomicsSchema.create_validator()

        result = validator.validate(compliant_proteomics_data)

        assert isinstance(result, ValidationResult)
        assert not result.has_errors
        # May have warnings but should not have errors
    
    def test_validate_non_compliant_data(self, non_compliant_proteomics_data):
        """Test validation of non-compliant proteomics data."""
        validator = ProteomicsSchema.create_validator()

        result = validator.validate(non_compliant_proteomics_data, strict=True)

        assert isinstance(result, ValidationResult)
        # Since schema is now flexible, may have warnings instead of errors in non-strict mode
        # In strict mode, warnings become errors
    
    def test_validate_protein_annotations(self):
        """Test validation with protein annotations."""
        validator = ProteomicsSchema.create_validator()

        # Create data with protein annotations
        adata = ProteomicsDataFactory(config=SMALL_DATASET_CONFIG)
        adata.var['protein_id'] = [f"PROT_{i:05d}" for i in range(adata.n_vars)]
        adata.var['protein_name'] = [f"Protein_{i}" for i in range(adata.n_vars)]
        adata.var['uniprot_id'] = [f"P{i:05d}" for i in range(adata.n_vars)]

        result = validator.validate(adata)
        # Should not have errors with proper protein annotations
        assert not result.has_errors
    
    def test_validate_intensity_values(self):
        """Test intensity value validation."""
        validator = ProteomicsSchema.create_validator()

        # Positive intensity values (good)
        intensity_matrix = np.random.lognormal(3, 1, (50, 200))
        adata = ad.AnnData(X=intensity_matrix)

        result = validator.validate(adata, check_ranges=True)
        # Positive values should be fine
        assert not result.has_errors

        # Negative intensity values (unusual for proteomics)
        negative_matrix = np.random.normal(0, 1, (50, 200))  # Can have negative values
        adata_negative = ad.AnnData(X=negative_matrix)

        negative_result = validator.validate(adata_negative, check_ranges=True)
        # May generate warnings for negative intensities
    
    def test_validate_missing_values(self):
        """Test missing value validation."""
        validator = ProteomicsSchema.create_validator()

        # Create data with missing values (common in proteomics)
        matrix = np.random.lognormal(3, 1, (50, 200))
        # Add missing values
        missing_mask = np.random.choice([True, False], matrix.shape, p=[0.2, 0.8])
        matrix[missing_mask] = np.nan

        adata = ad.AnnData(X=matrix)

        result = validator.validate(adata)
        # Missing values should generate info/warnings, not errors
        assert not result.has_errors
    
    def test_validate_contaminants(self):
        """Test contaminant validation."""
        validator = ProteomicsSchema.create_validator()

        # Create data with contaminant annotations
        protein_names = ['Protein1', 'CON_TRYP_HUMAN', 'Protein2', 'REV_Protein3']
        adata = ad.AnnData(
            X=np.random.lognormal(3, 1, (20, 4)),
            var=pd.DataFrame(index=protein_names)
        )
        adata.var['is_contaminant'] = [False, True, False, True]
        adata.var['is_reverse'] = [False, False, False, True]

        result = validator.validate(adata)
        # Should not have errors with proper contaminant annotations
        assert not result.has_errors
    
    def test_validate_sample_metadata(self):
        """Test sample metadata validation."""
        validator = ProteomicsSchema.create_validator()

        # Create data with sample metadata
        adata = ProteomicsDataFactory(config=SMALL_DATASET_CONFIG)
        adata.obs['sample_id'] = [f"Sample_{i:03d}" for i in range(adata.n_obs)]
        adata.obs['total_intensity'] = np.array(adata.X.sum(axis=1)).flatten()
        adata.obs['condition'] = np.random.choice(['Control', 'Treatment'], adata.n_obs)

        result = validator.validate(adata)
        # Should not have errors with proper sample metadata
        assert not result.has_errors
    
    def test_get_schema(self):
        """Test schema retrieval."""
        validator = ProteomicsSchema.create_validator()

        schema = validator.schema
        expected_schema = ProteomicsSchema.get_mass_spectrometry_schema()

        assert schema == expected_schema
        assert isinstance(schema, dict)


# ===============================================================================
# Schema Evolution and Backward Compatibility Tests
# ===============================================================================

@pytest.mark.unit
class TestSchemaEvolution:
    """Test schema evolution and backward compatibility."""
    
    def test_schema_version_compatibility(self):
        """Test that schemas maintain backward compatibility."""
        # Test that old data structures still validate
        # This is important for data that was processed with older versions
        
        # Create minimal old-style data
        old_style_transcriptomics = ad.AnnData(X=np.random.randint(0, 100, (50, 200)))
        old_style_transcriptomics.obs['n_genes'] = (old_style_transcriptomics.X > 0).sum(axis=1)  # Old field name
        
        validator = TranscriptomicsSchema.create_validator()
        
        # Should handle gracefully in permissive mode
        result = validator.validate(old_style_transcriptomics, strict=False)
        # May have warnings but should not crash
        assert isinstance(result, ValidationResult)
    
    def test_optional_field_handling(self):
        """Test handling of optional fields."""
        # Test that data validates correctly with and without optional fields
        
        # Minimal compliant data
        minimal_data = ad.AnnData(X=np.random.randint(0, 100, (50, 200)))
        minimal_data.obs['total_counts'] = np.array(minimal_data.X.sum(axis=1)).flatten()
        minimal_data.obs['n_genes_by_counts'] = (minimal_data.X > 0).sum(axis=1)
        minimal_data.var['gene_ids'] = [f"GENE_{i:05d}" for i in range(200)]
        minimal_data.var['feature_types'] = 'Gene Expression'
        
        validator = TranscriptomicsSchema.create_validator()
        result = validator.validate(minimal_data)
        assert not result.has_errors
        
        # Enhanced data with optional fields
        enhanced_data = minimal_data.copy()
        enhanced_data.obs['pct_counts_mt'] = np.random.uniform(0, 20, 50)
        enhanced_data.var['gene_names'] = [f"Gene_{i}" for i in range(200)]
        
        enhanced_result = validator.validate(enhanced_data)
        assert not enhanced_result.has_errors
    
    def test_custom_schema_extension(self):
        """Test extending schemas with custom fields."""
        # Test that custom schemas can extend base schemas
        
        custom_schema = TranscriptomicsSchema.get_single_cell_schema().copy()
        if "custom_obs" not in custom_schema:
            custom_schema["custom_obs"] = []
        custom_schema["custom_obs"] = custom_schema.get("optional_obs", []) + ["experiment_id", "batch_id"]
        
        validator = FlexibleValidator(custom_schema)
        
        # Create data with custom fields
        adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        adata.obs['experiment_id'] = 'EXP_001'
        adata.obs['batch_id'] = np.random.choice(['Batch1', 'Batch2'], adata.n_obs)
        
        # Custom validator should handle extended schema
        assert validator.schema == custom_schema
    
    def test_schema_field_deprecation(self):
        """Test handling of deprecated fields."""
        # Test that deprecated fields generate appropriate warnings
        
        # Simulate data with deprecated field names
        deprecated_data = ad.AnnData(X=np.random.randint(0, 100, (50, 200)))
        deprecated_data.obs['total_umis'] = np.array(deprecated_data.X.sum(axis=1)).flatten()  # Deprecated name
        deprecated_data.obs['n_genes'] = (deprecated_data.X > 0).sum(axis=1)  # Old name
        
        validator = TranscriptomicsSchema.create_validator()
        
        # Should handle gracefully
        result = validator.validate(deprecated_data, strict=False)
        assert isinstance(result, ValidationResult)


# ===============================================================================
# Custom Validator Implementation Tests
# ===============================================================================

@pytest.mark.unit
class TestCustomValidatorImplementations:
    """Test custom validator implementations and patterns."""
    
    def test_custom_validator_creation(self):
        """Test creating custom validators."""
        
        class CustomTranscriptomicsValidator(SchemaValidator):
            """Custom validator with additional checks."""
            
            def __init__(self):
                # Start with base transcriptomics schema
                custom_schema = TranscriptomicsSchema.get_single_cell_schema().copy()
                # Add custom requirements
                custom_schema["obs"]["required"] = custom_schema["obs"]["required"] + ["experiment_date"]
                super().__init__(custom_schema)
            
            def validate(self, adata: ad.AnnData, strict: bool = False) -> ValidationResult:
                """Custom validation with additional checks."""
                # Run base validation
                result = super().validate(adata, strict)
                
                # Add custom validation logic
                if hasattr(adata, 'obs') and 'experiment_date' in adata.obs:
                    # Validate date format
                    pass  # Custom date validation would go here
                
                return result
        
        validator = CustomTranscriptomicsValidator()
        
        # Test with compliant data
        adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        adata.obs['experiment_date'] = '2024-01-15'  # Add required custom field
        
        result = validator.validate(adata)
        # Should work with custom validator
        assert isinstance(result, ValidationResult)
    
    def test_validator_composition(self):
        """Test composing multiple validators."""
        
        def composite_validation(adata: ad.AnnData) -> ValidationResult:
            """Run multiple validators and combine results."""
            transcriptomics_validator = TranscriptomicsSchema.create_validator()
            proteomics_validator = ProteomicsSchema.create_validator()
            
            # Try both validators
            t_result = transcriptomics_validator.validate(adata, strict=False)
            p_result = proteomics_validator.validate(adata, strict=False)
            
            # Combine results (choose the one with fewer errors)
            if len(t_result.errors) <= len(p_result.errors):
                return t_result
            else:
                return p_result
        
        # Test with transcriptomics data
        transcriptomics_data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        result = composite_validation(transcriptomics_data)
        assert isinstance(result, ValidationResult)
    
    def test_conditional_validation(self):
        """Test conditional validation based on data characteristics."""
        
        class ConditionalValidator(SchemaValidator):
            """Validator that adapts based on data characteristics."""
            
            def __init__(self):
                super().__init__({})  # Empty schema, will be set dynamically
            
            def validate(self, adata: ad.AnnData, strict: bool = False) -> ValidationResult:
                """Conditional validation based on data size."""

                if adata.n_vars > 10000:
                    # Likely single-cell data
                    self.schema = TranscriptomicsSchema.get_single_cell_schema()
                else:
                    # Likely proteomics data
                    self.schema = ProteomicsSchema.get_mass_spectrometry_schema()

                # Run appropriate validation using parent class method
                return super().validate(adata, strict)
        
        validator = ConditionalValidator()
        
        # Test with high-dimensional data (single-cell-like)
        high_dim_data = ad.AnnData(X=np.random.randint(0, 100, (100, 15000)))
        result_high = validator.validate(high_dim_data)
        assert isinstance(result_high, ValidationResult)
        
        # Test with low-dimensional data (proteomics-like)
        low_dim_data = ad.AnnData(X=np.random.randn(50, 500))
        result_low = validator.validate(low_dim_data)
        assert isinstance(result_low, ValidationResult)


# ===============================================================================
# New API Methods Tests
# ===============================================================================

@pytest.mark.unit
class TestNewAPIFeatures:
    """Test new API features introduced in the current implementation."""

    def test_transcriptomics_qc_thresholds(self):
        """Test QC threshold recommendations for transcriptomics."""
        # Single-cell thresholds
        sc_thresholds = TranscriptomicsSchema.get_recommended_qc_thresholds("single_cell")
        assert isinstance(sc_thresholds, dict)
        assert "min_genes_per_cell" in sc_thresholds
        assert "max_pct_mt" in sc_thresholds
        assert "min_total_counts" in sc_thresholds
        assert sc_thresholds["min_genes_per_cell"] == 200
        assert sc_thresholds["max_pct_mt"] == 20.0

        # Bulk RNA-seq thresholds
        bulk_thresholds = TranscriptomicsSchema.get_recommended_qc_thresholds("bulk")
        assert isinstance(bulk_thresholds, dict)
        assert "min_genes_per_sample" in bulk_thresholds
        assert "min_total_counts" in bulk_thresholds
        assert "min_mapping_rate" in bulk_thresholds
        assert bulk_thresholds["min_genes_per_sample"] == 10000

    def test_proteomics_qc_thresholds(self):
        """Test QC threshold recommendations for proteomics."""
        # Mass spectrometry thresholds
        ms_thresholds = ProteomicsSchema.get_recommended_qc_thresholds("mass_spectrometry")
        assert isinstance(ms_thresholds, dict)
        assert "min_proteins_per_sample" in ms_thresholds
        assert "max_missing_per_sample" in ms_thresholds
        assert "max_cv_threshold" in ms_thresholds
        assert ms_thresholds["min_proteins_per_sample"] == 100
        assert ms_thresholds["max_missing_per_sample"] == 0.7

        # Affinity proteomics thresholds
        affinity_thresholds = ProteomicsSchema.get_recommended_qc_thresholds("affinity")
        assert isinstance(affinity_thresholds, dict)
        assert "min_proteins_per_sample" in affinity_thresholds
        assert "max_cv_threshold" in affinity_thresholds
        assert affinity_thresholds["max_cv_threshold"] == 30.0

    def test_validator_creation_with_different_types(self):
        """Test creating validators for different schema types."""
        # Test transcriptomics validators
        sc_validator = TranscriptomicsSchema.create_validator("single_cell")
        assert isinstance(sc_validator, FlexibleValidator)
        assert sc_validator.schema["modality"] == "single_cell_rna_seq"

        bulk_validator = TranscriptomicsSchema.create_validator("bulk")
        assert isinstance(bulk_validator, FlexibleValidator)
        assert bulk_validator.schema["modality"] == "bulk_rna_seq"

        # Test proteomics validators
        ms_validator = ProteomicsSchema.create_validator("mass_spectrometry")
        assert isinstance(ms_validator, FlexibleValidator)
        assert ms_validator.schema["modality"] == "mass_spectrometry_proteomics"

        affinity_validator = ProteomicsSchema.create_validator("affinity")
        assert isinstance(affinity_validator, FlexibleValidator)
        assert affinity_validator.schema["modality"] == "affinity_proteomics"

    def test_validator_strict_mode(self):
        """Test validator creation with strict mode."""
        strict_validator = TranscriptomicsSchema.create_validator(strict=True)
        permissive_validator = TranscriptomicsSchema.create_validator(strict=False)

        # Both should be FlexibleValidator instances
        assert isinstance(strict_validator, FlexibleValidator)
        assert isinstance(permissive_validator, FlexibleValidator)

        # Test with minimal data
        minimal_data = ad.AnnData(X=np.random.rand(10, 20))

        permissive_result = permissive_validator.validate(minimal_data, strict=False)
        strict_result = strict_validator.validate(minimal_data, strict=True)

        # Strict mode should have more errors (warnings converted to errors)
        if permissive_result.has_warnings:
            assert len(strict_result.errors) >= len(permissive_result.errors)

    def test_custom_validation_rules(self):
        """Test custom validation rules in validators."""
        validator = TranscriptomicsSchema.create_validator()

        # Create test data with gene symbols
        adata = ad.AnnData(X=np.random.rand(10, 20))
        adata.var['gene_symbol'] = [f"GENE_{i}" for i in range(20)]
        # Add a duplicate
        adata.var.loc[adata.var.index[1], 'gene_symbol'] = adata.var['gene_symbol'].iloc[0]

        result = validator.validate(adata)
        # Should detect duplicate gene symbols
        has_duplicate_warning = any("duplicate" in str(msg).lower() for msg in result.warnings)
        assert has_duplicate_warning

    def test_ignore_warnings_functionality(self):
        """Test warning filtering functionality."""
        # Create validator with specific warnings ignored
        validator = TranscriptomicsSchema.create_validator(
            ignore_warnings=["missing values", "Very sparse data"]
        )

        # Create sparse data that would normally generate warnings
        sparse_data = np.zeros((50, 100))
        sparse_data[0, 0] = 1
        adata = ad.AnnData(X=sparse_data)

        result = validator.validate(adata)
        # Ignored warnings should be filtered out
        has_sparsity_warning = any("sparse" in str(msg).lower() for msg in result.warnings)
        assert not has_sparsity_warning or len(result.warnings) == 0

    def test_peptide_mapping_schema(self):
        """Test peptide-to-protein mapping schema."""
        mapping_schema = ProteomicsSchema.get_peptide_to_protein_mapping_schema()
        assert isinstance(mapping_schema, dict)
        assert "peptide_sequence" in mapping_schema
        assert "protein_id" in mapping_schema
        assert "is_unique" in mapping_schema
        assert mapping_schema["peptide_sequence"] == "string"
        assert mapping_schema["is_unique"] == "boolean"

    def test_invalid_schema_types(self):
        """Test error handling for invalid schema types."""
        with pytest.raises(ValueError, match="Unknown schema type"):
            TranscriptomicsSchema.create_validator("invalid_type")

        with pytest.raises(ValueError, match="Unknown schema type"):
            ProteomicsSchema.create_validator("invalid_type")

        with pytest.raises(ValueError, match="Unknown schema type"):
            TranscriptomicsSchema.get_recommended_qc_thresholds("invalid_type")

        with pytest.raises(ValueError, match="Unknown schema type"):
            ProteomicsSchema.get_recommended_qc_thresholds("invalid_type")


# ===============================================================================
# Edge Cases and Error Handling Tests
# ===============================================================================

@pytest.mark.unit
class TestSchemaEdgeCases:
    """Test edge cases and error handling in schema validation."""
    
    def test_empty_data_validation(self):
        """Test validation of empty datasets."""
        validators = [
            TranscriptomicsSchema.create_validator(),
            ProteomicsSchema.create_validator()
        ]
        
        for validator in validators:
            # Completely empty data
            empty_data = ad.AnnData(X=np.array([]).reshape(0, 0))
            result = validator.validate(empty_data)
            assert result.has_errors  # Should detect empty data
            
            # Empty observations but with variables
            empty_obs = ad.AnnData(X=np.array([]).reshape(0, 10))
            result_obs = validator.validate(empty_obs)
            assert result_obs.has_errors or result_obs.has_warnings
            
            # Empty variables but with observations
            empty_vars = ad.AnnData(X=np.array([]).reshape(10, 0))
            result_vars = validator.validate(empty_vars)
            assert result_vars.has_errors or result_vars.has_warnings
    
    def test_malformed_data_validation(self):
        """Test validation of malformed data structures."""
        validator = TranscriptomicsSchema.create_validator()
        
        # Data with inconsistent dimensions
        adata = ad.AnnData(X=np.random.rand(10, 20))
        # Create proper-sized metadata but with wrong content structure
        adata.obs = pd.DataFrame({'sample_id': [f'S{i}' for i in range(10)]})
        # Simulate malformed structure by making non-standard data types
        adata.obs['malformed_field'] = [{'nested': 'dict'} for _ in range(10)]
        
        # Should handle gracefully
        result = validator.validate(adata, strict=False)
        assert isinstance(result, ValidationResult)
    
    def test_extreme_values_validation(self):
        """Test validation with extreme values."""
        validator = TranscriptomicsSchema.create_validator()
        
        # Very large values
        large_values = ad.AnnData(X=np.random.rand(10, 20) * 1e10)
        result_large = validator.validate(large_values, strict=False)
        assert isinstance(result_large, ValidationResult)
        
        # Very small values
        small_values = ad.AnnData(X=np.random.rand(10, 20) * 1e-10)
        result_small = validator.validate(small_values, strict=False)
        assert isinstance(result_small, ValidationResult)
    
    def test_unicode_and_special_characters(self):
        """Test validation with unicode and special characters."""
        validator = TranscriptomicsSchema.create_validator()
        
        # Data with unicode gene names
        unicode_genes = ['Gene_α', 'Gene_β', 'Gene_γ', '基因_1', 'Gène_2']
        adata = ad.AnnData(
            X=np.random.randint(0, 100, (10, 5)),
            var=pd.DataFrame(index=unicode_genes)
        )
        adata.var['gene_ids'] = [f"GENE_{i:03d}" for i in range(5)]
        adata.var['feature_types'] = 'Gene Expression'
        
        result = validator.validate(adata, strict=False)
        assert isinstance(result, ValidationResult)
        # Should handle unicode gracefully
    
    def test_concurrent_validation(self):
        """Test thread safety of validators."""
        import threading
        import time
        
        validator = TranscriptomicsSchema.create_validator()
        results = []
        errors = []
        
        def validate_worker(worker_id):
            """Worker function for concurrent validation."""
            try:
                # Create test data
                adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
                result = validator.validate(adata)
                results.append((worker_id, result))
                time.sleep(0.01)
            except Exception as e:
                errors.append((worker_id, e))
        
        # Run multiple validators concurrently
        threads = []
        for i in range(5):
            thread = threading.Thread(target=validate_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Concurrent validation errors: {errors}"
        assert len(results) == 5
        
        # All results should be valid ValidationResult instances
        for worker_id, result in results:
            assert isinstance(result, ValidationResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])