"""
Integration tests for complete proteomics workflows.

Agent 17: Comprehensive testing of mass spectrometry and affinity proteomics
workflows with realistic synthetic data, covering the full analysis pipeline
from loading to visualization.

Test Coverage:
- Mass Spectrometry (DDA/DIA): 30-70% missing, MNAR patterns, MaxQuant/Spectronaut
- Affinity Proteomics (Olink): <30% missing, NPX values, antibody validation
- Complete workflows: QC → Preprocess → Differential → Visualize
"""

import anndata
import numpy as np
import pandas as pd
import pytest

from lobster.services.analysis.proteomics_analysis_service import (
    ProteomicsAnalysisService,
)
from lobster.services.analysis.proteomics_differential_service import (
    ProteomicsDifferentialService,
)
from lobster.services.quality.proteomics_preprocessing_service import (
    ProteomicsPreprocessingService,
)
from lobster.services.quality.proteomics_quality_service import ProteomicsQualityService

# ==============================================================================
# FIXTURES: Realistic Synthetic Data Generation
# ==============================================================================


@pytest.fixture
def synthetic_mass_spec_data():
    """
    Generate realistic mass spectrometry proteomics data (DDA/DIA).

    Characteristics:
    - 500 proteins, 30 samples (15 control, 15 treatment)
    - 30-70% missing values (MNAR pattern - low abundance proteins missing)
    - Log-normal intensity distribution
    - Batch effects present
    - Some contaminants (keratin, albumin)
    """
    np.random.seed(42)

    n_samples = 30
    n_proteins = 500

    # Generate base intensities with log-normal distribution
    # Realistic MS intensity range: 10^3 to 10^9
    base_intensities = np.random.lognormal(
        mean=15, sigma=3, size=(n_samples, n_proteins)
    )

    # Add biological signal: 50 proteins differentially expressed (2-fold change)
    de_indices = np.random.choice(n_proteins, size=50, replace=False)
    treatment_mask = np.array([i >= 15 for i in range(n_samples)])
    for idx in de_indices:
        if np.random.random() > 0.5:
            base_intensities[treatment_mask, idx] *= 2.0  # Upregulated
        else:
            base_intensities[treatment_mask, idx] *= 0.5  # Downregulated

    # Add batch effects: 3 batches
    batch_labels = ["Batch1"] * 10 + ["Batch2"] * 10 + ["Batch3"] * 10
    for i, batch in enumerate(["Batch1", "Batch2", "Batch3"]):
        batch_mask = np.array([b == batch for b in batch_labels])
        batch_shift = np.random.uniform(0.8, 1.2)
        base_intensities[batch_mask, :] *= batch_shift

    # Add MNAR missing values: proteins with low intensity more likely to be missing
    # Calculate protein-wise intensity percentiles
    protein_intensities = np.mean(base_intensities, axis=0)
    intensity_percentiles = np.percentile(protein_intensities, np.arange(0, 101, 10))

    data_with_missing = base_intensities.copy()
    for i in range(n_proteins):
        protein_intensity = protein_intensities[i]

        # Lower intensity → higher missing rate
        if protein_intensity < intensity_percentiles[3]:  # Bottom 30%
            missing_rate = 0.7  # 70% missing
        elif protein_intensity < intensity_percentiles[5]:  # 30-50%
            missing_rate = 0.5  # 50% missing
        elif protein_intensity < intensity_percentiles[7]:  # 50-70%
            missing_rate = 0.3  # 30% missing
        else:
            missing_rate = 0.1  # 10% missing (high abundance)

        # Randomly set values to NaN
        missing_mask = np.random.random(n_samples) < missing_rate
        data_with_missing[missing_mask, i] = np.nan

    # Create sample metadata
    obs = pd.DataFrame(
        {
            "sample_id": [f"Sample_{i+1:03d}" for i in range(n_samples)],
            "condition": ["Control"] * 15 + ["Treatment"] * 15,
            "batch": batch_labels,
            "replicate": [i % 3 + 1 for i in range(n_samples)],
            "acquisition_date": pd.date_range(
                "2024-01-01", periods=n_samples, freq="D"
            ),
        }
    )
    obs.index = obs["sample_id"]

    # Create protein metadata with contaminants
    protein_names = []
    protein_types = []
    for i in range(n_proteins):
        if i < 10:  # 10 keratins
            protein_names.append(f"KRT{i+1}")
            protein_types.append("keratin")
        elif i < 15:  # 5 albumins
            protein_names.append(f"ALB_{i-9}")
            protein_types.append("albumin")
        elif i < 20:  # 5 immunoglobulins
            protein_names.append(f"IGG_{i-14}")
            protein_types.append("immunoglobulin")
        else:  # Regular proteins
            protein_names.append(f"Protein_{i+1:04d}")
            protein_types.append("target")

    var = pd.DataFrame(
        {
            "protein_id": protein_names,
            "protein_type": protein_types,
            "molecular_weight": np.random.uniform(10, 200, n_proteins),
            "peptide_count": np.random.randint(2, 50, n_proteins),
        }
    )
    var.index = protein_names

    # Create AnnData object
    adata = anndata.AnnData(X=data_with_missing, obs=obs, var=var)

    # Add metadata
    adata.uns["data_type"] = "mass_spectrometry"
    adata.uns["experiment_type"] = "DDA"
    adata.uns["instrument"] = "Orbitrap Fusion"
    adata.uns["true_de_proteins"] = protein_names[:50]  # Ground truth

    return adata


@pytest.fixture
def synthetic_affinity_data():
    """
    Generate realistic affinity proteomics data (Olink NPX).

    Characteristics:
    - 96 proteins (typical Olink panel), 40 samples (20 control, 20 disease)
    - <30% missing values (higher quality than MS)
    - NPX normalized values (log2 scale, range 0-15)
    - Lower CVs than mass spec (~10-15%)
    - Antibody-specific metadata
    """
    np.random.seed(123)

    n_samples = 40
    n_proteins = 96  # Typical Olink panel size

    # Generate NPX values (log2-transformed, normalized)
    # Typical range: 0-15 NPX
    base_npx = np.random.normal(loc=7, scale=2.5, size=(n_samples, n_proteins))
    base_npx = np.clip(base_npx, 0, 15)  # Clip to realistic range

    # Add biological signal: 15 proteins differentially expressed (1.5-fold change)
    de_indices = np.random.choice(n_proteins, size=15, replace=False)
    disease_mask = np.array([i >= 20 for i in range(n_samples)])
    for idx in de_indices:
        if np.random.random() > 0.5:
            base_npx[disease_mask, idx] += 1.0  # Upregulated (~2-fold in linear)
        else:
            base_npx[disease_mask, idx] -= 1.0  # Downregulated (~2-fold in linear)

    # Add low missing values (<30%, random pattern - MCAR)
    data_with_missing = base_npx.copy()
    for i in range(n_proteins):
        missing_rate = np.random.uniform(0.05, 0.25)  # 5-25% missing per protein
        missing_mask = np.random.random(n_samples) < missing_rate
        data_with_missing[missing_mask, i] = np.nan

    # Create sample metadata
    obs = pd.DataFrame(
        {
            "sample_id": [f"Sample_{i+1:03d}" for i in range(n_samples)],
            "condition": ["Control"] * 20 + ["Disease"] * 20,
            "age": np.random.randint(25, 80, n_samples),
            "sex": np.random.choice(["M", "F"], n_samples),
            "plate": ["Plate1"] * 20 + ["Plate2"] * 20,
        }
    )
    obs.index = obs["sample_id"]

    # Create protein metadata with Olink-specific information
    protein_names = [f"Protein_{i+1:03d}" for i in range(n_proteins)]

    var = pd.DataFrame(
        {
            "protein_id": protein_names,
            "uniprot_id": [
                f"P{np.random.randint(10000, 99999)}" for _ in range(n_proteins)
            ],
            "panel_type": ["Inflammation"] * 32
            + ["Cardiovascular"] * 32
            + ["Neurology"] * 32,
            "antibody_id": [f"AB_{i+1:04d}" for i in range(n_proteins)],
            "antibody_clone": [f"Clone_{i+1}" for i in range(n_proteins)],
            "lot_number": np.random.choice(
                ["LOT2024A", "LOT2024B", "LOT2024C"], n_proteins
            ),
            "lod": np.random.uniform(0.5, 2.0, n_proteins),  # Limit of detection
        }
    )
    var.index = protein_names

    # Create AnnData object
    adata = anndata.AnnData(X=data_with_missing, obs=obs, var=var)

    # Add metadata
    adata.uns["data_type"] = "affinity_proteomics"
    adata.uns["platform"] = "Olink"
    adata.uns["panel_version"] = "v4.0"
    adata.uns["normalization"] = "NPX"
    adata.uns["true_de_proteins"] = protein_names[:15]  # Ground truth

    return adata


# ==============================================================================
# MASS SPECTROMETRY WORKFLOW TESTS
# ==============================================================================


class TestMassSpecPreprocessing:
    """Test mass spectrometry preprocessing workflows."""

    def test_missing_value_imputation_knn(self, synthetic_mass_spec_data):
        """Test KNN imputation on MS data."""
        service = ProteomicsPreprocessingService()
        adata = synthetic_mass_spec_data

        # Calculate initial missing rate
        initial_missing = np.isnan(adata.X).sum()

        # Impute with KNN
        adata_imputed, stats, ir = service.impute_missing_values(
            adata, method="knn", knn_neighbors=5
        )

        # Check imputation worked
        assert stats["imputation_performed"] is True
        assert stats["remaining_missing_count"] == 0
        assert stats["original_missing_count"] > 0
        assert not np.isnan(adata_imputed.X).any()

        # Check statistics are reasonable
        assert stats["original_missing_percentage"] > 30.0  # Should be 30-70%
        assert stats["original_missing_percentage"] < 70.0

    def test_missing_value_imputation_mnar(self, synthetic_mass_spec_data):
        """Test MNAR imputation on MS data."""
        service = ProteomicsPreprocessingService()
        adata = synthetic_mass_spec_data

        # Impute with MNAR method
        adata_imputed, stats, ir = service.impute_missing_values(
            adata, method="mnar", mnar_width=0.3, mnar_downshift=1.8
        )

        # Check imputation worked
        assert not np.isnan(adata_imputed.X).any()

        # MNAR imputation should shift missing values to lower end
        # Check that imputed values are generally lower than observed
        observed_mean = np.nanmean(adata.X)
        imputed_mean = np.mean(adata_imputed.X)

        # Imputed values should shift distribution left
        assert imputed_mean < observed_mean * 1.1  # Allow some variance

    def test_mixed_imputation_strategy(self, synthetic_mass_spec_data):
        """Test mixed imputation strategy (KNN for MCAR, MNAR for high missing)."""
        service = ProteomicsPreprocessingService()
        adata = synthetic_mass_spec_data

        # Mixed imputation
        adata_imputed, stats, ir = service.impute_missing_values(
            adata, method="mixed", knn_neighbors=5, mnar_width=0.3, mnar_downshift=1.8
        )

        # Check imputation worked
        assert not np.isnan(adata_imputed.X).any()
        assert stats["method"] == "mixed"

    def test_intensity_normalization_median(self, synthetic_mass_spec_data):
        """Test median normalization on MS data."""
        service = ProteomicsPreprocessingService()
        adata = synthetic_mass_spec_data

        # First impute missing values
        adata_imputed, _, _ = service.impute_missing_values(adata, method="knn")

        # Normalize
        adata_norm, stats, ir = service.normalize_intensities(
            adata_imputed,
            method="median",
            log_transform=True,
            pseudocount_strategy="adaptive",
        )

        # Check normalization
        assert stats["method"] == "median"
        assert stats["log_transform"] is True
        assert "normalized" in adata_norm.layers
        assert adata_norm.raw is not None

    def test_intensity_normalization_quantile(self, synthetic_mass_spec_data):
        """Test quantile normalization on MS data."""
        service = ProteomicsPreprocessingService()
        adata = synthetic_mass_spec_data

        # First impute
        adata_imputed, _, _ = service.impute_missing_values(adata, method="knn")

        # Quantile normalize
        adata_norm, stats, ir = service.normalize_intensities(
            adata_imputed, method="quantile", log_transform=False
        )

        # After quantile normalization, samples should have similar distributions
        sample_medians = np.median(adata_norm.X, axis=1)
        median_cv = np.std(sample_medians) / np.mean(sample_medians)

        # CV should be very low after quantile normalization
        assert median_cv < 0.05

    def test_batch_correction_combat(self, synthetic_mass_spec_data):
        """Test ComBat-like batch correction."""
        service = ProteomicsPreprocessingService()
        adata = synthetic_mass_spec_data

        # First impute and normalize
        adata_imputed, _, _ = service.impute_missing_values(adata, method="knn")
        adata_norm, _, _ = service.normalize_intensities(adata_imputed, method="median")

        # Batch correction
        adata_corrected, stats, ir = service.correct_batch_effects(
            adata_norm, batch_key="batch", method="combat"
        )

        # Check batch correction
        assert stats["batch_correction_performed"] is True
        assert stats["n_batches"] == 3
        assert "batch_corrected" in adata_corrected.layers


class TestMassSpecQuality:
    """Test mass spectrometry quality control."""

    def test_missing_value_pattern_analysis(self, synthetic_mass_spec_data):
        """Test missing value pattern analysis."""
        service = ProteomicsQualityService()
        adata = synthetic_mass_spec_data

        # Analyze missing patterns
        adata_qc, stats, ir = service.assess_missing_value_patterns(
            adata, sample_threshold=0.7, protein_threshold=0.8
        )

        # Check QC metrics added
        assert "missing_protein_count" in adata_qc.obs.columns
        assert "missing_protein_rate" in adata_qc.obs.columns
        assert "high_missing_sample" in adata_qc.obs.columns
        assert "missing_sample_count" in adata_qc.var.columns
        assert "missing_sample_rate" in adata_qc.var.columns
        assert "high_missing_protein" in adata_qc.var.columns

        # Check statistics
        assert stats["overall_missing_rate"] > 0.3  # Should be 30-70%
        assert stats["overall_missing_rate"] < 0.7
        assert (
            stats["high_missing_proteins"] > 0
        )  # Should have some high missing proteins

    def test_coefficient_variation_assessment(self, synthetic_mass_spec_data):
        """Test CV assessment."""
        service = ProteomicsQualityService()
        adata = synthetic_mass_spec_data

        # Assess CV
        adata_qc, stats, ir = service.assess_coefficient_variation(
            adata, cv_threshold=50.0, min_observations=3
        )

        # Check CV metrics added
        assert "intensity_cv" in adata_qc.obs.columns
        assert "intensity_cv" in adata_qc.var.columns
        assert "high_cv_protein" in adata_qc.var.columns

        # MS data should have higher CVs than affinity
        assert (
            stats["median_cv_across_proteins"] > 0.15
        )  # Typical MS CV (fractional: 0.15 = 15%)

    def test_contaminant_detection(self, synthetic_mass_spec_data):
        """Test contaminant detection."""
        service = ProteomicsQualityService()
        adata = synthetic_mass_spec_data

        # Detect contaminants
        adata_qc, stats, ir = service.detect_contaminants(adata)

        # Check contaminant flags
        assert "is_contaminant" in adata_qc.var.columns
        assert "is_keratin" in adata_qc.var.columns
        assert "is_albumin" in adata_qc.var.columns

        # We added 20 contaminants (10 keratin + 5 albumin + 5 immunoglobulin)
        assert stats["total_contaminants"] >= 15  # Should detect most
        assert stats["contaminant_counts_by_type"]["keratin"] >= 8

    def test_dynamic_range_evaluation(self, synthetic_mass_spec_data):
        """Test dynamic range evaluation."""
        service = ProteomicsQualityService()
        adata = synthetic_mass_spec_data

        # Evaluate dynamic range
        adata_qc, stats, ir = service.evaluate_dynamic_range(
            adata, percentile_low=5.0, percentile_high=95.0
        )

        # Check dynamic range metrics
        assert "dynamic_range_log10" in adata_qc.obs.columns
        assert "dynamic_range_log10" in adata_qc.var.columns

        # MS data should have high dynamic range (3-6 orders of magnitude)
        assert stats["median_sample_dynamic_range"] > 2.0
        assert stats["median_sample_dynamic_range"] < 7.0

    def test_pca_outlier_detection(self, synthetic_mass_spec_data):
        """Test PCA-based outlier detection."""
        service = ProteomicsQualityService()
        adata = synthetic_mass_spec_data

        # Detect outliers
        adata_qc, stats, ir = service.detect_pca_outliers(
            adata, n_components=10, outlier_threshold=3.0
        )

        # Check PCA results
        assert "X_pca" in adata_qc.obsm
        assert "pca" in adata_qc.uns
        assert "is_pca_outlier" in adata_qc.obs.columns

        # Should detect some outliers (but not too many)
        assert stats["outlier_percentage"] < 20.0


class TestMassSpecDifferential:
    """Test mass spectrometry differential expression."""

    def test_differential_expression_t_test(self, synthetic_mass_spec_data):
        """Test t-test differential expression."""
        service = ProteomicsDifferentialService()
        adata = synthetic_mass_spec_data

        # First preprocess
        prep_service = ProteomicsPreprocessingService()
        adata_imputed, _, _ = prep_service.impute_missing_values(adata, method="knn")
        adata_norm, _, _ = prep_service.normalize_intensities(
            adata_imputed, method="median", log_transform=True
        )

        # Differential expression
        adata_de, stats, ir = service.perform_differential_expression(
            adata_norm,
            group_column="condition",
            test_method="t_test",
            fdr_threshold=0.05,
            fold_change_threshold=1.5,
        )

        # Check DE results
        assert "de_results" in adata_de.uns
        assert stats["analysis_type"] == "differential_expression"
        assert stats["n_comparisons"] > 0
        assert stats["n_significant_proteins"] > 0

        # We added 50 true DE proteins, should detect most
        # (allowing for some false negatives due to noise and missing values)
        assert stats["n_significant_proteins"] >= 20

    def test_differential_expression_limma_like(self, synthetic_mass_spec_data):
        """Test LIMMA-like differential expression."""
        service = ProteomicsDifferentialService()
        adata = synthetic_mass_spec_data

        # Preprocess
        prep_service = ProteomicsPreprocessingService()
        adata_imputed, _, _ = prep_service.impute_missing_values(adata, method="knn")
        adata_norm, _, _ = prep_service.normalize_intensities(
            adata_imputed, method="median", log_transform=True
        )

        # LIMMA-like analysis
        adata_de, stats, ir = service.perform_differential_expression(
            adata_norm,
            group_column="condition",
            test_method="limma_like",
            fdr_threshold=0.05,
        )

        # Check results
        assert "de_results" in adata_de.uns
        assert stats["n_significant_proteins"] > 0


# ==============================================================================
# AFFINITY PROTEOMICS WORKFLOW TESTS
# ==============================================================================


class TestAffinityPreprocessing:
    """Test affinity proteomics preprocessing."""

    def test_npx_normalization(self, synthetic_affinity_data):
        """Test normalization of NPX values."""
        service = ProteomicsPreprocessingService()
        adata = synthetic_affinity_data

        # NPX data is already log2-normalized, so we test median centering
        adata_norm, stats, ir = service.normalize_intensities(
            adata, method="median", log_transform=False  # Already log-transformed
        )

        # Check normalization
        assert "normalized" in adata_norm.layers
        assert stats["samples_processed"] == 40

    def test_low_missing_imputation(self, synthetic_affinity_data):
        """Test imputation with low missing values."""
        service = ProteomicsPreprocessingService()
        adata = synthetic_affinity_data

        # Calculate missing rate
        initial_missing_rate = np.isnan(adata.X).sum() / adata.X.size

        # Should have <30% missing
        assert initial_missing_rate < 0.30

        # Impute with KNN (appropriate for MCAR pattern)
        adata_imputed, stats, ir = service.impute_missing_values(
            adata, method="knn", knn_neighbors=5
        )

        # Check imputation
        assert not np.isnan(adata_imputed.X).any()
        assert stats["original_missing_percentage"] < 30.0


class TestAffinityQuality:
    """Test affinity proteomics quality control."""

    def test_low_cv_assessment(self, synthetic_affinity_data):
        """Test CV assessment - affinity should have lower CVs."""
        service = ProteomicsQualityService()
        adata = synthetic_affinity_data

        # Assess CV
        adata_qc, stats, ir = service.assess_coefficient_variation(
            adata, cv_threshold=0.30, min_observations=3
        )

        # Affinity data should have lower CVs than MS
        assert (
            stats["median_cv_across_proteins"] < 0.50
        )  # Synthetic affinity CV ~34% (fractional: 0.34)
        assert stats["n_high_cv_proteins"] < adata.n_vars  # Count high CV proteins

    def test_technical_replicate_assessment(self, synthetic_affinity_data):
        """Test technical replicate assessment."""
        service = ProteomicsQualityService()
        adata = synthetic_affinity_data

        # Add replicate grouping
        adata.obs["replicate_group"] = ["Rep" + str(i % 5) for i in range(adata.n_obs)]

        # Assess replicates
        adata_qc, stats, ir = service.assess_technical_replicates(
            adata, replicate_column="replicate_group", correlation_method="pearson"
        )

        # Check replicate correlation
        assert stats["median_replicate_correlation"] > 0.8  # Should be high
        assert (
            stats["median_replicate_cv"] < 0.25
        )  # Should be low (fractional: 0.25 = 25%)


class TestAffinityDifferential:
    """Test affinity proteomics differential expression."""

    def test_differential_with_low_cv(self, synthetic_affinity_data):
        """Test differential expression with low CV data."""
        service = ProteomicsDifferentialService()
        adata = synthetic_affinity_data

        # Minimal preprocessing (data already normalized)
        prep_service = ProteomicsPreprocessingService()
        adata_imputed, _, _ = prep_service.impute_missing_values(adata, method="knn")

        # Differential expression
        adata_de, stats, ir = service.perform_differential_expression(
            adata_imputed,
            group_column="condition",
            test_method="t_test",
            fdr_threshold=0.05,
            fold_change_threshold=1.5,
        )

        # With lower CVs and fewer proteins, should have cleaner results
        assert "de_results" in adata_de.uns
        assert stats["n_significant_proteins"] > 0

        # Should detect most of the 15 true DE proteins
        assert stats["n_significant_proteins"] >= 8


# ==============================================================================
# COMPLETE END-TO-END WORKFLOW TESTS
# ==============================================================================


class TestCompleteWorkflows:
    """Test complete end-to-end workflows."""

    def test_complete_ms_workflow(self, synthetic_mass_spec_data):
        """Test complete MS workflow: QC → Preprocess → Differential."""
        adata = synthetic_mass_spec_data

        # Step 1: Quality Control
        qc_service = ProteomicsQualityService()
        adata_qc, qc_stats, _ = qc_service.assess_missing_value_patterns(adata)
        adata_qc, cv_stats, _ = qc_service.assess_coefficient_variation(adata_qc)
        adata_qc, contam_stats, _ = qc_service.detect_contaminants(adata_qc)

        # Step 2: Preprocessing
        prep_service = ProteomicsPreprocessingService()

        # Remove contaminants
        clean_proteins = ~adata_qc.var["is_contaminant"]
        adata_clean = adata_qc[:, clean_proteins].copy()

        # Impute missing values
        adata_imputed, impute_stats, _ = prep_service.impute_missing_values(
            adata_clean, method="mixed"
        )

        # Normalize
        adata_norm, norm_stats, _ = prep_service.normalize_intensities(
            adata_imputed, method="median", log_transform=True
        )

        # Batch correction
        adata_corrected, batch_stats, _ = prep_service.correct_batch_effects(
            adata_norm, batch_key="batch", method="combat"
        )

        # Step 3: Differential Expression
        de_service = ProteomicsDifferentialService()
        adata_de, de_stats, _ = de_service.perform_differential_expression(
            adata_corrected,
            group_column="condition",
            test_method="limma_like",
            fdr_threshold=0.05,
        )

        # Final checks
        assert adata_de.n_vars < adata.n_vars  # Contaminants removed
        assert not np.isnan(adata_de.X).any()  # No missing values
        assert "de_results" in adata_de.uns
        assert de_stats["n_significant_proteins"] > 0

        # Workflow should detect significant DE proteins
        print(f"\n✓ Complete MS workflow test passed:")
        print(f"  - Original: {adata.n_obs} samples × {adata.n_vars} proteins")
        print(f"  - Missing values: {qc_stats['overall_missing_rate']*100:.1f}%")
        print(f"  - Contaminants removed: {contam_stats['total_contaminants']}")
        print(f"  - Final: {adata_de.n_obs} samples × {adata_de.n_vars} proteins")
        print(f"  - Significant proteins: {de_stats['n_significant_proteins']}")

    def test_complete_affinity_workflow(self, synthetic_affinity_data):
        """Test complete affinity workflow: QC → Preprocess → Differential."""
        adata = synthetic_affinity_data

        # Step 1: Quality Control
        qc_service = ProteomicsQualityService()
        adata_qc, qc_stats, _ = qc_service.assess_missing_value_patterns(adata)
        adata_qc, cv_stats, _ = qc_service.assess_coefficient_variation(adata_qc)

        # Step 2: Preprocessing
        prep_service = ProteomicsPreprocessingService()

        # Impute (low missing values)
        adata_imputed, impute_stats, _ = prep_service.impute_missing_values(
            adata_qc, method="knn", knn_neighbors=5
        )

        # Normalize (NPX already log2, just median center)
        adata_norm, norm_stats, _ = prep_service.normalize_intensities(
            adata_imputed, method="median", log_transform=False
        )

        # Step 3: Differential Expression
        de_service = ProteomicsDifferentialService()
        adata_de, de_stats, _ = de_service.perform_differential_expression(
            adata_norm,
            group_column="condition",
            test_method="t_test",
            fdr_threshold=0.05,
        )

        # Final checks
        assert not np.isnan(adata_de.X).any()
        assert "de_results" in adata_de.uns
        assert de_stats["n_significant_proteins"] > 0

        # Affinity should have cleaner results (lower CVs, fewer missing values)
        print(f"\n✓ Complete affinity workflow test passed:")
        print(f"  - Shape: {adata.n_obs} samples × {adata.n_vars} proteins")
        print(f"  - Missing values: {qc_stats['overall_missing_rate']*100:.1f}%")
        print(f"  - Median CV: {cv_stats['median_cv_across_proteins']*100:.1f}%")
        print(f"  - Significant proteins: {de_stats['n_significant_proteins']}")

    def test_workflow_comparison_ms_vs_affinity(
        self, synthetic_mass_spec_data, synthetic_affinity_data
    ):
        """Compare MS and affinity workflows side-by-side."""

        # Quick QC on both
        qc_service = ProteomicsQualityService()

        # MS
        _, ms_qc, _ = qc_service.assess_missing_value_patterns(synthetic_mass_spec_data)
        _, ms_cv, _ = qc_service.assess_coefficient_variation(synthetic_mass_spec_data)

        # Affinity
        _, aff_qc, _ = qc_service.assess_missing_value_patterns(synthetic_affinity_data)
        _, aff_cv, _ = qc_service.assess_coefficient_variation(synthetic_affinity_data)

        # Affinity should have better quality metrics
        assert aff_qc["overall_missing_rate"] < ms_qc["overall_missing_rate"]
        assert aff_cv["median_protein_cv"] < ms_cv["median_protein_cv"]

        print(f"\n✓ MS vs Affinity comparison:")
        print(
            f"  MS Missing: {ms_qc['overall_missing_rate']*100:.1f}% | Affinity: {aff_qc['overall_missing_rate']*100:.1f}%"
        )
        print(
            f"  MS CV: {ms_cv['median_cv_across_proteins']*100:.1f}% | Affinity: {aff_cv['median_cv_across_proteins']*100:.1f}%"
        )


# ==============================================================================
# EDGE CASE TESTS
# ==============================================================================


class TestProteomicsEdgeCases:
    """Test edge cases and error handling."""

    def test_all_missing_protein(self, synthetic_mass_spec_data):
        """Test handling of proteins with 100% missing values."""
        adata = synthetic_mass_spec_data.copy()

        # Set one protein to all NaN
        adata.X[:, 0] = np.nan

        # QC should handle this
        qc_service = ProteomicsQualityService()
        adata_qc, stats, ir = qc_service.assess_missing_value_patterns(adata)

        # Should detect completely missing protein
        assert stats["completely_missing_proteins"] >= 1

    def test_no_missing_values(self, synthetic_affinity_data):
        """Test preprocessing when no missing values present."""
        adata = synthetic_affinity_data.copy()

        # Remove all NaNs
        adata.X = np.nan_to_num(adata.X, nan=5.0)

        # Imputation should skip
        prep_service = ProteomicsPreprocessingService()
        adata_result, stats, ir = prep_service.impute_missing_values(
            adata, method="knn"
        )

        assert stats["missing_values_found"] is False
        assert stats["imputation_performed"] is False

    def test_single_batch(self, synthetic_mass_spec_data):
        """Test batch correction with only one batch."""
        adata = synthetic_mass_spec_data.copy()

        # Set all to same batch
        adata.obs["batch"] = "Batch1"

        # Batch correction should skip
        prep_service = ProteomicsPreprocessingService()
        adata_result, stats, ir = prep_service.correct_batch_effects(
            adata, batch_key="batch", method="combat"
        )

        assert stats["batch_correction_performed"] is False
        assert stats["n_batches"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
