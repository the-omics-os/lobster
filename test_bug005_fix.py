#!/usr/bin/env python3
"""
Test script for BUG-005: Metadata Loss in QC/Clustering Pipeline

This script validates that biological metadata (patient_id, tissue_region, condition, sample_id)
is preserved through QC filtering and clustering operations.

Expected behavior AFTER fix:
- QC filtering preserves all original obs columns
- Clustering preserves all original obs columns
- Pseudobulk workflows can access patient_id and other metadata
"""

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix

from lobster.services.quality.preprocessing_service import PreprocessingService
from lobster.services.analysis.clustering_service import ClusteringService


def create_test_adata_with_metadata():
    """Create synthetic AnnData with biological metadata."""
    np.random.seed(42)

    # Create expression matrix (100 cells x 500 genes)
    n_cells = 100
    n_genes = 500

    # Sparse count matrix
    counts = np.random.negative_binomial(5, 0.3, size=(n_cells, n_genes))
    X = csr_matrix(counts.astype(np.float32))

    # Gene names (470 regular + 20 MT + 10 ribosomal = 500 total)
    mt_genes = [f"MT-Gene_{i}" for i in range(20)]
    ribo_genes = [f"RPS{i}" for i in range(5)] + [f"RPL{i}" for i in range(5)]
    regular_genes = [f"Gene_{i}" for i in range(470)]

    var = pd.DataFrame(index=regular_genes + mt_genes + ribo_genes)

    # Cell metadata with biological information (THIS IS WHAT WE'RE TESTING)
    obs = pd.DataFrame({
        'patient_id': [f'PT{(i % 3) + 1:02d}' for i in range(n_cells)],
        'tissue_region': ['Tumor' if i < 50 else 'Normal' for i in range(n_cells)],
        'condition': ['CRC' if i < 70 else 'Control' for i in range(n_cells)],
        'sample_id': [f'Sample_{(i % 5) + 1}' for i in range(n_cells)],
        'batch': [f'Batch_{(i % 2) + 1}' for i in range(n_cells)],
    }, index=[f"Cell_{i}" for i in range(n_cells)])

    adata = AnnData(X=X, obs=obs, var=var)

    print(f"Created test AnnData: {adata.shape[0]} cells x {adata.shape[1]} genes")
    print(f"Original obs columns: {list(adata.obs.columns)}")

    return adata


def test_preprocessing_metadata_preservation():
    """Test that QC filtering preserves metadata."""
    print("\n" + "="*80)
    print("TEST 1: PreprocessingService.filter_and_normalize_cells")
    print("="*80)

    # Create test data
    adata = create_test_adata_with_metadata()
    original_obs_cols = set(adata.obs.columns)
    original_metadata_cols = {'patient_id', 'tissue_region', 'condition', 'sample_id', 'batch'}

    # Run preprocessing with permissive filters for synthetic data
    service = PreprocessingService()
    adata_filtered, stats, ir = service.filter_and_normalize_cells(
        adata=adata,
        min_genes_per_cell=10,  # Very permissive for synthetic data
        max_genes_per_cell=10000,  # Very permissive
        min_cells_per_gene=1,  # Very permissive
        max_mito_percent=100.0,  # No filtering
        max_ribo_percent=100.0,  # No filtering
        target_sum=10000,
    )

    # Check results
    filtered_obs_cols = set(adata_filtered.obs.columns)

    print(f"\nOriginal cells: {adata.n_obs}")
    print(f"Filtered cells: {adata_filtered.n_obs}")
    print(f"Cells retained: {stats['cells_retained_pct']:.1f}%")

    print(f"\nOriginal obs columns: {sorted(original_obs_cols)}")
    print(f"Filtered obs columns: {sorted(filtered_obs_cols)}")

    # Check metadata preservation
    missing_metadata = original_metadata_cols - filtered_obs_cols
    if missing_metadata:
        print(f"\n[FAIL] Missing metadata columns: {missing_metadata}")
        return False
    else:
        print(f"\n[PASS] All metadata columns preserved: {original_metadata_cols}")

    # Verify pseudobulk-critical columns
    pseudobulk_required = ['patient_id', 'tissue_region']
    if all(col in adata_filtered.obs.columns for col in pseudobulk_required):
        print(f"[PASS] Pseudobulk workflow columns present: {pseudobulk_required}")

        # Show sample data
        print("\nSample metadata from filtered AnnData:")
        print(adata_filtered.obs[['patient_id', 'tissue_region', 'condition']].head(10))
        return True
    else:
        print(f"[FAIL] Pseudobulk workflow blocked - missing: {[col for col in pseudobulk_required if col not in adata_filtered.obs.columns]}")
        return False


def test_clustering_metadata_preservation():
    """Test that clustering preserves metadata."""
    print("\n" + "="*80)
    print("TEST 2: ClusteringService.cluster_and_visualize")
    print("="*80)

    # Create and preprocess test data
    adata = create_test_adata_with_metadata()

    # Quick preprocessing (no filtering for simplicity)
    import scanpy as sc
    sc.pp.normalize_total(adata, target_sum=10000)
    sc.pp.log1p(adata)

    # Add metadata
    adata.obs['patient_id'] = [f'PT{(i % 3) + 1:02d}' for i in range(adata.n_obs)]
    adata.obs['tissue_region'] = ['Tumor' if i < 50 else 'Normal' for i in range(adata.n_obs)]

    original_metadata_cols = {'patient_id', 'tissue_region', 'condition', 'sample_id', 'batch'}

    # Run clustering
    service = ClusteringService()
    adata_clustered, stats, ir = service.cluster_and_visualize(
        adata=adata,
        resolution=0.5,
        demo_mode=True,  # Fast mode
        subsample_size=None,  # No subsampling
        feature_selection_method='deviance',
        n_features=200,  # Fewer features for speed
    )

    # Check results
    clustered_obs_cols = set(adata_clustered.obs.columns)

    print(f"\nCells clustered: {adata_clustered.n_obs}")
    print(f"Clusters identified: {stats['n_clusters']}")
    print(f"Clustered obs columns: {sorted(clustered_obs_cols)}")

    # Check metadata preservation
    missing_metadata = original_metadata_cols - clustered_obs_cols
    if missing_metadata:
        print(f"\n[FAIL] Missing metadata columns: {missing_metadata}")
        return False
    else:
        print(f"\n[PASS] All metadata columns preserved: {original_metadata_cols}")

    # Verify pseudobulk-critical columns
    pseudobulk_required = ['patient_id', 'tissue_region', 'leiden']
    if all(col in adata_clustered.obs.columns for col in pseudobulk_required):
        print(f"[PASS] Pseudobulk workflow columns present: {pseudobulk_required}")

        # Show sample data
        print("\nSample metadata from clustered AnnData:")
        print(adata_clustered.obs[['patient_id', 'tissue_region', 'leiden']].head(10))
        return True
    else:
        print(f"[FAIL] Pseudobulk workflow blocked - missing: {[col for col in pseudobulk_required if col not in adata_clustered.obs.columns]}")
        return False


def test_end_to_end_pseudobulk_workflow():
    """Test complete QC → Clustering → Pseudobulk workflow."""
    print("\n" + "="*80)
    print("TEST 3: End-to-End Pseudobulk Workflow (QC → Clustering → Pseudobulk)")
    print("="*80)

    # Create test data
    adata = create_test_adata_with_metadata()

    # Step 1: QC filtering with permissive filters for synthetic data
    preprocessing_service = PreprocessingService()
    adata_filtered, _, _ = preprocessing_service.filter_and_normalize_cells(
        adata=adata,
        min_genes_per_cell=10,
        max_genes_per_cell=10000,
        min_cells_per_gene=1,
        max_mito_percent=100.0,
        max_ribo_percent=100.0,
        target_sum=10000,
    )

    print(f"\n[STEP 1] QC Filtering: {adata.n_obs} → {adata_filtered.n_obs} cells")

    # Step 2: Clustering
    clustering_service = ClusteringService()
    adata_clustered, _, _ = clustering_service.cluster_and_visualize(
        adata=adata_filtered,
        resolution=0.5,
        demo_mode=True,
        feature_selection_method='deviance',
        n_features=200,
    )

    print(f"[STEP 2] Clustering: Identified {len(adata_clustered.obs['leiden'].unique())} clusters")

    # Step 3: Try to create pseudobulk (this is what was failing in STRESS_TEST_08)
    try:
        # Check if we can group by patient_id and tissue_region
        if 'patient_id' not in adata_clustered.obs.columns:
            print(f"\n[FAIL] patient_id column not found - pseudobulk workflow blocked")
            return False

        if 'tissue_region' not in adata_clustered.obs.columns:
            print(f"\n[FAIL] tissue_region column not found - pseudobulk workflow blocked")
            return False

        # Simulate pseudobulk grouping
        pseudobulk_groups = adata_clustered.obs.groupby(['patient_id', 'tissue_region']).size()

        print(f"[STEP 3] Pseudobulk: Successfully created {len(pseudobulk_groups)} pseudobulk samples")
        print("\nPseudobulk groups:")
        print(pseudobulk_groups)

        print("\n[PASS] End-to-end pseudobulk workflow successful!")
        return True

    except Exception as e:
        print(f"\n[FAIL] Pseudobulk workflow failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "#"*80)
    print("# BUG-005 Validation Test Suite: Metadata Preservation")
    print("#"*80)

    results = []

    # Test 1: Preprocessing
    results.append(("Preprocessing", test_preprocessing_metadata_preservation()))

    # Test 2: Clustering
    results.append(("Clustering", test_clustering_metadata_preservation()))

    # Test 3: End-to-end
    results.append(("End-to-End", test_end_to_end_pseudobulk_workflow()))

    # Summary
    print("\n" + "#"*80)
    print("# Test Summary")
    print("#"*80)

    for test_name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {test_name}")

    all_passed = all(result[1] for result in results)

    if all_passed:
        print("\n[SUCCESS] All tests passed - BUG-005 is fixed!")
        return 0
    else:
        print("\n[FAILURE] Some tests failed - BUG-005 not fully resolved")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
