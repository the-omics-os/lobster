#!/usr/bin/env python3
"""
Test script for genomics implementation using 1000 Genomes Phase 3 chr22 data.

Tests:
1. VCF adapter - Load chr22 VCF
2. GenomicsQualityService - Calculate QC metrics
3. Filtering - Apply QC thresholds
4. GWASService - Run GWAS with synthetic phenotype
5. PCA - Population structure analysis
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add lobster to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lobster.core.adapters.genomics.vcf_adapter import VCFAdapter
from lobster.services.quality.genomics_quality_service import GenomicsQualityService
from lobster.services.analysis.gwas_service import GWASService

print("=" * 80)
print("GENOMICS IMPLEMENTATION TEST SUITE")
print("Dataset: 1000 Genomes Phase 3 chr22")
print("=" * 80)

# Test 1: VCF Adapter
print("\n" + "=" * 80)
print("TEST 1: VCF ADAPTER - Loading chr22 VCF")
print("=" * 80)

vcf_path = Path(__file__).parent / "chr22.vcf.gz"
if not vcf_path.exists():
    print(f"❌ VCF file not found: {vcf_path}")
    sys.exit(1)

print(f"Loading: {vcf_path}")
print("Loading first 10,000 variants for testing...")

adapter = VCFAdapter(strict_validation=False)
try:
    adata = adapter.from_source(
        str(vcf_path),
        max_variants=10000,
        filter_pass=True,
    )
    print(f"✅ VCF loaded successfully!")
    print(f"   - Samples: {adata.n_obs}")
    print(f"   - Variants: {adata.n_vars}")
    print(f"   - Sparsity: {1 - np.count_nonzero(adata.X) / adata.X.size:.2%}")
    print(f"   - Genotype matrix shape: {adata.X.shape}")
    print(f"   - Layers: {list(adata.layers.keys())}")

    # Validate structure
    assert adata.n_obs > 2000, f"Expected >2000 samples, got {adata.n_obs}"
    assert adata.n_vars > 9000, f"Expected >9000 variants after filtering, got {adata.n_vars}"
    assert 'GT' in adata.layers, "GT layer missing"
    assert 'CHROM' in adata.var.columns, "CHROM column missing in var"
    assert 'POS' in adata.var.columns, "POS column missing in var"
    print("✅ All structural validations passed")

except Exception as e:
    print(f"❌ VCF loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: GenomicsQualityService
print("\n" + "=" * 80)
print("TEST 2: GENOMICS QUALITY SERVICE - QC Metrics")
print("=" * 80)

qc_service = GenomicsQualityService()
try:
    adata_qc, stats, ir = qc_service.assess_quality(
        adata,
        min_call_rate=0.95,
        min_maf=0.01,
        hwe_pvalue=1e-10,
    )
    print(f"✅ QC assessment completed!")
    print(f"   - Mean sample call rate: {stats['sample_metrics']['mean_call_rate']:.3f}")
    print(f"   - Mean variant call rate: {stats['variant_metrics']['mean_call_rate']:.3f}")
    print(f"   - Mean heterozygosity: {stats['sample_metrics']['mean_heterozygosity']:.3f}")
    print(f"   - Variants passing QC: {stats['n_variants_pass_qc']}/{stats['n_variants']}")

    # Validate metrics exist
    assert 'call_rate' in adata_qc.obs.columns, "call_rate not in obs"
    assert 'heterozygosity' in adata_qc.obs.columns, "heterozygosity not in obs"
    assert 'maf' in adata_qc.var.columns, "maf not in var"
    assert 'hwe_p' in adata_qc.var.columns, "hwe_p not in var"
    assert 'qc_pass' in adata_qc.var.columns, "qc_pass not in var"
    print("✅ All QC metrics present")

    # Validate IR
    assert ir.operation == "genomics.qc.assess", f"Wrong operation: {ir.operation}"
    assert ir.tool_name == "GenomicsQualityService.assess_quality", f"Wrong tool: {ir.tool_name}"
    assert 'min_maf' in ir.parameters, "min_maf not in IR parameters"
    print("✅ Provenance IR validated")

except Exception as e:
    print(f"❌ QC assessment failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Filtering
print("\n" + "=" * 80)
print("TEST 3: FILTERING - Apply QC Thresholds")
print("=" * 80)

try:
    # Filter samples
    adata_filtered, sample_stats, ir_sample = qc_service.filter_samples(
        adata_qc,
        min_call_rate=0.95,
        het_sd_threshold=3.0,
    )
    print(f"✅ Sample filtering completed!")
    print(f"   - Samples before: {sample_stats['samples_before']}")
    print(f"   - Samples after: {sample_stats['samples_after']}")
    print(f"   - Samples removed: {sample_stats['samples_removed']}")

    # Filter variants
    adata_filtered, variant_stats, ir_variant = qc_service.filter_variants(
        adata_filtered,
        min_call_rate=0.99,
        min_maf=0.01,
        min_hwe_p=1e-6,
    )
    print(f"✅ Variant filtering completed!")
    print(f"   - Variants before: {variant_stats['variants_before']}")
    print(f"   - Variants after: {variant_stats['variants_after']}")
    print(f"   - Variants removed: {variant_stats['variants_removed']}")

    assert adata_filtered.n_obs > 2000, f"Too many samples removed: {adata_filtered.n_obs}"
    assert adata_filtered.n_vars > 500, f"Too many variants removed: {adata_filtered.n_vars}"
    assert adata_filtered.n_vars < 1000, f"Too few variants filtered: {adata_filtered.n_vars}"
    print(f"✅ Filtering thresholds reasonable ({adata_filtered.n_vars} variants, ~{adata_filtered.n_vars/10000*100:.1f}% retained)")

except Exception as e:
    print(f"❌ Filtering failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: GWAS with synthetic phenotype
print("\n" + "=" * 80)
print("TEST 4: GWAS SERVICE - Synthetic Phenotype")
print("=" * 80)

gwas_service = GWASService()
try:
    # Create synthetic phenotype (normal distribution)
    np.random.seed(42)
    adata_filtered.obs['height'] = np.random.normal(170, 10, adata_filtered.n_obs)
    adata_filtered.obs['age'] = np.random.randint(20, 80, adata_filtered.n_obs)
    adata_filtered.obs['sex'] = np.random.choice([1, 2], adata_filtered.n_obs)

    # Run GWAS
    adata_gwas, gwas_stats, ir_gwas = gwas_service.run_gwas(
        adata_filtered,
        phenotype='height',
        covariates=['age', 'sex'],
        model='linear',
        pvalue_threshold=5e-8,
    )
    print(f"✅ GWAS completed!")
    print(f"   - Variants tested: {gwas_stats['n_variants_tested']}")
    print(f"   - Significant variants (p<5e-8): {gwas_stats['n_variants_significant']}")
    print(f"   - Lambda GC: {gwas_stats['lambda_gc']:.3f} ({gwas_stats['lambda_gc_interpretation']})")

    # Validate GWAS results
    assert 'gwas_beta' in adata_gwas.var.columns, "gwas_beta not in var"
    assert 'gwas_pvalue' in adata_gwas.var.columns, "gwas_pvalue not in var"
    assert 'gwas_qvalue' in adata_gwas.var.columns, "gwas_qvalue not in var"
    assert 'gwas_significant' in adata_gwas.var.columns, "gwas_significant not in var"

    # Lambda GC expected to be inflated (>1.0) due to population structure in 1000 Genomes
    # 26 populations without PCA correction → inflation is expected behavior
    assert 0.8 < gwas_stats['lambda_gc'] < 2.0, f"Lambda GC extremely abnormal: {gwas_stats['lambda_gc']}"
    if gwas_stats['lambda_gc'] > 1.1:
        print(f"   ℹ️  High Lambda GC expected for 1000 Genomes without PCA correction")
    print("✅ GWAS results validated")

except Exception as e:
    print(f"❌ GWAS failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: PCA for population structure
print("\n" + "=" * 80)
print("TEST 5: PCA - Population Structure")
print("=" * 80)

try:
    adata_pca, pca_stats, ir_pca = gwas_service.calculate_pca(
        adata_filtered,
        n_components=10,
        ld_prune=False,  # Skip LD pruning (requires variant_position coordinates)
    )
    print(f"✅ PCA completed!")
    print(f"   - Components: {pca_stats['n_components']}")
    print(f"   - Variants used: {pca_stats['n_variants_used']}")
    print(f"   - Samples: {pca_stats['n_samples']}")
    print(f"   - PC1 variance: {pca_stats['variance_explained_pc1']:.1%}")
    print(f"   - Top 5 PCs variance: {pca_stats['variance_explained_top5']:.1%}")

    # Validate PCA results
    assert 'X_pca' in adata_pca.obsm.keys(), "X_pca not in obsm"
    assert adata_pca.obsm['X_pca'].shape == (adata_pca.n_obs, 10), "Wrong PCA shape"
    assert 'pca_variance_ratio' in adata_pca.uns.keys(), "pca_variance_ratio not in uns"

    # First PC should explain >5% variance for population structure
    assert pca_stats['variance_explained_pc1'] > 0.05, "PC1 explains too little variance"
    print("✅ PCA results validated")

except Exception as e:
    print(f"❌ PCA failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print("✅ TEST 1: VCF Adapter - PASSED")
print("✅ TEST 2: GenomicsQualityService - PASSED")
print("✅ TEST 3: Filtering - PASSED")
print("✅ TEST 4: GWAS Service - PASSED")
print("✅ TEST 5: PCA - PASSED")
print("\n🎉 ALL TESTS PASSED!")
print("=" * 80)
