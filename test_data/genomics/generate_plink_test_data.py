#!/usr/bin/env python3
"""
Generate PLINK test data from the existing chr22.vcf.gz file.

This script:
1. Loads a subset of chr22.vcf.gz (100 samples, 1000 variants)
2. Converts to PLINK binary format (.bed/.bim/.fam)
3. Saves to plink_test/ directory for testing PLINKAdapter

Requires: cyvcf2, bed-reader (or pandas-plink)
"""

import sys
from pathlib import Path
import numpy as np

# Add lobster to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lobster.core.adapters.genomics.vcf_adapter import VCFAdapter

print("=" * 80)
print("GENERATING PLINK TEST DATA FROM CHR22.VCF.GZ")
print("=" * 80)

# Step 1: Load subset of VCF
vcf_path = Path(__file__).parent / "chr22.vcf.gz"
if not vcf_path.exists():
    print(f"❌ VCF file not found: {vcf_path}")
    sys.exit(1)

print(f"\n1. Loading VCF: {vcf_path}")
print("   Selecting first 100 samples, 1000 variants...")

adapter = VCFAdapter(strict_validation=False)
try:
    adata = adapter.from_source(
        str(vcf_path),
        max_variants=1000,
        filter_pass=True,
    )
    print(f"✅ Loaded: {adata.n_obs} samples × {adata.n_vars} variants")

    # Subsample to first 100 individuals for faster testing
    if adata.n_obs > 100:
        adata = adata[:100, :].copy()
        print(f"✅ Subsampled to: {adata.n_obs} samples × {adata.n_vars} variants")

except Exception as e:
    print(f"❌ VCF loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 2: Convert to PLINK format
print("\n2. Converting to PLINK format...")

output_dir = Path(__file__).parent / "plink_test"
output_prefix = output_dir / "test_chr22"

try:
    # Extract genotype matrix (samples × variants)
    gt = adata.X.copy()
    if hasattr(gt, 'toarray'):
        gt = gt.toarray()

    # PLINK .fam file (sample information)
    # Format: FamilyID IndividualID FatherID MotherID Sex Phenotype (TAB-separated)
    print("   Writing .fam file...")
    fam_lines = []
    for i, sample_id in enumerate(adata.obs.index):
        # FID IID FatherID MotherID Sex(1=male,2=female,0=unknown) Phenotype(-9=missing)
        fam_lines.append(f"FAM{i+1}\t{sample_id}\t0\t0\t0\t-9")

    with open(f"{output_prefix}.fam", "w") as f:
        f.write("\n".join(fam_lines) + "\n")
    print(f"   ✅ Wrote {len(fam_lines)} samples to {output_prefix}.fam")

    # PLINK .bim file (variant information)
    # Format: chr snp_id genetic_distance bp_position allele1 allele2 (TAB-separated)
    print("   Writing .bim file...")
    bim_lines = []
    for i, (var_id, var_row) in enumerate(adata.var.iterrows()):
        chrom = var_row['CHROM'].replace('chr', '')
        pos = var_row['POS']
        snp_id = var_row.get('ID', f'snp_{i+1}')
        if snp_id == '.' or snp_id == '':
            snp_id = f'chr{chrom}:{pos}'
        ref = var_row['REF']
        alt = var_row['ALT']

        # PLINK format: chr snp_id genetic_dist bp_pos allele1 allele2 (TAB-separated)
        bim_lines.append(f"{chrom}\t{snp_id}\t0\t{pos}\t{ref}\t{alt}")

    with open(f"{output_prefix}.bim", "w") as f:
        f.write("\n".join(bim_lines) + "\n")
    print(f"   ✅ Wrote {len(bim_lines)} variants to {output_prefix}.bim")

    # PLINK .bed file (binary genotype data)
    # Format: Magic bytes (0x6c, 0x1b, 0x01) + genotype data
    print("   Writing .bed file...")

    # PLINK bed format: SNP-major mode (variants as rows)
    # Each genotype coded in 2 bits: 00=homozygous ref, 01=missing, 10=het, 11=homozygous alt

    n_samples = gt.shape[0]
    n_variants = gt.shape[1]

    # Write magic bytes
    with open(f"{output_prefix}.bed", "wb") as f:
        # Magic number for PLINK binary
        f.write(bytes([0x6c, 0x1b]))
        # SNP-major mode (0x01)
        f.write(bytes([0x01]))

        # Each byte stores 4 genotypes (2 bits each)
        bytes_per_variant = (n_samples + 3) // 4  # Ceiling division

        for var_idx in range(n_variants):
            # Get genotypes for this variant (all samples)
            var_gts = gt[:, var_idx]

            # Pack into bytes (4 genotypes per byte)
            variant_bytes = []
            for byte_idx in range(bytes_per_variant):
                byte_val = 0
                for bit_idx in range(4):
                    sample_idx = byte_idx * 4 + bit_idx
                    if sample_idx < n_samples:
                        gt_val = var_gts[sample_idx]

                        # Convert 0/1/2/-1 encoding to PLINK 2-bit encoding
                        if gt_val == -1 or np.isnan(gt_val):
                            plink_code = 0b01  # Missing
                        elif gt_val == 0:
                            plink_code = 0b00  # Homozygous ref
                        elif gt_val == 1:
                            plink_code = 0b10  # Heterozygous
                        elif gt_val == 2:
                            plink_code = 0b11  # Homozygous alt
                        else:
                            plink_code = 0b01  # Unknown -> missing

                        # Pack into byte (least significant bits first)
                        byte_val |= (plink_code << (bit_idx * 2))

                variant_bytes.append(byte_val)

            f.write(bytes(variant_bytes))

    print(f"   ✅ Wrote binary genotypes to {output_prefix}.bed")

    # Verify file sizes
    bed_size = Path(f"{output_prefix}.bed").stat().st_size
    expected_size = 3 + (n_variants * bytes_per_variant)
    print(f"\n3. Verification:")
    print(f"   .bed file size: {bed_size} bytes (expected: {expected_size})")
    print(f"   .bim lines: {len(bim_lines)}")
    print(f"   .fam lines: {len(fam_lines)}")

    if bed_size == expected_size:
        print("   ✅ File sizes correct")
    else:
        print(f"   ⚠️  Size mismatch: {bed_size} != {expected_size}")

except Exception as e:
    print(f"❌ PLINK conversion failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ PLINK TEST DATA GENERATED SUCCESSFULLY")
print("=" * 80)
print(f"Files created:")
print(f"  - {output_prefix}.bed ({n_variants} variants × {n_samples} samples)")
print(f"  - {output_prefix}.bim ({n_variants} variants)")
print(f"  - {output_prefix}.fam ({n_samples} samples)")
print("\nThese files can now be used to test PLINKAdapter.")
