#!/usr/bin/env python3
"""
Direct test of adaptive 10X loader fix for GSE182227.
Tests that single-column genes files are handled correctly.
"""
import sys
import tempfile
from pathlib import Path
import gzip

# Test the format detection
def test_format_detection():
    """Test _detect_features_format() with single-column file"""
    print("=" * 80)
    print("TEST 1: Format Detection")
    print("=" * 80)

    # Create temp file with single-column format (like GSE182227)
    temp_dir = Path(tempfile.mkdtemp(prefix="test_"))
    features_path = temp_dir / "genes.txt.gz"

    # Write single-column format
    with gzip.open(features_path, 'wt') as f:
        f.write("RP11-34P13.3\n")
        f.write("FAM138A\n")
        f.write("OR4F5\n")

    # Test detection
    from lobster.services.data_access.geo_service import GEOService
    from lobster.core.data_manager_v2 import DataManagerV2

    data_manager = DataManagerV2()
    geo_service = GEOService(data_manager)

    format_type = geo_service._detect_features_format(features_path)

    print(f"✅ Format detected: {format_type}")
    assert format_type == "symbols_only", f"Expected 'symbols_only', got '{format_type}'"
    print("✅ TEST PASSED: Single-column file detected correctly")

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)

    return True

# Test the manual loader
def test_manual_loader():
    """Test _load_10x_manual() with minimal data"""
    print("\n" + "=" * 80)
    print("TEST 2: Manual Loader")
    print("=" * 80)

    from scipy.io import mmwrite
    from scipy.sparse import csr_matrix
    import numpy as np
    import gzip

    # Create temp directory with MTX trio
    temp_dir = Path(tempfile.mkdtemp(prefix="test_"))

    # Create minimal matrix (10 cells × 5 genes)
    X = csr_matrix(np.random.poisson(1, (5, 10)))  # genes × cells for MTX format
    mmwrite(temp_dir / "matrix.mtx", X)

    #Compress it
    with open(temp_dir / "matrix.mtx", 'rb') as f_in:
        with gzip.open(temp_dir / "matrix.mtx.gz", 'wb') as f_out:
            f_out.write(f_in.read())

    # Create barcodes (10 cells)
    with gzip.open(temp_dir / "barcodes.tsv.gz", 'wt') as f:
        for i in range(10):
            f.write(f"CELL_{i}\n")

    # Create single-column genes file (5 genes)
    with gzip.open(temp_dir / "genes.txt.gz", 'wt') as f:
        f.write("GENE1\n")
        f.write("GENE2\n")
        f.write("GENE3\n")
        f.write("GENE4\n")
        f.write("GENE5\n")

    # Test manual loader
    from lobster.services.data_access.geo_service import GEOService
    from lobster.core.data_manager_v2 import DataManagerV2

    data_manager = DataManagerV2()
    geo_service = GEOService(data_manager)

    adata = geo_service._load_10x_manual(temp_dir, "symbols_only", "TEST")

    print(f"✅ Loaded AnnData: {adata.n_obs} cells × {adata.n_vars} genes")
    assert adata.n_obs == 10, f"Expected 10 cells, got {adata.n_obs}"
    assert adata.n_vars == 5, f"Expected 5 genes, got {adata.n_vars}"
    assert 'gene_ids' in adata.var.columns, "Missing gene_ids column"
    assert 'feature_types' in adata.var.columns, "Missing feature_types column"

    print("✅ TEST PASSED: Manual loader works correctly")

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)

    return True

if __name__ == "__main__":
    try:
        print("Testing Adaptive 10X Loader Fix")
        print("=" * 80)
        print()

        # Run tests
        test_format_detection()
        test_manual_loader()

        print("\n" + "=" * 80)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 80)
        print()
        print("The adaptive loader is working correctly:")
        print("✅ Detects single-column features files")
        print("✅ Loads MTX data manually when needed")
        print("✅ Creates proper AnnData with correct dimensions")
        print()
        print("Next step: Test with real GSE182227 dataset")
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
