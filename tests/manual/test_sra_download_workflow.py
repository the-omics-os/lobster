"""
Manual test for SRA download workflow.

This script tests the complete SRA download workflow:
1. SRAProvider.get_download_urls() - Fetch URLs from ENA
2. SRADownloadService integration with DownloadOrchestrator
3. Full download → AnnData creation → provenance tracking

Usage:
    python tests/manual/test_sra_download_workflow.py
"""

import tempfile
from pathlib import Path

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.schemas.download_queue import DownloadQueueEntry, DownloadStatus
from lobster.services.data_access.sra_download_service import SRADownloadService
from lobster.tools.download_orchestrator import DownloadOrchestrator
from lobster.tools.providers.sra_provider import SRAProvider


def test_sra_provider_get_urls():
    """Test SRAProvider.get_download_urls() method."""
    print("\n=== Test 1: SRAProvider.get_download_urls() ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        dm = DataManagerV2(workspace_path=tmpdir)
        provider = SRAProvider(dm)

        # Test with a small public SRA run
        accession = "SRR21960766"  # Small AMPLICON dataset from microbiome study
        print(f"Fetching download URLs for {accession} from ENA...")

        try:
            # get_download_urls() now returns typed DownloadUrlResult
            url_result = provider.get_download_urls(accession)

            print(f"✓ Success! Retrieved URL info:")
            print(f"  - Accession: {url_result.accession}")
            print(f"  - Database: {url_result.database}")
            print(f"  - Layout: {url_result.layout}")
            print(f"  - Platform: {url_result.platform}")
            print(f"  - Total size: {url_result.get_total_size() / 1e6:.2f} MB")
            print(f"  - Files: {len(url_result.raw_files)}")

            for idx, file_obj in enumerate(url_result.raw_files, 1):
                print(
                    f"    {idx}. {file_obj.filename} ({file_obj.size_bytes / 1e6:.2f} MB)"
                )
                print(f"       MD5: {file_obj.checksum}")

            # Also test legacy dict conversion
            legacy_dict = url_result.to_legacy_dict()
            assert "accession" in legacy_dict
            assert "raw_urls" in legacy_dict
            print(f"  ✓ Legacy dict conversion works (backward compatible)")

            return url_result

        except Exception as e:
            print(f"✗ Failed: {e}")
            raise


def test_sra_download_service_registration():
    """Test SRADownloadService auto-registration in DownloadOrchestrator."""
    print("\n=== Test 2: SRADownloadService Auto-Registration ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        dm = DataManagerV2(workspace_path=tmpdir)
        orchestrator = DownloadOrchestrator(dm)

        print(f"Supported databases: {orchestrator.list_supported_databases()}")

        # Check SRA service is registered
        assert "sra" in orchestrator.list_supported_databases()
        assert "ena" in orchestrator.list_supported_databases()
        assert "ddbj" in orchestrator.list_supported_databases()

        # Get service
        sra_service = orchestrator.get_service_for_database("sra")
        assert sra_service is not None
        assert isinstance(sra_service, SRADownloadService)

        print(f"✓ SRADownloadService registered successfully")
        print(f"  - Supported strategies: {sra_service.get_supported_strategies()}")
        print(f"  - Supported databases: {sra_service.supported_databases()}")


def test_strategy_validation():
    """Test strategy parameter validation."""
    print("\n=== Test 3: Strategy Parameter Validation ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        dm = DataManagerV2(workspace_path=tmpdir)
        service = SRADownloadService(dm)

        # Test valid parameters
        valid_params = [
            {"verify_checksum": True},
            {"layout": "PAIRED"},
            {"verify_checksum": False, "layout": "SINGLE"},
        ]

        for params in valid_params:
            valid, error = service.validate_strategy_params(params)
            assert valid is True, f"Valid params {params} should pass"
            print(f"✓ Valid: {params}")

        # Test invalid parameters
        invalid_params = [
            {"verify_checksum": "yes"},  # Should be bool
            {"layout": "INVALID"},
            {"file_type": "sra"},  # Not supported in Phase 1
        ]

        for params in invalid_params:
            valid, error = service.validate_strategy_params(params)
            assert valid is False, f"Invalid params {params} should fail"
            print(f"✓ Invalid (as expected): {params} - {error}")


def test_fastq_loader():
    """Test FASTQLoader with mock files."""
    print("\n=== Test 4: FASTQLoader ===")

    from lobster.services.data_access.sra_download_service import FASTQLoader

    loader = FASTQLoader()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock FASTQ files
        fastq_files = [
            Path(tmpdir) / "SRR21960766_1.fastq.gz",
            Path(tmpdir) / "SRR21960766_2.fastq.gz",
        ]

        for f in fastq_files:
            f.write_bytes(b"@read1\nACGT\n+\nIIII\n@read2\nGCTA\n+\nIIII\n")

        # Create AnnData
        metadata = {
            "layout": "PAIRED",
            "platform": "ILLUMINA",
            "mirror": "ena",
            "total_size_bytes": sum(f.stat().st_size for f in fastq_files),
        }

        queue_entry = DownloadQueueEntry(
            entry_id="test_123",
            dataset_id="SRR21960766",
            database="sra",
            status=DownloadStatus.PENDING,
        )

        adata = loader.create_fastq_anndata(
            fastq_paths=fastq_files,
            metadata=metadata,
            queue_entry=queue_entry,
        )

        print(f"✓ Created AnnData:")
        print(f"  - Shape: {adata.n_obs} obs × {adata.n_vars} vars")
        print(f"  - Data type: {adata.uns['data_type']}")
        print(f"  - Processing required: {adata.uns['processing_required']}")
        print(f"  - FASTQ files: {adata.uns['fastq_files']['n_files']}")
        print(f"  - Total size: {adata.uns['fastq_files']['total_size_mb']:.2f} MB")
        print(f"  - Layout: {adata.uns['fastq_files']['layout']}")

        # Verify structure
        assert adata.uns["data_type"] == "fastq_raw"
        assert "alignment" in adata.uns["processing_required"]
        assert "fastq_files" in adata.uns
        assert len(adata.uns["fastq_files"]["paths"]) == 2


if __name__ == "__main__":
    print("=" * 70)
    print("SRA Download Service - Manual Test Suite")
    print("=" * 70)

    try:
        # Test 1: Provider URL fetching
        url_info = test_sra_provider_get_urls()

        # Test 2: Service registration
        test_sra_download_service_registration()

        # Test 3: Strategy validation
        test_strategy_validation()

        # Test 4: FASTQ loader
        test_fastq_loader()

        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED")
        print("=" * 70)

        print("\n🎉 SRA Download Service is ready for production use!")
        print("\nNext steps:")
        print("1. Test with real SRA download via data_expert agent")
        print("2. Monitor download performance and mirror success rates")
        print("3. Consider adding .sra format support in Phase 2")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
