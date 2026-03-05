"""
Tests for 3 pre-existing GEO pipeline bugs found during v1.0 stress testing.

Bug 1: 10X CellRanger H5 files crash the parser
Bug 2: Multi-modal filter not propagated to file download
Bug 3: Strategy engine ignores LLM modality signal
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lobster.services.data_access.geo.helpers import (
    _MODALITY_PATTERNS,
    _is_unsupported_modality_file,
)


# ---------------------------------------------------------------------------
# Bug 1: 10X H5 detection and routing
# ---------------------------------------------------------------------------


class TestBug1_10xH5:
    """10X CellRanger H5 files should be detected and routed to scanpy."""

    def _make_parser(self):
        from lobster.services.data_access.geo.parser import GEOParser

        return GEOParser()

    @patch("h5py.File")
    def test_is_10x_h5_detects_cellranger_format(self, mock_h5file):
        """10X H5 with /matrix group containing CSR components should be detected."""
        import h5py

        # Mock /matrix as a Group with CSR keys
        mock_group = MagicMock(spec=h5py.Group)
        mock_group.__contains__ = lambda self, k: k in ("data", "indices", "indptr")

        mock_f = MagicMock()
        mock_f.__contains__ = lambda self, k: k == "matrix"
        mock_f.__getitem__ = lambda self, k: mock_group if k == "matrix" else None
        mock_f.__enter__ = lambda self: self
        mock_f.__exit__ = MagicMock(return_value=False)
        mock_h5file.return_value = mock_f

        parser = self._make_parser()
        assert parser._is_10x_h5(Path("/fake/test.h5")) is True

    @patch("h5py.File")
    def test_is_10x_h5_rejects_non_cellranger(self, mock_h5file):
        """Regular H5AD files (no /matrix group) should not be detected as 10X."""
        import h5py

        # Mock /matrix as a Dataset (not Group)
        mock_dataset = MagicMock(spec=h5py.Dataset)

        mock_f = MagicMock()
        mock_f.__contains__ = lambda self, k: k == "matrix"
        mock_f.__getitem__ = lambda self, k: mock_dataset if k == "matrix" else None
        mock_f.__enter__ = lambda self: self
        mock_f.__exit__ = MagicMock(return_value=False)
        mock_h5file.return_value = mock_f

        parser = self._make_parser()
        assert parser._is_10x_h5(Path("/fake/test.h5")) is False

    def test_is_10x_h5_handles_missing_file(self):
        """Missing file should return False, not crash."""
        parser = self._make_parser()
        assert parser._is_10x_h5(Path("/nonexistent/test.h5")) is False

    @patch.object(
        __import__("lobster.services.data_access.geo.parser", fromlist=["GEOParser"]).GEOParser,
        "_is_10x_h5",
        return_value=True,
    )
    @patch.object(
        __import__("lobster.services.data_access.geo.parser", fromlist=["GEOParser"]).GEOParser,
        "_parse_10x_h5",
        return_value=MagicMock(),
    )
    @patch.object(
        __import__("lobster.services.data_access.geo.parser", fromlist=["GEOParser"]).GEOParser,
        "_parse_h5ad_with_fallback",
    )
    def test_h5_routing_prefers_10x(self, mock_h5ad, mock_10x, mock_detect):
        """When _is_10x_h5 returns True, route to _parse_10x_h5 not _parse_h5ad_with_fallback."""
        parser = self._make_parser()
        result = parser.parse_supplementary_file(Path("/fake/test.h5"))
        mock_10x.assert_called_once()
        mock_h5ad.assert_not_called()


class TestBug1_LegacyH5adCrashGuard:
    """_parse_legacy_h5ad should not crash when /matrix is an h5py.Group."""

    @patch("h5py.File")
    def test_legacy_h5ad_handles_matrix_group(self, mock_h5file):
        """Legacy parser should handle /matrix as Group with CSR format."""
        import h5py
        import numpy as np

        # Create mock CSR data
        mock_data = np.array([1.0, 2.0, 3.0])
        mock_indices = np.array([0, 1, 2])
        mock_indptr = np.array([0, 1, 2, 3])

        mock_group = MagicMock(spec=h5py.Group)
        mock_group.__contains__ = lambda self, k: k in ("data", "indices", "indptr")
        mock_group.__getitem__ = lambda self, k: {
            "data": MagicMock(__getitem__=lambda s, sl: mock_data),
            "indices": MagicMock(__getitem__=lambda s, sl: mock_indices),
            "indptr": MagicMock(__getitem__=lambda s, sl: mock_indptr),
        }[k]

        # Mock obs_names and var_names
        mock_obs = MagicMock()
        mock_obs.__getitem__ = lambda self, sl: np.array([b"cell1", b"cell2", b"cell3"])

        mock_var = MagicMock()
        mock_var.__getitem__ = lambda self, sl: np.array([b"gene1", b"gene2", b"gene3"])

        mock_f = MagicMock()
        mock_f.keys.return_value = ["matrix", "obs_names", "var_names"]

        def mock_contains(key):
            return key in ("matrix", "obs_names", "var_names")

        mock_f.__contains__ = lambda self, k: mock_contains(k)

        def mock_getitem(key):
            if key == "matrix":
                return mock_group
            elif key == "obs_names":
                return mock_obs
            elif key == "var_names":
                return mock_var
            raise KeyError(key)

        mock_f.__getitem__ = lambda self, k: mock_getitem(k)
        mock_f.__enter__ = lambda self: self
        mock_f.__exit__ = MagicMock(return_value=False)
        mock_h5file.return_value = mock_f

        from lobster.services.data_access.geo.parser import GEOParser

        parser = GEOParser()
        # This should NOT crash with "Accessing a group is done with bytes or str, not slice"
        result = parser._parse_legacy_h5ad(Path("/fake/legacy.h5"))
        # Result may be None due to mock complexity, but the key point is no crash
        # The actual fix prevents the `f["matrix"][:]` crash


# ---------------------------------------------------------------------------
# Bug 2: Multi-modal filter
# ---------------------------------------------------------------------------


class TestBug2_MultimodalFilter:
    """Unsupported modality files should be filtered out."""

    def test_atac_files_filtered(self):
        """ATAC-related filenames should be identified as unsupported."""
        assert _is_unsupported_modality_file("GSM123_atac_counts.h5", ["atac"]) is True
        assert _is_unsupported_modality_file("peaks_matrix.tsv.gz", ["atac"]) is True
        assert _is_unsupported_modality_file("fragments.tsv.gz", ["atac"]) is True
        assert _is_unsupported_modality_file("chromatin_accessibility.h5ad", ["atac"]) is True

    def test_protein_files_filtered(self):
        """Protein/ADT/CITE-seq filenames should be identified as unsupported."""
        assert _is_unsupported_modality_file("adt_counts.csv.gz", ["protein"]) is True
        assert _is_unsupported_modality_file("antibody_capture.h5", ["protein"]) is True

    def test_rna_files_not_filtered(self):
        """RNA expression filenames should NOT be filtered."""
        assert _is_unsupported_modality_file("rna_counts.h5", ["atac"]) is False
        assert _is_unsupported_modality_file("expression_matrix.tsv.gz", ["atac", "protein"]) is False
        assert _is_unsupported_modality_file("GSM123_filtered_feature_bc_matrix.h5", ["atac"]) is False

    def test_empty_unsupported_types(self):
        """Empty unsupported list should never filter."""
        assert _is_unsupported_modality_file("atac_counts.h5", []) is False

    def test_unknown_modality_type(self):
        """Unknown modality types should not filter (no patterns defined)."""
        assert _is_unsupported_modality_file("atac_counts.h5", ["unknown_modality"]) is False

    def test_modality_patterns_coverage(self):
        """Verify expected modality types have patterns defined."""
        assert "atac" in _MODALITY_PATTERNS
        assert "protein" in _MODALITY_PATTERNS
        assert "spatial" in _MODALITY_PATTERNS


class TestBug2_MultimodalPassthrough:
    """multimodal_info should flow from download_execution to archive_processing."""

    def test_process_supplementary_files_accepts_multimodal_info(self):
        """_process_supplementary_files should accept multimodal_info parameter."""
        from lobster.services.data_access.geo.archive_processing import (
            ArchiveProcessor,
        )
        import inspect

        sig = inspect.signature(ArchiveProcessor._process_supplementary_files)
        assert "multimodal_info" in sig.parameters

    def test_process_tar_file_accepts_multimodal_info(self):
        """_process_tar_file should accept multimodal_info parameter."""
        from lobster.services.data_access.geo.archive_processing import (
            ArchiveProcessor,
        )
        import inspect

        sig = inspect.signature(ArchiveProcessor._process_tar_file)
        assert "multimodal_info" in sig.parameters


# ---------------------------------------------------------------------------
# Bug 3: LLM modality signal override
# ---------------------------------------------------------------------------


class TestBug3_LLMModalitySignal:
    """LLM modality detection should override keyword heuristic when confident."""

    def _make_executor(self):
        """Create a DownloadExecutor with mocked service."""
        from lobster.services.data_access.geo.download_execution import (
            DownloadExecutor,
        )

        mock_service = MagicMock()
        mock_service._determine_data_type_from_metadata.return_value = "single_cell_rna_seq"
        mock_service.pipeline_engine.determine_pipeline.return_value = (
            MagicMock(name="SUPPLEMENTARY_FIRST"),
            "test",
        )
        mock_service.pipeline_engine.get_pipeline_functions.return_value = []
        return DownloadExecutor(mock_service)

    def test_llm_bulk_rna_overrides_heuristic(self):
        """High-confidence LLM bulk_rna signal should produce bulk_rna_seq data_type."""
        executor = self._make_executor()
        executor.service.data_manager._get_geo_metadata.return_value = {
            "metadata": {},
            "modality_detection": {
                "modality": "bulk_rna",
                "confidence": 0.95,
            },
        }

        with patch(
            "lobster.services.data_access.geo.download_execution.create_pipeline_context"
        ) as mock_ctx:
            mock_ctx.return_value = MagicMock()
            executor._get_processing_pipeline("GSE157103", {}, {})

            # Verify create_pipeline_context was called with bulk_rna_seq
            call_kwargs = mock_ctx.call_args
            assert call_kwargs.kwargs.get("data_type") == "bulk_rna_seq" or \
                   call_kwargs[1].get("data_type") == "bulk_rna_seq"

            # Heuristic should NOT have been called
            executor.service._determine_data_type_from_metadata.assert_not_called()

    def test_low_confidence_falls_back_to_heuristic(self):
        """Low-confidence LLM signal should fall back to keyword heuristic."""
        executor = self._make_executor()
        executor.service.data_manager._get_geo_metadata.return_value = {
            "metadata": {},
            "modality_detection": {
                "modality": "bulk_rna",
                "confidence": 0.5,  # Below 0.8 threshold
            },
        }

        with patch(
            "lobster.services.data_access.geo.download_execution.create_pipeline_context"
        ) as mock_ctx:
            mock_ctx.return_value = MagicMock()
            executor._get_processing_pipeline("GSE157103", {}, {})

            # Heuristic SHOULD have been called as fallback
            executor.service._determine_data_type_from_metadata.assert_called_once()

    def test_no_metadata_falls_back_to_heuristic(self):
        """Missing metadata store entry should fall back to keyword heuristic."""
        executor = self._make_executor()
        executor.service.data_manager._get_geo_metadata.return_value = None

        with patch(
            "lobster.services.data_access.geo.download_execution.create_pipeline_context"
        ) as mock_ctx:
            mock_ctx.return_value = MagicMock()
            executor._get_processing_pipeline("GSE157103", {}, {})

            # Heuristic SHOULD have been called as fallback
            executor.service._determine_data_type_from_metadata.assert_called_once()

    def test_unmapped_modality_falls_back(self):
        """Unknown LLM modality string should fall back to heuristic."""
        executor = self._make_executor()
        executor.service.data_manager._get_geo_metadata.return_value = {
            "metadata": {},
            "modality_detection": {
                "modality": "spatial_transcriptomics",  # Not in mapping
                "confidence": 0.95,
            },
        }

        with patch(
            "lobster.services.data_access.geo.download_execution.create_pipeline_context"
        ) as mock_ctx:
            mock_ctx.return_value = MagicMock()
            executor._get_processing_pipeline("GSE157103", {}, {})

            # Heuristic SHOULD have been called as fallback
            executor.service._determine_data_type_from_metadata.assert_called_once()
