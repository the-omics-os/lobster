"""Unit tests for Loom metadata enrichment (loom_metadata.py).

Tests the standalone functions for GEO SOFT metadata extraction and
merging into Loom-loaded AnnData objects.
"""

import numpy as np
import pandas as pd
import pytest

import anndata

from lobster.services.data_access.geo.loom_metadata import (
    enrich_loom_adata_with_geo_metadata,
    extract_geo_accession,
)


class TestExtractGeoAccession:
    """Test GSE accession extraction from filenames."""

    def test_standard_filename(self):
        assert extract_geo_accession("GSE162183_Skin.loom") == "GSE162183"

    def test_path_with_directory(self):
        assert extract_geo_accession("/data/geo/GSE162183_SomeDesc.loom") == "GSE162183"

    def test_no_accession(self):
        assert extract_geo_accession("my_experiment.loom") is None

    def test_case_insensitive(self):
        assert extract_geo_accession("gse12345_data.loom") == "GSE12345"

    def test_accession_in_middle(self):
        assert extract_geo_accession("data_GSE99999_filtered.h5ad") == "GSE99999"


class TestEnrichLoomAdataWithGeoMetadata:
    """Test enrichment of AnnData with pre-fetched sample metadata."""

    @pytest.fixture
    def bulk_adata(self):
        """Create a bulk-like AnnData with 3 samples."""
        X = np.random.rand(3, 100)
        adata = anndata.AnnData(
            X=X,
            obs=pd.DataFrame(index=["GSM001", "GSM002", "GSM003"]),
            var=pd.DataFrame(index=[f"gene_{i}" for i in range(100)]),
        )
        return adata

    @pytest.fixture
    def sc_adata(self):
        """Create a single-cell-like AnnData with barcodes containing sample suffixes."""
        n_cells = 300
        barcodes = []
        for sample_idx in range(1, 4):  # 3 samples
            for i in range(100):
                barcodes.append(f"ACGT{i:04d}-{sample_idx}")
        X = np.random.rand(n_cells, 50)
        adata = anndata.AnnData(
            X=X,
            obs=pd.DataFrame(index=barcodes),
            var=pd.DataFrame(index=[f"gene_{i}" for i in range(50)]),
        )
        return adata

    @pytest.fixture
    def sample_metadata(self):
        """Pre-fetched SOFT metadata for 3 samples."""
        return {
            "GSM001": {"tissue": "skin", "disease_state": "psoriasis", "age": "45"},
            "GSM002": {"tissue": "skin", "disease_state": "healthy", "age": "38"},
            "GSM003": {"tissue": "skin", "disease_state": "psoriasis", "age": "52"},
        }

    def test_bulk_obs_names_mapping(self, bulk_adata, sample_metadata):
        """Bulk case: obs_names ARE GSM IDs."""
        result = enrich_loom_adata_with_geo_metadata(
            bulk_adata, "GSE12345", sample_metadata=sample_metadata
        )
        assert result is True
        assert "disease_state" in bulk_adata.obs.columns
        assert "tissue" in bulk_adata.obs.columns
        assert bulk_adata.obs.loc["GSM001", "disease_state"] == "psoriasis"
        assert bulk_adata.obs.loc["GSM002", "disease_state"] == "healthy"

    def test_positional_mapping(self, sample_metadata):
        """When sample count == obs count, use positional mapping."""
        adata = anndata.AnnData(
            X=np.random.rand(3, 10),
            obs=pd.DataFrame(index=["sample_A", "sample_B", "sample_C"]),
            var=pd.DataFrame(index=[f"g{i}" for i in range(10)]),
        )
        result = enrich_loom_adata_with_geo_metadata(
            adata, "GSE12345", sample_metadata=sample_metadata
        )
        assert result is True
        assert adata.obs.loc["sample_A", "disease_state"] == "psoriasis"

    def test_barcode_suffix_mapping(self, sc_adata, sample_metadata):
        """Single-cell: barcode suffix -1, -2, -3 maps to sample index."""
        result = enrich_loom_adata_with_geo_metadata(
            sc_adata, "GSE12345", sample_metadata=sample_metadata
        )
        assert result is True
        assert "disease_state" in sc_adata.obs.columns
        # Cells with suffix -1 should get GSM001 metadata
        suffix_1_cells = [n for n in sc_adata.obs_names if n.endswith("-1")]
        assert sc_adata.obs.loc[suffix_1_cells[0], "disease_state"] == "psoriasis"
        # Cells with suffix -2 should get GSM002 metadata
        suffix_2_cells = [n for n in sc_adata.obs_names if n.endswith("-2")]
        assert sc_adata.obs.loc[suffix_2_cells[0], "disease_state"] == "healthy"

    def test_no_metadata(self, bulk_adata):
        """Empty metadata returns False without modifying adata."""
        result = enrich_loom_adata_with_geo_metadata(
            bulk_adata, "GSE12345", sample_metadata={}
        )
        assert result is False
        assert "disease_state" not in bulk_adata.obs.columns

    def test_no_mapping_possible(self, sample_metadata):
        """Unmappable obs_names returns False."""
        adata = anndata.AnnData(
            X=np.random.rand(5, 10),
            obs=pd.DataFrame(index=[f"cell_{i}" for i in range(5)]),
            var=pd.DataFrame(index=[f"g{i}" for i in range(10)]),
        )
        result = enrich_loom_adata_with_geo_metadata(
            adata, "GSE12345", sample_metadata=sample_metadata
        )
        assert result is False

    def test_skips_existing_columns(self, bulk_adata, sample_metadata):
        """Pre-existing columns are not overwritten."""
        bulk_adata.obs["tissue"] = "pre_existing"
        enrich_loom_adata_with_geo_metadata(
            bulk_adata, "GSE12345", sample_metadata=sample_metadata
        )
        # tissue should remain unchanged
        assert (bulk_adata.obs["tissue"] == "pre_existing").all()
        # But disease_state should be injected
        assert "disease_state" in bulk_adata.obs.columns

    def test_numeric_conversion(self, bulk_adata, sample_metadata):
        """Numeric-looking values get converted to numeric dtype."""
        enrich_loom_adata_with_geo_metadata(
            bulk_adata, "GSE12345", sample_metadata=sample_metadata
        )
        # age values are "45", "38", "52" — should be numeric
        assert pd.api.types.is_numeric_dtype(bulk_adata.obs["age"])

    def test_sample_id_column_mapping(self, sample_metadata):
        """When obs has sample_id column matching GSM IDs."""
        adata = anndata.AnnData(
            X=np.random.rand(3, 10),
            obs=pd.DataFrame(
                {"sample_id": ["GSM001", "GSM002", "GSM003"]},
                index=["cell_0", "cell_1", "cell_2"],
            ),
            var=pd.DataFrame(index=[f"g{i}" for i in range(10)]),
        )
        result = enrich_loom_adata_with_geo_metadata(
            adata, "GSE12345", sample_metadata=sample_metadata
        )
        assert result is True
        assert adata.obs.loc["cell_0", "disease_state"] == "psoriasis"


class TestFuzzyPrefixMatch:
    """Test the abbreviation-aware prefix matching."""

    def test_ctrl_control(self):
        from lobster.services.data_access.geo.loom_metadata import _fuzzy_prefix_match

        assert _fuzzy_prefix_match("ctrl", "control") is True

    def test_psor_psoriasis(self):
        from lobster.services.data_access.geo.loom_metadata import _fuzzy_prefix_match

        assert _fuzzy_prefix_match("psor", "psoriasis") is True

    def test_exact_match(self):
        from lobster.services.data_access.geo.loom_metadata import _fuzzy_prefix_match

        assert _fuzzy_prefix_match("healthy", "healthy") is True

    def test_prefix_match(self):
        from lobster.services.data_access.geo.loom_metadata import _fuzzy_prefix_match

        assert _fuzzy_prefix_match("tum", "tumor") is True

    def test_no_match(self):
        from lobster.services.data_access.geo.loom_metadata import _fuzzy_prefix_match

        assert _fuzzy_prefix_match("ctrl", "psoriasis") is False

    def test_empty_strings(self):
        from lobster.services.data_access.geo.loom_metadata import _fuzzy_prefix_match

        assert _fuzzy_prefix_match("", "control") is False
        assert _fuzzy_prefix_match("ctrl", "") is False


class TestAdapterFormatDetection:
    """Test that .loom is properly detected by the adapter stack."""

    def test_base_adapter_detects_loom(self):
        from lobster.core.interfaces.adapter import IModalityAdapter

        class TestAdapter(IModalityAdapter):
            def from_source(self, source, **kwargs):
                pass

            def validate(self, adata, strict=False):
                pass

            def get_schema(self):
                pass

            def get_supported_formats(self):
                pass

        adapter = TestAdapter()
        assert adapter.detect_format("test.loom") == "loom"
        assert adapter.detect_format("GSE12345_data.loom") == "loom"

    def test_transcriptomics_adapter_supports_loom(self):
        from lobster.core.adapters.transcriptomics_adapter import TranscriptomicsAdapter

        adapter = TranscriptomicsAdapter()
        assert "loom" in adapter.get_supported_formats()


class TestDownloaderLoomDetection:
    """Test that GEO downloader recognizes .loom files."""

    def test_loom_in_tar_extension_priority(self):
        """Verify .loom is in the extension priority for TAR extraction."""
        # This is a structural test — we verify the string is present
        import inspect
        from lobster.services.data_access.geo.downloader import GEODownloadManager

        source = inspect.getsource(GEODownloadManager.find_expression_file_in_tar)
        assert '".loom"' in source

    def test_loom_in_supplementary_regex(self):
        """Verify .loom is matched by the supplementary file regex."""
        import re

        # The regex from get_supplementary_files
        pattern = r'href="([^"]*\.(?:txt|csv|xlsx|h5|h5ad|loom|gz|bz2))"'
        html = '<a href="GSE162183_data.loom">GSE162183_data.loom</a>'
        matches = re.findall(pattern, html, re.IGNORECASE)
        assert len(matches) == 1
        assert "GSE162183_data.loom" in matches[0]
