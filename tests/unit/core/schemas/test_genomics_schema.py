"""
Unit tests for genomics schema validation.

Tests cover WGS chromosome name validation via GenomicsSchema.create_validator().
"""

import anndata as ad
import numpy as np
import pytest

from lobster.core.schemas.genomics import GenomicsSchema

STANDARD_CHROMOSOMES = [
    "1",
    "22",
    "X",
    "Y",
    "M",
    "MT",
    "chr1",
    "chr2",
    "chr22",
    "chrX",
    "chrY",
    "chrM",
    "chrMT",
]

NON_STANDARD_CHROMOSOME_WARNING = "non-standard chromosome names"


def _chromosome_warnings(warnings: list[str]) -> list[str]:
    """Return warnings related to chromosome naming."""
    return [w for w in warnings if NON_STANDARD_CHROMOSOME_WARNING in w]


def _make_wgs_adata(chrom: str) -> ad.AnnData:
    """Build minimal WGS AnnData for validator integration tests."""
    adata = ad.AnnData(X=np.zeros((1, 1)))
    adata.var["CHROM"] = [chrom]
    adata.var["POS"] = [100]
    adata.var["REF"] = ["A"]
    adata.var["ALT"] = ["G"]
    adata.layers["GT"] = np.zeros((1, 1), dtype=int)
    return adata


@pytest.fixture
def wgs_validator():
    """WGS validator with genomics custom rules registered."""
    return GenomicsSchema.create_validator("wgs")


@pytest.mark.unit
class TestWGSChromosomeFormatValidation:
    """Test chromosome naming via the WGS schema validator."""

    @pytest.mark.parametrize("chrom", STANDARD_CHROMOSOMES)
    def test_standard_chromosomes_do_not_warn(self, wgs_validator, chrom):
        """Standard chromosome names should not trigger non-standard warnings."""
        adata = _make_wgs_adata(chrom)

        result = wgs_validator.validate(adata)

        assert _chromosome_warnings(result.warnings) == []

    def test_non_standard_chromosome_warns(self, wgs_validator):
        """Truly non-standard chromosome names should produce a warning."""
        adata = _make_wgs_adata("chrUn")

        result = wgs_validator.validate(adata)

        chromosome_warnings = _chromosome_warnings(result.warnings)
        assert len(chromosome_warnings) == 1
        assert "chrUn" in chromosome_warnings[0]
