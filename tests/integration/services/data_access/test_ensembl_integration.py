"""
Integration tests for EnsemblService with live API calls.

These tests hit the real Ensembl REST API and should only run when
explicitly selected:
    pytest tests/integration/services/data_access/test_ensembl_integration.py -v -m real_api
"""

import pytest

from lobster.services.data_access.ensembl_service import (
    EnsemblNotFoundError,
    EnsemblService,
)

pytestmark = [pytest.mark.real_api, pytest.mark.integration]


@pytest.fixture(scope="module")
def service():
    """Shared service instance for all tests in this module."""
    return EnsemblService(timeout=30)


# =========================================================================
# lookup_gene
# =========================================================================


def test_lookup_gene_by_ensembl_id(service):
    """ENSG00000141510 = TP53 gene."""
    result = service.lookup_gene("ENSG00000141510")

    assert result["id"] == "ENSG00000141510"
    assert result["display_name"] == "TP53"
    assert result["biotype"] == "protein_coding"
    assert result["species"] == "homo_sapiens"
    assert "seq_region_name" in result
    assert "start" in result
    assert "end" in result


def test_lookup_gene_by_symbol(service):
    """Look up TP53 by gene symbol for human."""
    result = service.lookup_gene("TP53", species="human")

    assert result["display_name"] == "TP53"
    assert result["species"] == "homo_sapiens"


def test_lookup_gene_species_normalization(service):
    """Taxonomy ID 9606 should resolve to homo_sapiens."""
    result = service.lookup_gene("TP53", species="9606")

    assert result["display_name"] == "TP53"
    assert result["species"] == "homo_sapiens"


def test_lookup_gene_not_found(service):
    """Invalid gene ID should raise EnsemblNotFoundError."""
    with pytest.raises(EnsemblNotFoundError):
        service.lookup_gene("ENSG99999999999")


def test_lookup_gene_with_expand(service):
    """Expand=True should include Transcript details."""
    result = service.lookup_gene("ENSG00000141510", expand=True)

    transcripts = result.get("Transcript", [])
    assert len(transcripts) > 0
    assert "id" in transcripts[0]
    assert transcripts[0]["id"].startswith("ENST")


# =========================================================================
# get_variant_consequences (VEP)
# =========================================================================


def test_vep_hgvs_notation(service):
    """Test VEP with HGVS genomic notation for a TP53 variant."""
    # TP53 missense variant (R248W)
    result = service.get_variant_consequences(
        notation="17:g.7674220C>T",
        species="human",
        notation_type="hgvs",
    )

    assert isinstance(result, list)
    assert len(result) > 0

    entry = result[0]
    assert "most_severe_consequence" in entry


def test_vep_rsid(service):
    """Test VEP with rsID notation."""
    result = service.get_variant_consequences(
        notation="rs1042522",
        species="human",
        notation_type="id",
    )

    assert isinstance(result, list)
    assert len(result) > 0


# =========================================================================
# get_sequence
# =========================================================================


def test_get_cdna_sequence(service):
    """Get cDNA sequence for TP53 transcript."""
    result = service.get_sequence("ENST00000269305", seq_type="cdna")

    assert "seq" in result
    assert len(result["seq"]) > 0
    assert result["id"] == "ENST00000269305"


def test_get_protein_sequence(service):
    """Get protein sequence for TP53."""
    result = service.get_sequence("ENSP00000269305", seq_type="protein")

    assert "seq" in result
    assert len(result["seq"]) > 0
    # Protein sequences should start with M (methionine)
    assert result["seq"].startswith("M")


# =========================================================================
# get_xrefs
# =========================================================================


def test_get_xrefs_all(service):
    """Get all cross-references for TP53 gene."""
    result = service.get_xrefs("ENSG00000141510")

    assert isinstance(result, list)
    assert len(result) > 0

    # Should have at least primary_id and dbname
    first = result[0]
    assert "primary_id" in first
    assert "dbname" in first


def test_get_xrefs_uniprot_filter(service):
    """Filter xrefs to UniProt/SWISSPROT only."""
    result = service.get_xrefs(
        "ENSG00000141510", external_db="UniProt/SWISSPROT"
    )

    assert isinstance(result, list)
    assert len(result) > 0

    # All results should be from UniProt
    for xref in result:
        assert "UniProt" in xref.get("dbname", "")
