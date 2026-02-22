"""
Integration tests for UniProtService with live API calls.

These tests hit the real UniProt REST API and should only run when
explicitly selected:
    pytest tests/integration/services/data_access/test_uniprot_integration.py -v -m real_api
"""

import pytest

from lobster.services.data_access.uniprot_service import (
    UniProtNotFoundError,
    UniProtService,
)

pytestmark = [pytest.mark.real_api, pytest.mark.integration]


@pytest.fixture(scope="module")
def service():
    """Shared service instance for all tests in this module."""
    return UniProtService(timeout=30)


# =========================================================================
# get_protein
# =========================================================================


def test_get_protein_p04637(service):
    """P04637 = human TP53 protein, one of the most-studied proteins."""
    result = service.get_protein("P04637")

    assert result["primaryAccession"] == "P04637"
    assert "proteinDescription" in result
    assert "organism" in result
    assert result["organism"]["scientificName"] == "Homo sapiens"

    # Should have gene info
    genes = result.get("genes", [])
    assert len(genes) > 0
    assert genes[0]["geneName"]["value"] == "TP53"


def test_get_protein_not_found(service):
    """Invalid accession should raise UniProtNotFoundError."""
    with pytest.raises(UniProtNotFoundError):
        service.get_protein("INVALID_ACCESSION_999")


# =========================================================================
# search_proteins
# =========================================================================


def test_search_proteins_tp53_human(service):
    """Search for TP53 in human should return results."""
    result = service.search_proteins("TP53", max_results=5, organism="9606")

    results_list = result.get("results", [])
    assert len(results_list) > 0
    assert len(results_list) <= 5

    # First result should be related to TP53
    first = results_list[0]
    assert "primaryAccession" in first


def test_search_proteins_reviewed_only(service):
    """Filtering to reviewed (Swiss-Prot) should work."""
    result = service.search_proteins(
        "insulin", max_results=3, organism="9606", reviewed=True
    )
    results_list = result.get("results", [])
    assert len(results_list) > 0


# =========================================================================
# map_ids
# =========================================================================


def test_map_ids_gene_name_to_uniprot(service):
    """Map gene name TP53 to UniProt accession."""
    result = service.map_ids(
        from_db="Gene_Name",
        to_db="UniProtKB",
        ids=["TP53"],
    )

    results_list = result.get("results", [])
    assert len(results_list) > 0

    # Should find P04637 among results
    accessions = []
    for r in results_list:
        to_entry = r.get("to", {})
        if isinstance(to_entry, dict):
            accessions.append(to_entry.get("primaryAccession", ""))
        else:
            accessions.append(str(to_entry))

    assert "P04637" in accessions


def test_map_ids_empty_list(service):
    """Empty ID list should return empty results."""
    result = service.map_ids(
        from_db="Gene_Name", to_db="UniProtKB", ids=[]
    )
    assert result == {"results": []}
