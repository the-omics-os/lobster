"""
Unit tests for EnsemblService.

Tests the Ensembl REST API client covering:
- Gene/transcript lookup by Ensembl ID and gene symbol
- Variant consequence prediction (VEP) via hgvs, region, and id notation
- Sequence retrieval (genomic, cdna, cds, protein)
- Cross-database reference queries (xrefs)
- Species normalization (aliases and NCBI taxonomy IDs)
- Rate limit awareness via X-RateLimit-Remaining header
- Error handling (404, 429, 400, connection errors, timeouts)
- LRU cache behavior for lookup_gene and get_sequence

All HTTP calls are mocked — no network access required.

Running Tests:
```bash
pytest tests/unit/services/data_access/test_ensembl_service.py -v
```
"""

import time
from unittest.mock import MagicMock, patch

import pytest
import requests
from requests.adapters import HTTPAdapter

from lobster.services.data_access.ensembl_service import (
    BASE_URL,
    SPECIES_ALIASES,
    EnsemblNotFoundError,
    EnsemblRateLimitError,
    EnsemblService,
    EnsemblServiceError,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def service():
    """Create a fresh EnsemblService instance with caches cleared."""
    svc = EnsemblService(timeout=10)
    # Clear LRU caches to ensure test isolation
    svc._lookup_gene_cached.cache_clear()
    svc._get_sequence_cached.cache_clear()
    return svc


@pytest.fixture
def mock_response():
    """Factory fixture that creates a mock requests.Response with configurable attributes."""

    def _make_response(
        status_code=200,
        json_data=None,
        headers=None,
        url="https://rest.ensembl.org/test",
        text="",
        ok=None,
    ):
        resp = MagicMock(spec=requests.Response)
        resp.status_code = status_code
        resp.ok = ok if ok is not None else (200 <= status_code < 400)
        resp.url = url
        resp.text = text
        resp.headers = headers or {}
        resp.json.return_value = json_data if json_data is not None else {}
        return resp

    return _make_response


# =============================================================================
# Sample response data
# =============================================================================

GENE_LOOKUP_RESPONSE = {
    "id": "ENSG00000141510",
    "display_name": "TP53",
    "description": "tumor protein p53 [Source:HGNC Symbol;Acc:HGNC:11998]",
    "biotype": "protein_coding",
    "species": "homo_sapiens",
    "assembly_name": "GRCh38",
    "seq_region_name": "17",
    "start": 7661779,
    "end": 7687550,
    "strand": -1,
    "object_type": "Gene",
}

GENE_LOOKUP_EXPANDED_RESPONSE = {
    **GENE_LOOKUP_RESPONSE,
    "Transcript": [
        {
            "id": "ENST00000269305",
            "display_name": "TP53-201",
            "biotype": "protein_coding",
            "Exon": [
                {"id": "ENSE00001657961", "start": 7687377, "end": 7687550}
            ],
        }
    ],
}

SYMBOL_LOOKUP_RESPONSE = {
    "id": "ENSG00000141510",
    "display_name": "TP53",
    "description": "tumor protein p53",
    "biotype": "protein_coding",
    "species": "homo_sapiens",
    "object_type": "Gene",
}

VEP_HGVS_RESPONSE = [
    {
        "most_severe_consequence": "missense_variant",
        "transcript_consequences": [
            {
                "gene_id": "ENSG00000141510",
                "consequence_terms": ["missense_variant"],
                "amino_acids": "R/H",
                "impact": "MODERATE",
            }
        ],
    }
]

VEP_REGION_RESPONSE = [
    {
        "most_severe_consequence": "missense_variant",
        "input": "9:22125503-22125503:1/C",
    }
]

VEP_ID_RESPONSE = [
    {
        "most_severe_consequence": "missense_variant",
        "id": "rs1042522",
        "colocated_variants": [{"id": "rs1042522", "allele_string": "G/C"}],
    }
]

SEQUENCE_GENOMIC_RESPONSE = {
    "id": "ENSG00000141510",
    "seq": "ATGCGATCGATCGATCG" * 10,
    "molecule": "dna",
    "desc": "chromosome:GRCh38:17:7661779:7687550:-1",
}

SEQUENCE_CDNA_RESPONSE = {
    "id": "ENST00000269305",
    "seq": "ATGGAGGAGCCGCAGTCAGATCC",
    "molecule": "dna",
    "desc": "cdna:known",
}

SEQUENCE_CDS_RESPONSE = {
    "id": "ENST00000269305",
    "seq": "ATGGAGGAGCCGCAGTCAGATCC",
    "molecule": "dna",
    "desc": "cds:known",
}

SEQUENCE_PROTEIN_RESPONSE = {
    "id": "ENSP00000269305",
    "seq": "MEEPQSDPSVEPPLSQETFSDLWKLL",
    "molecule": "protein",
    "desc": "pep:known",
}

XREFS_RESPONSE = [
    {
        "primary_id": "P04637",
        "display_id": "P53_HUMAN",
        "dbname": "Uniprot/SWISSPROT",
        "description": "Cellular tumor antigen p53",
        "info_type": "DIRECT",
    },
    {
        "primary_id": "11998",
        "display_id": "TP53",
        "dbname": "HGNC",
        "description": "",
        "info_type": "DIRECT",
    },
    {
        "primary_id": "NM_000546",
        "display_id": "NM_000546",
        "dbname": "RefSeq_mRNA",
        "description": "",
        "info_type": "DEPENDENT",
    },
]

XREFS_FILTERED_RESPONSE = [
    {
        "primary_id": "P04637",
        "display_id": "P53_HUMAN",
        "dbname": "Uniprot/SWISSPROT",
        "description": "Cellular tumor antigen p53",
        "info_type": "DIRECT",
    },
]


# =============================================================================
# Tests: lookup_gene
# =============================================================================


def test_lookup_gene_with_ensembl_id(service, mock_response):
    """lookup_gene with an Ensembl stable ID (ENSG*) uses /lookup/id/ endpoint."""
    resp = mock_response(json_data=GENE_LOOKUP_RESPONSE)

    with patch.object(service._session, "request", return_value=resp) as mock_req:
        result = service.lookup_gene("ENSG00000141510")

    mock_req.assert_called_once_with(
        "GET",
        f"{BASE_URL}/lookup/id/ENSG00000141510",
        params={},
        timeout=10,
    )
    assert result["id"] == "ENSG00000141510"
    assert result["display_name"] == "TP53"
    assert result["biotype"] == "protein_coding"


def test_lookup_gene_with_gene_symbol(service, mock_response):
    """lookup_gene with a gene symbol (e.g. TP53) uses /lookup/symbol/ endpoint."""
    resp = mock_response(json_data=SYMBOL_LOOKUP_RESPONSE)

    with patch.object(service._session, "request", return_value=resp) as mock_req:
        result = service.lookup_gene("TP53", species="human")

    mock_req.assert_called_once_with(
        "GET",
        f"{BASE_URL}/lookup/symbol/homo_sapiens/TP53",
        params={},
        timeout=10,
    )
    assert result["id"] == "ENSG00000141510"
    assert result["display_name"] == "TP53"


def test_lookup_gene_with_expand(service, mock_response):
    """lookup_gene with expand=True includes transcripts and exons."""
    resp = mock_response(json_data=GENE_LOOKUP_EXPANDED_RESPONSE)

    with patch.object(service._session, "request", return_value=resp) as mock_req:
        result = service.lookup_gene("ENSG00000141510", expand=True)

    mock_req.assert_called_once_with(
        "GET",
        f"{BASE_URL}/lookup/id/ENSG00000141510",
        params={"expand": "1"},
        timeout=10,
    )
    assert "Transcript" in result
    assert len(result["Transcript"]) == 1
    assert result["Transcript"][0]["id"] == "ENST00000269305"
    assert "Exon" in result["Transcript"][0]


def test_lookup_gene_strips_whitespace(service, mock_response):
    """lookup_gene strips leading/trailing whitespace from identifier."""
    resp = mock_response(json_data=GENE_LOOKUP_RESPONSE)

    with patch.object(service._session, "request", return_value=resp) as mock_req:
        service.lookup_gene("  ENSG00000141510  ")

    mock_req.assert_called_once_with(
        "GET",
        f"{BASE_URL}/lookup/id/ENSG00000141510",
        params={},
        timeout=10,
    )


def test_lookup_gene_symbol_expand(service, mock_response):
    """lookup_gene with gene symbol and expand=True passes expand param."""
    resp = mock_response(json_data=GENE_LOOKUP_EXPANDED_RESPONSE)

    with patch.object(service._session, "request", return_value=resp) as mock_req:
        result = service.lookup_gene("TP53", species="homo_sapiens", expand=True)

    mock_req.assert_called_once_with(
        "GET",
        f"{BASE_URL}/lookup/symbol/homo_sapiens/TP53",
        params={"expand": "1"},
        timeout=10,
    )
    assert "Transcript" in result


def test_lookup_gene_transcript_id(service, mock_response):
    """lookup_gene with ENST* ID uses /lookup/id/ endpoint (auto-detect)."""
    transcript_response = {
        "id": "ENST00000269305",
        "display_name": "TP53-201",
        "object_type": "Transcript",
    }
    resp = mock_response(json_data=transcript_response)

    with patch.object(service._session, "request", return_value=resp) as mock_req:
        result = service.lookup_gene("ENST00000269305")

    mock_req.assert_called_once_with(
        "GET",
        f"{BASE_URL}/lookup/id/ENST00000269305",
        params={},
        timeout=10,
    )
    assert result["object_type"] == "Transcript"


def test_lookup_gene_protein_id(service, mock_response):
    """lookup_gene with ENSP* ID uses /lookup/id/ endpoint (auto-detect)."""
    protein_response = {
        "id": "ENSP00000269305",
        "object_type": "Translation",
    }
    resp = mock_response(json_data=protein_response)

    with patch.object(service._session, "request", return_value=resp) as mock_req:
        result = service.lookup_gene("ENSP00000269305")

    mock_req.assert_called_once_with(
        "GET",
        f"{BASE_URL}/lookup/id/ENSP00000269305",
        params={},
        timeout=10,
    )
    assert result["object_type"] == "Translation"


# =============================================================================
# Tests: get_variant_consequences
# =============================================================================


def test_get_variant_consequences_hgvs(service, mock_response):
    """get_variant_consequences with hgvs notation type uses /vep/{species}/hgvs/ endpoint."""
    resp = mock_response(json_data=VEP_HGVS_RESPONSE)

    with patch.object(service._session, "request", return_value=resp) as mock_req:
        result = service.get_variant_consequences(
            "9:g.22125503G>C", species="human", notation_type="hgvs"
        )

    mock_req.assert_called_once_with(
        "GET",
        f"{BASE_URL}/vep/homo_sapiens/hgvs/9:g.22125503G>C",
        params=None,
        timeout=10,
    )
    assert len(result) == 1
    assert result[0]["most_severe_consequence"] == "missense_variant"
    assert result[0]["transcript_consequences"][0]["amino_acids"] == "R/H"


def test_get_variant_consequences_region(service, mock_response):
    """get_variant_consequences with region notation type uses /vep/{species}/region/ endpoint."""
    resp = mock_response(json_data=VEP_REGION_RESPONSE)

    with patch.object(service._session, "request", return_value=resp) as mock_req:
        result = service.get_variant_consequences(
            "9:22125503-22125503:1/C", species="homo_sapiens", notation_type="region"
        )

    mock_req.assert_called_once_with(
        "GET",
        f"{BASE_URL}/vep/homo_sapiens/region/9:22125503-22125503:1/C",
        params=None,
        timeout=10,
    )
    assert len(result) == 1
    assert result[0]["input"] == "9:22125503-22125503:1/C"


def test_get_variant_consequences_id(service, mock_response):
    """get_variant_consequences with id notation type uses /vep/{species}/id/ endpoint for rsIDs."""
    resp = mock_response(json_data=VEP_ID_RESPONSE)

    with patch.object(service._session, "request", return_value=resp) as mock_req:
        result = service.get_variant_consequences(
            "rs1042522", species="human", notation_type="id"
        )

    mock_req.assert_called_once_with(
        "GET",
        f"{BASE_URL}/vep/homo_sapiens/id/rs1042522",
        params=None,
        timeout=10,
    )
    assert len(result) == 1
    assert result[0]["id"] == "rs1042522"
    assert result[0]["colocated_variants"][0]["allele_string"] == "G/C"


def test_get_variant_consequences_invalid_notation_type(service):
    """get_variant_consequences raises EnsemblServiceError for invalid notation_type."""
    with pytest.raises(EnsemblServiceError, match="Invalid notation_type: vcf"):
        service.get_variant_consequences(
            "9:g.22125503G>C", notation_type="vcf"
        )


def test_get_variant_consequences_strips_whitespace(service, mock_response):
    """get_variant_consequences strips whitespace from notation."""
    resp = mock_response(json_data=VEP_HGVS_RESPONSE)

    with patch.object(service._session, "request", return_value=resp) as mock_req:
        service.get_variant_consequences("  9:g.22125503G>C  ")

    mock_req.assert_called_once_with(
        "GET",
        f"{BASE_URL}/vep/homo_sapiens/hgvs/9:g.22125503G>C",
        params=None,
        timeout=10,
    )


def test_get_variant_consequences_default_notation_type(service, mock_response):
    """get_variant_consequences defaults to hgvs notation_type."""
    resp = mock_response(json_data=VEP_HGVS_RESPONSE)

    with patch.object(service._session, "request", return_value=resp) as mock_req:
        service.get_variant_consequences("ENST00000269305.9:c.817C>T")

    mock_req.assert_called_once_with(
        "GET",
        f"{BASE_URL}/vep/homo_sapiens/hgvs/ENST00000269305.9:c.817C>T",
        params=None,
        timeout=10,
    )


# =============================================================================
# Tests: get_sequence
# =============================================================================


def test_get_sequence_genomic(service, mock_response):
    """get_sequence with seq_type='genomic' retrieves genomic DNA sequence."""
    resp = mock_response(json_data=SEQUENCE_GENOMIC_RESPONSE)

    with patch.object(service._session, "request", return_value=resp) as mock_req:
        result = service.get_sequence("ENSG00000141510", seq_type="genomic")

    mock_req.assert_called_once_with(
        "GET",
        f"{BASE_URL}/sequence/id/ENSG00000141510",
        params={"type": "genomic"},
        timeout=10,
    )
    assert result["id"] == "ENSG00000141510"
    assert result["molecule"] == "dna"
    assert len(result["seq"]) > 0


def test_get_sequence_cdna(service, mock_response):
    """get_sequence with seq_type='cdna' retrieves complementary DNA sequence."""
    resp = mock_response(json_data=SEQUENCE_CDNA_RESPONSE)

    with patch.object(service._session, "request", return_value=resp) as mock_req:
        result = service.get_sequence("ENST00000269305", seq_type="cdna")

    mock_req.assert_called_once_with(
        "GET",
        f"{BASE_URL}/sequence/id/ENST00000269305",
        params={"type": "cdna"},
        timeout=10,
    )
    assert result["id"] == "ENST00000269305"
    assert result["molecule"] == "dna"


def test_get_sequence_cds(service, mock_response):
    """get_sequence with seq_type='cds' retrieves coding sequence."""
    resp = mock_response(json_data=SEQUENCE_CDS_RESPONSE)

    with patch.object(service._session, "request", return_value=resp) as mock_req:
        result = service.get_sequence("ENST00000269305", seq_type="cds")

    mock_req.assert_called_once_with(
        "GET",
        f"{BASE_URL}/sequence/id/ENST00000269305",
        params={"type": "cds"},
        timeout=10,
    )
    assert result["desc"] == "cds:known"


def test_get_sequence_protein(service, mock_response):
    """get_sequence with seq_type='protein' retrieves amino acid sequence."""
    resp = mock_response(json_data=SEQUENCE_PROTEIN_RESPONSE)

    with patch.object(service._session, "request", return_value=resp) as mock_req:
        result = service.get_sequence("ENSP00000269305", seq_type="protein")

    mock_req.assert_called_once_with(
        "GET",
        f"{BASE_URL}/sequence/id/ENSP00000269305",
        params={"type": "protein"},
        timeout=10,
    )
    assert result["molecule"] == "protein"
    assert result["seq"].startswith("MEEPQ")


def test_get_sequence_invalid_seq_type(service):
    """get_sequence raises EnsemblServiceError for unsupported seq_type."""
    with pytest.raises(EnsemblServiceError, match="Invalid seq_type: mrna"):
        service.get_sequence("ENSG00000141510", seq_type="mrna")


def test_get_sequence_strips_whitespace(service, mock_response):
    """get_sequence strips whitespace from ensembl_id."""
    resp = mock_response(json_data=SEQUENCE_GENOMIC_RESPONSE)

    with patch.object(service._session, "request", return_value=resp) as mock_req:
        service.get_sequence("  ENSG00000141510  ", seq_type="genomic")

    mock_req.assert_called_once_with(
        "GET",
        f"{BASE_URL}/sequence/id/ENSG00000141510",
        params={"type": "genomic"},
        timeout=10,
    )


def test_get_sequence_default_type_is_genomic(service, mock_response):
    """get_sequence defaults to seq_type='genomic' when not specified."""
    resp = mock_response(json_data=SEQUENCE_GENOMIC_RESPONSE)

    with patch.object(service._session, "request", return_value=resp) as mock_req:
        service.get_sequence("ENSG00000141510")

    mock_req.assert_called_once_with(
        "GET",
        f"{BASE_URL}/sequence/id/ENSG00000141510",
        params={"type": "genomic"},
        timeout=10,
    )


# =============================================================================
# Tests: get_xrefs
# =============================================================================


def test_get_xrefs_no_filter(service, mock_response):
    """get_xrefs without external_db returns all cross-references."""
    resp = mock_response(json_data=XREFS_RESPONSE)

    with patch.object(service._session, "request", return_value=resp) as mock_req:
        result = service.get_xrefs("ENSG00000141510")

    mock_req.assert_called_once_with(
        "GET",
        f"{BASE_URL}/xrefs/id/ENSG00000141510",
        params={},
        timeout=10,
    )
    assert len(result) == 3
    db_names = [x["dbname"] for x in result]
    assert "Uniprot/SWISSPROT" in db_names
    assert "HGNC" in db_names
    assert "RefSeq_mRNA" in db_names


def test_get_xrefs_with_external_db_filter(service, mock_response):
    """get_xrefs with external_db filter restricts results to a single database."""
    resp = mock_response(json_data=XREFS_FILTERED_RESPONSE)

    with patch.object(service._session, "request", return_value=resp) as mock_req:
        result = service.get_xrefs("ENSG00000141510", external_db="UniProt/SWISSPROT")

    mock_req.assert_called_once_with(
        "GET",
        f"{BASE_URL}/xrefs/id/ENSG00000141510",
        params={"external_db": "UniProt/SWISSPROT"},
        timeout=10,
    )
    assert len(result) == 1
    assert result[0]["primary_id"] == "P04637"
    assert result[0]["dbname"] == "Uniprot/SWISSPROT"


def test_get_xrefs_strips_whitespace(service, mock_response):
    """get_xrefs strips whitespace from ensembl_id."""
    resp = mock_response(json_data=XREFS_RESPONSE)

    with patch.object(service._session, "request", return_value=resp) as mock_req:
        service.get_xrefs("  ENSG00000141510  ")

    mock_req.assert_called_once_with(
        "GET",
        f"{BASE_URL}/xrefs/id/ENSG00000141510",
        params={},
        timeout=10,
    )


# =============================================================================
# Tests: species normalization
# =============================================================================


def test_normalize_species_human_alias(service, mock_response):
    """Species alias 'human' normalizes to 'homo_sapiens'."""
    resp = mock_response(json_data=SYMBOL_LOOKUP_RESPONSE)

    with patch.object(service._session, "request", return_value=resp) as mock_req:
        service.lookup_gene("TP53", species="human")

    url_called = mock_req.call_args[0][1]
    assert "/homo_sapiens/" in url_called


def test_normalize_species_mouse_alias(service, mock_response):
    """Species alias 'mouse' normalizes to 'mus_musculus'."""
    resp = mock_response(json_data=SYMBOL_LOOKUP_RESPONSE)

    with patch.object(service._session, "request", return_value=resp) as mock_req:
        service.lookup_gene("Trp53", species="mouse")

    url_called = mock_req.call_args[0][1]
    assert "/mus_musculus/" in url_called


def test_normalize_species_ncbi_taxonomy_id_human(service, mock_response):
    """NCBI taxonomy ID '9606' normalizes to 'homo_sapiens'."""
    resp = mock_response(json_data=SYMBOL_LOOKUP_RESPONSE)

    with patch.object(service._session, "request", return_value=resp) as mock_req:
        service.lookup_gene("TP53", species="9606")

    url_called = mock_req.call_args[0][1]
    assert "/homo_sapiens/" in url_called


def test_normalize_species_ncbi_taxonomy_id_mouse(service, mock_response):
    """NCBI taxonomy ID '10090' normalizes to 'mus_musculus'."""
    resp = mock_response(json_data=SYMBOL_LOOKUP_RESPONSE)

    with patch.object(service._session, "request", return_value=resp) as mock_req:
        service.lookup_gene("Trp53", species="10090")

    url_called = mock_req.call_args[0][1]
    assert "/mus_musculus/" in url_called


def test_normalize_species_case_insensitive(service, mock_response):
    """Species normalization is case-insensitive."""
    resp = mock_response(json_data=SYMBOL_LOOKUP_RESPONSE)

    with patch.object(service._session, "request", return_value=resp) as mock_req:
        service.lookup_gene("TP53", species="Human")

    url_called = mock_req.call_args[0][1]
    assert "/homo_sapiens/" in url_called


def test_normalize_species_already_normalized(service, mock_response):
    """Already-normalized species name passes through unchanged."""
    resp = mock_response(json_data=SYMBOL_LOOKUP_RESPONSE)

    with patch.object(service._session, "request", return_value=resp) as mock_req:
        service.lookup_gene("TP53", species="homo_sapiens")

    url_called = mock_req.call_args[0][1]
    assert "/homo_sapiens/" in url_called


def test_normalize_species_strips_whitespace(service, mock_response):
    """Species normalization strips whitespace."""
    resp = mock_response(json_data=SYMBOL_LOOKUP_RESPONSE)

    with patch.object(service._session, "request", return_value=resp) as mock_req:
        service.lookup_gene("TP53", species="  human  ")

    url_called = mock_req.call_args[0][1]
    assert "/homo_sapiens/" in url_called


def test_normalize_species_unknown_passes_through(service, mock_response):
    """Unknown species name passes through as-is (lowered)."""
    resp = mock_response(json_data=SYMBOL_LOOKUP_RESPONSE)

    with patch.object(service._session, "request", return_value=resp) as mock_req:
        service.lookup_gene("gene1", species="Arabidopsis_Thaliana")

    url_called = mock_req.call_args[0][1]
    assert "/arabidopsis_thaliana/" in url_called


def test_normalize_species_all_aliases_covered():
    """All SPECIES_ALIASES entries are correctly mapped."""
    svc = EnsemblService.__new__(EnsemblService)
    assert svc._normalize_species("human") == "homo_sapiens"
    assert svc._normalize_species("mouse") == "mus_musculus"
    assert svc._normalize_species("rat") == "rattus_norvegicus"
    assert svc._normalize_species("zebrafish") == "danio_rerio"
    assert svc._normalize_species("fly") == "drosophila_melanogaster"
    assert svc._normalize_species("worm") == "caenorhabditis_elegans"
    assert svc._normalize_species("yeast") == "saccharomyces_cerevisiae"
    assert svc._normalize_species("chicken") == "gallus_gallus"
    assert svc._normalize_species("pig") == "sus_scrofa"
    assert svc._normalize_species("dog") == "canis_lupus_familiaris"
    assert svc._normalize_species("9606") == "homo_sapiens"
    assert svc._normalize_species("10090") == "mus_musculus"
    assert svc._normalize_species("10116") == "rattus_norvegicus"
    assert svc._normalize_species("7955") == "danio_rerio"
    assert svc._normalize_species("7227") == "drosophila_melanogaster"
    assert svc._normalize_species("6239") == "caenorhabditis_elegans"


def test_normalize_species_used_by_vep(service, mock_response):
    """get_variant_consequences applies species normalization."""
    resp = mock_response(json_data=VEP_HGVS_RESPONSE)

    with patch.object(service._session, "request", return_value=resp) as mock_req:
        service.get_variant_consequences("9:g.22125503G>C", species="9606")

    url_called = mock_req.call_args[0][1]
    assert "/vep/homo_sapiens/" in url_called


# =============================================================================
# Tests: rate limit handling
# =============================================================================


def test_rate_limit_no_header_no_sleep(service, mock_response):
    """No sleep when X-RateLimit-Remaining header is absent."""
    resp = mock_response(json_data=GENE_LOOKUP_RESPONSE, headers={})

    with patch.object(service._session, "request", return_value=resp):
        with patch("lobster.services.data_access.ensembl_service.time.sleep") as mock_sleep:
            service.lookup_gene("ENSG00000141510")

    mock_sleep.assert_not_called()


def test_rate_limit_plenty_remaining_no_sleep(service, mock_response):
    """No sleep when plenty of rate limit budget remains."""
    resp = mock_response(
        json_data=GENE_LOOKUP_RESPONSE,
        headers={"X-RateLimit-Remaining": "50"},
    )

    with patch.object(service._session, "request", return_value=resp):
        with patch("lobster.services.data_access.ensembl_service.time.sleep") as mock_sleep:
            service.lookup_gene("ENSG00000141510")

    mock_sleep.assert_not_called()


def test_rate_limit_exactly_one_remaining_triggers_sleep(service, mock_response):
    """Sleep triggered when X-RateLimit-Remaining is exactly 1."""
    resp = mock_response(
        json_data=GENE_LOOKUP_RESPONSE,
        headers={
            "X-RateLimit-Remaining": "1",
            "X-RateLimit-Reset": "2",
        },
    )

    with patch.object(service._session, "request", return_value=resp):
        with patch("lobster.services.data_access.ensembl_service.time.sleep") as mock_sleep:
            service.lookup_gene("ENSG00000141510")

    mock_sleep.assert_called_once_with(2.0)


def test_rate_limit_zero_remaining_triggers_sleep(service, mock_response):
    """Sleep triggered when X-RateLimit-Remaining is 0."""
    resp = mock_response(
        json_data=GENE_LOOKUP_RESPONSE,
        headers={
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": "5",
        },
    )

    with patch.object(service._session, "request", return_value=resp):
        with patch("lobster.services.data_access.ensembl_service.time.sleep") as mock_sleep:
            service.lookup_gene("ENSG00000141510")

    mock_sleep.assert_called_once_with(5.0)


def test_rate_limit_sleep_capped_at_ten_seconds(service, mock_response):
    """Sleep time is capped at 10 seconds even if X-RateLimit-Reset is higher."""
    resp = mock_response(
        json_data=GENE_LOOKUP_RESPONSE,
        headers={
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": "60",
        },
    )

    with patch.object(service._session, "request", return_value=resp):
        with patch("lobster.services.data_access.ensembl_service.time.sleep") as mock_sleep:
            service.lookup_gene("ENSG00000141510")

    mock_sleep.assert_called_once_with(10.0)


def test_rate_limit_missing_reset_header_defaults_to_one(service, mock_response):
    """When X-RateLimit-Reset header is missing, defaults to sleeping 1 second."""
    resp = mock_response(
        json_data=GENE_LOOKUP_RESPONSE,
        headers={"X-RateLimit-Remaining": "0"},
    )

    with patch.object(service._session, "request", return_value=resp):
        with patch("lobster.services.data_access.ensembl_service.time.sleep") as mock_sleep:
            service.lookup_gene("ENSG00000141510")

    mock_sleep.assert_called_once_with(1.0)


def test_rate_limit_malformed_remaining_header_ignored(service, mock_response):
    """Malformed X-RateLimit-Remaining header is silently ignored."""
    resp = mock_response(
        json_data=GENE_LOOKUP_RESPONSE,
        headers={"X-RateLimit-Remaining": "not_a_number"},
    )

    with patch.object(service._session, "request", return_value=resp):
        with patch("lobster.services.data_access.ensembl_service.time.sleep") as mock_sleep:
            service.lookup_gene("ENSG00000141510")

    mock_sleep.assert_not_called()


# =============================================================================
# Tests: error handling
# =============================================================================


def test_error_404_raises_not_found(service, mock_response):
    """404 response raises EnsemblNotFoundError with URL in message."""
    resp = mock_response(
        status_code=404,
        url=f"{BASE_URL}/lookup/id/ENSG99999999999",
    )

    with patch.object(service._session, "request", return_value=resp):
        with pytest.raises(EnsemblNotFoundError, match="Not found"):
            service.lookup_gene("ENSG99999999999")


def test_error_429_raises_rate_limit_error(service, mock_response):
    """429 response raises EnsemblRateLimitError with retry info."""
    resp = mock_response(
        status_code=429,
        headers={
            "Retry-After": "30",
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": "1",
        },
    )

    with patch.object(service._session, "request", return_value=resp):
        with patch("lobster.services.data_access.ensembl_service.time.sleep"):
            with pytest.raises(EnsemblRateLimitError, match="Rate limit exceeded"):
                service.lookup_gene("ENSG00000141510")


def test_error_429_includes_retry_after(service, mock_response):
    """429 error message includes the Retry-After value."""
    resp = mock_response(
        status_code=429,
        headers={"Retry-After": "45", "X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1"},
    )

    with patch.object(service._session, "request", return_value=resp):
        with patch("lobster.services.data_access.ensembl_service.time.sleep"):
            with pytest.raises(EnsemblRateLimitError, match="45"):
                service.lookup_gene("ENSG00000141510")


def test_error_400_raises_service_error_with_detail(service, mock_response):
    """400 response raises EnsemblServiceError with error detail from JSON body."""
    resp = mock_response(
        status_code=400,
        json_data={"error": "Invalid HGVS notation"},
        text="Invalid HGVS notation",
    )

    with patch.object(service._session, "request", return_value=resp):
        with pytest.raises(EnsemblServiceError, match="Bad request.*Invalid HGVS notation"):
            service.get_variant_consequences("invalid_notation")


def test_error_400_fallback_to_text_on_json_failure(service, mock_response):
    """400 response falls back to response.text when JSON parsing fails."""
    resp = mock_response(
        status_code=400,
        text="plain text error",
    )
    resp.json.side_effect = Exception("JSON decode failed")

    with patch.object(service._session, "request", return_value=resp):
        with pytest.raises(EnsemblServiceError, match="Bad request.*plain text error"):
            service.get_variant_consequences("bad_notation")


def test_error_500_raises_http_error(service, mock_response):
    """500 response raises HTTPError via raise_for_status (after retry exhaustion)."""
    resp = mock_response(status_code=500)
    resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "500 Server Error"
    )

    with patch.object(service._session, "request", return_value=resp):
        with pytest.raises(requests.exceptions.HTTPError):
            service.lookup_gene("ENSG00000141510")


def test_error_connection_error(service):
    """Connection errors are wrapped in EnsemblServiceError."""
    with patch.object(
        service._session,
        "request",
        side_effect=requests.exceptions.ConnectionError("DNS failure"),
    ):
        with pytest.raises(EnsemblServiceError, match="Connection error"):
            service.lookup_gene("ENSG00000141510")


def test_error_timeout(service):
    """Timeout errors are wrapped in EnsemblServiceError."""
    with patch.object(
        service._session,
        "request",
        side_effect=requests.exceptions.Timeout("Request timed out"),
    ):
        with pytest.raises(EnsemblServiceError, match="timed out"):
            service.lookup_gene("ENSG00000141510")


def test_error_json_decode_error(service, mock_response):
    """Invalid JSON in response body raises EnsemblServiceError."""
    resp = mock_response(json_data=None)
    resp.json.side_effect = requests.exceptions.JSONDecodeError(
        "Expecting value", "doc", 0
    )

    with patch.object(service._session, "request", return_value=resp):
        with pytest.raises(EnsemblServiceError, match="Invalid JSON response"):
            service.lookup_gene("ENSG00000141510")


# =============================================================================
# Tests: LRU cache behavior
# =============================================================================


def test_cache_lookup_gene_same_args_cached(service, mock_response):
    """Repeated lookup_gene calls with same arguments hit cache, not network."""
    resp = mock_response(json_data=GENE_LOOKUP_RESPONSE)

    with patch.object(service._session, "request", return_value=resp) as mock_req:
        result1 = service.lookup_gene("ENSG00000141510")
        result2 = service.lookup_gene("ENSG00000141510")

    # Session request should only be called once (second call served from cache)
    mock_req.assert_called_once()
    assert result1 == result2


def test_cache_lookup_gene_different_args_not_cached(service, mock_response):
    """lookup_gene calls with different identifiers make separate requests."""
    resp1 = mock_response(json_data=GENE_LOOKUP_RESPONSE)
    resp2 = mock_response(
        json_data={**GENE_LOOKUP_RESPONSE, "id": "ENSG00000012048", "display_name": "BRCA1"}
    )

    with patch.object(service._session, "request", side_effect=[resp1, resp2]) as mock_req:
        result1 = service.lookup_gene("ENSG00000141510")
        result2 = service.lookup_gene("ENSG00000012048")

    assert mock_req.call_count == 2
    assert result1["display_name"] == "TP53"
    assert result2["display_name"] == "BRCA1"


def test_cache_lookup_gene_expand_difference(service, mock_response):
    """lookup_gene with expand=True and expand=False are cached separately."""
    resp_no_expand = mock_response(json_data=GENE_LOOKUP_RESPONSE)
    resp_expanded = mock_response(json_data=GENE_LOOKUP_EXPANDED_RESPONSE)

    with patch.object(
        service._session, "request", side_effect=[resp_no_expand, resp_expanded]
    ) as mock_req:
        result1 = service.lookup_gene("ENSG00000141510", expand=False)
        result2 = service.lookup_gene("ENSG00000141510", expand=True)

    assert mock_req.call_count == 2
    assert "Transcript" not in result1
    assert "Transcript" in result2


def test_cache_get_sequence_same_args_cached(service, mock_response):
    """Repeated get_sequence calls with same arguments hit cache, not network."""
    resp = mock_response(json_data=SEQUENCE_PROTEIN_RESPONSE)

    with patch.object(service._session, "request", return_value=resp) as mock_req:
        result1 = service.get_sequence("ENSP00000269305", seq_type="protein")
        result2 = service.get_sequence("ENSP00000269305", seq_type="protein")

    mock_req.assert_called_once()
    assert result1 == result2


def test_cache_get_sequence_different_seq_types(service, mock_response):
    """get_sequence calls with different seq_types are cached separately."""
    resp_genomic = mock_response(json_data=SEQUENCE_GENOMIC_RESPONSE)
    resp_protein = mock_response(json_data=SEQUENCE_PROTEIN_RESPONSE)

    with patch.object(
        service._session, "request", side_effect=[resp_genomic, resp_protein]
    ) as mock_req:
        result1 = service.get_sequence("ENSG00000141510", seq_type="genomic")
        result2 = service.get_sequence("ENSG00000141510", seq_type="protein")

    assert mock_req.call_count == 2
    assert result1["molecule"] == "dna"
    assert result2["molecule"] == "protein"


def test_cache_isolated_between_instances(mock_response):
    """Each EnsemblService instance has its own LRU cache."""
    svc1 = EnsemblService(timeout=10)
    svc2 = EnsemblService(timeout=10)
    svc1._lookup_gene_cached.cache_clear()
    svc2._lookup_gene_cached.cache_clear()

    resp = mock_response(json_data=GENE_LOOKUP_RESPONSE)

    with patch.object(svc1._session, "request", return_value=resp) as mock_req1:
        svc1.lookup_gene("ENSG00000141510")

    with patch.object(svc2._session, "request", return_value=resp) as mock_req2:
        svc2.lookup_gene("ENSG00000141510")

    # Both instances should make their own network call
    mock_req1.assert_called_once()
    mock_req2.assert_called_once()


def test_cache_info_accessible(service, mock_response):
    """LRU cache_info is accessible for monitoring cache hit rates."""
    resp = mock_response(json_data=GENE_LOOKUP_RESPONSE)

    with patch.object(service._session, "request", return_value=resp):
        service.lookup_gene("ENSG00000141510")
        service.lookup_gene("ENSG00000141510")

    info = service._lookup_gene_cached.cache_info()
    assert info.hits == 1
    assert info.misses == 1


# =============================================================================
# Tests: session and initialization
# =============================================================================


def test_service_default_timeout():
    """EnsemblService defaults to 30-second timeout."""
    svc = EnsemblService()
    assert svc._timeout == 30


def test_service_custom_timeout():
    """EnsemblService accepts custom timeout."""
    svc = EnsemblService(timeout=60)
    assert svc._timeout == 60


def test_session_has_retry_adapter():
    """Session is configured with retry adapter for HTTPS."""
    svc = EnsemblService()
    adapter = svc._session.get_adapter("https://rest.ensembl.org")
    assert isinstance(adapter, HTTPAdapter)


def test_session_has_correct_headers():
    """Session headers include Content-Type and User-Agent."""
    svc = EnsemblService()
    assert svc._session.headers["Content-Type"] == "application/json"
    assert "lobster-ai" in svc._session.headers["User-Agent"]


def test_base_url_constant():
    """BASE_URL points to Ensembl REST API."""
    assert BASE_URL == "https://rest.ensembl.org"


# =============================================================================
# Tests: exception hierarchy
# =============================================================================


def test_not_found_error_is_service_error():
    """EnsemblNotFoundError is a subclass of EnsemblServiceError."""
    assert issubclass(EnsemblNotFoundError, EnsemblServiceError)


def test_rate_limit_error_is_service_error():
    """EnsemblRateLimitError is a subclass of EnsemblServiceError."""
    assert issubclass(EnsemblRateLimitError, EnsemblServiceError)


def test_service_error_is_base_exception():
    """EnsemblServiceError is a subclass of Exception."""
    assert issubclass(EnsemblServiceError, Exception)


def test_catching_base_catches_all():
    """Catching EnsemblServiceError catches both NotFound and RateLimit errors."""
    with pytest.raises(EnsemblServiceError):
        raise EnsemblNotFoundError("test")

    with pytest.raises(EnsemblServiceError):
        raise EnsemblRateLimitError("test")
