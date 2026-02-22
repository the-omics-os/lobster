"""
Unit tests for UniProtService.

Tests the stateless HTTP client for https://rest.uniprot.org covering:
- Single protein lookup by accession (get_protein)
- Keyword/field search across UniProtKB (search_proteins)
- Cross-database ID mapping with async job flow (map_ids)
- Error handling: 404, 429, 400, connection errors, timeouts, bad JSON
- LRU cache behaviour for repeated protein lookups

All HTTP calls are mocked via unittest.mock.patch on the session's request/get/post
methods. No real network calls are made.

Running Tests:
```bash
pytest tests/unit/services/data_access/test_uniprot_service.py -v
```
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
import requests
from requests.adapters import HTTPAdapter

from lobster.services.data_access.uniprot_service import (
    BASE_URL,
    ID_MAPPING_URL,
    UniProtNotFoundError,
    UniProtRateLimitError,
    UniProtService,
    UniProtServiceError,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def service():
    """Create a fresh UniProtService instance with default timeout."""
    return UniProtService()


@pytest.fixture
def service_short_timeout():
    """Create a UniProtService instance with short timeout for timeout tests."""
    return UniProtService(timeout=5)


def _mock_response(
    status_code=200,
    json_data=None,
    headers=None,
    url="https://rest.uniprot.org/mock",
    raise_for_status_effect=None,
):
    """Build a mock requests.Response object with the specified attributes."""
    resp = Mock(spec=requests.Response)
    resp.status_code = status_code
    resp.url = url
    resp.headers = headers or {}
    resp.text = str(json_data) if json_data else ""

    if json_data is not None:
        resp.json.return_value = json_data
    else:
        resp.json.side_effect = requests.exceptions.JSONDecodeError(
            "No JSON", "", 0
        )

    if raise_for_status_effect:
        resp.raise_for_status.side_effect = raise_for_status_effect
    else:
        resp.raise_for_status.return_value = None

    return resp


# =============================================================================
# Sample data
# =============================================================================

SAMPLE_PROTEIN_P04637 = {
    "primaryAccession": "P04637",
    "uniProtkbId": "P53_HUMAN",
    "proteinDescription": {
        "recommendedName": {
            "fullName": {"value": "Cellular tumor antigen p53"}
        }
    },
    "organism": {
        "scientificName": "Homo sapiens",
        "taxonId": 9606,
    },
    "genes": [{"geneName": {"value": "TP53"}}],
    "sequence": {
        "value": "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVL...",
        "length": 393,
        "molWeight": 43653,
    },
}

SAMPLE_SEARCH_RESULTS = {
    "results": [
        {
            "primaryAccession": "P04637",
            "uniProtkbId": "P53_HUMAN",
            "organism": {"scientificName": "Homo sapiens"},
        },
        {
            "primaryAccession": "P02340",
            "uniProtkbId": "P53_MOUSE",
            "organism": {"scientificName": "Mus musculus"},
        },
    ]
}

SAMPLE_MAPPING_RESULTS = {
    "results": [
        {"from": "TP53", "to": {"primaryAccession": "P04637"}},
        {"from": "BRCA1", "to": {"primaryAccession": "P38398"}},
    ]
}


# =============================================================================
# get_protein tests
# =============================================================================


def test_get_protein_valid_accession(service):
    """get_protein returns protein data for a valid accession like P04637."""
    mock_resp = _mock_response(json_data=SAMPLE_PROTEIN_P04637)

    with patch.object(service._session, "request", return_value=mock_resp) as mock_req:
        result = service.get_protein("P04637")

    assert result["primaryAccession"] == "P04637"
    assert result["uniProtkbId"] == "P53_HUMAN"
    assert result["organism"]["taxonId"] == 9606
    mock_req.assert_called_once_with(
        "GET",
        f"{BASE_URL}/uniprotkb/P04637",
        params=None,
        timeout=service._timeout,
    )


def test_get_protein_strips_whitespace(service):
    """get_protein strips leading/trailing whitespace from the accession."""
    mock_resp = _mock_response(json_data=SAMPLE_PROTEIN_P04637)

    with patch.object(service._session, "request", return_value=mock_resp):
        result = service.get_protein("  P04637  ")

    assert result["primaryAccession"] == "P04637"


def test_get_protein_entry_name(service):
    """get_protein accepts entry names like P53_HUMAN, not just accessions."""
    mock_resp = _mock_response(json_data=SAMPLE_PROTEIN_P04637)

    with patch.object(service._session, "request", return_value=mock_resp):
        result = service.get_protein("P53_HUMAN")

    assert result["primaryAccession"] == "P04637"


def test_get_protein_not_found_raises(service):
    """get_protein raises UniProtNotFoundError for a 404 response."""
    mock_resp = _mock_response(
        status_code=404,
        url=f"{BASE_URL}/uniprotkb/INVALID123",
    )

    with patch.object(service._session, "request", return_value=mock_resp):
        with pytest.raises(UniProtNotFoundError, match="Not found"):
            service.get_protein("INVALID123")


def test_get_protein_rate_limit_raises(service):
    """get_protein raises UniProtRateLimitError when the API returns 429."""
    mock_resp = _mock_response(
        status_code=429,
        headers={"Retry-After": "60"},
        url=f"{BASE_URL}/uniprotkb/P04637",
    )

    with patch.object(service._session, "request", return_value=mock_resp):
        with pytest.raises(UniProtRateLimitError, match="Retry after 60s"):
            service.get_protein("P04637")


def test_get_protein_rate_limit_unknown_retry_after(service):
    """get_protein handles 429 when Retry-After header is missing."""
    mock_resp = _mock_response(
        status_code=429,
        headers={},
        url=f"{BASE_URL}/uniprotkb/P04637",
    )

    with patch.object(service._session, "request", return_value=mock_resp):
        with pytest.raises(UniProtRateLimitError, match="Retry after unknowns"):
            service.get_protein("P04637")


def test_get_protein_bad_request_with_json_detail(service):
    """get_protein raises UniProtServiceError with detail on 400 with JSON body."""
    error_detail = {"messages": ["Invalid accession format"]}
    mock_resp = _mock_response(
        status_code=400,
        json_data=error_detail,
        url=f"{BASE_URL}/uniprotkb/!!!",
    )

    with patch.object(service._session, "request", return_value=mock_resp):
        with pytest.raises(UniProtServiceError, match="Bad request"):
            service.get_protein("!!!")


def test_get_protein_bad_request_with_text_fallback(service):
    """get_protein shows text body when 400 response has no valid JSON."""
    mock_resp = _mock_response(
        status_code=400,
        url=f"{BASE_URL}/uniprotkb/bad",
    )
    mock_resp.json.side_effect = ValueError("No JSON")
    mock_resp.text = "Invalid request format"

    with patch.object(service._session, "request", return_value=mock_resp):
        with pytest.raises(UniProtServiceError, match="Bad request"):
            service.get_protein("bad")


def test_get_protein_server_error_raises_for_status(service):
    """get_protein falls through to raise_for_status for unexpected status codes."""
    mock_resp = _mock_response(
        status_code=503,
        url=f"{BASE_URL}/uniprotkb/P04637",
    )
    mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "503 Service Unavailable"
    )

    with patch.object(service._session, "request", return_value=mock_resp):
        with pytest.raises(requests.exceptions.HTTPError, match="503"):
            service.get_protein("P04637")


# =============================================================================
# get_protein -- LRU cache tests
# =============================================================================


def test_get_protein_caches_result(service):
    """Calling get_protein twice with the same accession hits the network only once."""
    mock_resp = _mock_response(json_data=SAMPLE_PROTEIN_P04637)

    with patch.object(service._session, "request", return_value=mock_resp) as mock_req:
        result1 = service.get_protein("P04637")
        result2 = service.get_protein("P04637")

    assert result1 is result2
    mock_req.assert_called_once()


def test_get_protein_cache_different_accessions(service):
    """Different accessions are cached independently."""
    protein_a = {"primaryAccession": "P04637", "uniProtkbId": "P53_HUMAN"}
    protein_b = {"primaryAccession": "P38398", "uniProtkbId": "BRCA1_HUMAN"}

    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _mock_response(json_data=protein_a)
        return _mock_response(json_data=protein_b)

    with patch.object(service._session, "request", side_effect=side_effect):
        result_a = service.get_protein("P04637")
        result_b = service.get_protein("P38398")
        result_a_again = service.get_protein("P04637")

    assert result_a["primaryAccession"] == "P04637"
    assert result_b["primaryAccession"] == "P38398"
    assert result_a_again is result_a
    assert call_count == 2


def test_get_protein_cache_not_shared_between_instances():
    """Each UniProtService instance has its own LRU cache."""
    service_a = UniProtService()
    service_b = UniProtService()

    mock_resp = _mock_response(json_data=SAMPLE_PROTEIN_P04637)

    with patch.object(service_a._session, "request", return_value=mock_resp) as mock_a:
        service_a.get_protein("P04637")

    with patch.object(service_b._session, "request", return_value=mock_resp) as mock_b:
        service_b.get_protein("P04637")

    # Both instances should have made their own HTTP call
    mock_a.assert_called_once()
    mock_b.assert_called_once()


def test_get_protein_cache_strips_whitespace_for_key(service):
    """Whitespace-stripped accession uses the same cache slot."""
    mock_resp = _mock_response(json_data=SAMPLE_PROTEIN_P04637)

    with patch.object(service._session, "request", return_value=mock_resp) as mock_req:
        service.get_protein("P04637")
        service.get_protein("  P04637  ")

    # Both calls resolve to "P04637" after strip(), so only one request is made
    mock_req.assert_called_once()


# =============================================================================
# search_proteins tests
# =============================================================================


def test_search_proteins_basic_query(service):
    """search_proteins sends a basic keyword query to UniProtKB."""
    mock_resp = _mock_response(json_data=SAMPLE_SEARCH_RESULTS)

    with patch.object(service._session, "request", return_value=mock_resp) as mock_req:
        result = service.search_proteins("TP53")

    assert len(result["results"]) == 2
    assert result["results"][0]["primaryAccession"] == "P04637"

    mock_req.assert_called_once()
    call_args = mock_req.call_args
    assert call_args[0] == ("GET", f"{BASE_URL}/uniprotkb/search")
    params = call_args[1]["params"]
    assert params["query"] == "TP53"
    assert params["size"] == 10
    assert params["format"] == "json"


def test_search_proteins_with_organism_filter(service):
    """search_proteins appends organism_id filter to the query."""
    mock_resp = _mock_response(json_data=SAMPLE_SEARCH_RESULTS)

    with patch.object(service._session, "request", return_value=mock_resp) as mock_req:
        service.search_proteins("kinase", organism="9606")

    params = mock_req.call_args[1]["params"]
    assert params["query"] == "kinase AND organism_id:9606"


def test_search_proteins_with_reviewed_true(service):
    """search_proteins appends reviewed:true filter when reviewed=True."""
    mock_resp = _mock_response(json_data=SAMPLE_SEARCH_RESULTS)

    with patch.object(service._session, "request", return_value=mock_resp) as mock_req:
        service.search_proteins("TP53", reviewed=True)

    params = mock_req.call_args[1]["params"]
    assert params["query"] == "TP53 AND reviewed:true"


def test_search_proteins_with_reviewed_false(service):
    """search_proteins appends reviewed:false filter when reviewed=False."""
    mock_resp = _mock_response(json_data=SAMPLE_SEARCH_RESULTS)

    with patch.object(service._session, "request", return_value=mock_resp) as mock_req:
        service.search_proteins("TP53", reviewed=False)

    params = mock_req.call_args[1]["params"]
    assert params["query"] == "TP53 AND reviewed:false"


def test_search_proteins_all_filters_combined(service):
    """search_proteins combines organism and reviewed filters with AND."""
    mock_resp = _mock_response(json_data=SAMPLE_SEARCH_RESULTS)

    with patch.object(service._session, "request", return_value=mock_resp) as mock_req:
        service.search_proteins("BRCA1", organism="9606", reviewed=True)

    params = mock_req.call_args[1]["params"]
    assert params["query"] == "BRCA1 AND organism_id:9606 AND reviewed:true"


def test_search_proteins_max_results_capped_at_500(service):
    """search_proteins caps max_results to 500 even if a larger value is passed."""
    mock_resp = _mock_response(json_data={"results": []})

    with patch.object(service._session, "request", return_value=mock_resp) as mock_req:
        service.search_proteins("anything", max_results=1000)

    params = mock_req.call_args[1]["params"]
    assert params["size"] == 500


def test_search_proteins_custom_max_results(service):
    """search_proteins passes custom max_results when below 500."""
    mock_resp = _mock_response(json_data={"results": []})

    with patch.object(service._session, "request", return_value=mock_resp) as mock_req:
        service.search_proteins("kinase", max_results=25)

    params = mock_req.call_args[1]["params"]
    assert params["size"] == 25


def test_search_proteins_empty_results(service):
    """search_proteins returns empty results list for no-match query."""
    mock_resp = _mock_response(json_data={"results": []})

    with patch.object(service._session, "request", return_value=mock_resp):
        result = service.search_proteins("nonexistent_protein_xyzzy_12345")

    assert result["results"] == []


# =============================================================================
# map_ids tests
# =============================================================================


def test_map_ids_empty_list_returns_early(service):
    """map_ids returns empty results immediately for an empty ID list."""
    result = service.map_ids("Gene_Name", "UniProtKB", [])
    assert result == {"results": []}


def test_map_ids_complete_flow_with_303_redirect(service):
    """map_ids follows the submit -> poll (303) -> fetch results flow."""
    # Step 1: Submit response
    submit_resp = _mock_response(
        status_code=200,
        json_data={"jobId": "abc123"},
    )

    # Step 2: Poll response -- 303 redirect
    poll_resp = _mock_response(status_code=303, json_data=None)
    poll_resp.headers = {"Location": f"{ID_MAPPING_URL}/results/abc123"}
    poll_resp.json.return_value = {}
    poll_resp.json.side_effect = None

    # Step 3: Results response (fetched via _request)
    results_resp = _mock_response(json_data=SAMPLE_MAPPING_RESULTS)

    with patch.object(service._session, "post", return_value=submit_resp) as mock_post, \
         patch.object(service._session, "get", return_value=poll_resp) as mock_get, \
         patch.object(service._session, "request", return_value=results_resp) as mock_req:
        result = service.map_ids("Gene_Name", "UniProtKB", ["TP53", "BRCA1"])

    assert len(result["results"]) == 2
    assert result["results"][0]["from"] == "TP53"
    assert result["results"][1]["from"] == "BRCA1"

    # Verify submit call
    mock_post.assert_called_once_with(
        f"{ID_MAPPING_URL}/run",
        data={"from": "Gene_Name", "to": "UniProtKB", "ids": "TP53,BRCA1"},
        timeout=service._timeout,
    )


def test_map_ids_complete_flow_with_finished_status(service):
    """map_ids handles poll returning jobStatus=FINISHED instead of 303."""
    submit_resp = _mock_response(
        status_code=200,
        json_data={"jobId": "def456"},
    )

    poll_resp = _mock_response(
        status_code=200,
        json_data={"jobStatus": "FINISHED"},
    )

    results_resp = _mock_response(json_data=SAMPLE_MAPPING_RESULTS)

    with patch.object(service._session, "post", return_value=submit_resp), \
         patch.object(service._session, "get", return_value=poll_resp), \
         patch.object(service._session, "request", return_value=results_resp):
        result = service.map_ids("Gene_Name", "UniProtKB", ["TP53"])

    assert len(result["results"]) == 2


def test_map_ids_poll_running_then_finished(service):
    """map_ids polls multiple times when jobStatus is RUNNING before FINISHED."""
    submit_resp = _mock_response(
        status_code=200,
        json_data={"jobId": "running_job"},
    )

    running_resp = _mock_response(
        status_code=200,
        json_data={"jobStatus": "RUNNING"},
    )
    finished_resp = _mock_response(
        status_code=200,
        json_data={"jobStatus": "FINISHED"},
    )

    results_resp = _mock_response(json_data=SAMPLE_MAPPING_RESULTS)

    with patch.object(service._session, "post", return_value=submit_resp), \
         patch.object(
             service._session, "get",
             side_effect=[running_resp, running_resp, finished_resp]
         ) as mock_get, \
         patch.object(service._session, "request", return_value=results_resp), \
         patch("lobster.services.data_access.uniprot_service.time.sleep") as mock_sleep:
        result = service.map_ids(
            "Gene_Name", "UniProtKB", ["TP53"],
            poll_interval=1.0,
        )

    assert len(result["results"]) == 2
    # Two RUNNING polls trigger sleep, the third (FINISHED) does not
    assert mock_sleep.call_count == 2
    assert mock_get.call_count == 3


def test_map_ids_results_inline_in_status_response(service):
    """map_ids returns directly when status response includes results inline."""
    submit_resp = _mock_response(
        status_code=200,
        json_data={"jobId": "inline_job"},
    )

    inline_resp = _mock_response(
        status_code=200,
        json_data={"results": [{"from": "TP53", "to": {"primaryAccession": "P04637"}}]},
    )

    with patch.object(service._session, "post", return_value=submit_resp), \
         patch.object(service._session, "get", return_value=inline_resp):
        result = service.map_ids("Gene_Name", "UniProtKB", ["TP53"])

    assert len(result["results"]) == 1
    assert result["results"][0]["from"] == "TP53"


def test_map_ids_no_job_id_raises(service):
    """map_ids raises UniProtServiceError when no jobId is returned."""
    submit_resp = _mock_response(
        status_code=200,
        json_data={"error": "something went wrong"},
    )

    with patch.object(service._session, "post", return_value=submit_resp):
        with pytest.raises(UniProtServiceError, match="No jobId"):
            service.map_ids("Gene_Name", "UniProtKB", ["TP53"])


def test_map_ids_job_failure_raises(service):
    """map_ids raises UniProtServiceError when job reports a failed status."""
    submit_resp = _mock_response(
        status_code=200,
        json_data={"jobId": "fail_job"},
    )

    failed_resp = _mock_response(
        status_code=200,
        json_data={"jobStatus": "INTERNAL_ERROR"},
    )

    with patch.object(service._session, "post", return_value=submit_resp), \
         patch.object(service._session, "get", return_value=failed_resp):
        with pytest.raises(UniProtServiceError, match="failed with status: INTERNAL_ERROR"):
            service.map_ids("Gene_Name", "UniProtKB", ["TP53"])


def test_map_ids_timeout_raises(service):
    """map_ids raises UniProtServiceError when max_polls is exhausted."""
    submit_resp = _mock_response(
        status_code=200,
        json_data={"jobId": "timeout_job"},
    )

    # Always returns something that is neither FINISHED, 303, nor has "results"
    unknown_resp = _mock_response(
        status_code=200,
        json_data={"someUnknownField": True},
    )

    with patch.object(service._session, "post", return_value=submit_resp), \
         patch.object(service._session, "get", return_value=unknown_resp), \
         patch("lobster.services.data_access.uniprot_service.time.sleep"):
        with pytest.raises(UniProtServiceError, match="timed out"):
            service.map_ids(
                "Gene_Name", "UniProtKB", ["TP53"],
                poll_interval=0.01,
                max_polls=3,
            )


def test_map_ids_303_without_location_header(service):
    """map_ids constructs results URL from jobId when Location header is missing."""
    submit_resp = _mock_response(
        status_code=200,
        json_data={"jobId": "noloc_job"},
    )

    redirect_resp = _mock_response(status_code=303, json_data=None)
    redirect_resp.headers = {}
    redirect_resp.json.return_value = {}
    redirect_resp.json.side_effect = None

    results_resp = _mock_response(json_data=SAMPLE_MAPPING_RESULTS)

    with patch.object(service._session, "post", return_value=submit_resp), \
         patch.object(service._session, "get", return_value=redirect_resp), \
         patch.object(service._session, "request", return_value=results_resp) as mock_req:
        service.map_ids("Gene_Name", "UniProtKB", ["TP53"])

    # Should fall back to constructing URL from job_id
    call_args = mock_req.call_args
    assert call_args[0][1] == f"{ID_MAPPING_URL}/results/noloc_job"


def test_map_ids_submit_error_propagates(service):
    """map_ids propagates errors from the submit step (e.g. 400)."""
    submit_resp = _mock_response(
        status_code=400,
        json_data={"messages": ["Invalid 'from' database"]},
        url=f"{ID_MAPPING_URL}/run",
    )

    with patch.object(service._session, "post", return_value=submit_resp):
        with pytest.raises(UniProtServiceError, match="Bad request"):
            service.map_ids("INVALID_DB", "UniProtKB", ["TP53"])


def test_map_ids_joins_ids_with_comma(service):
    """map_ids joins the ID list with commas in the POST body."""
    submit_resp = _mock_response(
        status_code=200,
        json_data={"jobId": "comma_job"},
    )

    inline_resp = _mock_response(
        status_code=200,
        json_data={"results": []},
    )

    with patch.object(service._session, "post", return_value=submit_resp) as mock_post, \
         patch.object(service._session, "get", return_value=inline_resp):
        service.map_ids("Gene_Name", "UniProtKB", ["TP53", "BRCA1", "EGFR"])

    submitted_data = mock_post.call_args[1]["data"]
    assert submitted_data["ids"] == "TP53,BRCA1,EGFR"


# =============================================================================
# Error handling tests (connection, timeout, bad JSON)
# =============================================================================


def test_connection_error_raises_service_error(service):
    """Connection errors are wrapped in UniProtServiceError."""
    with patch.object(
        service._session, "request",
        side_effect=requests.exceptions.ConnectionError("DNS resolution failed"),
    ):
        with pytest.raises(UniProtServiceError, match="Connection error"):
            service.get_protein("P04637")


def test_timeout_error_raises_service_error(service_short_timeout):
    """Timeout errors are wrapped in UniProtServiceError."""
    with patch.object(
        service_short_timeout._session, "request",
        side_effect=requests.exceptions.Timeout("Read timed out"),
    ):
        with pytest.raises(UniProtServiceError, match="timed out"):
            service_short_timeout.get_protein("P04637")


def test_bad_json_response_raises_service_error(service):
    """Invalid JSON responses are wrapped in UniProtServiceError."""
    mock_resp = _mock_response(status_code=200)
    mock_resp.json.side_effect = requests.exceptions.JSONDecodeError(
        "Expecting value", "", 0
    )

    with patch.object(service._session, "request", return_value=mock_resp):
        with pytest.raises(UniProtServiceError, match="Invalid JSON"):
            service.get_protein("P04637")


def test_search_connection_error(service):
    """search_proteins wraps connection errors in UniProtServiceError."""
    with patch.object(
        service._session, "request",
        side_effect=requests.exceptions.ConnectionError("Network unreachable"),
    ):
        with pytest.raises(UniProtServiceError, match="Connection error"):
            service.search_proteins("TP53")


def test_search_timeout_error(service):
    """search_proteins wraps timeout errors in UniProtServiceError."""
    with patch.object(
        service._session, "request",
        side_effect=requests.exceptions.Timeout("Connection timed out"),
    ):
        with pytest.raises(UniProtServiceError, match="timed out"):
            service.search_proteins("TP53")


# =============================================================================
# Session and constructor tests
# =============================================================================


def test_default_timeout():
    """Default timeout is 30 seconds."""
    svc = UniProtService()
    assert svc._timeout == 30


def test_custom_timeout():
    """Custom timeout is stored correctly."""
    svc = UniProtService(timeout=60)
    assert svc._timeout == 60


def test_session_has_retry_adapter():
    """Session is configured with HTTPAdapter for HTTPS retries."""
    svc = UniProtService()
    adapter = svc._session.get_adapter("https://example.com")
    assert isinstance(adapter, HTTPAdapter)
    assert adapter.max_retries.total == 3


def test_session_has_correct_headers():
    """Session has Accept and User-Agent headers set."""
    svc = UniProtService()
    assert svc._session.headers["Accept"] == "application/json"
    assert "lobster-ai" in svc._session.headers["User-Agent"]


def test_session_retry_includes_429_and_5xx():
    """Retry strategy covers 429, 500, 502, 503, 504 status codes."""
    svc = UniProtService()
    adapter = svc._session.get_adapter("https://example.com")
    assert set(adapter.max_retries.status_forcelist) == {429, 500, 502, 503, 504}


def test_session_retry_allows_get_and_post():
    """Retry strategy allows both GET and POST methods."""
    svc = UniProtService()
    adapter = svc._session.get_adapter("https://example.com")
    assert set(adapter.max_retries.allowed_methods) == {"GET", "POST"}


# =============================================================================
# Exception hierarchy tests
# =============================================================================


def test_not_found_is_subclass_of_service_error():
    """UniProtNotFoundError is a subclass of UniProtServiceError."""
    assert issubclass(UniProtNotFoundError, UniProtServiceError)


def test_rate_limit_is_subclass_of_service_error():
    """UniProtRateLimitError is a subclass of UniProtServiceError."""
    assert issubclass(UniProtRateLimitError, UniProtServiceError)


def test_service_error_is_subclass_of_exception():
    """UniProtServiceError is a standard Exception subclass."""
    assert issubclass(UniProtServiceError, Exception)


def test_catch_service_error_catches_not_found():
    """Catching UniProtServiceError also catches UniProtNotFoundError."""
    with pytest.raises(UniProtServiceError):
        raise UniProtNotFoundError("test not found")


def test_catch_service_error_catches_rate_limit():
    """Catching UniProtServiceError also catches UniProtRateLimitError."""
    with pytest.raises(UniProtServiceError):
        raise UniProtRateLimitError("test rate limit")


# =============================================================================
# Module-level constants
# =============================================================================


def test_base_url_value():
    """BASE_URL points to the UniProt REST API."""
    assert BASE_URL == "https://rest.uniprot.org"


def test_id_mapping_url_derived_from_base():
    """ID_MAPPING_URL is derived from BASE_URL."""
    assert ID_MAPPING_URL == f"{BASE_URL}/idmapping"
