"""Debug script to inspect raw NCBI XML responses."""

import logging
import time
import urllib.parse
import urllib.request

import xmltodict

from lobster.utils.ssl_utils import create_ssl_context

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configuration
email = "kevin.yar@omics-os.com"
ssl_context = create_ssl_context()

print("=" * 80)
print("NCBI XML DEBUG - Phase 1 Investigation")
print("=" * 80)

# Step 1: Execute esearch for "cancer"
print("\n[STEP 1] Execute esearch for 'cancer'")
print("-" * 80)

esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
esearch_params = {
    "db": "sra",
    "term": "cancer",
    "retmax": "3",
    "retmode": "xml",
    "email": email,
}

esearch_full_url = f"{esearch_url}?{urllib.parse.urlencode(esearch_params)}"
print(f"URL: {esearch_full_url}")

time.sleep(1)  # Manual rate limiting
response = urllib.request.urlopen(esearch_full_url, context=ssl_context, timeout=30)
esearch_content = response.read()

print(f"\nRaw XML Response ({len(esearch_content)} bytes):")
print(esearch_content.decode("utf-8")[:500])

# Parse XML
esearch_result = xmltodict.parse(esearch_content)
print("\nParsed XML structure:")
print(f"Top-level keys: {list(esearch_result.keys())}")

id_list = esearch_result.get("eSearchResult", {}).get("IdList", {}).get("Id", [])
if isinstance(id_list, str):
    id_list = [id_list]

print(f"\nExtracted IDs: {id_list}")

if not id_list:
    print("ERROR: No IDs found from esearch!")
    exit(1)

# Step 2: Execute esummary with the IDs
print("\n" + "=" * 80)
print(f"[STEP 2] Execute esummary for {len(id_list)} IDs")
print("-" * 80)

esummary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
esummary_params = {
    "db": "sra",
    "id": ",".join(id_list),
    "retmode": "xml",
    "email": email,
}

esummary_full_url = f"{esummary_url}?{urllib.parse.urlencode(esummary_params)}"
print(f"URL: {esummary_full_url}")

time.sleep(1)  # Manual rate limiting
response = urllib.request.urlopen(esummary_full_url, context=ssl_context, timeout=30)
esummary_content = response.read()

print(f"\nRaw XML Response ({len(esummary_content)} bytes):")
print(esummary_content.decode("utf-8")[:1000])

# Parse XML
esummary_result = xmltodict.parse(esummary_content)
print("\nParsed XML structure:")
print(f"Top-level keys: {list(esummary_result.keys())}")

if "eSummaryResult" in esummary_result:
    print(f"eSummaryResult keys: {list(esummary_result['eSummaryResult'].keys())}")

    doc_summaries = esummary_result.get("eSummaryResult", {}).get("DocumentSummary", [])
    print(f"\nDocumentSummary type: {type(doc_summaries)}")

    if isinstance(doc_summaries, dict):
        doc_summaries = [doc_summaries]

    print(f"Number of DocumentSummary entries: {len(doc_summaries)}")

    if doc_summaries:
        print("\n[FIRST RECORD STRUCTURE]")
        print("-" * 80)
        first_doc = doc_summaries[0]
        print(f"Keys in first DocumentSummary: {list(first_doc.keys())[:10]}")

        # Show a few fields
        if "ExpXml" in first_doc:
            print("\nExpXml structure (first 500 chars):")
            print(str(first_doc["ExpXml"])[:500])

        if "Runs" in first_doc:
            print("\nRuns structure (first 500 chars):")
            print(str(first_doc["Runs"])[:500])

print("\n" + "=" * 80)
print("XML DEBUG COMPLETE")
print("=" * 80)
