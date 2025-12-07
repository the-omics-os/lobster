"""
Debug script to understand GDS search results and why we're getting GDS IDs instead of GSE IDs.

Run: python tests/manual/debug_gds_search.py
"""

import json
import ssl
import urllib.parse
import urllib.request
from pathlib import Path

# Create SSL context that doesn't verify certificates (for debugging only)
ssl_context = ssl._create_unverified_context()


def search_geo_datasets(query: str, max_results: int = 5):
    """Execute eSearch against GEO database."""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

    url_params = {
        "db": "gds",
        "term": query,
        "retmode": "json",
        "retmax": str(max_results),
        "usehistory": "y",
        "tool": "lobster_debug",
        "email": "debug@test.com",
    }

    url = f"{base_url}?" + urllib.parse.urlencode(url_params)
    print(f"\n=== eSearch Request ===")
    print(f"URL: {url}")

    with urllib.request.urlopen(url, context=ssl_context) as response:
        data = json.loads(response.read())

    print(f"\n=== eSearch Response ===")
    print(json.dumps(data, indent=2))

    esearch_result = data.get("esearchresult", {})
    ids = esearch_result.get("idlist", [])
    web_env = esearch_result.get("webenv")
    query_key = esearch_result.get("querykey")

    return ids, web_env, query_key


def get_summaries(ids: list, web_env: str = None, query_key: str = None):
    """Get summaries using eSummary."""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

    url_params = {
        "db": "gds",
        "retmode": "json",
        "tool": "lobster_debug",
        "email": "debug@test.com",
    }

    # Force use of ID list instead of webenv (webenv seems to return all 2230 results)
    url_params["id"] = ",".join(ids)

    url = f"{base_url}?" + urllib.parse.urlencode(url_params)
    print(f"\n=== eSummary Request ===")
    print(f"URL: {url}")

    with urllib.request.urlopen(url, context=ssl_context) as response:
        data = json.loads(response.read())

    print(f"\n=== eSummary Response (full JSON) ===")
    print(json.dumps(data, indent=2)[:5000])  # First 5000 chars
    print("\n")

    result = data.get("result", {})

    summaries = []
    for i, uid in enumerate(ids[:2]):  # Only show first 2 for readability
        if uid in result:
            summary = result[uid]
            print(f"\n--- Record {i+1} (UID: {uid}) ---")
            print(json.dumps(summary, indent=2))
            summaries.append(summary)

            # Extract key fields
            accession = summary.get("accession", f"GDS{uid}")
            entry_type = summary.get("entrytype", "unknown")
            title = summary.get("title", "N/A")

            print(f"\n📌 Key Fields:")
            print(f"  - UID: {uid}")
            print(f"  - Accession: {accession}")
            print(f"  - Entry Type: {entry_type}")
            print(f"  - Title: {title[:100]}...")

    return summaries


def main():
    """Run debug tests."""
    print("=" * 80)
    print("GDS SEARCH DEBUG SCRIPT")
    print("=" * 80)

    # Test query from user's log
    query = '"10x Genomics" single cell RNA-seq'
    print(f"\nQuery: {query}")
    print(f"Max Results: 5")

    # Step 1: Search
    ids, web_env, query_key = search_geo_datasets(query, max_results=5)

    print(f"\n=== Search Results Summary ===")
    print(f"Found {len(ids)} IDs")
    print(f"IDs: {ids}")
    print(f"WebEnv: {web_env}")
    print(f"QueryKey: {query_key}")

    # Step 2: Get summaries
    if ids:
        summaries = get_summaries(ids, web_env, query_key)

        print("\n" + "=" * 80)
        print("ANALYSIS")
        print("=" * 80)

        for i, summary in enumerate(summaries, 1):
            uid = summary.get("uid")
            accession = summary.get("accession", f"GDS{uid}")
            entry_type = summary.get("entrytype", "unknown")

            print(f"\n{i}. UID {uid}:")
            print(f"   - Accession field: {accession}")
            print(f"   - Entry type: {entry_type}")
            print(f"   - Is GDS prefix correct? {accession.startswith('GDS')}")
            print(f"   - Should be GSE? {entry_type == 'GSE'}")

            if entry_type == "GSE":
                print(f"   ⚠️  ISSUE: Entry type is GSE but accession is {accession}")
            elif not accession.startswith("GDS"):
                print(f"   ✅ Accession correctly starts with {accession[:3]}")
            else:
                print(f"   ℹ️  GDS dataset (curated subset)")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("""
The NCBI 'gds' database returns mixed entry types (GSE, GDS, GPL, GSM).
The eSummary response contains an 'accession' field with the correct prefix.

Current bug in format_geo_search_results():
- Line 1074 blindly prefixes UIDs with 'GDS', creating invalid accessions
- Should use summary.get('accession') instead

Fix: Always fetch summaries and use the 'accession' field from eSummary.
""")


if __name__ == "__main__":
    main()
