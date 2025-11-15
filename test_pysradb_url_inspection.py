#!/usr/bin/env python3
"""
Inspect the actual URLs being constructed by pysradb for NCBI queries.
"""

import requests
from pysradb.search import SraSearch

print("="*80)
print("PYSRADB URL INSPECTION")
print("="*80)

# Test different query patterns
test_cases = [
    {"query": "microbiome", "description": "Single keyword"},
    {"query": "gut microbiome", "description": "Two keywords"},
    {"query": "cancer", "description": "Common term"},
    {"query": "SARS-CoV-2", "description": "Virus name"},
]

for i, test in enumerate(test_cases, 1):
    print(f"\n{i}. {test['description']}: '{test['query']}'")
    print("-"*80)

    # Create SraSearch instance
    instance = SraSearch(
        verbosity=2,
        return_max=5,
        query=test['query']
    )

    # Get the formatted request payload
    payload = instance._format_request()
    query_string = instance._format_query_string()

    print(f"Query String: {query_string}")
    print(f"Payload: {payload}")

    # Construct the actual URL that would be queried
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    print(f"\nFull URL: {url}?{requests.compat.urlencode(payload)}")

    # Actually make the request to see what NCBI returns
    print("\nMaking request to NCBI...")
    try:
        r = requests.get(url, params=payload, timeout=20)
        r.raise_for_status()
        response = r.json()

        print(f"Response Status: {r.status_code}")
        print(f"Response Keys: {response.keys()}")

        if "esearchresult" in response:
            result = response["esearchresult"]
            print(f"Count: {result.get('count', 'N/A')}")
            print(f"RetMax: {result.get('retmax', 'N/A')}")
            print(f"RetStart: {result.get('retstart', 'N/A')}")
            print(f"IdList: {result.get('idlist', [])}")
            print(f"IdList Length: {len(result.get('idlist', []))}")

            if result.get('idlist'):
                print(f"✅ SUCCESS - Found {len(result['idlist'])} UIDs")
            else:
                print(f"❌ FAIL - Empty idlist returned by NCBI")

        print(f"\nFull Response:")
        print(response)

    except Exception as e:
        print(f"ERROR: {e}")

print("\n" + "="*80)
print("Now testing with filters...")
print("="*80)

# Test with filters
filter_tests = [
    {
        "query": "microbiome",
        "organism": "Homo sapiens",
        "description": "microbiome + organism filter"
    },
    {
        "query": "microbiome",
        "organism": "Homo sapiens",
        "strategy": "AMPLICON",
        "description": "microbiome + both filters"
    },
]

for i, test in enumerate(filter_tests, 1):
    print(f"\n{i}. {test['description']}")
    print("-"*80)

    # Create SraSearch instance
    kwargs = {k: v for k, v in test.items() if k != 'description'}
    instance = SraSearch(verbosity=2, return_max=5, **kwargs)

    # Get the formatted request payload
    payload = instance._format_request()
    query_string = instance._format_query_string()

    print(f"Query String: {query_string}")

    # Make request
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    try:
        r = requests.get(url, params=payload, timeout=20)
        response = r.json()

        if "esearchresult" in response:
            result = response["esearchresult"]
            print(f"Count: {result.get('count', 'N/A')}")
            print(f"IdList Length: {len(result.get('idlist', []))}")

            if result.get('idlist'):
                print(f"✅ SUCCESS - Found {len(result['idlist'])} UIDs")
            else:
                print(f"❌ FAIL - Empty idlist")

    except Exception as e:
        print(f"ERROR: {e}")

print("\n" + "="*80)
print("INSPECTION COMPLETE")
print("="*80)
