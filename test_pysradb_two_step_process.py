#!/usr/bin/env python3
"""
Test pysradb's two-step process:
1. esearch.fcgi to get UIDs
2. efetch.fcgi to get detailed information

This will help identify where the breakdown occurs.
"""

import requests
import xml.etree.ElementTree as Et
from io import BytesIO

print("="*80)
print("PYSRADB TWO-STEP PROCESS INVESTIGATION")
print("="*80)

# Test case: simple "microbiome" query
query_term = "microbiome"
return_max = 5

print(f"\nQuery: '{query_term}'")
print(f"Return Max: {return_max}")
print("="*80)

# STEP 1: esearch to get UIDs
print("\nSTEP 1: esearch.fcgi - Get UIDs")
print("-"*80)

esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
esearch_payload = {
    "db": "sra",
    "term": query_term,
    "retmode": "json",
    "retmax": return_max,
}

try:
    r1 = requests.get(esearch_url, params=esearch_payload, timeout=20)
    r1.raise_for_status()
    response = r1.json()

    uids = response["esearchresult"]["idlist"]
    print(f"✅ SUCCESS: Retrieved {len(uids)} UIDs")
    print(f"UIDs: {uids}")

except Exception as e:
    print(f"❌ ERROR in Step 1: {e}")
    exit(1)

# STEP 2: efetch to get detailed information
print("\n\nSTEP 2: efetch.fcgi - Get Detailed Information")
print("-"*80)

efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
efetch_payload = {
    "db": "sra",
    "retmode": "xml",
    "id": ",".join(uids),
}

print(f"Fetching details for UIDs: {','.join(uids)}")

try:
    r2 = requests.get(efetch_url, params=efetch_payload, timeout=20, stream=True)
    r2.raise_for_status()

    print(f"✅ SUCCESS: Received response")
    print(f"Status Code: {r2.status_code}")
    print(f"Content-Type: {r2.headers.get('Content-Type', 'N/A')}")
    print(f"Content-Length: {r2.headers.get('Content-Length', 'N/A')}")

    # Read the content
    r2.raw.decode_content = True
    content = r2.content

    print(f"Downloaded {len(content)} bytes")

    # Parse the XML to see if it contains experiment packages
    print("\nParsing XML...")
    root = Et.fromstring(content)
    print(f"Root tag: {root.tag}")
    print(f"Root attrib: {root.attrib}")

    # Count EXPERIMENT_PACKAGE elements
    packages = root.findall(".//EXPERIMENT_PACKAGE")
    print(f"Found {len(packages)} EXPERIMENT_PACKAGE elements")

    if len(packages) > 0:
        print("\n✅ XML contains experiment data!")
        # Show first package structure
        first_package = packages[0]
        print("\nFirst EXPERIMENT_PACKAGE structure:")
        for child in first_package:
            print(f"  - {child.tag}: {len(child)} sub-elements")

        # Try to extract basic info
        exp = first_package.find(".//EXPERIMENT")
        if exp is not None:
            exp_acc = exp.get("accession", "N/A")
            print(f"\nFirst Experiment Accession: {exp_acc}")

            title = exp.find(".//TITLE")
            if title is not None:
                print(f"First Experiment Title: {title.text[:100] if title.text else 'N/A'}...")

    else:
        print("\n❌ No EXPERIMENT_PACKAGE elements found!")
        print("\nXML Structure:")
        for child in root:
            print(f"  - {child.tag}")

        # Show first 500 chars of XML
        print("\nFirst 500 characters of XML:")
        print(content[:500].decode('utf-8', errors='ignore'))

except Exception as e:
    print(f"❌ ERROR in Step 2: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("Now testing full pysradb.search() method...")
print("="*80)

from pysradb.search import SraSearch

instance = SraSearch(verbosity=2, return_max=5, query=query_term)
print(f"\nCalling instance.search()...")

result = instance.search()

print(f"\nResult type: {type(result)}")
print(f"Result: {result}")

if result is not None:
    print(f"Result shape: {result.shape if hasattr(result, 'shape') else 'N/A'}")
    if hasattr(result, 'columns'):
        print(f"Columns: {result.columns.tolist()}")
else:
    print("❌ pysradb returned None")

print("\n" + "="*80)
print("INVESTIGATION COMPLETE")
print("="*80)
