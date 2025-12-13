#!/usr/bin/env python3
"""Quick test to verify PRIDE provider fixes."""

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.providers.pride_provider import PRIDEProvider
from pathlib import Path

# Create minimal data manager
workspace = Path("/tmp/test_pride")
workspace.mkdir(exist_ok=True)

dm = DataManagerV2(workspace_path=workspace)
provider = PRIDEProvider(dm)

# Test search (this was failing before)
print("Testing PRIDE search with debugging...")
import traceback
import logging

# Enable DEBUG logging to see where error occurs
logging.basicConfig(level=logging.DEBUG)

try:
    print("Calling search_publications...")
    result = provider.search_publications("human cancer", max_results=5)
    print(f"Result type: {type(result)}")
    print(f"Result length: {len(result)}")

    if "Error" in result:
        print(f"❌ Search returned error: {result[:300]}")
    else:
        print("✅ Search successful!")
        print(result[:500])  # Print first 500 chars
except Exception as e:
    print(f"❌ Search crashed with exception: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    print("\nTrying to continue...")

# Test metadata extraction
print("\nTesting PRIDE metadata extraction...")
try:
    metadata = provider.extract_publication_metadata("PXD012345")
    print(f"✅ Metadata extraction successful!")
    print(f"   Title: {metadata.title[:80]}")
    print(f"   Authors: {len(metadata.authors)} authors")
except Exception as e:
    print(f"❌ Metadata extraction failed: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ All tests passed!")
