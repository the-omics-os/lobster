"""Diagnostic script to test SRA Provider Phase 1 implementation."""

import logging
import os
from pathlib import Path

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.providers.sra_provider import SRAProvider, SRAProviderConfig

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create temporary workspace
workspace = Path("/tmp/sra_test_workspace")
workspace.mkdir(exist_ok=True)

# Initialize provider
dm = DataManagerV2(workspace_path=str(workspace))
config = SRAProviderConfig(max_results=5)
provider = SRAProvider(data_manager=dm, config=config)

print("=" * 80)
print("DIAGNOSTIC TEST: SRAProvider Phase 1")
print("=" * 80)

# Test 1: Simple keyword search (no filters)
print("\n[TEST 1] Simple keyword search: 'cancer'")
print("-" * 80)
result1 = provider.search_publications("cancer", max_results=3)
print(f"\nResult length: {len(result1)} characters")
print(f"\nFirst 500 chars:\n{result1[:500]}")

# Test 2: Keyword with filter
print("\n" + "=" * 80)
print("[TEST 2] Keyword with filter: 'microbiome' + organism='Homo sapiens'")
print("-" * 80)
result2 = provider.search_publications(
    "microbiome", max_results=3, filters={"organism": "Homo sapiens"}
)
print(f"\nResult length: {len(result2)} characters")
print(f"\nFirst 500 chars:\n{result2[:500]}")

# Test 3: Accession lookup
print("\n" + "=" * 80)
print("[TEST 3] Accession lookup: 'SRP033351'")
print("-" * 80)
result3 = provider.search_publications("SRP033351", max_results=3)
print(f"\nResult length: {len(result3)} characters")
print(f"\nFirst 500 chars:\n{result3[:500]}")

# Test 4: Check NCBI API key
print("\n" + "=" * 80)
print("[CONFIGURATION]")
print("-" * 80)
print(f"NCBI API KEY set: {bool(os.environ.get('NCBI_API_KEY'))}")
print(f"Config API KEY set: {bool(config.api_key)}")
print(f"Config email: {config.email}")
print(f"Max results: {config.max_results}")

print("\n" + "=" * 80)
print("DIAGNOSTIC TEST COMPLETE")
print("=" * 80)
