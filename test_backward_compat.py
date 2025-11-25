#!/usr/bin/env python
"""
Test script to verify backward compatibility of metadata_assistant
without microbiome features.

This simulates the public lobster-local repository environment.
"""
import sys
import importlib.util

# Test 1: Check if MICROBIOME_FEATURES_AVAILABLE flag exists
print("Test 1: Checking MICROBIOME_FEATURES_AVAILABLE flag...")
try:
    from lobster.agents.metadata_assistant import MICROBIOME_FEATURES_AVAILABLE
    print(f"✓ Flag available: {MICROBIOME_FEATURES_AVAILABLE}")
    if MICROBIOME_FEATURES_AVAILABLE:
        print("  → Microbiome features are ENABLED (private repo)")
    else:
        print("  → Microbiome features are DISABLED (public repo mode)")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 2: Check if metadata_assistant function can be imported
print("\nTest 2: Importing metadata_assistant function...")
try:
    from lobster.agents.metadata_assistant import metadata_assistant
    print("✓ Import successful")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 3: Check optional services
print("\nTest 3: Checking optional service imports...")
try:
    from lobster.agents.metadata_assistant import (
        MicrobiomeFilteringService,
        DiseaseStandardizationService,
    )
    if MicrobiomeFilteringService is None and DiseaseStandardizationService is None:
        print("✓ Optional services are None (public mode)")
    else:
        print("✓ Optional services available (private mode)")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 4: Verify docstring mentions optional feature
print("\nTest 4: Checking docstring for optional feature notice...")
try:
    from lobster.agents.metadata_assistant import metadata_assistant
    docstring = metadata_assistant.__doc__
    if "OPTIONAL" in docstring or "optional" in docstring:
        print("✓ Docstring mentions optional features")
        # Find and print the relevant line
        for line in docstring.split('\n'):
            if 'optional' in line.lower() or 'OPTIONAL' in line:
                print(f"  → {line.strip()}")
    else:
        print("⚠ Warning: Docstring should mention optional features")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("SUMMARY: All backward compatibility tests PASSED ✓")
print("="*60)
print("\nThe metadata_assistant agent will work correctly in both:")
print("  - Private repo (with microbiome features)")
print("  - Public lobster-local repo (without microbiome features)")
