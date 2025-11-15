#!/usr/bin/env python3
"""
Diagnostic test to understand why pysradb SraSearch is failing.
"""

import sys
from pysradb.search import SraSearch

print("="*80)
print("PYSRADB DIAGNOSTIC TEST")
print("="*80)

# Test 1: Most basic query possible
print("\n1. Testing most basic query: 'cancer'")
print("-"*80)
try:
    instance = SraSearch(verbosity=3, return_max=3, query="cancer")
    print(f"SraSearch instance created: {instance}")
    print(f"Query: {instance.query}")
    print(f"Return max: {instance.return_max}")

    # Check what attributes the instance has
    print("\nInstance attributes:")
    for attr in dir(instance):
        if not attr.startswith('_'):
            try:
                val = getattr(instance, attr)
                if not callable(val):
                    print(f"  {attr}: {val}")
            except:
                pass

    print("\nExecuting search...")
    result = instance.search()
    print(f"Result type: {type(result)}")
    print(f"Result: {result}")

    if result is not None:
        print(f"Result length: {len(result)}")
        if len(result) > 0:
            print(f"Result columns: {result.columns.tolist() if hasattr(result, 'columns') else 'N/A'}")
            print(f"First row: {result.iloc[0] if hasattr(result, 'iloc') else 'N/A'}")
    else:
        print("Result is None - no results found")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Try with different parameters
print("\n\n2. Testing with accession instead of query")
print("-"*80)
try:
    instance = SraSearch(verbosity=3, return_max=3, accession="SRP000001")
    result = instance.search()
    print(f"Result: {result}")
    if result is not None:
        print(f"Result length: {len(result)}")
except Exception as e:
    print(f"ERROR: {e}")

# Test 3: Try without any filters
print("\n\n3. Testing simplest possible initialization")
print("-"*80)
try:
    instance = SraSearch(query="covid")
    result = instance.search()
    print(f"Result: {result}")
    if result is not None:
        print(f"Result length: {len(result)}")
except Exception as e:
    print(f"ERROR: {e}")

print("\n" + "="*80)
print("DIAGNOSTIC TEST COMPLETE")
print("="*80)
