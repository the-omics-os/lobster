#!/usr/bin/env python3
"""
Debug pysradb's XML parsing to see why it's not creating a DataFrame.
"""

import sys
from pysradb.search import SraSearch

# Monkey-patch the _format_result method to add debugging
original_format_result = SraSearch._format_result

def debug_format_result(self):
    print("\n" + "="*80)
    print("DEBUG: _format_result() called")
    print("="*80)

    print(f"Number of entries in self.number_entries: {getattr(self, 'number_entries', 'N/A')}")
    print(f"Keys in self.entries dict: {list(self.entries.keys()) if hasattr(self, 'entries') else 'No entries'}")

    if hasattr(self, 'entries') and self.entries:
        print(f"Number of keys in self.entries: {len(self.entries)}")
        # Show first few entries
        for i, (key, values) in enumerate(list(self.entries.items())[:5]):
            print(f"  {i+1}. '{key}': {len(values)} values, first value: {values[0] if values else 'EMPTY'}")

    # Call original method
    original_format_result(self)

    print(f"\nAfter calling original _format_result:")
    print(f"self.df type: {type(self.df)}")
    print(f"self.df shape: {self.df.shape if hasattr(self.df, 'shape') else 'N/A'}")
    print(f"self.df.empty: {self.df.empty if hasattr(self.df, 'empty') else 'N/A'}")

    if hasattr(self.df, 'empty') and not self.df.empty:
        print(f"self.df columns: {self.df.columns.tolist()}")
        print(f"self.df first row:\n{self.df.iloc[0] if len(self.df) > 0 else 'N/A'}")
    print("="*80 + "\n")

SraSearch._format_result = debug_format_result

# Monkey-patch _format_response to add debugging
original_format_response = SraSearch._format_response

def debug_format_response(self, content):
    print("\n" + "="*80)
    print("DEBUG: _format_response() called")
    print("="*80)
    print(f"Content type: {type(content)}")
    print(f"self.number_entries at start: {getattr(self, 'number_entries', 'NOT SET')}")

    # Call original method
    original_format_response(self, content)

    print(f"\nAfter parsing XML:")
    print(f"self.number_entries: {getattr(self, 'number_entries', 'N/A')}")
    print(f"Keys in self.entries: {len(self.entries.keys()) if hasattr(self, 'entries') else 'N/A'}")
    print("="*80 + "\n")

SraSearch._format_response = debug_format_response

# Now run the search
print("="*80)
print("RUNNING PYSRADB SEARCH WITH DEBUG INSTRUMENTATION")
print("="*80)

instance = SraSearch(verbosity=2, return_max=5, query="microbiome")
print(f"\nSraSearch instance created")
print(f"Instance attributes before search:")
print(f"  - verbosity: {instance.verbosity}")
print(f"  - return_max: {instance.return_max}")
print(f"  - number_entries: {getattr(instance, 'number_entries', 'NOT SET')}")
print(f"  - entries: {hasattr(instance, 'entries')}")

print("\nCalling search()...")
result = instance.search()

print("\n" + "="*80)
print("SEARCH COMPLETE")
print("="*80)
print(f"Result type: {type(result)}")
print(f"Result: {result}")

if result is not None:
    print(f"Result shape: {result.shape}")
    print(f"Result columns: {result.columns.tolist()}")
else:
    print("❌ Search returned None")

print("\n" + "="*80)
print("DEBUG COMPLETE")
print("="*80)
