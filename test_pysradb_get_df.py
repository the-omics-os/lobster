#!/usr/bin/env python3
"""
Test if pysradb requires calling .get_df() to retrieve results.
"""

from pysradb.search import SraSearch

print("="*80)
print("TESTING PYSRADB API: .search() vs .get_df()")
print("="*80)

instance = SraSearch(verbosity=2, return_max=5, query="microbiome")

print("\n1. Calling .search()...")
result = instance.search()
print(f"   .search() returned: {type(result)}")

print("\n2. Checking instance.df...")
print(f"   instance.df type: {type(instance.df)}")
print(f"   instance.df shape: {instance.df.shape if hasattr(instance.df, 'shape') else 'N/A'}")

print("\n3. Trying .get_df()...")
try:
    df = instance.get_df()
    print(f"   .get_df() returned: {type(df)}")
    if df is not None:
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {df.columns.tolist()}")
        print(f"\n   First 3 rows:")
        print(df.head(3))
        print("\n   ✅ SUCCESS - Data retrieved via .get_df()!")
except Exception as e:
    print(f"   ERROR: {e}")

print("\n4. Checking instance.df directly after search...")
if hasattr(instance, 'df') and not instance.df.empty:
    print(f"   instance.df shape: {instance.df.shape}")
    print(f"   instance.df columns: {instance.df.columns.tolist()}")
    print(f"\n   First 3 rows:")
    print(instance.df.head(3))
    print("\n   ✅ SUCCESS - Data accessible via instance.df!")

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)
print("pysradb.search.SraSearch API:")
print("  - .search() returns None (just populates internal .df)")
print("  - Use .get_df() or access .df directly to get results")
print("="*80)
