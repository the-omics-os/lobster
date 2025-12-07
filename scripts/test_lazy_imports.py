#!/usr/bin/env python3
"""
Test the performance improvement from lazy imports.
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("🧪 Testing Lazy Import Performance\n")
print("=" * 60)

# Test 1: Time cli.py module import
print("\n📦 Test 1: Import lobster.cli module")
print("-" * 60)

start = time.perf_counter()
import lobster.cli
elapsed = time.perf_counter() - start

print(f"✅ Module imported in: {elapsed * 1000:.0f}ms")

# Test 2: Compare to profiler expectations
print("\n📊 Test 2: Expected Improvement")
print("-" * 60)
print("Before lazy imports: ~5.4s")
print(f"After lazy imports:  {elapsed:.2f}s")
print(f"Speed improvement:   {5.4 / elapsed if elapsed > 0 else 0:.1f}x faster")
print(f"Time saved:          {(5.4 - elapsed) * 1000:.0f}ms")

# Test 3: Verify animation can start quickly
print("\n🎬 Test 3: Animation Readiness")
print("-" * 60)
if elapsed < 0.1:
    print("✅ EXCELLENT: Animation can start instantly (<100ms)")
elif elapsed < 0.5:
    print("✅ GOOD: Animation starts quickly (<500ms)")
elif elapsed < 1.0:
    print("⚠️  OK: Some delay but acceptable (<1s)")
else:
    print(f"❌ POOR: Still too slow ({elapsed:.2f}s)")

print("\n" + "=" * 60)
print("✅ Test complete!")
print("=" * 60)
