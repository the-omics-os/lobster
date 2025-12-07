#!/usr/bin/env python3
"""
Profile lobster CLI startup to identify performance bottlenecks.

Usage:
    python scripts/profile_startup.py
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def time_import(module_name: str) -> float:
    """Time how long it takes to import a module."""
    start = time.perf_counter()
    try:
        __import__(module_name)
        elapsed = time.perf_counter() - start
        return elapsed
    except ImportError as e:
        print(f"  ❌ Failed to import {module_name}: {e}")
        return 0.0


def format_time(seconds: float) -> str:
    """Format time with appropriate units."""
    if seconds < 0.001:
        return f"{seconds * 1000000:.0f}µs"
    elif seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    else:
        return f"{seconds:.2f}s"


print("🔬 Profiling Lobster CLI Startup\n")
print("=" * 60)

# Track total time
total_start = time.perf_counter()

results = []

print("\n📦 STANDARD LIBRARY IMPORTS")
print("-" * 60)
for module in ["os", "sys", "pathlib", "typing", "json", "time", "threading"]:
    elapsed = time_import(module)
    results.append((module, elapsed, "stdlib"))
    print(f"  {module:30s} {format_time(elapsed):>10s}")

print("\n📦 THIRD-PARTY IMPORTS (Light)")
print("-" * 60)
for module in ["typer", "rich"]:
    elapsed = time_import(module)
    results.append((module, elapsed, "light"))
    print(f"  {module:30s} {format_time(elapsed):>10s}")

print("\n📦 THIRD-PARTY IMPORTS (Heavy)")
print("-" * 60)
for module in ["numpy", "pandas"]:
    elapsed = time_import(module)
    results.append((module, elapsed, "heavy"))
    print(f"  {module:30s} {format_time(elapsed):>10s}")

print("\n📦 LOBSTER CORE IMPORTS")
print("-" * 60)
lobster_modules = [
    "lobster.version",
    "lobster.config.settings",
    "lobster.core.workspace",
    "lobster.ui",
    "lobster.utils",
    "lobster.core.client",  # This is the big one
]
for module in lobster_modules:
    elapsed = time_import(module)
    results.append((module, elapsed, "lobster"))
    print(f"  {module:30s} {format_time(elapsed):>10s}")

print("\n📦 BIOINFORMATICS LIBRARIES (via AgentClient)")
print("-" * 60)
bio_modules = [
    "scanpy",
    "anndata",
]
for module in bio_modules:
    elapsed = time_import(module)
    results.append((module, elapsed, "bio"))
    print(f"  {module:30s} {format_time(elapsed):>10s}")

total_elapsed = time.perf_counter() - total_start

print("\n" + "=" * 60)
print("📊 SUMMARY")
print("=" * 60)

# Group by category
categories = {}
for name, elapsed, category in results:
    if category not in categories:
        categories[category] = []
    categories[category].append((name, elapsed))

for category, items in categories.items():
    total_cat = sum(t for _, t in items)
    print(f"\n{category.upper():20s} {format_time(total_cat):>10s}")
    for name, elapsed in sorted(items, key=lambda x: x[1], reverse=True)[:3]:
        pct = (elapsed / total_elapsed) * 100
        print(f"  └─ {name:26s} {format_time(elapsed):>10s} ({pct:>5.1f}%)")

print(f"\n{'TOTAL IMPORT TIME':20s} {format_time(total_elapsed):>10s}")

# Find top 5 slowest imports
print("\n🐌 TOP 5 SLOWEST IMPORTS")
print("-" * 60)
sorted_results = sorted(results, key=lambda x: x[1], reverse=True)[:5]
for i, (name, elapsed, category) in enumerate(sorted_results, 1):
    pct = (elapsed / total_elapsed) * 100
    print(f"  {i}. {name:30s} {format_time(elapsed):>10s} ({pct:>5.1f}%)")

print("\n" + "=" * 60)
print("💡 RECOMMENDATIONS")
print("=" * 60)

# Analyze and provide recommendations
heavy_imports = [(n, e) for n, e, c in results if e > 0.1]  # > 100ms
if heavy_imports:
    print(f"\n⚠️  Found {len(heavy_imports)} imports taking >100ms:")
    for name, elapsed in heavy_imports:
        print(f"  • {name} ({format_time(elapsed)})")
    print("\n  → Consider lazy loading these imports inside functions")

very_heavy = [(n, e) for n, e, c in results if e > 0.5]  # > 500ms
if very_heavy:
    print(f"\n🔥 Critical bottlenecks (>500ms):")
    for name, elapsed in very_heavy:
        print(f"  • {name} ({format_time(elapsed)})")
    print("\n  → These MUST be lazy loaded for instant startup")

print("\n" + "=" * 60)
print("✅ Profile complete!")
print("=" * 60)
