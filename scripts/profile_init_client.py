#!/usr/bin/env python3
"""
Profile what happens during init_client() to find bottlenecks.
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("🔬 Profiling init_client() Bottlenecks\n")
print("=" * 60)

# Step 1: Import cli module (should be fast now with lazy loading)
step_start = time.perf_counter()
from lobster import cli
step_time = time.perf_counter() - step_start
print(f"✅ CLI module import: {step_time * 1000:.0f}ms\n")

print("=" * 60)
print("📦 PROFILING CRITICAL IMPORTS INSIDE init_client()")
print("=" * 60)

results = []

# Test the imports that happen INSIDE init_client()
critical_imports = [
    ("AgentClient", "lobster.core.client"),
    ("Settings", "lobster.config.settings"),
    ("DataManagerV2", "lobster.core.data_manager_v2"),
]

for name, module in critical_imports:
    start = time.perf_counter()
    try:
        __import__(module)
        elapsed = time.perf_counter() - start
        results.append((name, module, elapsed))
        print(f"\n{name}")
        print(f"  Module: {module}")
        print(f"  Time:   {elapsed * 1000:.0f}ms")
    except Exception as e:
        print(f"\n{name}")
        print(f"  Module: {module}")
        print(f"  ERROR:  {e}")

print("\n" + "=" * 60)
print("📊 BREAKDOWN")
print("=" * 60)

total = sum(e for _, _, e in results)
for name, module, elapsed in sorted(results, key=lambda x: x[2], reverse=True):
    pct = (elapsed / total * 100) if total > 0 else 0
    print(f"{name:20s} {elapsed * 1000:>6.0f}ms ({pct:>5.1f}%)")

print(f"{'─' * 40}")
print(f"{'TOTAL':20s} {total * 1000:>6.0f}ms")

print("\n" + "=" * 60)
print("💡 WHAT'S SLOW?")
print("=" * 60)

# Check what AgentClient imports
print("\nLet's see what makes AgentClient slow...")
print("Testing AgentClient dependencies:")

agent_deps = [
    "langchain",
    "langchain_core",
    "langchain_anthropic",
    "langgraph",
    "scanpy",
    "anndata",
]

print()
for dep in agent_deps:
    start = time.perf_counter()
    try:
        __import__(dep)
        elapsed = time.perf_counter() - start
        if elapsed > 0.05:  # Only show slow ones (>50ms)
            print(f"  {dep:30s} {elapsed * 1000:>6.0f}ms")
    except ImportError:
        pass

print("\n" + "=" * 60)
print("✅ Profile complete!")
print("=" * 60)
