#!/usr/bin/env python3
"""
Test actual startup experience timing - from command to ready prompt.
This simulates what the user experiences.
"""

import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("🧪 Testing Actual Startup Experience\n")
print("=" * 60)

# Measure full flow
total_start = time.perf_counter()

print("\n⏱️  Step 1: Import lobster.cli module")
step1_start = time.perf_counter()
import lobster.cli
step1_time = time.perf_counter() - step1_start
print(f"   ✅ Complete in {step1_time * 1000:.0f}ms")

print("\n⏱️  Step 2: Simulate animation (without actual display)")
step2_start = time.perf_counter()
# Simulate animation duration
animation_duration = 0.7
time.sleep(animation_duration)
step2_time = time.perf_counter() - step2_start
print(f"   ✅ Animation plays for {step2_time * 1000:.0f}ms")

print("\n⏱️  Step 3: Would call init_client() here")
print("   (Not testing actual init to avoid full setup)")

total_time = time.perf_counter() - total_start

print("\n" + "=" * 60)
print("📊 TIMING BREAKDOWN")
print("=" * 60)
print(f"  Module load:     {step1_time * 1000:>6.0f}ms")
print(f"  Animation:       {step2_time * 1000:>6.0f}ms")
print(f"  ─────────────────────────")
print(f"  Total (so far):  {total_time * 1000:>6.0f}ms")
print()
print("  (Init would add ~4-5s for AgentClient creation)")

print("\n" + "=" * 60)
print("💡 USER PERCEPTION")
print("=" * 60)

if step1_time < 0.1:
    print(f"✅ INSTANT: User sees animation start in {step1_time * 1000:.0f}ms")
elif step1_time < 0.5:
    print(f"✅ FAST: Animation starts in {step1_time * 1000:.0f}ms (feels instant)")
elif step1_time < 1.0:
    print(f"⚠️  NOTICEABLE: {step1_time * 1000:.0f}ms delay before animation")
else:
    print(f"❌ SLOW: {step1_time:.1f}s delay is too long")

print(f"\nAnimation plays while user waits = distraction from init time!")
print("\n" + "=" * 60)
print("✅ Test complete!")
print("=" * 60)
