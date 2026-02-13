"""
Memory Exhaustion Attack Tests for CustomCodeExecutionService.

Tests memory allocation attacks including:
- Large list/string allocation
- numpy array bombs
- Recursive data structures
- Memory leaks via circular references

SAFETY: Uses 500MB max per test (safe for testing)
REAL ATTACK: Could allocate 10GB+ and trigger OOM killer

Run with: pytest tests/manual/custom_code_execution/02_resource_exhaustion/test_memory_bombs.py -v -s
"""

import sys
from pathlib import Path

import pytest

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.execution.custom_code_execution_service import (
    CodeExecutionError,
    CustomCodeExecutionService,
)


@pytest.fixture
def service(tmp_path):
    """Create service instance with temporary workspace."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    data_manager = DataManagerV2(workspace_path=workspace)
    return CustomCodeExecutionService(data_manager)


class TestMemoryAllocationBombs:
    """Test memory allocation attacks that could exhaust system RAM."""

    def test_large_list_allocation_EXPECT_SUCCESS(self, service):
        """
        VULNERABILITY: Large list allocation (400MB)

        Test: Allocate ~400MB list
        Expected: SUCCESS - No memory limit enforced
        Impact: HIGH

        SAFE LIMIT: 10^8 elements = ~400MB
        REAL ATTACK: 10^9 elements = ~4GB (could trigger OOM)

        Mitigation: cgroups memory.max, Docker --memory flag
        """
        code = """
import sys
# Allocate ~400MB (safe for testing)
x = [0] * (10**8)
result = f"Allocated {sys.getsizeof(x) / 1024**2:.0f}MB"
"""

        try:
            result, stats, ir = service.execute(code, persist=False, timeout=30)
            print(f"\n⚠️  VULNERABILITY CONFIRMED: Memory allocation succeeded")
            print(f"    Result: {result}")
            print(f"    Duration: {stats['duration_seconds']}s")
            print(f"    🔥 REAL ATTACK: Could allocate 10GB+ and crash system")
            assert stats["success"] is True
            assert "Allocated" in result
        except CodeExecutionError as e:
            print(f"\n✅ PROTECTED: Memory allocation blocked: {e}")
            pytest.fail("Expected vulnerability but got protection (good!)")

    def test_large_string_allocation_EXPECT_SUCCESS(self, service):
        """
        VULNERABILITY: Large string allocation (100MB)

        Test: Allocate 100MB string
        Expected: SUCCESS - No memory limit
        Impact: HIGH

        SAFE LIMIT: 10^8 chars = 100MB
        REAL ATTACK: 10^9 chars = 1GB string

        Mitigation: cgroups memory.max
        """
        code = """
import sys
# Allocate 100MB string (safe)
s = 'A' * (10**8)
result = f"String size: {sys.getsizeof(s) / 1024**2:.0f}MB"
"""

        try:
            result, stats, ir = service.execute(code, persist=False, timeout=30)
            print(f"\n⚠️  VULNERABILITY CONFIRMED: Large string allocated")
            print(f"    Result: {result}")
            print(f"    🔥 REAL ATTACK: 1GB+ strings could exhaust RAM")
            assert stats["success"] is True
        except CodeExecutionError as e:
            print(f"\n✅ PROTECTED: String allocation blocked: {e}")
            pytest.fail("Expected vulnerability")

    def test_numpy_array_bomb_EXPECT_SUCCESS(self, service):
        """
        VULNERABILITY: numpy array memory bomb

        Test: Create 100MB numpy array
        Expected: SUCCESS - No limits
        Impact: HIGH

        SAFE LIMIT: 10^4 x 10^4 float64 = ~800MB
        REAL ATTACK: 10^5 x 10^5 = 80GB (would crash)

        Mitigation: cgroups + Docker memory limits
        """
        code = """
import numpy as np
# Create ~800MB array (safe)
arr = np.zeros((10**4, 10**4), dtype=np.float64)
result = f"Array size: {arr.nbytes / 1024**2:.0f}MB, shape: {arr.shape}"
"""

        try:
            result, stats, ir = service.execute(code, persist=False, timeout=60)
            print(f"\n⚠️  VULNERABILITY CONFIRMED: Large numpy array created")
            print(f"    Result: {result}")
            print(f"    🔥 REAL ATTACK: 80GB arrays would trigger OOM killer")
            assert stats["success"] is True
            assert "MB" in result
        except CodeExecutionError as e:
            print(f"\n✅ PROTECTED: numpy array blocked: {e}")
            pytest.fail("Expected vulnerability")

    def test_multiple_allocations_EXPECT_SUCCESS(self, service):
        """
        VULNERABILITY: Death by a thousand allocations

        Test: Allocate 10 x 50MB = 500MB total
        Expected: SUCCESS - No tracking of cumulative memory
        Impact: HIGH

        REAL ATTACK: 100 x 100MB = 10GB death by cuts

        Mitigation: cgroups tracks total process memory
        """
        code = """
import sys
# Allocate 10 separate 50MB lists
allocations = []
for i in range(10):
    allocations.append([0] * (5 * 10**7))  # 50MB each

total_mb = sum(sys.getsizeof(x) for x in allocations) / 1024**2
result = f"Total allocated: {total_mb:.0f}MB across {len(allocations)} lists"
"""

        try:
            result, stats, ir = service.execute(code, persist=False, timeout=30)
            print(f"\n⚠️  VULNERABILITY CONFIRMED: Multiple allocations succeeded")
            print(f"    Result: {result}")
            print(f"    🔥 REAL ATTACK: 100 x 100MB = 10GB incremental exhaustion")
            assert stats["success"] is True
        except CodeExecutionError as e:
            print(f"\n✅ PROTECTED: Multiple allocations blocked: {e}")
            pytest.fail("Expected vulnerability")


class TestRecursiveDataStructures:
    """Test memory exhaustion via recursive/circular data structures."""

    def test_recursive_list_bomb_EXPECT_SUCCESS(self, service):
        """
        VULNERABILITY: Recursive list creation

        Test: Create nested lists (safe depth)
        Expected: SUCCESS - No depth limit
        Impact: MEDIUM

        SAFE LIMIT: 100 levels
        REAL ATTACK: 10000 levels could exhaust stack + heap

        Mitigation: Python recursion limit (default 1000) + memory limits
        """
        code = """
# Create nested list structure (100 levels - safe)
def create_nested_list(depth):
    if depth == 0:
        return []
    return [create_nested_list(depth - 1)]

nested = create_nested_list(100)

# Measure depth
def measure_depth(obj):
    if not isinstance(obj, list) or not obj:
        return 0
    return 1 + measure_depth(obj[0])

result = f"Nested list depth: {measure_depth(nested)}"
"""

        try:
            result, stats, ir = service.execute(code, persist=False, timeout=10)
            print(f"\n⚠️  VULNERABILITY: Recursive structures allowed")
            print(f"    Result: {result}")
            print(f"    🔥 REAL ATTACK: 10000 levels could cause stack overflow")
            assert stats["success"] is True
        except CodeExecutionError as e:
            print(f"\n✅ PROTECTED: Recursion blocked: {e}")
            pytest.fail("Expected vulnerability")

    def test_circular_reference_memory_leak_EXPECT_SUCCESS(self, service):
        """
        VULNERABILITY: Circular references (GC still works)

        Test: Create circular references
        Expected: SUCCESS - Python GC handles this
        Impact: LOW (Python has cycle detector)

        Note: Not a real vulnerability in Python but shows no external monitoring
        """
        code = """
import gc
import sys

# Create circular references
class Node:
    def __init__(self):
        self.ref = None
        self.data = [0] * 1000000  # 1MB per node

nodes = []
for i in range(100):  # 100MB total
    node = Node()
    if nodes:
        node.ref = nodes[-1]
        nodes[-1].ref = node
    nodes.append(node)

# Force garbage collection
collected = gc.collect()

result = f"Created {len(nodes)} circular refs (~100MB), GC collected {collected} objects"
"""

        try:
            result, stats, ir = service.execute(code, persist=False, timeout=10)
            print(f"\n✅ Python GC handles circular refs (not a vulnerability)")
            print(f"    Result: {result}")
            print(f"    Note: External memory monitoring still recommended")
            assert stats["success"] is True
        except CodeExecutionError as e:
            print(f"\n✅ PROTECTED: Circular refs blocked: {e}")


class TestPandasMemoryBombs:
    """Test memory exhaustion via pandas DataFrames."""

    def test_large_dataframe_allocation_EXPECT_SUCCESS(self, service):
        """
        VULNERABILITY: Large pandas DataFrame

        Test: Create 100MB DataFrame
        Expected: SUCCESS - No limits
        Impact: HIGH

        SAFE LIMIT: 10^6 rows x 10 cols = ~100MB
        REAL ATTACK: 10^8 rows = 10GB DataFrame

        Mitigation: Memory limits via cgroups
        """
        code = """
import pandas as pd
import numpy as np

# Create ~100MB DataFrame (safe)
df = pd.DataFrame({
    f'col_{i}': np.random.rand(10**6)
    for i in range(10)
})

result = f"DataFrame: {df.shape}, Memory: {df.memory_usage(deep=True).sum() / 1024**2:.0f}MB"
"""

        try:
            result, stats, ir = service.execute(code, persist=False, timeout=30)
            print(f"\n⚠️  VULNERABILITY CONFIRMED: Large DataFrame created")
            print(f"    Result: {result}")
            print(f"    🔥 REAL ATTACK: 10GB DataFrames would exhaust RAM")
            assert stats["success"] is True
        except CodeExecutionError as e:
            print(f"\n✅ PROTECTED: DataFrame blocked: {e}")
            pytest.fail("Expected vulnerability")

    def test_dataframe_concat_bomb_EXPECT_SUCCESS(self, service):
        """
        VULNERABILITY: Memory bomb via repeated concatenation

        Test: Concatenate DataFrames in loop
        Expected: SUCCESS - No tracking
        Impact: HIGH

        SAFE LIMIT: 100 concatenations = ~100MB
        REAL ATTACK: 1000 concatenations = 1GB+
        """
        code = """
import pandas as pd
import numpy as np

# Start with small DataFrame
df = pd.DataFrame({'a': [1, 2, 3]})

# Repeatedly concatenate (doubles each time - exponential growth!)
for i in range(15):  # 2^15 = 32768 rows
    df = pd.concat([df, df], ignore_index=True)

result = f"Final DataFrame: {len(df)} rows, {df.memory_usage(deep=True).sum() / 1024:.0f}KB"
"""

        try:
            result, stats, ir = service.execute(code, persist=False, timeout=30)
            print(f"\n⚠️  VULNERABILITY CONFIRMED: Exponential DataFrame growth")
            print(f"    Result: {result}")
            print(f"    🔥 REAL ATTACK: 20 iterations = 1M+ rows, memory explosion")
            assert stats["success"] is True
        except CodeExecutionError as e:
            print(f"\n✅ PROTECTED: DataFrame growth blocked: {e}")
            pytest.fail("Expected vulnerability")


class TestMemoryExhaustionSummary:
    """Summary test documenting all memory vulnerabilities."""

    def test_memory_vulnerability_summary(self, service):
        """
        SUMMARY: Memory Exhaustion Vulnerabilities

        CONFIRMED VULNERABILITIES:
        1. ✅ Large list allocation (400MB+ possible)
        2. ✅ Large string allocation (100MB+ possible)
        3. ✅ numpy array bombs (800MB+ possible)
        4. ✅ Multiple allocations (cumulative 500MB+)
        5. ✅ Recursive data structures (100+ levels)
        6. ✅ Large pandas DataFrames (100MB+ possible)
        7. ✅ Exponential DataFrame growth

        CURRENT PROTECTION: NONE (only 300s timeout)

        IMPACT: HIGH
        - Can exhaust system RAM
        - Trigger OOM killer
        - Crash other processes
        - Denial of service

        RECOMMENDED MITIGATIONS:
        1. cgroups memory.max (e.g., 2GB per execution)
        2. Docker --memory flag
        3. systemd MemoryMax directive
        4. Monitor RSS via psutil before subprocess
        5. Kubernetes ResourceQuota

        Example Docker mitigation:
        ```bash
        docker run --memory=2g --memory-swap=2g lobster
        ```

        Example cgroups mitigation (Linux):
        ```python
        # In subprocess setup
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
        ```
        """
        print("\n" + "=" * 70)
        print("MEMORY EXHAUSTION VULNERABILITY SUMMARY")
        print("=" * 70)
        print(self.test_memory_vulnerability_summary.__doc__)

        # Just a marker test
        assert True
