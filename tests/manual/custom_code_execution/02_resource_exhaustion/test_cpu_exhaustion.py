"""
CPU Exhaustion Attack Tests for CustomCodeExecutionService.

Tests CPU-intensive operations including:
- Infinite loops (timeout test)
- Prime number calculations
- Nested loops (O(n^3) complexity)
- Cryptographic operations
- Regex catastrophic backtracking

SAFETY: Uses short timeouts (5-10s) to avoid long waits
PROTECTION: Only 300s timeout (no CPU throttling)

Run with: pytest tests/manual/custom_code_execution/02_resource_exhaustion/test_cpu_exhaustion.py -v -s
"""

import time

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


class TestInfiniteLoops:
    """Test timeout protection against infinite loops."""

    def test_infinite_loop_hits_timeout_EXPECT_ERROR(self, service):
        """
        PARTIAL PROTECTION: Infinite loop hits 300s timeout

        Test: Run infinite loop with 5s timeout
        Expected: ERROR after 5s timeout
        Impact: MEDIUM (timeout works, but 300s is LONG)

        VULNERABILITY: Default 300s allows 5 minutes of CPU burn
        REAL ATTACK: Spin all cores at 100% for 5 minutes

        Mitigation: Reduce default timeout to 30s, add CPU limits
        """
        code = """
# Infinite loop (will hit timeout)
while True:
    x = 1 + 1
"""

        start = time.time()
        try:
            result, stats, ir = service.execute(code, persist=False, timeout=5)
            pytest.fail("Expected timeout error, got success")
        except CodeExecutionError as e:
            elapsed = time.time() - start
            print(f"\n✅ TIMEOUT WORKS: Infinite loop killed after {elapsed:.1f}s")
            print(f"    Error: {e}")
            print(f"    ⚠️  BUT: Default 300s timeout is too long!")
            print(f"    🔥 REAL ATTACK: 300s of CPU burn = denial of service")
            assert "timeout" in str(e).lower()
            assert elapsed >= 4.5 and elapsed <= 6  # Should be ~5s

    def test_infinite_loop_default_timeout_WARNING(self, service):
        """
        WARNING: Default 300s timeout is dangerously long

        This test documents the vulnerability but doesn't actually run for 300s.

        VULNERABILITY: Default timeout allows 5 minutes of CPU abuse
        Impact: HIGH
        - Burns CPU for 5 full minutes
        - Blocks other work (if running sequentially)
        - Multiple executions = extended DoS

        Mitigation: Reduce default to 30s or add CPU% limits
        """
        print("\n" + "=" * 70)
        print("⚠️  WARNING: Default 300s timeout vulnerability")
        print("=" * 70)
        print("Code like 'while True: pass' would run for 5 MINUTES")
        print("This allows sustained CPU exhaustion attacks")
        print("")
        print("Mitigation needed:")
        print("1. Reduce DEFAULT_TIMEOUT to 30s")
        print("2. Add CPU throttling via cgroups cpu.max")
        print("3. Add per-user rate limits")
        print("=" * 70)
        assert True  # Just documentation

    def test_cpu_intensive_loop_EXPECT_SUCCESS(self, service):
        """
        VULNERABILITY: CPU-intensive loop (no CPU limits)

        Test: Run tight loop for 5 seconds
        Expected: SUCCESS (burns CPU, no protection)
        Impact: HIGH

        REAL ATTACK: Spin all cores at 100% until timeout

        Mitigation: cgroups cpu.max (e.g., 50% of 1 core)
        """
        code = """
import time
# Burn CPU for 5 seconds
start = time.time()
count = 0
while time.time() - start < 5:
    count += 1

result = f"Executed {count:,} iterations in 5 seconds (CPU at 100%)"
"""

        try:
            result, stats, ir = service.execute(code, persist=False, timeout=10)
            print(f"\n⚠️  VULNERABILITY CONFIRMED: CPU burn succeeded")
            print(f"    Result: {result}")
            print(f"    Duration: {stats['duration_seconds']}s")
            print(f"    🔥 REAL ATTACK: Could run for 300s at 100% CPU")
            assert stats["success"] is True
            assert "iterations" in result
        except CodeExecutionError as e:
            print(f"\n✅ PROTECTED: CPU burn blocked: {e}")
            pytest.fail("Expected vulnerability")


class TestComputationallyExpensiveOperations:
    """Test computationally expensive operations."""

    def test_prime_number_calculation_EXPECT_SUCCESS(self, service):
        """
        VULNERABILITY: Expensive prime calculation

        Test: Find primes up to 100,000
        Expected: SUCCESS (no CPU limits)
        Impact: HIGH

        SAFE LIMIT: 100,000
        REAL ATTACK: 10,000,000+ would burn CPU for minutes

        Mitigation: CPU throttling via cgroups
        """
        code = """
# Calculate primes up to 100,000 (computationally expensive)
def find_primes(n):
    primes = []
    for num in range(2, n):
        is_prime = True
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
    return primes

primes = find_primes(100000)
result = f"Found {len(primes)} primes up to 100,000"
"""

        try:
            result, stats, ir = service.execute(code, persist=False, timeout=60)
            print(f"\n⚠️  VULNERABILITY CONFIRMED: Expensive computation allowed")
            print(f"    Result: {result}")
            print(f"    Duration: {stats['duration_seconds']}s")
            print(f"    🔥 REAL ATTACK: find_primes(10_000_000) = minutes of CPU")
            assert stats["success"] is True
        except CodeExecutionError as e:
            print(f"\n✅ PROTECTED: Prime calculation blocked: {e}")
            pytest.fail("Expected vulnerability")

    def test_nested_loops_cubic_complexity_EXPECT_SUCCESS(self, service):
        """
        VULNERABILITY: O(n^3) nested loops

        Test: 100^3 = 1,000,000 iterations
        Expected: SUCCESS (no complexity limits)
        Impact: HIGH

        SAFE LIMIT: n=100
        REAL ATTACK: n=1000 = 1 billion iterations (minutes of CPU)

        Mitigation: Timeout helps, but CPU throttling needed
        """
        code = """
# O(n^3) nested loops
n = 100
count = 0
for i in range(n):
    for j in range(n):
        for k in range(n):
            count += 1

result = f"Executed {count:,} iterations (O(n^3) with n={n})"
"""

        try:
            result, stats, ir = service.execute(code, persist=False, timeout=30)
            print(f"\n⚠️  VULNERABILITY CONFIRMED: O(n^3) loops allowed")
            print(f"    Result: {result}")
            print(f"    Duration: {stats['duration_seconds']}s")
            print(f"    🔥 REAL ATTACK: n=1000 = 1 billion iterations")
            assert stats["success"] is True
        except CodeExecutionError as e:
            print(f"\n✅ PROTECTED: Nested loops blocked: {e}")
            pytest.fail("Expected vulnerability")

    def test_matrix_multiplication_bomb_EXPECT_SUCCESS(self, service):
        """
        VULNERABILITY: Large matrix operations

        Test: 1000x1000 matrix multiplication
        Expected: SUCCESS (CPU intensive, no limits)
        Impact: HIGH

        SAFE LIMIT: 1000x1000
        REAL ATTACK: 5000x5000 = minutes of CPU + GB of RAM

        Mitigation: CPU + memory limits
        """
        code = """
import numpy as np
import time

# Create large matrices and multiply (CPU intensive)
start = time.time()
A = np.random.rand(1000, 1000)
B = np.random.rand(1000, 1000)
C = np.matmul(A, B)
elapsed = time.time() - start

result = f"Matrix multiply (1000x1000): {elapsed:.2f}s"
"""

        try:
            result, stats, ir = service.execute(code, persist=False, timeout=30)
            print(f"\n⚠️  VULNERABILITY CONFIRMED: Large matrix ops allowed")
            print(f"    Result: {result}")
            print(f"    🔥 REAL ATTACK: 5000x5000 matrices = extended CPU burn")
            assert stats["success"] is True
        except CodeExecutionError as e:
            print(f"\n✅ PROTECTED: Matrix ops blocked: {e}")
            pytest.fail("Expected vulnerability")


class TestCryptographicOperations:
    """Test expensive cryptographic operations."""

    def test_bcrypt_high_rounds_EXPECT_SUCCESS(self, service):
        """
        VULNERABILITY: Expensive bcrypt operations

        Test: bcrypt with 12 rounds (standard)
        Expected: SUCCESS (intentionally slow)
        Impact: MEDIUM

        Note: bcrypt is DESIGNED to be slow (good for passwords)
        But repeated executions = DoS

        SAFE LIMIT: 12 rounds
        REAL ATTACK: 100 bcrypt(rounds=14) calls = minutes

        Mitigation: Rate limiting + timeout
        """
        code = """
# Note: bcrypt may not be installed, use hashlib as alternative
import hashlib
import time

# Simulate expensive hashing with repeated SHA256
password = b"test_password"
start = time.time()

# Perform 100,000 rounds of hashing (simulates bcrypt work)
result = password
for i in range(100000):
    result = hashlib.sha256(result).digest()

elapsed = time.time() - start
result = f"100k hash rounds: {elapsed:.2f}s"
"""

        try:
            result, stats, ir = service.execute(code, persist=False, timeout=30)
            print(f"\n⚠️  VULNERABILITY CONFIRMED: Expensive hashing allowed")
            print(f"    Result: {result}")
            print(f"    🔥 REAL ATTACK: Repeated expensive crypto = CPU exhaustion")
            assert stats["success"] is True
        except CodeExecutionError as e:
            print(f"\n✅ PROTECTED: Hashing blocked: {e}")
            pytest.fail("Expected vulnerability")


class TestRegexBacktracking:
    """Test regex catastrophic backtracking."""

    def test_regex_catastrophic_backtracking_EXPECT_SUCCESS(self, service):
        """
        VULNERABILITY: Regex catastrophic backtracking

        Test: Evil regex pattern (safe input size)
        Expected: SUCCESS (but exponential complexity)
        Impact: HIGH

        SAFE LIMIT: 20 character input
        REAL ATTACK: 30+ character input = minutes of CPU

        Mitigation: Timeout helps, but regex validation recommended

        Example evil patterns:
        - (a+)+b
        - (a*)*b
        - (a|a)*b
        """
        code = """
import re
import time

# Evil regex pattern (exponential backtracking)
# Pattern: (a+)+ trying to match string of all 'a's (no 'b' = backtracking)
pattern = r'^(a+)+$'

# Safe input size (20 chars)
text = 'a' * 20 + 'b'  # 'b' at end causes backtracking

start = time.time()
try:
    match = re.match(pattern, text, timeout=5)  # Python 3.11+ has timeout
except Exception as e:
    pass
elapsed = time.time() - start

result = f"Regex backtracking test: {elapsed:.3f}s (20 char input)"
"""

        try:
            result, stats, ir = service.execute(code, persist=False, timeout=10)
            print(f"\n⚠️  VULNERABILITY: Regex backtracking possible")
            print(f"    Result: {result}")
            print(f"    🔥 REAL ATTACK: 30+ char input = exponential CPU burn")
            print(f"    Note: Python 3.11+ has regex timeout (good!)")
            assert stats["success"] is True
        except CodeExecutionError as e:
            print(f"\n✅ PROTECTED: Regex blocked: {e}")
            pytest.fail("Expected vulnerability")


class TestCPUExhaustionSummary:
    """Summary test documenting all CPU vulnerabilities."""

    def test_cpu_vulnerability_summary(self, service):
        """
        SUMMARY: CPU Exhaustion Vulnerabilities

        CONFIRMED VULNERABILITIES:
        1. ✅ Infinite loops (timeout at 300s)
        2. ✅ CPU-intensive loops (100% CPU allowed)
        3. ✅ Prime calculations (expensive algorithms)
        4. ✅ O(n^3) nested loops (cubic complexity)
        5. ✅ Large matrix operations (CPU + memory)
        6. ✅ Expensive cryptography (repeated hashing)
        7. ✅ Regex catastrophic backtracking

        CURRENT PROTECTION: 300s timeout ONLY
        - No CPU throttling
        - No CPU percentage limits
        - No algorithmic complexity checks

        IMPACT: HIGH
        - 300s of 100% CPU burn per execution
        - Can spin all cores simultaneously
        - Blocks other work (sequential execution)
        - Multiple users = compounding DoS

        RECOMMENDED MITIGATIONS:
        1. Reduce DEFAULT_TIMEOUT to 30s (not 300s)
        2. cgroups cpu.max (e.g., 50% of 1 core)
        3. Docker --cpus flag (e.g., --cpus=0.5)
        4. Per-user rate limiting (e.g., 5 executions/minute)
        5. CPU accounting via cgroups cpu.stat
        6. Priority reduction (nice value)

        Example Docker mitigation:
        ```bash
        docker run --cpus=0.5 --memory=2g lobster
        # Limits to 50% of 1 CPU core + 2GB RAM
        ```

        Example cgroups mitigation (Linux):
        ```python
        # In subprocess setup
        import os
        # Set CPU quota (50% of 1 core = 50000/100000)
        os.system('echo 50000 > /sys/fs/cgroup/cpu.max')
        ```

        Example systemd mitigation:
        ```ini
        [Service]
        CPUQuota=50%
        MemoryMax=2G
        ```
        """
        print("\n" + "=" * 70)
        print("CPU EXHAUSTION VULNERABILITY SUMMARY")
        print("=" * 70)
        print(self.test_cpu_vulnerability_summary.__doc__)
        assert True
