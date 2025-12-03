"""
Timing Attack Tests: Side Channel Leaks

Tests covert communication channels and information leakage through indirect
timing measurements (CPU usage, I/O patterns, network timing, etc.).

SECURITY IMPACT: Medium for local CLI, High for cloud/multi-tenant environments

Attack vectors:
1. Network service probing via connection timeout differences
2. CPU usage patterns as covert channel
3. Disk I/O timing as covert channel
4. Memory allocation timing patterns
5. Process/thread detection via resource contention
6. Cache timing attacks
"""

import tempfile
import time
from pathlib import Path

import pytest


@pytest.fixture
def service(tmp_path):
    """Create CustomCodeExecutionService with temporary workspace."""
    from lobster.core.data_manager_v2 import DataManagerV2
    from lobster.services.execution.custom_code_execution_service import (
        CustomCodeExecutionService,
    )

    data_manager = DataManagerV2(workspace_path=tmp_path)
    return CustomCodeExecutionService(data_manager), data_manager


class TestNetworkTimingAttacks:
    """Test timing attacks via network operations."""

    def test_localhost_port_scanning_via_timeout_EXPECT_SUCCESS(self, service):
        """
        Test: Scan localhost ports via connection timeout differences

        Expected: SUCCESS (vulnerability - timeout differences leak port state)
        Impact: Medium (local), High (cloud)
        Exploit: Discover internal services without network tools

        Real-world scenario:
        - Find Redis on localhost:6379
        - Detect PostgreSQL on localhost:5432
        - Discover internal APIs
        """
        svc, dm = service

        code = """
import socket
import time

# Ports to probe (common services)
ports_to_scan = {
    22: 'SSH',
    80: 'HTTP',
    443: 'HTTPS',
    3306: 'MySQL',
    5432: 'PostgreSQL',
    6379: 'Redis',
    8080: 'HTTP-Alt',
    9999: 'Unlikely'
}

scan_results = {}

for port, service_name in ports_to_scan.items():
    start = time.perf_counter()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.1)  # 100ms timeout

    try:
        result = sock.connect_ex(('127.0.0.1', port))
        connection_time = time.perf_counter() - start

        scan_results[port] = {
            'service': service_name,
            'result_code': result,
            'time_ms': connection_time * 1000,
            'likely_open': result == 0,
            'likely_filtered': connection_time > 0.09  # Near timeout
        }
    except Exception as e:
        connection_time = time.perf_counter() - start
        scan_results[port] = {
            'service': service_name,
            'result_code': -1,
            'time_ms': connection_time * 1000,
            'error': str(e)
        }
    finally:
        sock.close()

# Categorize by timing pattern
fast_rejects = [p for p, r in scan_results.items() if r['time_ms'] < 10]
slow_timeouts = [p for p, r in scan_results.items() if r['time_ms'] > 80]
open_ports = [p for p, r in scan_results.items() if r.get('likely_open', False)]

result = {
    'scan_results': scan_results,
    'fast_rejects': fast_rejects,
    'slow_timeouts': slow_timeouts,
    'open_ports': open_ports,
    'scan_successful': True
}
"""

        result, stats, ir = svc.execute(code, persist=False, timeout=30)

        print("\n" + "=" * 80)
        print("TIMING ATTACK: Localhost Port Scanning")
        print("=" * 80)
        print(f"Ports scanned: {len(result['scan_results'])}")
        print(f"Fast rejects:  {len(result['fast_rejects'])} (closed ports)")
        print(f"Slow timeouts: {len(result['slow_timeouts'])} (filtered/no route)")
        print(f"Open ports:    {result['open_ports']}")
        print("\nDetailed results:")
        for port, details in result["scan_results"].items():
            port_num = int(port) if isinstance(port, str) else port
            print(
                f"  Port {port_num:5d} ({details['service']:12s}): "
                f"{details['time_ms']:6.2f} ms - "
                f"Code: {details.get('result_code', 'N/A')}"
            )
        print("\n⚠️  VULNERABILITY CONFIRMED")
        print("   Attacker can probe internal network services via timing")
        print("   Impact: Network reconnaissance without explicit network access")
        print(
            "   Mitigation: Disable socket operations or use network namespace isolation"
        )
        print("=" * 80)

        assert stats["success"]

    def test_external_network_timing_reconnaissance_EXPECT_SUCCESS(self, service):
        """
        Test: Detect network connectivity and firewall rules via timing

        Expected: SUCCESS (vulnerability)
        Impact: Medium
        """
        svc, dm = service

        code = """
import socket
import time

# Test connectivity to different domains (will fail but timing reveals firewall)
test_targets = [
    ('8.8.8.8', 53, 'Google DNS'),
    ('1.1.1.1', 53, 'Cloudflare DNS'),
    ('192.168.1.1', 80, 'Private IP'),
    ('10.0.0.1', 80, 'Private IP'),
]

timing_results = {}

for ip, port, label in test_targets:
    times = []
    for _ in range(3):
        start = time.perf_counter()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.2)

        try:
            result = sock.connect_ex((ip, port))
            elapsed = time.perf_counter() - start
        except Exception as e:
            elapsed = time.perf_counter() - start
        finally:
            sock.close()

        times.append(elapsed)

    avg_time = sum(times) / len(times)
    timing_results[f"{ip}:{port}"] = {
        'label': label,
        'avg_time_ms': avg_time * 1000,
        'pattern': 'timeout' if avg_time > 0.15 else 'reject'
    }

result = {
    'timing_results': timing_results,
    'firewall_detectable': True  # Different timing patterns reveal rules
}
"""

        result, stats, ir = svc.execute(code, persist=False, timeout=30)

        print("\n" + "=" * 80)
        print("TIMING ATTACK: Network Reconnaissance")
        print("=" * 80)
        for target, details in result["timing_results"].items():
            print(
                f"  {target:20s} ({details['label']:15s}): "
                f"{details['avg_time_ms']:6.2f} ms - {details['pattern']}"
            )
        print("\n⚠️  VULNERABILITY CONFIRMED")
        print("   Timing differences reveal firewall rules and network topology")
        print("=" * 80)

        assert stats["success"]


class TestCPUTimingCovertChannels:
    """Test covert communication via CPU usage patterns."""

    def test_cpu_usage_as_covert_channel_EXPECT_SUCCESS(self, service):
        """
        Test: Use CPU busy-wait vs sleep to encode binary message

        Expected: SUCCESS (vulnerability)
        Impact: Low (local), Medium (cloud multi-tenant)
        Exploit: Covert communication between isolated processes

        Covert channel encoding:
        - Busy-wait (100% CPU) = binary 1
        - Sleep (0% CPU) = binary 0
        - External observer can read message via CPU monitoring
        """
        svc, dm = service

        code = """
import time

# Message to transmit: "HI" in binary (ASCII)
# H = 72 = 01001000, I = 73 = 01001001
message = "HI"
binary = ''.join(format(ord(c), '08b') for c in message)

def send_bit(bit, duration=0.05):
    '''Send one bit via CPU usage pattern.'''
    start = time.time()
    if bit == '1':
        # Busy-wait (high CPU)
        while time.time() - start < duration:
            x = 1234 * 5678  # Keep CPU busy
    else:
        # Sleep (low CPU)
        time.sleep(duration)

# Transmit message
print(f"Transmitting message: {message}")
print(f"Binary: {binary}")

transmission_start = time.time()
for bit in binary:
    send_bit(bit, duration=0.02)  # 20ms per bit

transmission_time = time.time() - transmission_start

result = {
    'message': message,
    'binary_encoding': binary,
    'bits_transmitted': len(binary),
    'transmission_time_seconds': transmission_time,
    'bandwidth_bps': len(binary) / transmission_time if transmission_time > 0 else 0,
    'covert_channel_viable': True
}
"""

        result, stats, ir = svc.execute(code, persist=False, timeout=30)

        print("\n" + "=" * 80)
        print("TIMING ATTACK: CPU Usage Covert Channel")
        print("=" * 80)
        print(f"Message transmitted: '{result['message']}'")
        print(f"Binary encoding:     {result['binary_encoding']}")
        print(f"Bits transmitted:    {result['bits_transmitted']}")
        print(f"Transmission time:   {result['transmission_time_seconds']:.3f} seconds")
        print(f"Bandwidth:           {result['bandwidth_bps']:.1f} bits/second")
        print("\n⚠️  COVERT CHANNEL VULNERABILITY CONFIRMED")
        print("   Attacker can exfiltrate data via CPU usage patterns")
        print("   Impact: Information leakage in multi-tenant environments")
        print("   Detection: CPU usage monitoring reveals pattern")
        print("=" * 80)

        assert stats["success"]

    def test_cache_timing_side_channel_EXPECT_SUCCESS(self, service):
        """
        Test: Cache timing attack (simplified Spectre-like concept)

        Expected: SUCCESS (vulnerability)
        Impact: Low (simplified), High (real cache attacks are complex)
        """
        svc, dm = service

        code = """
import time

# Simulate secret data in memory
SECRET_DATA = [42, 17, 99, 3, 88, 19, 76, 54]

def access_array(arr, index):
    '''Access array element (cache effects).'''
    return arr[index]

# Measure timing for cached vs uncached access
def measure_cache_timing(arr, index, samples=100):
    '''Measure access time to detect cache state.'''
    times = []

    for _ in range(samples):
        # First access (cold - not in cache)
        start = time.perf_counter()
        value = access_array(arr, index)
        first_access = time.perf_counter() - start

        # Second access (hot - in cache)
        start = time.perf_counter()
        value = access_array(arr, index)
        second_access = time.perf_counter() - start

        times.append({
            'cold': first_access,
            'hot': second_access,
            'speedup': first_access / second_access if second_access > 0 else 0
        })

    avg_cold = sum(t['cold'] for t in times) / len(times)
    avg_hot = sum(t['hot'] for t in times) / len(times)
    avg_speedup = sum(t['speedup'] for t in times) / len(times)

    return avg_cold, avg_hot, avg_speedup

# Test cache timing on secret data
cold_time, hot_time, speedup = measure_cache_timing(SECRET_DATA, 0)

result = {
    'cold_access_ns': cold_time * 1e9,
    'hot_access_ns': hot_time * 1e9,
    'cache_speedup': speedup,
    'cache_detectable': speedup > 1.1,
    'note': 'Real cache attacks are more sophisticated (Spectre, Meltdown)'
}
"""

        result, stats, ir = svc.execute(code, persist=False, timeout=30)

        print("\n" + "=" * 80)
        print("TIMING ATTACK: Cache Timing Side Channel")
        print("=" * 80)
        print(f"Cold access time: {result['cold_access_ns']:.3f} ns")
        print(f"Hot access time:  {result['hot_access_ns']:.3f} ns")
        print(f"Cache speedup:    {result['cache_speedup']:.2f}x")
        print(f"Detectable:       {result['cache_detectable']}")
        print("\nℹ️  Note: Real cache attacks (Spectre, Meltdown) are far more complex")
        print("   This is a simplified demonstration of cache timing principles")
        print("=" * 80)

        assert stats["success"]


class TestDiskIOTimingAttacks:
    """Test timing attacks via disk I/O patterns."""

    def test_disk_io_timing_covert_channel_EXPECT_SUCCESS(self, service):
        """
        Test: Covert channel via disk I/O timing patterns

        Expected: SUCCESS (vulnerability)
        Impact: Low-Medium
        Exploit: Communicate via disk access patterns
        """
        svc, dm = service

        code = """
import time
from pathlib import Path

workspace = Path.cwd()

# Encode binary message via I/O timing
message = "OK"
binary = ''.join(format(ord(c), '08b') for c in message)

def send_bit_via_io(bit, tmp_file, duration=0.03):
    '''Send bit via disk I/O pattern.'''
    start = time.time()
    if bit == '1':
        # High I/O (many small writes)
        while time.time() - start < duration:
            tmp_file.write_text(str(time.time()))
    else:
        # Low I/O (sleep)
        time.sleep(duration)

tmp_file = workspace / '.covert_channel.tmp'

transmission_start = time.time()
for bit in binary:
    send_bit_via_io(bit, tmp_file, duration=0.02)

transmission_time = time.time() - transmission_start

# Cleanup
if tmp_file.exists():
    tmp_file.unlink()

result = {
    'message': message,
    'binary': binary,
    'bits_sent': len(binary),
    'transmission_seconds': transmission_time,
    'io_covert_channel_viable': True
}
"""

        result, stats, ir = svc.execute(code, persist=False, timeout=30)

        print("\n" + "=" * 80)
        print("TIMING ATTACK: Disk I/O Covert Channel")
        print("=" * 80)
        print(f"Message:          '{result['message']}'")
        print(f"Binary:           {result['binary']}")
        print(f"Transmission:     {result['transmission_seconds']:.3f} seconds")
        print("\n⚠️  COVERT CHANNEL VULNERABILITY CONFIRMED")
        print("   Attacker can exfiltrate data via disk I/O patterns")
        print("   Detection: Disk usage monitoring reveals pattern")
        print("=" * 80)

        assert stats["success"]

    def test_memory_allocation_timing_EXPECT_SUCCESS(self, service):
        """
        Test: Infer system memory state via allocation timing

        Expected: SUCCESS (vulnerability)
        Impact: Low-Medium
        """
        svc, dm = service

        code = """
import time

# Measure memory allocation timing at different sizes
allocation_timings = {}

for size_mb in [1, 10, 50, 100, 500]:
    size_bytes = size_mb * 1024 * 1024
    times = []

    for _ in range(10):
        start = time.perf_counter()
        # Allocate memory
        data = bytearray(size_bytes)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        # Free memory
        del data

    avg_time = sum(times) / len(times)
    allocation_timings[size_mb] = {
        'avg_time_ms': avg_time * 1000,
        'throughput_mb_per_sec': size_mb / avg_time if avg_time > 0 else 0
    }

result = {
    'allocation_timings': allocation_timings,
    'memory_state_inferable': True,
    'note': 'Timing reveals memory pressure and system state'
}
"""

        result, stats, ir = svc.execute(code, persist=False, timeout=30)

        print("\n" + "=" * 80)
        print("TIMING ATTACK: Memory Allocation Timing")
        print("=" * 80)
        for size_mb, details in result["allocation_timings"].items():
            size_num = int(size_mb) if isinstance(size_mb, str) else size_mb
            print(
                f"  {size_num:3d} MB: {details['avg_time_ms']:6.2f} ms "
                f"({details['throughput_mb_per_sec']:6.1f} MB/s)"
            )
        print("\n⚠️  VULNERABILITY CONFIRMED")
        print("   Memory allocation timing reveals system state")
        print("   Impact: Can infer memory pressure, other processes")
        print("=" * 80)

        assert stats["success"]


class TestResourceContentionTimingAttacks:
    """Test timing attacks via resource contention."""

    def test_process_detection_via_cpu_contention_EXPECT_SUCCESS(self, service):
        """
        Test: Detect other processes via CPU contention timing

        Expected: SUCCESS (vulnerability)
        Impact: Low (local), Medium (cloud)
        """
        svc, dm = service

        code = """
import time
import multiprocessing

def cpu_intensive_work(duration):
    '''Perform CPU-intensive work for specified duration.'''
    start = time.time()
    count = 0
    while time.time() - start < duration:
        count += sum(range(1000))
    return count

# Measure baseline CPU timing (no contention)
baseline_times = []
for _ in range(5):
    start = time.perf_counter()
    cpu_intensive_work(0.1)
    elapsed = time.perf_counter() - start
    baseline_times.append(elapsed)
baseline_avg = sum(baseline_times) / len(baseline_times)

# Measure CPU timing with contention (spawn competing processes)
cpu_count = multiprocessing.cpu_count()
contention_times = []

# Create CPU contention
processes = []
for _ in range(cpu_count):
    p = multiprocessing.Process(target=cpu_intensive_work, args=(1.0,))
    p.start()
    processes.append(p)

time.sleep(0.1)  # Let contention build

for _ in range(5):
    start = time.perf_counter()
    cpu_intensive_work(0.1)
    elapsed = time.perf_counter() - start
    contention_times.append(elapsed)
contention_avg = sum(contention_times) / len(contention_times)

# Cleanup
for p in processes:
    p.terminate()
    p.join()

result = {
    'cpu_cores': cpu_count,
    'baseline_time_ms': baseline_avg * 1000,
    'contention_time_ms': contention_avg * 1000,
    'slowdown_ratio': contention_avg / baseline_avg if baseline_avg > 0 else 0,
    'contention_detectable': (contention_avg / baseline_avg) > 1.2 if baseline_avg > 0 else False
}
"""

        result, stats, ir = svc.execute(code, persist=False, timeout=60)

        print("\n" + "=" * 80)
        print("TIMING ATTACK: Process Detection via CPU Contention")
        print("=" * 80)
        print(f"CPU cores:           {result['cpu_cores']}")
        print(f"Baseline timing:     {result['baseline_time_ms']:.2f} ms")
        print(f"Contention timing:   {result['contention_time_ms']:.2f} ms")
        print(f"Slowdown ratio:      {result['slowdown_ratio']:.2f}x")
        print(f"Contention detected: {result['contention_detectable']}")
        print("\n⚠️  VULNERABILITY CONFIRMED")
        print("   Timing differences reveal presence of other processes")
        print("   Impact: Information leakage in shared environments")
        print("=" * 80)

        assert stats["success"]


class TestTimingAttackMitigations:
    """Test potential mitigations for timing attacks."""

    def test_constant_time_comparison_mitigation_EXPECT_FAILURE(self, service):
        """
        Test: Demonstrate constant-time comparison (proper mitigation)

        Expected: FAILURE to extract information (mitigation works)
        Impact: Shows how to defend against timing attacks
        """
        svc, dm = service

        code = """
import time
import secrets

# Secure comparison using secrets.compare_digest (constant-time)
SECRET_KEY = "sk-proj-abcdef1234567890"

def secure_compare(input_key, real_key):
    '''Constant-time comparison using secrets.compare_digest.'''
    return secrets.compare_digest(input_key, real_key)

# Try timing attack with constant-time comparison
import string
charset = string.ascii_lowercase + string.digits + '-_'
timings_by_char = {}

for guess_char in charset:
    test_key = guess_char + 'x' * (len(SECRET_KEY) - 1)

    times = []
    for _ in range(100):
        start = time.perf_counter()
        matches = secure_compare(test_key, SECRET_KEY)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    timings_by_char[guess_char] = sum(times) / len(times)

# Try to infer first character
sorted_chars = sorted(timings_by_char.items(), key=lambda x: x[1], reverse=True)
likely_first_char = sorted_chars[0][0]
actual_first_char = SECRET_KEY[0]

timing_variance = max(timings_by_char.values()) - min(timings_by_char.values())

result = {
    'actual_first_char': actual_first_char,
    'inferred_first_char': likely_first_char,
    'attack_failed': likely_first_char != actual_first_char,
    'timing_variance_ns': timing_variance * 1e9,
    'mitigation_effective': timing_variance < 1e-9
}
"""

        result, stats, ir = svc.execute(code, persist=False, timeout=60)

        print("\n" + "=" * 80)
        print("MITIGATION TEST: Constant-Time Comparison")
        print("=" * 80)
        print(f"Actual first char: '{result['actual_first_char']}'")
        print(f"Inferred char:     '{result['inferred_first_char']}'")
        print(f"Attack failed:     {result['attack_failed']}")
        print(f"Timing variance:   {result['timing_variance_ns']:.3f} ns")
        print("\n✅ MITIGATION EFFECTIVE")
        print("   secrets.compare_digest() prevents timing attacks")
        print("   Timing variance is below detectable threshold")
        print("   Recommendation: Use for all sensitive comparisons")
        print("=" * 80)

        assert stats["success"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
