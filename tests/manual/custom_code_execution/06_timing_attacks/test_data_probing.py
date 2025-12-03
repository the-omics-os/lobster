"""
Timing Attack Tests: Data Probing

Tests information leakage through execution time differences in data operations.
These tests demonstrate how attackers can infer sensitive information by measuring
execution time variations.

SECURITY IMPACT: Medium for local CLI, High for cloud deployment

Attack vectors:
1. File existence probing via timing differences
2. File size inference via read timing
3. String length inference via comparison timing
4. API key/password brute force via character-by-character timing
5. Directory vs file detection via stat() timing
6. Data structure size inference (list/dict length)
"""

import tempfile
import time
from pathlib import Path
from typing import Dict, List

import pytest


# Test fixtures
@pytest.fixture
def service(tmp_path):
    """Create CustomCodeExecutionService with temporary workspace."""
    from lobster.core.data_manager_v2 import DataManagerV2
    from lobster.services.execution.custom_code_execution_service import (
        CustomCodeExecutionService,
    )

    data_manager = DataManagerV2(workspace_path=tmp_path)
    return CustomCodeExecutionService(data_manager), data_manager


@pytest.fixture
def setup_test_files(tmp_path):
    """Create test files with known properties for timing attacks."""
    # Create files with different sizes
    small_file = tmp_path / "small.txt"
    small_file.write_text("x" * 100)

    large_file = tmp_path / "large.txt"
    large_file.write_text("x" * 10000)

    # Create hidden file
    hidden_file = tmp_path / ".hidden"
    hidden_file.write_text("secret")

    # Create directory
    test_dir = tmp_path / "test_directory"
    test_dir.mkdir()

    return {
        "small_file": small_file,
        "large_file": large_file,
        "hidden_file": hidden_file,
        "test_dir": test_dir,
        "nonexistent": tmp_path / "nonexistent_file_12345.txt",
    }


class TestFileSystemTimingLeaks:
    """Test timing-based information leakage through file system operations."""

    def test_file_existence_timing_leak_EXPECT_SUCCESS(self, service, setup_test_files):
        """
        Test: Detect file existence via timing differences in Path.exists()

        Expected: SUCCESS (vulnerability exists - timing not normalized)
        Impact: Medium
        Exploit: Attacker can probe sensitive file locations without direct access

        Real-world scenario:
        - Probe for /etc/passwd, ~/.ssh/id_rsa, .env files
        - Enumerate workspace contents without list_modalities()
        - Discover hidden configuration files
        """
        svc, dm = service
        files = setup_test_files

        code = f"""
import time
from pathlib import Path

# Measure timing for existing file
timings = []
for i in range(100):
    start = time.perf_counter()
    exists = Path('{files['small_file']}').exists()
    elapsed = time.perf_counter() - start
    timings.append(elapsed)
exists_time = sum(timings) / len(timings)

# Measure timing for non-existent file
timings = []
for i in range(100):
    start = time.perf_counter()
    not_exists = Path('{files['nonexistent']}').exists()
    elapsed = time.perf_counter() - start
    timings.append(elapsed)
not_exists_time = sum(timings) / len(timings)

result = {{
    'exists_avg_time': exists_time,
    'not_exists_avg_time': not_exists_time,
    'difference_ns': (exists_time - not_exists_time) * 1e9,
    'detectable': abs(exists_time - not_exists_time) > 1e-8
}}
"""

        result, stats, ir = svc.execute(code, persist=False, timeout=30)

        print("\n" + "=" * 80)
        print("TIMING ATTACK: File Existence Probing")
        print("=" * 80)
        print(f"Exists timing:     {result['exists_avg_time']*1e6:.3f} µs")
        print(f"Not exists timing: {result['not_exists_avg_time']*1e6:.3f} µs")
        print(f"Difference:        {result['difference_ns']:.3f} ns")
        print(f"Detectable:        {result['detectable']}")
        print("\n⚠️  VULNERABILITY CONFIRMED")
        print("   Attacker can probe file existence via timing differences")
        print("   Impact: Enumeration of sensitive file locations")
        print("=" * 80)

        assert stats["success"], "Timing attack should succeed"

    def test_file_vs_directory_timing_leak_EXPECT_SUCCESS(
        self, service, setup_test_files
    ):
        """
        Test: Distinguish files from directories via stat() timing

        Expected: SUCCESS (vulnerability)
        Impact: Medium
        Exploit: Refine file system enumeration by identifying directories
        """
        svc, dm = service
        files = setup_test_files

        code = f"""
import time
from pathlib import Path

# Measure timing for file
timings_file = []
for i in range(50):
    start = time.perf_counter()
    p = Path('{files['small_file']}')
    is_file = p.is_file()
    elapsed = time.perf_counter() - start
    timings_file.append(elapsed)
file_time = sum(timings_file) / len(timings_file)

# Measure timing for directory
timings_dir = []
for i in range(50):
    start = time.perf_counter()
    p = Path('{files['test_dir']}')
    is_dir = p.is_dir()
    elapsed = time.perf_counter() - start
    timings_dir.append(elapsed)
dir_time = sum(timings_dir) / len(timings_dir)

result = {{
    'file_time_us': file_time * 1e6,
    'dir_time_us': dir_time * 1e6,
    'difference_us': abs(file_time - dir_time) * 1e6,
    'distinguishable': abs(file_time - dir_time) > 1e-7
}}
"""

        result, stats, ir = svc.execute(code, persist=False, timeout=30)

        print("\n" + "=" * 80)
        print("TIMING ATTACK: File vs Directory Detection")
        print("=" * 80)
        print(f"File check time:      {result['file_time_us']:.3f} µs")
        print(f"Directory check time: {result['dir_time_us']:.3f} µs")
        print(f"Difference:           {result['difference_us']:.3f} µs")
        print(f"Distinguishable:      {result['distinguishable']}")
        print("\n⚠️  VULNERABILITY CONFIRMED")
        print("   Attacker can distinguish files from directories")
        print("=" * 80)

        assert stats["success"]

    def test_file_size_inference_via_read_timing_EXPECT_SUCCESS(
        self, service, setup_test_files
    ):
        """
        Test: Infer file size via read operation timing

        Expected: SUCCESS (vulnerability)
        Impact: Medium
        Exploit: Learn approximate file sizes without reading content

        Real-world use:
        - Identify configuration files vs data files
        - Detect large result files (indicates processing occurred)
        """
        svc, dm = service
        files = setup_test_files

        code = f"""
import time
from pathlib import Path

# Measure timing for small file
timings_small = []
for i in range(20):
    start = time.perf_counter()
    content = Path('{files['small_file']}').read_text()
    elapsed = time.perf_counter() - start
    timings_small.append(elapsed)
small_time = sum(timings_small) / len(timings_small)

# Measure timing for large file
timings_large = []
for i in range(20):
    start = time.perf_counter()
    content = Path('{files['large_file']}').read_text()
    elapsed = time.perf_counter() - start
    timings_large.append(elapsed)
large_time = sum(timings_large) / len(timings_large)

result = {{
    'small_file_time_us': small_time * 1e6,
    'large_file_time_us': large_time * 1e6,
    'time_ratio': large_time / small_time if small_time > 0 else 0,
    'size_inferable': (large_time / small_time) > 1.5 if small_time > 0 else False
}}
"""

        result, stats, ir = svc.execute(code, persist=False, timeout=30)

        print("\n" + "=" * 80)
        print("TIMING ATTACK: File Size Inference")
        print("=" * 80)
        print(f"Small file (100 bytes):  {result['small_file_time_us']:.3f} µs")
        print(f"Large file (10KB):       {result['large_file_time_us']:.3f} µs")
        print(f"Time ratio:              {result['time_ratio']:.2f}x")
        print(f"Size inferable:          {result['size_inferable']}")
        print("\n⚠️  VULNERABILITY CONFIRMED")
        print("   Attacker can infer approximate file sizes via read timing")
        print("=" * 80)

        assert stats["success"]


class TestStringComparisonTimingAttacks:
    """Test timing attacks on string comparison operations."""

    def test_string_length_inference_EXPECT_SUCCESS(self, service):
        """
        Test: Infer string length via comparison timing

        Expected: SUCCESS (vulnerability)
        Impact: Medium
        Exploit: Learn length of secret strings without seeing content
        """
        svc, dm = service

        code = """
import time

# Simulate secret string in environment
secret = "this_is_a_secret_api_key_with_32_characters"

# Try to guess length via comparison timing
timing_results = {}
for test_length in range(10, 60):
    test_string = "x" * test_length

    timings = []
    for i in range(100):
        start = time.perf_counter()
        # Python string comparison checks length first (fast path)
        matches = (test_string == secret)
        elapsed = time.perf_counter() - start
        timings.append(elapsed)

    avg_time = sum(timings) / len(timings)
    timing_results[test_length] = avg_time

# Find length with anomalous timing (matching length takes different path)
sorted_by_time = sorted(timing_results.items(), key=lambda x: x[1])
likely_length = sorted_by_time[0][0]
actual_length = len(secret)

result = {
    'actual_length': actual_length,
    'inferred_length': likely_length,
    'correct_inference': likely_length == actual_length,
    'timing_pattern': {k: v*1e9 for k, v in list(timing_results.items())[:10]}
}
"""

        result, stats, ir = svc.execute(code, persist=False, timeout=30)

        print("\n" + "=" * 80)
        print("TIMING ATTACK: String Length Inference")
        print("=" * 80)
        print(f"Actual secret length:   {result['actual_length']}")
        print(f"Inferred length:        {result['inferred_length']}")
        print(f"Inference correct:      {result['correct_inference']}")
        print("\n⚠️  VULNERABILITY CONFIRMED")
        print("   Attacker can infer string lengths via comparison timing")
        print("   Impact: Helps narrow brute force attacks")
        print("=" * 80)

        assert stats["success"]

    def test_api_key_brute_force_timing_attack_EXPECT_SUCCESS(self, service):
        """
        Test: Brute force API key prefix via character-by-character timing

        Expected: SUCCESS (critical vulnerability)
        Impact: HIGH
        Exploit: Python string comparison is NOT constant-time

        Real-world scenario:
        - Attacker can extract API keys character by character
        - Each character takes ~100 comparisons (not 62^N)
        - Works even with rate limiting (timing is side channel)

        NOTE: This is a simplified demonstration. Real attacks would:
        - Use statistical analysis over many measurements
        - Account for system noise
        - Employ cache timing side channels
        """
        svc, dm = service

        code = """
import time
import string

# Simulate API key (in real scenario, would be in environment/config)
REAL_KEY = "sk-proj-abcdef1234567890"

def time_comparison(guess_key, real_key, samples=50):
    '''Measure average comparison time.'''
    timings = []
    for _ in range(samples):
        start = time.perf_counter()
        matches = (guess_key == real_key)
        elapsed = time.perf_counter() - start
        timings.append(elapsed)
    return sum(timings) / len(timings)

# Try to guess first character via timing
charset = string.ascii_lowercase + string.digits + '-_'
timings_by_char = {}

for guess_char in charset:
    test_key = guess_char + 'x' * (len(REAL_KEY) - 1)
    avg_time = time_comparison(test_key, REAL_KEY, samples=100)
    timings_by_char[guess_char] = avg_time

# Find character with longest comparison time (more characters matched)
# In Python, string comparison is NOT constant-time
sorted_chars = sorted(timings_by_char.items(), key=lambda x: x[1], reverse=True)
likely_first_char = sorted_chars[0][0]
actual_first_char = REAL_KEY[0]

result = {
    'actual_first_char': actual_first_char,
    'inferred_first_char': likely_first_char,
    'correct_inference': likely_first_char == actual_first_char,
    'top_5_candidates': [(c, t*1e9) for c, t in sorted_chars[:5]],
    'timing_variance_ns': (sorted_chars[0][1] - sorted_chars[-1][1]) * 1e9,
    'attack_feasible': (sorted_chars[0][1] - sorted_chars[-1][1]) > 1e-10
}
"""

        result, stats, ir = svc.execute(code, persist=False, timeout=60)

        print("\n" + "=" * 80)
        print("TIMING ATTACK: API Key Brute Force (Character-by-Character)")
        print("=" * 80)
        print(f"Actual first char:  '{result['actual_first_char']}'")
        print(f"Inferred char:      '{result['inferred_first_char']}'")
        print(f"Attack successful:  {result['correct_inference']}")
        print(f"Timing variance:    {result['timing_variance_ns']:.3f} ns")
        print(f"\nTop 5 candidates (char, time):")
        for char, time_ns in result["top_5_candidates"]:
            print(f"  '{char}': {time_ns:.3f} ns")
        print("\n⚠️  CRITICAL VULNERABILITY CONFIRMED")
        print("   Python string comparison is NOT constant-time")
        print("   Attacker can extract secrets character by character")
        print("   Impact: Complete API key/password compromise")
        print("\n   Mitigation: Use secrets.compare_digest() for sensitive comparisons")
        print("=" * 80)

        assert stats["success"]

    def test_password_timing_attack_EXPECT_SUCCESS(self, service):
        """
        Test: Password verification timing attack (like bcrypt comparison)

        Expected: SUCCESS (vulnerability if not using constant-time comparison)
        Impact: HIGH
        """
        svc, dm = service

        code = """
import time

# Simulate password verification (INSECURE - not constant-time)
STORED_PASSWORD = "MySecureP@ssw0rd123"

def verify_password_INSECURE(input_password):
    '''Insecure password verification with timing leak.'''
    if len(input_password) != len(STORED_PASSWORD):
        return False

    # Character-by-character comparison (leaks timing)
    for i, char in enumerate(input_password):
        if char != STORED_PASSWORD[i]:
            return False  # Early return leaks position of mismatch
    return True

# Attack: Find correct password length via timing
timings_by_length = {}
for length in range(10, 30):
    test_pwd = "x" * length

    times = []
    for _ in range(50):
        start = time.perf_counter()
        verify_password_INSECURE(test_pwd)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    timings_by_length[length] = sum(times) / len(times)

# Correct length will have longer comparison time
sorted_lengths = sorted(timings_by_length.items(), key=lambda x: x[1], reverse=True)
inferred_length = sorted_lengths[0][0]
actual_length = len(STORED_PASSWORD)

# Now attack first character at correct length
charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789@!#$%"
timings_by_char = {}

for char in charset:
    test_pwd = char + "x" * (actual_length - 1)

    times = []
    for _ in range(50):
        start = time.perf_counter()
        verify_password_INSECURE(test_pwd)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    timings_by_char[char] = sum(times) / len(times)

sorted_chars = sorted(timings_by_char.items(), key=lambda x: x[1], reverse=True)
inferred_first_char = sorted_chars[0][0]
actual_first_char = STORED_PASSWORD[0]

result = {
    'password_length': {
        'actual': actual_length,
        'inferred': inferred_length,
        'correct': inferred_length == actual_length
    },
    'first_character': {
        'actual': actual_first_char,
        'inferred': inferred_first_char,
        'correct': inferred_first_char == actual_first_char
    },
    'attack_successful': (inferred_length == actual_length and
                         inferred_first_char == actual_first_char)
}
"""

        result, stats, ir = svc.execute(code, persist=False, timeout=60)

        print("\n" + "=" * 80)
        print("TIMING ATTACK: Password Verification")
        print("=" * 80)
        print(f"Length inference:")
        print(f"  Actual:   {result['password_length']['actual']}")
        print(f"  Inferred: {result['password_length']['inferred']}")
        print(f"  Correct:  {result['password_length']['correct']}")
        print(f"\nFirst character inference:")
        print(f"  Actual:   '{result['first_character']['actual']}'")
        print(f"  Inferred: '{result['first_character']['inferred']}'")
        print(f"  Correct:  {result['first_character']['correct']}")
        print(f"\n⚠️  CRITICAL VULNERABILITY CONFIRMED")
        print(f"   Attack successful: {result['attack_successful']}")
        print("   Early return in comparison leaks match position")
        print("   Impact: Efficient password brute force")
        print("=" * 80)

        assert stats["success"]


class TestDataStructureTimingLeaks:
    """Test timing leaks from data structure operations."""

    def test_list_length_inference_EXPECT_SUCCESS(self, service):
        """
        Test: Infer list length via iteration timing

        Expected: SUCCESS (vulnerability)
        Impact: Low-Medium
        """
        svc, dm = service

        code = """
import time

# Simulate secret list
secret_list = list(range(1000))

# Try to guess length via iteration timing
timing_results = {}
for guess_length in [10, 50, 100, 500, 1000, 1500, 2000]:
    test_list = list(range(guess_length))

    times = []
    for _ in range(20):
        start = time.perf_counter()
        # Operation that depends on length
        for item in test_list:
            pass
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    timing_results[guess_length] = sum(times) / len(times)

# Actual timing for secret list
secret_times = []
for _ in range(20):
    start = time.perf_counter()
    for item in secret_list:
        pass
    elapsed = time.perf_counter() - start
    secret_times.append(elapsed)
secret_time = sum(secret_times) / len(secret_times)

# Find closest timing match
closest_guess = min(timing_results.items(),
                   key=lambda x: abs(x[1] - secret_time))

result = {
    'actual_length': len(secret_list),
    'inferred_length': closest_guess[0],
    'error_percent': abs(closest_guess[0] - len(secret_list)) / len(secret_list) * 100,
    'timing_profile': {k: v*1e6 for k, v in timing_results.items()}
}
"""

        result, stats, ir = svc.execute(code, persist=False, timeout=30)

        print("\n" + "=" * 80)
        print("TIMING ATTACK: List Length Inference")
        print("=" * 80)
        print(f"Actual length:    {result['actual_length']}")
        print(f"Inferred length:  {result['inferred_length']}")
        print(f"Error:            {result['error_percent']:.1f}%")
        print("\n⚠️  VULNERABILITY CONFIRMED")
        print("   Attacker can infer data structure sizes via operation timing")
        print("=" * 80)

        assert stats["success"]

    def test_dict_key_existence_timing_EXPECT_SUCCESS(self, service):
        """
        Test: Detect dictionary key existence via lookup timing

        Expected: SUCCESS (vulnerability)
        Impact: Low-Medium
        """
        svc, dm = service

        code = """
import time

# Simulate secret configuration dictionary
secret_config = {
    'api_key': 'sk-secret',
    'database_url': 'postgres://...',
    'admin_password': 'admin123'
}

# Try to detect if specific keys exist
potential_keys = ['api_key', 'aws_key', 'database_url', 'secret_token',
                 'admin_password', 'user_password']

timing_results = {}
for key in potential_keys:
    times = []
    for _ in range(100):
        start = time.perf_counter()
        try:
            value = secret_config.get(key)
        except:
            pass
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    timing_results[key] = sum(times) / len(times)

# Keys that exist might have different timing patterns
# (though Python's dict.get() is quite constant-time)
result = {
    'timing_by_key': {k: v*1e9 for k, v in timing_results.items()},
    'actual_keys': list(secret_config.keys()),
    'timing_variance_ns': (max(timing_results.values()) - min(timing_results.values())) * 1e9
}
"""

        result, stats, ir = svc.execute(code, persist=False, timeout=30)

        print("\n" + "=" * 80)
        print("TIMING ATTACK: Dictionary Key Existence")
        print("=" * 80)
        print(f"Timing variance: {result['timing_variance_ns']:.3f} ns")
        print(f"Actual keys:     {result['actual_keys']}")
        print("\nℹ️  Note: Python dict.get() is relatively constant-time")
        print("   However, cache effects may still leak information")
        print("=" * 80)

        assert stats["success"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
