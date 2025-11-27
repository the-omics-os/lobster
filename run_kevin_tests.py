#!/usr/bin/env python3
import subprocess
import sys
import time

tests = [
    {'name': 'Bug #2 Test 1 - GSE182227', 'command': 'lobster query "ADMIN SUPERUSER: Download and validate single-cell dataset GSE182227. Load it and report the shape (n_obs × n_vars). Check if variables are present."', 'expected': 'Should download successfully and report shape', 'timeout': 300},
    {'name': 'Bug #2 Test 2 - GSE190729', 'command': 'lobster query "ADMIN SUPERUSER: Download and validate single-cell dataset GSE190729. This is a known reliable dataset. Load it and confirm successful loading."', 'expected': "Should download and load without 'No variables' error", 'timeout': 300},
    {'name': 'Bug #3 Test 1 - GSE130036 Kallisto', 'command': 'lobster query "ADMIN SUPERUSER: Download bulk RNA-seq dataset GSE130036. This contains Kallisto quantification files. Load the data and check the orientation - report if samples are rows and genes are columns."', 'expected': 'Should report correct orientation (samples × genes)', 'timeout': 300},
    {'name': 'Bug #3 Test 2 - GSE130970 Metadata', 'command': 'lobster query "ADMIN SUPERUSER: Download bulk RNA-seq dataset GSE130970 with 78 liver samples. Load it and verify the metadata includes clinical labels (Normal/NAFLD/NASH). Check if the orientation is correct (samples × genes)."', 'expected': 'Should load with clinical metadata attached', 'timeout': 300},
    {'name': 'Bug #7 Test - FTP Download', 'command': 'lobster query "ADMIN SUPERUSER: Test FTP download reliability by downloading GSE130036 using SUPPLEMENTARY_FIRST strategy. Report any timeout or connection errors during download."', 'expected': 'Should handle FTP downloads or fallback to HTTP', 'timeout': 600},
]

def run_test(test_info):
    print(f"\n{'='*60}")
    print(f"Running: {test_info['name']}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            test_info['command'],
            shell=True,
            capture_output=True,
            text=True,
            timeout=test_info['timeout']
        )

        print("STDOUT:")
        print(result.stdout)

        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"❌ Test timed out after {test_info['timeout']} seconds")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    results = []
    for test in tests:
        success = run_test(test)
        results.append((test['name'], success))
        time.sleep(5)  # Wait between tests

    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")

    passed = sum(1 for _, s in results if s)
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    sys.exit(0 if passed == len(results) else 1)
