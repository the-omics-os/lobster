#!/usr/bin/env python3
"""
Test script for Kevin's bugs that require actual GEO data downloads.
Uses ADMIN SUPERUSER mode to bypass confirmation prompts.

Author: Bug Verification Team
Date: 2025-11-25
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any

# Test configuration
TEST_DATASETS = {
    "bug2_singlecell": {
        "datasets": ["GSE182227", "GSE190729"],
        "description": "Bug #2: GEO single-cell downloads validation failure"
    },
    "bug3_bulk": {
        "datasets": ["GSE130036", "GSE130970"],
        "description": "Bug #3: Bulk RNA-seq orientation issues"
    },
    "bug7_ftp": {
        "datasets": ["GSE130036"],
        "description": "Bug #7: FTP reliability issues"
    }
}

# Test prompts for each bug using ADMIN SUPERUSER mode
TEST_PROMPTS = {
    "bug2_test1": """ADMIN SUPERUSER: Download and validate single-cell dataset GSE182227. Load it and report the shape (n_obs × n_vars). Check if variables are present.""",

    "bug2_test2": """ADMIN SUPERUSER: Download and validate single-cell dataset GSE190729. This is a known reliable dataset. Load it and confirm successful loading.""",

    "bug3_test1": """ADMIN SUPERUSER: Download bulk RNA-seq dataset GSE130036. This contains Kallisto quantification files. Load the data and check the orientation - report if samples are rows and genes are columns.""",

    "bug3_test2": """ADMIN SUPERUSER: Download bulk RNA-seq dataset GSE130970 with 78 liver samples. Load it and verify the metadata includes clinical labels (Normal/NAFLD/NASH). Check if the orientation is correct (samples × genes).""",

    "bug7_test": """ADMIN SUPERUSER: Test FTP download reliability by downloading GSE130036 using SUPPLEMENTARY_FIRST strategy. Report any timeout or connection errors during download."""
}

def create_test_commands():
    """Create lobster commands for testing each bug."""

    commands = []

    # Bug #2: GEO Single-Cell Downloads
    commands.append({
        "name": "Bug #2 Test 1 - GSE182227",
        "command": 'lobster query "' + TEST_PROMPTS["bug2_test1"] + '"',
        "expected": "Should download successfully and report shape",
        "timeout": 300
    })

    commands.append({
        "name": "Bug #2 Test 2 - GSE190729",
        "command": 'lobster query "' + TEST_PROMPTS["bug2_test2"] + '"',
        "expected": "Should download and load without 'No variables' error",
        "timeout": 300
    })

    # Bug #3: Bulk RNA-seq Orientation
    commands.append({
        "name": "Bug #3 Test 1 - GSE130036 Kallisto",
        "command": 'lobster query "' + TEST_PROMPTS["bug3_test1"] + '"',
        "expected": "Should report correct orientation (samples × genes)",
        "timeout": 300
    })

    commands.append({
        "name": "Bug #3 Test 2 - GSE130970 Metadata",
        "command": 'lobster query "' + TEST_PROMPTS["bug3_test2"] + '"',
        "expected": "Should load with clinical metadata attached",
        "timeout": 300
    })

    # Bug #7: FTP Reliability
    commands.append({
        "name": "Bug #7 Test - FTP Download",
        "command": 'lobster query "' + TEST_PROMPTS["bug7_test"] + '"',
        "expected": "Should handle FTP downloads or fallback to HTTP",
        "timeout": 600
    })

    return commands

def print_test_header(test_name: str):
    """Print a formatted test header."""
    print("\n" + "="*70)
    print(f"TESTING: {test_name}")
    print("="*70)

def print_test_result(success: bool, message: str):
    """Print test result with formatting."""
    if success:
        print(f"✅ PASS: {message}")
    else:
        print(f"❌ FAIL: {message}")

def main():
    """Main test execution."""
    print("="*70)
    print("KEVIN'S BUG VERIFICATION - DATA DOWNLOAD TESTS")
    print("Using ADMIN SUPERUSER mode to bypass confirmations")
    print("="*70)

    # Check if Lobster is installed
    try:
        import lobster
        print("✅ Lobster is installed")
    except ImportError:
        print("❌ Lobster is not installed. Please install it first.")
        return 1

    # Generate test commands
    commands = create_test_commands()

    print(f"\nWill run {len(commands)} tests requiring network access and GEO data")
    print("Note: These tests require API keys and network connectivity")

    # Display test plan
    print("\n" + "="*70)
    print("TEST PLAN:")
    print("="*70)
    for i, cmd in enumerate(commands, 1):
        print(f"\n{i}. {cmd['name']}")
        print(f"   Expected: {cmd['expected']}")
        print(f"   Timeout: {cmd['timeout']}s")

    print("\n" + "="*70)
    print("INSTRUCTIONS FOR RUNNING TESTS:")
    print("="*70)
    print("""
1. Ensure environment variables are set:
   - AWS_BEDROCK_ACCESS_KEY
   - AWS_BEDROCK_SECRET_ACCESS_KEY
   - NCBI_API_KEY (optional, for higher rate limits)

2. Run each test command manually in a terminal:
""")

    for i, cmd in enumerate(commands, 1):
        print(f"\n   Test {i}: {cmd['name']}")
        print(f"   $ {cmd['command']}")

    print("""
3. Evaluate results based on expected outcomes:
   - Bug #2: Check if datasets download without "No variables" error
   - Bug #3: Verify orientation is samples × genes (not genes × samples)
   - Bug #7: Monitor for FTP timeouts and HTTP fallback behavior

4. Document results in bug_verification_report.md
""")

    # Save test commands to a shell script
    script_path = Path("run_kevin_tests.sh")
    with open(script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Automated test script for Kevin's bugs\n")
        f.write("# Generated: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
        f.write("# Ensure Lobster is activated\n")
        f.write("# source .venv/bin/activate\n\n")

        for i, cmd in enumerate(commands, 1):
            f.write(f"# Test {i}: {cmd['name']}\n")
            f.write(f"echo '{'='*60}'\n")
            f.write(f"echo 'Test {i}: {cmd['name']}'\n")
            f.write(f"echo '{'='*60}'\n")
            f.write(f"{cmd['command']}\n")
            f.write(f"echo '\\nTest {i} complete. Waiting 5 seconds...'\n")
            f.write("sleep 5\n\n")

    script_path.chmod(0o755)
    print(f"\n✅ Test script saved to: {script_path}")
    print("   Run it with: ./run_kevin_tests.sh")

    # Create a Python test runner
    runner_path = Path("run_kevin_tests.py")
    with open(runner_path, "w") as f:
        f.write("""#!/usr/bin/env python3
import subprocess
import sys
import time

tests = [
""")
        for cmd in commands:
            f.write(f"    {repr(cmd)},\n")
        f.write("""]

def run_test(test_info):
    print(f"\\n{'='*60}")
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

    print(f"\\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")

    passed = sum(1 for _, s in results if s)
    print(f"\\nTotal: {passed}/{len(results)} tests passed")
    sys.exit(0 if passed == len(results) else 1)
""")

    runner_path.chmod(0o755)
    print(f"✅ Python test runner saved to: {runner_path}")
    print("   Run it with: python run_kevin_tests.py")

    return 0

if __name__ == "__main__":
    sys.exit(main())