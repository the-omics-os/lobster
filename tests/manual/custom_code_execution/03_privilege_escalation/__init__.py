"""
Privilege Escalation Security Tests for CustomCodeExecutionService

This test suite assesses subprocess breakout and privilege escalation vulnerabilities.

Test Categories:
- test_subprocess_breakout.py: Process creation, import bypass, resource exhaustion
- test_signal_manipulation.py: Parent discovery, signal attacks, environment access
- test_process_injection.py: Memory injection, IPC channels, Docker escapes

⚠️ SAFETY: All tests are detection-only, no actual exploitation.

Usage:
    pytest 03_privilege_escalation/ -v -s                    # Run all tests
    pytest 03_privilege_escalation/test_subprocess_breakout.py -v -s  # Specific category
    pytest 03_privilege_escalation/ -v -s -k "SIGKILL"      # Specific vulnerability
"""
