"""
Resource Exhaustion Attack Tests for CustomCodeExecutionService.

This module tests for denial-of-service vulnerabilities through:
- Memory exhaustion (OOM attacks)
- CPU exhaustion (infinite loops, cryptographic operations)
- Disk exhaustion (large file creation)
- File descriptor exhaustion

SAFETY NOTICE:
Tests use SAFE limits to avoid crashing the test machine.
Each test documents what WOULD happen with larger attack values.
"""
