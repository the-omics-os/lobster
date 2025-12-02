"""
Data Exfiltration Security Test Suite

This package contains security tests for CustomCodeExecutionService,
focusing on data exfiltration vulnerabilities.

Test Categories:
- Network exfiltration (HTTP, sockets, DNS, etc.)
- Filesystem access (system files, SSH keys, credentials)
- Environment variable leakage (API keys, secrets)

Usage:
    pytest tests/manual/custom_code_execution/01_data_exfiltration/ -v

See: DATA_EXFILTRATION_REPORT.md for detailed findings.
"""
