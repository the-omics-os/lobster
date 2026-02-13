# Integration Attack Tests for CustomCodeExecutionService

This directory contains security tests for **multi-step exploits** and **agent chaining attacks** that demonstrate how individual vulnerabilities can be combined into devastating attack chains.

## Overview

While single-execution attacks may be contained by subprocess isolation, **multi-step attacks leverage workspace persistence** to:
- Install persistent backdoors across executions
- Compromise multiple agents via shared resources
- Evade detection through delayed triggers and obfuscation
- Achieve complete platform compromise

## Test Files

### 1. `test_multi_step_exploits.py`
Multi-step attacks requiring multiple executions to complete.

**Test Classes**:
- `TestPersistentBackdoors` - Import-time backdoors, delayed triggers, module hijacking
- `TestWorkspacePoisoning` - Session corruption, cache poisoning, queue manipulation
- `TestObfuscatedAttacks` - Hidden exfiltration in legitimate code
- `TestCredentialHarvesting` - Environment variable theft

**Key Tests**:
```bash
# Persistent backdoor via module import (CRITICAL)
pytest test_multi_step_exploits.py::TestPersistentBackdoors::test_backdoor_via_malicious_module_EXPECT_SUCCESS -v -s

# Delayed execution backdoor (CRITICAL)
pytest test_multi_step_exploits.py::TestPersistentBackdoors::test_delayed_execution_backdoor_EXPECT_SUCCESS -v -s

# Environment variable harvesting (CRITICAL)
pytest test_multi_step_exploits.py::TestCredentialHarvesting::test_env_var_harvesting_EXPECT_SUCCESS -v -s

# Obfuscated data exfiltration (HIGH)
pytest test_multi_step_exploits.py::TestObfuscatedAttacks::test_obfuscated_exfiltration_EXPECT_SUCCESS -v -s
```

---

### 2. `test_agent_chaining.py`
Cross-agent attacks via workspace manipulation.

**Test Classes**:
- `TestQueueManipulation` - Download queue injection, publication queue poisoning
- `TestCrossAgentAttacks` - Modality poisoning, provenance tampering
- `TestRealisticAttackScenarios` - Complete multi-step attack chains

**Key Tests**:
```bash
# Download queue injection (CRITICAL)
pytest test_agent_chaining.py::TestQueueManipulation::test_download_queue_injection_EXPECT_SUCCESS -v -s

# Modality data poisoning (HIGH)
pytest test_agent_chaining.py::TestCrossAgentAttacks::test_modality_poisoning_for_analysis_EXPECT_SUCCESS -v -s

# Complete attack chain (CRITICAL - MUST RUN)
pytest test_agent_chaining.py::TestRealisticAttackScenarios::test_complete_attack_chain_EXPECT_SUCCESS -v -s
```

---

## Critical Findings

### 🔴 CRITICAL Vulnerabilities (6)

1. **Persistent Backdoor via Import** (`test_backdoor_via_malicious_module_EXPECT_SUCCESS`)
   - **Attack**: Create malicious `.py` file → Import triggers backdoor
   - **Impact**: Persistent code execution across all future sessions
   - **Root Cause**: Python imports workspace modules without validation

2. **Complete Compromise Chain** (`test_complete_attack_chain_EXPECT_SUCCESS`)
   - **Attack**: Innocent request → Hidden backdoor → Credential theft → Queue poisoning
   - **Impact**: Complete platform compromise, affects all agents
   - **Root Cause**: No defense-in-depth, multiple weaknesses exploited

3. **Environment Variable Harvesting** (`test_env_var_harvesting_EXPECT_SUCCESS`)
   - **Attack**: Access `os.environ` → Exfiltrate API keys, credentials
   - **Impact**: Credential theft enables downstream attacks
   - **Root Cause**: No environment filtering in subprocess

4. **Download Queue Injection** (`test_download_queue_injection_EXPECT_SUCCESS`)
   - **Attack**: Modify `download_queue.jsonl` → Inject malicious dataset URLs
   - **Impact**: Supply chain attack, malicious data enters pipeline
   - **Root Cause**: Queue files writable, no signature verification

5. **Delayed Execution Backdoor** (`test_delayed_execution_backdoor_EXPECT_SUCCESS`)
   - **Attack**: Conditional trigger (time/count) → Evades immediate detection
   - **Impact**: Backdoor activates weeks after initial compromise
   - **Root Cause**: No anomaly detection across executions

6. **Module Name Hijacking** (`test_module_hijacking_EXPECT_SUCCESS`)
   - **Attack**: Create malicious `utils.py` → Shadows legitimate modules
   - **Impact**: Every import of 'utils' executes backdoor
   - **Root Cause**: Workspace first in Python import path

### 🟡 HIGH Risk Issues (5)

7. **Modality Data Poisoning** - Inject fake QC metrics, fake clustering results
8. **Publication Queue Poisoning** - Inject fake papers with malicious identifiers
9. **Provenance Tampering** - Hide evidence, inject fake audit records
10. **Obfuscated Exfiltration** - Hide theft in 200+ lines of legitimate code
11. **Session Corruption** - Manipulate agent state, hide operations

## Running Tests

### Run All Integration Attack Tests
```bash
cd /Users/tyo/GITHUB/omics-os/lobster
pytest tests/manual/custom_code_execution/08_integration_attacks/ -v -s
```

### Run Specific Categories
```bash
# Persistent backdoors only
pytest tests/manual/custom_code_execution/08_integration_attacks/test_multi_step_exploits.py::TestPersistentBackdoors -v -s

# Cross-agent attacks only
pytest tests/manual/custom_code_execution/08_integration_attacks/test_agent_chaining.py::TestCrossAgentAttacks -v -s

# Complete attack chains (CRITICAL)
pytest tests/manual/custom_code_execution/08_integration_attacks/test_agent_chaining.py::TestRealisticAttackScenarios -v -s
```

### Run Single Test
```bash
# Complete compromise demonstration
pytest tests/manual/custom_code_execution/08_integration_attacks/test_agent_chaining.py::TestRealisticAttackScenarios::test_complete_attack_chain_EXPECT_SUCCESS -v -s
```

## Expected Test Results

**All tests should PASS** (indicating vulnerabilities exist):
```
test_backdoor_via_malicious_module_EXPECT_SUCCESS ................. PASSED
test_delayed_execution_backdoor_EXPECT_SUCCESS .................... PASSED
test_module_hijacking_EXPECT_SUCCESS .............................. PASSED
test_session_file_corruption_EXPECT_SUCCESS ....................... PASSED
test_cache_poisoning_EXPECT_SUCCESS ............................... PASSED
test_obfuscated_exfiltration_EXPECT_SUCCESS ....................... PASSED
test_env_var_harvesting_EXPECT_SUCCESS ............................ PASSED
test_download_queue_injection_EXPECT_SUCCESS ...................... PASSED
test_publication_queue_poisoning_EXPECT_SUCCESS ................... PASSED
test_queue_status_manipulation_EXPECT_SUCCESS ..................... PASSED
test_modality_poisoning_for_analysis_EXPECT_SUCCESS ............... PASSED
test_provenance_log_tampering_EXPECT_SUCCESS ...................... PASSED
test_complete_attack_chain_EXPECT_SUCCESS ......................... PASSED

======================== 13 passed in X.XXs ========================
```

## Test Output Interpretation

### Success Indicators (Vulnerabilities Detected)
```
⚠️ CRITICAL: Backdoor executed successfully!
⚠️ CRITICAL: Environment harvesting succeeded!
⚠️ CRITICAL: Queue injection succeeded!
⚠️ CRITICAL: Complete attack chain succeeded!
```

### Protected Indicators (Would Show If Fixed)
```
✅ PROTECTED: Backdoor did not execute
✅ PROTECTED: No data exfiltration
✅ Queue integrity maintained
✅ Session integrity maintained
```

## Attack Demonstration Walkthrough

### Demo 1: Persistent Backdoor (5 minutes)
```bash
# Run persistent backdoor test
pytest test_multi_step_exploits.py::TestPersistentBackdoors::test_backdoor_via_malicious_module_EXPECT_SUCCESS -v -s
```

**What You'll See**:
1. Step 1: Code creates "analysis_helpers.py" module
2. Step 2: Next execution imports module
3. Backdoor activates automatically
4. Environment variables exfiltrated to `/tmp/`
5. Attack succeeds ⚠️

### Demo 2: Complete Attack Chain (10 minutes)
```bash
# Run complete compromise demonstration
pytest test_agent_chaining.py::TestRealisticAttackScenarios::test_complete_attack_chain_EXPECT_SUCCESS -v -s
```

**What You'll See**:
1. User asks innocent question: "Calculate 95th percentile"
2. Code performs calculation correctly
3. Hidden: Backdoor installed as "qc_helpers.py"
4. User sees legitimate result
5. Next execution: Backdoor activates
6. Environment exfiltrated
7. Download queue poisoned
8. Complete compromise achieved ⚠️

## Mitigation Testing

After implementing security fixes, tests should **FAIL** (indicating vulnerabilities patched):

```bash
# Expected after workspace segregation
pytest test_agent_chaining.py::TestQueueManipulation::test_download_queue_injection_EXPECT_SUCCESS -v -s
# Expected: FAILED - PermissionError: Cannot write to system/download_queue.jsonl

# Expected after import sandboxing
pytest test_multi_step_exploits.py::TestPersistentBackdoors::test_backdoor_via_malicious_module_EXPECT_SUCCESS -v -s
# Expected: FAILED - ImportError: Module 'analysis_helpers' not in allowed imports

# Expected after env filtering
pytest test_multi_step_exploits.py::TestCredentialHarvesting::test_env_var_harvesting_EXPECT_SUCCESS -v -s
# Expected: FAILED - Environment variables not accessible
```

## Defense Recommendations

### 🔴 CRITICAL (Implement Immediately)
1. **Workspace Segregation**: System files read-only, user files isolated
2. **Import Sandboxing**: Remove workspace from `sys.path`, whitelist imports
3. **File Integrity Monitoring**: Checksum validation before/after execution
4. **Environment Filtering**: Remove sensitive env vars from subprocess

### 🟡 HIGH (Within Sprint)
5. **Queue Cryptographic Signing**: research_agent signs entries, data_expert verifies
6. **Provenance Append-Only**: Cryptographic signatures prevent tampering
7. **Content Security Policy**: Extended AST validation for exfiltration patterns

### 🟢 MEDIUM (Next Quarter)
8. **Anomaly Detection**: Monitor file creation, env access, external writes
9. **Execution Rate Limiting**: Prevent rapid-fire attacks
10. **Audit Logging**: Immutable log of all custom code executions

## Related Documentation

- **Main Report**: `INTEGRATION_ATTACKS_REPORT.md` - Comprehensive findings
- **Test Files**:
  - `test_multi_step_exploits.py` - Persistence and obfuscation
  - `test_agent_chaining.py` - Cross-agent compromise
- **Other Security Reports**: `tests/manual/custom_code_execution/01_input_validation/` through `07_package_vulnerabilities/`

## Contact

For questions or security concerns:
- Review: `INTEGRATION_ATTACKS_REPORT.md`
- Tests: Run with `-v -s` for detailed output
- Issues: Report to security team with test output

---

**Last Updated**: 2025-11-30
**Agent**: 8 - Integration Attack Tester
**Status**: COMPLETE
**Severity**: CRITICAL
