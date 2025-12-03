"""
Environment Variable Leakage Security Tests

Tests for environment variable-based information disclosure in CustomCodeExecutionService.
User code running in subprocess inherits parent process environment variables, which may
contain API keys, credentials, and other sensitive configuration.

Test Naming Convention:
- test_*_EXPECT_SUCCESS: Attack should succeed (vulnerability exists)
- test_*_EXPECT_BLOCKED: Attack should be blocked (protection exists)

Run with: pytest tests/manual/custom_code_execution/01_data_exfiltration/test_environment_leakage.py -v
"""

import os
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import pytest

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.execution import CodeExecutionError, CustomCodeExecutionService


class TestEnvironmentLeakage:
    """Test environment variable leakage attack vectors."""

    @pytest.fixture
    def workspace(self, tmp_path):
        """Create test workspace."""
        workspace = tmp_path / ".lobster_workspace"
        workspace.mkdir()
        return workspace

    @pytest.fixture
    def data_manager(self, workspace):
        """Create DataManagerV2."""
        dm = DataManagerV2(workspace_path=workspace)

        # Add test modality
        adata = anndata.AnnData(
            X=np.array([[1, 2], [3, 4]]),
            obs=pd.DataFrame({"cell": ["c1", "c2"]}, index=["c1", "c2"]),
            var=pd.DataFrame({"gene": ["g1", "g2"]}, index=["g1", "g2"]),
        )
        dm.modalities["test"] = adata

        return dm

    @pytest.fixture
    def service(self, data_manager):
        """Create service instance."""
        return CustomCodeExecutionService(data_manager)

    @pytest.fixture(autouse=True)
    def setup_test_env_vars(self):
        """Set up test environment variables (simulate real credentials)."""
        test_vars = {
            "TEST_API_KEY": "sk-test-api-key-12345",
            "TEST_SECRET": "super-secret-token",
            "TEST_DATABASE_URL": "postgresql://user:password@localhost/db",
        }

        # Save original values
        original = {k: os.environ.get(k) for k in test_vars}

        # Set test values
        for k, v in test_vars.items():
            os.environ[k] = v

        yield

        # Restore original values
        for k, v in original.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    # ========== Complete Environment Dump ==========

    def test_dump_all_environment_variables_EXPECT_SUCCESS(self, service):
        """
        Test: Dump all environment variables
        Expected: SUCCESS (vulnerability - subprocess inherits all env vars)
        Impact: CRITICAL - Full environment disclosure
        """
        code = """
import os

try:
    env_vars = dict(os.environ)
    sensitive_keys = [k for k in env_vars.keys() if any(
        secret in k.lower() for secret in ['key', 'secret', 'token', 'password', 'api']
    )]

    result = f"VULNERABILITY: Dumped {len(env_vars)} env vars, {len(sensitive_keys)} sensitive: {sensitive_keys[:5]}"
except Exception as e:
    result = f"PROTECTED: {type(e).__name__}"
"""

        result, stats, ir = service.execute(code, persist=False)

        print(f"\n[Environment dump] Result: {result}")

        if "VULNERABILITY" in str(result):
            print("⚠️  CRITICAL VULNERABILITY: Can dump all environment variables")
            assert "VULNERABILITY" in str(result), "Environment not isolated"
        else:
            print("✅ PROTECTED: Environment access blocked")
            assert "PROTECTED" in str(result)

    # ========== Anthropic API Key Theft ==========

    def test_steal_anthropic_api_key_EXPECT_SUCCESS(self, service):
        """
        Test: Access ANTHROPIC_API_KEY
        Expected: SUCCESS if key present (vulnerability)
        Impact: CRITICAL - Claude API access theft
        """
        code = """
import os

try:
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if api_key:
        result = f"VULNERABILITY: Stole ANTHROPIC_API_KEY (starts with {api_key[:10]}...)"
    else:
        result = "INFO: ANTHROPIC_API_KEY not set in environment"
except Exception as e:
    result = f"PROTECTED: {type(e).__name__}"
"""

        result, stats, ir = service.execute(code, persist=False)

        print(f"\n[ANTHROPIC_API_KEY] Result: {result}")

        if "VULNERABILITY" in str(result):
            print("⚠️  CRITICAL VULNERABILITY: Can steal Anthropic API key")
            assert "VULNERABILITY" in str(result)
        elif "INFO" in str(result):
            print("ℹ️  Cannot test (API key not present)")
            pytest.skip("ANTHROPIC_API_KEY not set")
        else:
            print("✅ PROTECTED: API key access blocked")
            assert "PROTECTED" in str(result)

    # ========== AWS Credentials Theft ==========

    def test_steal_aws_access_keys_EXPECT_SUCCESS(self, service):
        """
        Test: Access AWS_* environment variables
        Expected: SUCCESS if keys present (vulnerability)
        Impact: CRITICAL - Cloud infrastructure access
        """
        code = """
import os

try:
    aws_vars = {
        'AWS_ACCESS_KEY_ID': os.environ.get('AWS_ACCESS_KEY_ID'),
        'AWS_SECRET_ACCESS_KEY': os.environ.get('AWS_SECRET_ACCESS_KEY'),
        'AWS_BEDROCK_ACCESS_KEY': os.environ.get('AWS_BEDROCK_ACCESS_KEY'),
        'AWS_BEDROCK_SECRET_ACCESS_KEY': os.environ.get('AWS_BEDROCK_SECRET_ACCESS_KEY'),
    }

    found = {k: v[:10] + '...' for k, v in aws_vars.items() if v}

    if found:
        result = f"VULNERABILITY: Stole AWS credentials: {list(found.keys())}"
    else:
        result = "INFO: No AWS credentials in environment"
except Exception as e:
    result = f"PROTECTED: {type(e).__name__}"
"""

        result, stats, ir = service.execute(code, persist=False)

        print(f"\n[AWS credentials] Result: {result}")

        if "VULNERABILITY" in str(result):
            print("⚠️  CRITICAL VULNERABILITY: Can steal AWS credentials")
            assert "VULNERABILITY" in str(result)
        elif "INFO" in str(result):
            print("ℹ️  Cannot test (AWS credentials not present)")
            pytest.skip("AWS credentials not set")
        else:
            print("✅ PROTECTED: AWS credentials blocked")
            assert "PROTECTED" in str(result)

    # ========== GitHub Token Theft ==========

    def test_steal_github_token_EXPECT_SUCCESS(self, service):
        """
        Test: Access GITHUB_TOKEN
        Expected: SUCCESS if token present (vulnerability)
        Impact: CRITICAL - Source code repository access
        """
        code = """
import os

try:
    tokens = {
        'GITHUB_TOKEN': os.environ.get('GITHUB_TOKEN'),
        'GH_TOKEN': os.environ.get('GH_TOKEN'),
        'GITHUB_PAT': os.environ.get('GITHUB_PAT'),
    }

    found = {k: v[:10] + '...' for k, v in tokens.items() if v}

    if found:
        result = f"VULNERABILITY: Stole GitHub token(s): {list(found.keys())}"
    else:
        result = "INFO: No GitHub tokens in environment"
except Exception as e:
    result = f"PROTECTED: {type(e).__name__}"
"""

        result, stats, ir = service.execute(code, persist=False)

        print(f"\n[GitHub tokens] Result: {result}")

        if "VULNERABILITY" in str(result):
            print("⚠️  CRITICAL VULNERABILITY: Can steal GitHub tokens")
            assert "VULNERABILITY" in str(result)
        elif "INFO" in str(result):
            print("ℹ️  Cannot test (GitHub tokens not present)")
            pytest.skip("GitHub tokens not set")
        else:
            print("✅ PROTECTED: GitHub tokens blocked")
            assert "PROTECTED" in str(result)

    # ========== Database Credentials Theft ==========

    def test_steal_database_urls_EXPECT_SUCCESS(self, service):
        """
        Test: Access database connection strings
        Expected: SUCCESS (vulnerability - test vars are set)
        Impact: CRITICAL - Database access theft
        """
        code = """
import os

try:
    db_vars = {
        'DATABASE_URL': os.environ.get('DATABASE_URL'),
        'DB_CONNECTION_STRING': os.environ.get('DB_CONNECTION_STRING'),
        'POSTGRES_URL': os.environ.get('POSTGRES_URL'),
        'MYSQL_URL': os.environ.get('MYSQL_URL'),
        'TEST_DATABASE_URL': os.environ.get('TEST_DATABASE_URL'),
    }

    found = {}
    for k, v in db_vars.items():
        if v:
            # Redact password for display
            if '@' in v:
                parts = v.split('@')
                found[k] = parts[0].split(':')[0] + ':***@' + parts[1]
            else:
                found[k] = v[:20] + '...'

    if found:
        result = f"VULNERABILITY: Stole database URLs: {found}"
    else:
        result = "INFO: No database URLs in environment"
except Exception as e:
    result = f"PROTECTED: {type(e).__name__}"
"""

        result, stats, ir = service.execute(code, persist=False)

        print(f"\n[Database URLs] Result: {result}")

        if "VULNERABILITY" in str(result):
            print("⚠️  CRITICAL VULNERABILITY: Can steal database credentials")
            assert "VULNERABILITY" in str(result)
        elif "INFO" in str(result):
            print("ℹ️  Cannot test (database URLs not present)")
            pytest.skip("Database URLs not set")
        else:
            print("✅ PROTECTED: Database credentials blocked")
            assert "PROTECTED" in str(result)

    # ========== Test Credential Theft (Known Present) ==========

    def test_steal_test_api_key_EXPECT_SUCCESS(self, service):
        """
        Test: Access TEST_API_KEY (set by fixture)
        Expected: SUCCESS (vulnerability - proves inheritance)
        Impact: CRITICAL - Confirms subprocess inherits parent env
        """
        code = """
import os

try:
    test_key = os.environ.get('TEST_API_KEY')
    if test_key:
        result = f"VULNERABILITY: Stole TEST_API_KEY: {test_key}"
    else:
        result = "ERROR: Test setup failed (key not present)"
except Exception as e:
    result = f"PROTECTED: {type(e).__name__}"
"""

        result, stats, ir = service.execute(code, persist=False)

        print(f"\n[TEST_API_KEY] Result: {result}")

        if "VULNERABILITY" in str(result):
            print("⚠️  CRITICAL VULNERABILITY: Subprocess inherits parent environment")
            assert "VULNERABILITY" in str(result)
        else:
            print("✅ PROTECTED: Environment inheritance blocked")
            assert "PROTECTED" in str(result)

    # ========== Shell and System Paths ==========

    def test_probe_system_paths_EXPECT_SUCCESS(self, service):
        """
        Test: Access PATH and other system configuration
        Expected: SUCCESS (vulnerability - system info disclosure)
        Impact: MEDIUM - System reconnaissance
        """
        code = """
import os

try:
    sys_vars = {
        'PATH': os.environ.get('PATH', ''),
        'HOME': os.environ.get('HOME', ''),
        'USER': os.environ.get('USER', ''),
        'SHELL': os.environ.get('SHELL', ''),
    }

    info = {k: v[:50] + '...' if len(v) > 50 else v for k, v in sys_vars.items() if v}

    result = f"VULNERABILITY: System info disclosed: {list(info.keys())}"
except Exception as e:
    result = f"PROTECTED: {type(e).__name__}"
"""

        result, stats, ir = service.execute(code, persist=False)

        print(f"\n[System paths] Result: {result}")

        if "VULNERABILITY" in str(result):
            print("⚠️  MEDIUM VULNERABILITY: System paths and user info accessible")
            assert "VULNERABILITY" in str(result)
        else:
            print("✅ PROTECTED: System info blocked")
            assert "PROTECTED" in str(result)

    # ========== Parent Process Information ==========

    def test_access_parent_process_environ_EXPECT_SUCCESS(self, service):
        """
        Test: Access parent process environment via /proc
        Expected: SUCCESS on Linux (vulnerability)
        Impact: HIGH - Parent process environment disclosure
        """
        code = """
import os
from pathlib import Path

try:
    # Get parent PID
    ppid = os.getppid()

    # Try to read parent's environment (Linux only)
    parent_environ_path = Path(f'/proc/{ppid}/environ')

    if parent_environ_path.exists():
        content = parent_environ_path.read_bytes()
        env_vars = content.split(b'\\x00')
        result = f"VULNERABILITY: Read parent process environ ({len(env_vars)} variables)"
    else:
        result = "INFO: /proc not available (not Linux)"
except PermissionError:
    result = "PROTECTED: Permission denied"
except Exception as e:
    result = f"PROTECTED: {type(e).__name__}"
"""

        result, stats, ir = service.execute(code, persist=False)

        print(f"\n[Parent process environ] Result: {result}")

        if "VULNERABILITY" in str(result):
            print("⚠️  HIGH VULNERABILITY: Can read parent process environment")
            assert "VULNERABILITY" in str(result)
        elif "INFO" in str(result):
            print("ℹ️  Cannot test (not Linux)")
            pytest.skip("Test requires Linux /proc")
        else:
            print("✅ PROTECTED: Parent environ access blocked")
            assert "PROTECTED" in str(result)

    # ========== Environment Variable Injection Attack ==========

    def test_modify_environment_for_next_execution_EXPECT_SUCCESS(self, service):
        """
        Test: Modify subprocess environment
        Expected: SUCCESS (vulnerability - can modify own environment)
        Impact: MEDIUM - Environment pollution
        """
        code = """
import os

try:
    # Modify environment
    os.environ['INJECTED_VAR'] = 'MALICIOUS_VALUE'

    # Verify modification
    if os.environ.get('INJECTED_VAR') == 'MALICIOUS_VALUE':
        result = "VULNERABILITY: Modified subprocess environment (pollution possible)"
    else:
        result = "PROTECTED: Environment modification failed"
except Exception as e:
    result = f"PROTECTED: {type(e).__name__}"
"""

        result, stats, ir = service.execute(code, persist=False)

        print(f"\n[Environment injection] Result: {result}")

        if "VULNERABILITY" in str(result):
            print("⚠️  MEDIUM VULNERABILITY: Can modify subprocess environment")
            assert "VULNERABILITY" in str(result)
        else:
            print("✅ PROTECTED: Environment modification blocked")
            assert "PROTECTED" in str(result)

    # ========== Sensitive Environment Variable Patterns ==========

    def test_pattern_based_credential_search_EXPECT_SUCCESS(self, service):
        """
        Test: Search environment for credential patterns
        Expected: SUCCESS (vulnerability - can grep for secrets)
        Impact: CRITICAL - Automated credential harvesting
        """
        code = """
import os
import re

try:
    env = dict(os.environ)

    # Common credential patterns
    patterns = {
        'api_keys': r'(?:api[_-]?key|apikey|key)',
        'secrets': r'(?:secret|token|password|pwd)',
        'aws': r'(?:aws|amazon)',
        'github': r'(?:github|gh)',
    }

    matches = {}
    for category, pattern in patterns.items():
        matching_vars = [k for k in env.keys() if re.search(pattern, k, re.IGNORECASE)]
        if matching_vars:
            matches[category] = matching_vars

    if matches:
        result = f"VULNERABILITY: Found credential patterns: {matches}"
    else:
        result = "INFO: No credential patterns found"
except Exception as e:
    result = f"PROTECTED: {type(e).__name__}"
"""

        result, stats, ir = service.execute(code, persist=False)

        print(f"\n[Credential pattern search] Result: {result}")

        if "VULNERABILITY" in str(result):
            print("⚠️  CRITICAL VULNERABILITY: Can search for credentials by pattern")
            assert "VULNERABILITY" in str(result)
        elif "INFO" in str(result):
            print("ℹ️  Cannot test (no matching patterns)")
            pytest.skip("No credential patterns found")
        else:
            print("✅ PROTECTED: Pattern search blocked")
            assert "PROTECTED" in str(result)


class TestEnvironmentLeakageSummary:
    """Generate summary report of environment leakage tests."""

    def test_generate_summary(self):
        """Print summary of environment leakage vectors."""
        print("\n" + "=" * 70)
        print("ENVIRONMENT VARIABLE LEAKAGE ATTACK SURFACE SUMMARY")
        print("=" * 70)
        print("\nTested Attack Vectors:")
        print("1. ⚠️  Dump all environment variables")
        print("2. ⚠️  Steal ANTHROPIC_API_KEY")
        print("3. ⚠️  Steal AWS credentials")
        print("4. ⚠️  Steal GitHub tokens")
        print("5. ⚠️  Steal database URLs")
        print("6. ⚠️  Steal test credentials (fixture)")
        print("7. ⚠️  Probe system paths (PATH, HOME, USER)")
        print("8. ⚠️  Access parent process environment")
        print("9. ⚠️  Modify subprocess environment")
        print("10. ⚠️ Pattern-based credential search")
        print("\nExpected Result: All applicable vulnerabilities should be confirmed")
        print("(Some tests require specific environment variables to be set)")
        print("=" * 70 + "\n")

        assert True  # Always pass - this is just a summary
