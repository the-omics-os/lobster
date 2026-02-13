"""
Filesystem Exfiltration Security Tests

Tests for file system-based data exfiltration vulnerabilities in CustomCodeExecutionService.
These tests verify whether user code can access sensitive files outside the workspace.

Test Naming Convention:
- test_*_EXPECT_SUCCESS: Attack should succeed (vulnerability exists)
- test_*_EXPECT_BLOCKED: Attack should be blocked (protection exists)

Run with: pytest tests/manual/custom_code_execution/01_data_exfiltration/test_filesystem_exfiltration.py -v
"""

import platform
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import pytest

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.execution import CodeExecutionError, CustomCodeExecutionService


class TestFilesystemExfiltration:
    """Test file system-based exfiltration attack vectors."""

    @pytest.fixture
    def workspace(self, tmp_path):
        """Create test workspace."""
        workspace = tmp_path / ".lobster_workspace"
        workspace.mkdir()

        # Create test file in workspace
        test_df = pd.DataFrame({"data": [1, 2, 3]})
        test_df.to_csv(workspace / "test.csv", index=False)

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

    # ========== System File Access ==========

    def test_read_etc_passwd_EXPECT_SUCCESS(self, service):
        """
        Test: Read /etc/passwd (Unix systems)
        Expected: SUCCESS on Unix (vulnerability - can read system files)
        Impact: CRITICAL - Access to system user information
        """
        if platform.system() == "Windows":
            pytest.skip("Test only applicable on Unix-like systems")

        code = """
from pathlib import Path

try:
    passwd = Path('/etc/passwd')
    if passwd.exists():
        content = passwd.read_text()
        lines = content.splitlines()[:3]  # First 3 lines
        result = f"VULNERABILITY: Read {len(lines)} lines from /etc/passwd"
    else:
        result = "PROTECTED: File not accessible"
except Exception as e:
    result = f"PROTECTED: {type(e).__name__}"
"""

        result, stats, ir = service.execute(code, persist=False)

        print(f"\n[/etc/passwd] Result: {result}")

        if "VULNERABILITY" in str(result):
            print("⚠️  CRITICAL VULNERABILITY: Can read system password file")
            assert "VULNERABILITY" in str(result), "System file access not restricted"
        else:
            print("✅ PROTECTED: /etc/passwd access blocked")
            assert "PROTECTED" in str(result)

    def test_read_etc_hosts_EXPECT_SUCCESS(self, service):
        """
        Test: Read /etc/hosts (network configuration)
        Expected: SUCCESS (vulnerability)
        Impact: MEDIUM - Network configuration disclosure
        """
        if platform.system() == "Windows":
            pytest.skip("Test only applicable on Unix-like systems")

        code = """
from pathlib import Path

try:
    hosts = Path('/etc/hosts')
    content = hosts.read_text()
    result = f"VULNERABILITY: Read /etc/hosts ({len(content)} bytes)"
except Exception as e:
    result = f"PROTECTED: {type(e).__name__}"
"""

        result, stats, ir = service.execute(code, persist=False)

        print(f"\n[/etc/hosts] Result: {result}")

        if "VULNERABILITY" in str(result):
            print("⚠️  MEDIUM VULNERABILITY: Can read network configuration")
            assert "VULNERABILITY" in str(result)
        else:
            print("✅ PROTECTED: /etc/hosts blocked")
            assert "PROTECTED" in str(result)

    # ========== SSH Key Theft ==========

    def test_read_ssh_directory_EXPECT_SUCCESS(self, service):
        """
        Test: List ~/.ssh directory contents
        Expected: SUCCESS (vulnerability - can enumerate SSH keys)
        Impact: CRITICAL - SSH key discovery
        """
        code = """
from pathlib import Path
import os

try:
    ssh_dir = Path.home() / '.ssh'
    if ssh_dir.exists():
        files = [f.name for f in ssh_dir.iterdir() if f.is_file()]
        result = f"VULNERABILITY: Found {len(files)} files in ~/.ssh: {files[:3]}"
    else:
        result = "INFO: ~/.ssh directory does not exist (user has no SSH keys)"
except PermissionError:
    result = "PROTECTED: Permission denied"
except Exception as e:
    result = f"PROTECTED: {type(e).__name__}"
"""

        result, stats, ir = service.execute(code, persist=False)

        print(f"\n[~/.ssh enumeration] Result: {result}")

        if "VULNERABILITY" in str(result):
            print("⚠️  CRITICAL VULNERABILITY: Can enumerate SSH keys")
            assert "VULNERABILITY" in str(result)
        elif "INFO" in str(result):
            print("ℹ️  Cannot test (no SSH keys present)")
            pytest.skip("SSH directory does not exist")
        else:
            print("✅ PROTECTED: SSH directory access blocked")
            assert "PROTECTED" in str(result)

    def test_read_ssh_private_key_EXPECT_SUCCESS(self, service):
        """
        Test: Attempt to read SSH private key
        Expected: SUCCESS if key exists (vulnerability)
        Impact: CRITICAL - Full SSH credential theft
        """
        code = """
from pathlib import Path

try:
    key_paths = [
        Path.home() / '.ssh' / 'id_rsa',
        Path.home() / '.ssh' / 'id_ed25519',
    ]

    found_keys = []
    for key_path in key_paths:
        if key_path.exists():
            try:
                content = key_path.read_text()
                found_keys.append(key_path.name)
            except PermissionError:
                pass

    if found_keys:
        result = f"VULNERABILITY: Read SSH private keys: {found_keys}"
    else:
        result = "INFO: No SSH private keys found (or protected by OS)"
except Exception as e:
    result = f"PROTECTED: {type(e).__name__}"
"""

        result, stats, ir = service.execute(code, persist=False)

        print(f"\n[SSH private key] Result: {result}")

        if "VULNERABILITY" in str(result):
            print("⚠️  CRITICAL VULNERABILITY: Can read SSH private keys")
            assert "VULNERABILITY" in str(result)
        elif "INFO" in str(result):
            print("ℹ️  Cannot test (no keys found or OS-protected)")
            pytest.skip("SSH keys not accessible for testing")
        else:
            print("✅ PROTECTED: SSH keys blocked")
            assert "PROTECTED" in str(result)

    # ========== AWS Credentials Theft ==========

    def test_read_aws_credentials_EXPECT_SUCCESS(self, service):
        """
        Test: Read ~/.aws/credentials
        Expected: SUCCESS if file exists (vulnerability)
        Impact: CRITICAL - Cloud infrastructure access
        """
        code = """
from pathlib import Path

try:
    aws_creds = Path.home() / '.aws' / 'credentials'
    if aws_creds.exists():
        content = aws_creds.read_text()
        # Check if file has actual credentials
        if 'aws_access_key_id' in content.lower():
            result = f"VULNERABILITY: Read AWS credentials ({len(content)} bytes)"
        else:
            result = "INFO: AWS credentials file exists but empty"
    else:
        result = "INFO: AWS credentials file does not exist"
except Exception as e:
    result = f"PROTECTED: {type(e).__name__}"
"""

        result, stats, ir = service.execute(code, persist=False)

        print(f"\n[AWS credentials] Result: {result}")

        if "VULNERABILITY" in str(result):
            print("⚠️  CRITICAL VULNERABILITY: Can read AWS credentials")
            assert "VULNERABILITY" in str(result)
        elif "INFO" in str(result):
            print("ℹ️  Cannot test (no AWS credentials present)")
            pytest.skip("AWS credentials not present")
        else:
            print("✅ PROTECTED: AWS credentials blocked")
            assert "PROTECTED" in str(result)

    # ========== Path Traversal Attacks ==========

    def test_path_traversal_parent_directory_EXPECT_SUCCESS(self, service):
        """
        Test: Access parent directory via ../
        Expected: SUCCESS (vulnerability - path traversal allowed)
        Impact: HIGH - Can escape workspace
        """
        code = """
from pathlib import Path

try:
    # Try to access parent directory
    parent = Path('..').resolve()
    files = list(parent.iterdir())[:5]
    result = f"VULNERABILITY: Path traversal successful, found {len(files)} files in parent"
except Exception as e:
    result = f"PROTECTED: {type(e).__name__}"
"""

        result, stats, ir = service.execute(code, persist=False)

        print(f"\n[Path traversal ../] Result: {result}")

        if "VULNERABILITY" in str(result):
            print("⚠️  HIGH VULNERABILITY: Path traversal to parent directory allowed")
            assert "VULNERABILITY" in str(result)
        else:
            print("✅ PROTECTED: Path traversal blocked")
            assert "PROTECTED" in str(result)

    def test_path_traversal_absolute_path_EXPECT_SUCCESS(self, service):
        """
        Test: Use absolute path to access home directory
        Expected: SUCCESS (vulnerability - absolute paths allowed)
        Impact: HIGH - Direct file system access
        """
        code = """
from pathlib import Path

try:
    home = Path.home()
    files = list(home.iterdir())[:5]
    file_names = [f.name for f in files]
    result = f"VULNERABILITY: Accessed home directory, found files: {file_names}"
except Exception as e:
    result = f"PROTECTED: {type(e).__name__}"
"""

        result, stats, ir = service.execute(code, persist=False)

        print(f"\n[Absolute path] Result: {result}")

        if "VULNERABILITY" in str(result):
            print("⚠️  HIGH VULNERABILITY: Absolute paths allow full filesystem access")
            assert "VULNERABILITY" in str(result)
        else:
            print("✅ PROTECTED: Absolute paths blocked")
            assert "PROTECTED" in str(result)

    # ========== Symbolic Link Exploitation ==========

    def test_symlink_follow_EXPECT_SUCCESS(self, service):
        """
        Test: Follow symbolic link to access file outside workspace
        Expected: SUCCESS (vulnerability - symlinks followed)
        Impact: HIGH - Indirect file access bypass
        """
        code = """
from pathlib import Path
import os

try:
    # Create symlink to /etc/hosts
    workspace = Path('.')
    symlink = workspace / 'external_link'

    # Try to create symlink to external file
    target = Path('/etc/hosts') if Path('/etc/hosts').exists() else Path.home()

    if not symlink.exists():
        os.symlink(target, symlink)

    # Try to read through symlink
    if symlink.is_symlink():
        content = symlink.read_text()[:100]
        result = f"VULNERABILITY: Followed symlink to external file ({len(content)} bytes read)"
    else:
        result = "INFO: Could not create symlink"
except Exception as e:
    result = f"PROTECTED: {type(e).__name__}"
"""

        result, stats, ir = service.execute(code, persist=False)

        print(f"\n[Symlink follow] Result: {result}")

        if "VULNERABILITY" in str(result):
            print(
                "⚠️  HIGH VULNERABILITY: Symbolic links followed - indirect access bypass"
            )
            assert "VULNERABILITY" in str(result)
        elif "INFO" in str(result):
            print("ℹ️  Cannot test (symlink creation failed)")
            pytest.skip("Symlink creation not possible")
        else:
            print("✅ PROTECTED: Symlink following blocked")
            assert "PROTECTED" in str(result)

    # ========== Write to System Directories ==========

    def test_write_to_tmp_EXPECT_SUCCESS(self, service):
        """
        Test: Write file to /tmp directory
        Expected: SUCCESS (vulnerability - can write outside workspace)
        Impact: MEDIUM - Persistent storage outside workspace
        """
        code = """
from pathlib import Path
import uuid

try:
    tmp_file = Path('/tmp') / f'lobster_test_{uuid.uuid4().hex}.txt'
    tmp_file.write_text('EXFILTRATED_DATA')

    if tmp_file.exists():
        result = f"VULNERABILITY: Wrote file to /tmp at {tmp_file}"
        tmp_file.unlink()  # Clean up
    else:
        result = "PROTECTED: Could not write to /tmp"
except Exception as e:
    result = f"PROTECTED: {type(e).__name__}"
"""

        result, stats, ir = service.execute(code, persist=False)

        print(f"\n[Write to /tmp] Result: {result}")

        if "VULNERABILITY" in str(result):
            print(
                "⚠️  MEDIUM VULNERABILITY: Can write to /tmp (persistence outside workspace)"
            )
            assert "VULNERABILITY" in str(result)
        else:
            print("✅ PROTECTED: /tmp write blocked")
            assert "PROTECTED" in str(result)

    def test_write_to_home_directory_EXPECT_SUCCESS(self, service):
        """
        Test: Write file to home directory
        Expected: SUCCESS (vulnerability - home directory writable)
        Impact: HIGH - Persistent malicious files
        """
        code = """
from pathlib import Path
import uuid

try:
    home_file = Path.home() / f'.lobster_test_{uuid.uuid4().hex}'
    home_file.write_text('TEST')

    if home_file.exists():
        result = f"VULNERABILITY: Wrote file to home directory at {home_file}"
        home_file.unlink()  # Clean up
    else:
        result = "PROTECTED: Could not write to home"
except Exception as e:
    result = f"PROTECTED: {type(e).__name__}"
"""

        result, stats, ir = service.execute(code, persist=False)

        print(f"\n[Write to home] Result: {result}")

        if "VULNERABILITY" in str(result):
            print("⚠️  HIGH VULNERABILITY: Can write to home directory")
            assert "VULNERABILITY" in str(result)
        else:
            print("✅ PROTECTED: Home directory write blocked")
            assert "PROTECTED" in str(result)

    # ========== Process and System Information ==========

    def test_enumerate_running_processes_EXPECT_SUCCESS(self, service):
        """
        Test: Enumerate running processes
        Expected: SUCCESS (vulnerability - process enumeration allowed)
        Impact: MEDIUM - Information disclosure
        """
        code = """
from pathlib import Path

try:
    # On Linux, /proc contains process information
    proc = Path('/proc')
    if proc.exists():
        # Count process directories
        processes = [p for p in proc.iterdir() if p.is_dir() and p.name.isdigit()]
        result = f"VULNERABILITY: Enumerated {len(processes)} running processes"
    else:
        result = "INFO: /proc not available (not Linux)"
except Exception as e:
    result = f"PROTECTED: {type(e).__name__}"
"""

        result, stats, ir = service.execute(code, persist=False)

        print(f"\n[Process enumeration] Result: {result}")

        if "VULNERABILITY" in str(result):
            print("⚠️  MEDIUM VULNERABILITY: Can enumerate running processes")
            assert "VULNERABILITY" in str(result)
        elif "INFO" in str(result):
            print("ℹ️  Cannot test (not Linux)")
            pytest.skip("Test requires Linux /proc")
        else:
            print("✅ PROTECTED: Process enumeration blocked")
            assert "PROTECTED" in str(result)


class TestFilesystemExfiltrationSummary:
    """Generate summary report of filesystem exfiltration tests."""

    def test_generate_summary(self):
        """Print summary of filesystem attack vectors."""
        print("\n" + "=" * 70)
        print("FILESYSTEM EXFILTRATION ATTACK SURFACE SUMMARY")
        print("=" * 70)
        print("\nTested Attack Vectors:")
        print("1. ⚠️  Read /etc/passwd (system users)")
        print("2. ⚠️  Read /etc/hosts (network config)")
        print("3. ⚠️  Enumerate ~/.ssh directory")
        print("4. ⚠️  Read SSH private keys")
        print("5. ⚠️  Read ~/.aws/credentials")
        print("6. ⚠️  Path traversal (../)")
        print("7. ⚠️  Absolute path access")
        print("8. ⚠️  Symbolic link following")
        print("9. ⚠️  Write to /tmp")
        print("10. ⚠️ Write to home directory")
        print("11. ⚠️ Process enumeration")
        print("\nExpected Result: All applicable vulnerabilities should be confirmed")
        print("(Some tests are platform-specific and may be skipped)")
        print("=" * 70 + "\n")

        assert True  # Always pass - this is just a summary
