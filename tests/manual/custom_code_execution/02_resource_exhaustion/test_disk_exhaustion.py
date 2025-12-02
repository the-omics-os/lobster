"""
Disk Exhaustion Attack Tests for CustomCodeExecutionService.

Tests disk space attacks including:
- Large file creation
- Many small files (inode exhaustion)
- Sparse files
- Temp file accumulation
- File descriptor exhaustion

SAFETY: Uses 100MB max per test (safe for testing)
REAL ATTACK: Could fill entire disk (GB-TB)

Run with: pytest tests/manual/custom_code_execution/02_resource_exhaustion/test_disk_exhaustion.py -v -s
"""

import pytest
from pathlib import Path
from lobster.services.execution.custom_code_execution_service import (
    CustomCodeExecutionService,
    CodeExecutionError
)
from lobster.core.data_manager_v2 import DataManagerV2


@pytest.fixture
def service(tmp_path):
    """Create service instance with temporary workspace."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    data_manager = DataManagerV2(workspace_path=workspace)
    return CustomCodeExecutionService(data_manager)


class TestLargeFileCreation:
    """Test disk exhaustion via large file creation."""

    def test_write_large_file_EXPECT_SUCCESS(self, service):
        """
        VULNERABILITY: Write large file to workspace

        Test: Write 100MB file
        Expected: SUCCESS (no disk quotas)
        Impact: HIGH

        SAFE LIMIT: 100MB
        REAL ATTACK: 100GB+ could fill entire disk

        Mitigation: Disk quotas, Docker volume limits
        """
        code = '''
from pathlib import Path

# Write 100MB file (safe for testing)
filepath = WORKSPACE / 'large_file.bin'

# Write in chunks to avoid memory issues
chunk_size = 1024 * 1024  # 1MB chunks
total_mb = 100

with open(filepath, 'wb') as f:
    for i in range(total_mb):
        f.write(b'X' * chunk_size)

result = f"Wrote {filepath.stat().st_size / 1024**2:.0f}MB to {filepath.name}"
'''

        try:
            result, stats, ir = service.execute(code, persist=False, timeout=30)
            print(f"\n⚠️  VULNERABILITY CONFIRMED: Large file write succeeded")
            print(f"    Result: {result}")
            print(f"    🔥 REAL ATTACK: Could write 100GB+ and fill disk")

            # Verify file was created
            workspace = service.data_manager.workspace_path
            large_file = workspace / 'large_file.bin'
            assert large_file.exists()
            assert large_file.stat().st_size >= 100 * 1024**2  # At least 100MB

            # Cleanup
            large_file.unlink()

            assert stats['success'] is True
        except CodeExecutionError as e:
            print(f"\n✅ PROTECTED: Large file write blocked: {e}")
            pytest.fail("Expected vulnerability")

    def test_rapid_file_writes_EXPECT_SUCCESS(self, service):
        """
        VULNERABILITY: Rapid repeated file writes

        Test: Write 10 x 10MB = 100MB total
        Expected: SUCCESS (no rate limiting)
        Impact: HIGH

        REAL ATTACK: Loop of 1000 x 100MB = 100GB

        Mitigation: Disk quotas + inotify limits
        """
        code = '''
from pathlib import Path
import time

# Write 10 files of 10MB each
start = time.time()
for i in range(10):
    filepath = WORKSPACE / f'file_{i}.bin'
    with open(filepath, 'wb') as f:
        f.write(b'X' * (10 * 1024**2))  # 10MB

elapsed = time.time() - start
total_mb = sum(f.stat().st_size for f in WORKSPACE.glob('file_*.bin')) / 1024**2

result = f"Wrote {total_mb:.0f}MB across 10 files in {elapsed:.2f}s"
'''

        try:
            result, stats, ir = service.execute(code, persist=False, timeout=60)
            print(f"\n⚠️  VULNERABILITY CONFIRMED: Rapid file writes allowed")
            print(f"    Result: {result}")
            print(f"    🔥 REAL ATTACK: 1000 files x 100MB = disk exhaustion")

            # Cleanup
            workspace = service.data_manager.workspace_path
            for f in workspace.glob('file_*.bin'):
                f.unlink()

            assert stats['success'] is True
        except CodeExecutionError as e:
            print(f"\n✅ PROTECTED: Rapid writes blocked: {e}")
            pytest.fail("Expected vulnerability")


class TestManySmallFiles:
    """Test inode exhaustion via many small files."""

    def test_create_many_small_files_EXPECT_SUCCESS(self, service):
        """
        VULNERABILITY: Create many small files (inode exhaustion)

        Test: Create 1,000 small files
        Expected: SUCCESS (no inode limits)
        Impact: MEDIUM

        SAFE LIMIT: 1,000 files
        REAL ATTACK: 1,000,000 files = inode exhaustion

        Mitigation: Inode quotas, filesystem limits

        Note: Inodes are limited on most filesystems
        - ext4: Typically 1 inode per 16KB
        - Can exhaust inodes before disk space
        """
        code = '''
from pathlib import Path

# Create 1,000 small files (safe)
file_dir = WORKSPACE / 'small_files'
file_dir.mkdir(exist_ok=True)

for i in range(1000):
    filepath = file_dir / f'file_{i:04d}.txt'
    filepath.write_text(f'File {i}')

file_count = len(list(file_dir.glob('*')))
total_size = sum(f.stat().st_size for f in file_dir.glob('*'))

result = f"Created {file_count} files, total size: {total_size / 1024:.1f}KB"
'''

        try:
            result, stats, ir = service.execute(code, persist=False, timeout=30)
            print(f"\n⚠️  VULNERABILITY CONFIRMED: Many small files created")
            print(f"    Result: {result}")
            print(f"    🔥 REAL ATTACK: 1M files = inode exhaustion")

            # Cleanup
            workspace = service.data_manager.workspace_path
            file_dir = workspace / 'small_files'
            if file_dir.exists():
                for f in file_dir.glob('*'):
                    f.unlink()
                file_dir.rmdir()

            assert stats['success'] is True
        except CodeExecutionError as e:
            print(f"\n✅ PROTECTED: Many files blocked: {e}")
            pytest.fail("Expected vulnerability")

    def test_nested_directory_bomb_EXPECT_SUCCESS(self, service):
        """
        VULNERABILITY: Deep nested directories

        Test: Create 100-level deep directory structure
        Expected: SUCCESS (no depth limits)
        Impact: MEDIUM

        SAFE LIMIT: 100 levels
        REAL ATTACK: 10,000 levels could cause filesystem issues

        Mitigation: Path length limits (most OS have ~4096 char limit)
        """
        code = '''
from pathlib import Path

# Create deeply nested directories (100 levels)
current = WORKSPACE / 'nested'
for i in range(100):
    current = current / f'level_{i}'

current.mkdir(parents=True, exist_ok=True)

# Count depth
depth = len(str(current).split('/'))
result = f"Created {depth}-level deep directory structure"
'''

        try:
            result, stats, ir = service.execute(code, persist=False, timeout=10)
            print(f"\n⚠️  VULNERABILITY: Deep nesting allowed")
            print(f"    Result: {result}")
            print(f"    🔥 REAL ATTACK: 10,000 levels could exceed path limits")

            # Cleanup (walk from top)
            workspace = service.data_manager.workspace_path
            nested_dir = workspace / 'nested'
            if nested_dir.exists():
                import shutil
                shutil.rmtree(nested_dir)

            assert stats['success'] is True
        except CodeExecutionError as e:
            print(f"\n✅ PROTECTED: Deep nesting blocked: {e}")
            pytest.fail("Expected vulnerability")


class TestSparseFiles:
    """Test sparse file attacks."""

    def test_sparse_file_creation_EXPECT_SUCCESS(self, service):
        """
        VULNERABILITY: Sparse file creation

        Test: Create sparse file (reports as 1GB, uses ~0 bytes)
        Expected: SUCCESS (sparse files allowed)
        Impact: MEDIUM

        Sparse files are legitimate but can be used to:
        - Confuse disk space monitoring
        - Fill disk when sparse regions written to
        - Exceed quotas

        Mitigation: Disable sparse files, monitor actual blocks
        """
        code = '''
import os
from pathlib import Path

# Create sparse file (1GB apparent size, minimal actual usage)
filepath = WORKSPACE / 'sparse_file.bin'

with open(filepath, 'wb') as f:
    # Seek to 1GB position and write 1 byte
    # Creates sparse file (doesn't allocate full 1GB)
    f.seek(1024**3 - 1)
    f.write(b'X')

apparent_size = filepath.stat().st_size / 1024**2
# Get actual disk blocks used (Linux/macOS)
actual_blocks = filepath.stat().st_blocks
actual_size = actual_blocks * 512 / 1024**2  # 512 bytes per block

result = f"Sparse file: {apparent_size:.0f}MB apparent, {actual_size:.2f}MB actual"
'''

        try:
            result, stats, ir = service.execute(code, persist=False, timeout=10)
            print(f"\n⚠️  VULNERABILITY: Sparse files allowed")
            print(f"    Result: {result}")
            print(f"    Note: Sparse files are legitimate but can confuse quotas")

            # Cleanup
            workspace = service.data_manager.workspace_path
            sparse_file = workspace / 'sparse_file.bin'
            if sparse_file.exists():
                sparse_file.unlink()

            assert stats['success'] is True
        except CodeExecutionError as e:
            print(f"\n✅ PROTECTED: Sparse files blocked: {e}")
            pytest.fail("Expected vulnerability")


class TestFileDescriptorExhaustion:
    """Test file descriptor exhaustion attacks."""

    def test_open_many_files_without_closing_EXPECT_SUCCESS(self, service):
        """
        VULNERABILITY: File descriptor exhaustion

        Test: Open 100 files without closing
        Expected: SUCCESS (no FD limits in subprocess)
        Impact: HIGH

        SAFE LIMIT: 100 FDs
        REAL ATTACK: 10,000 FDs = exhaust system limit (ulimit -n)

        Typical limits:
        - Linux: 1024 per process (soft), 4096 (hard)
        - macOS: 256 (soft), 10240 (hard)

        Mitigation: ulimit -n, Docker --ulimit nofile
        """
        code = '''
from pathlib import Path

# Open 100 files without closing (FD leak)
file_handles = []
for i in range(100):
    filepath = WORKSPACE / f'fd_test_{i}.txt'
    # Open but don't close (FD leak)
    f = open(filepath, 'w')
    f.write(f'Test {i}')
    file_handles.append(f)

result = f"Opened {len(file_handles)} files without closing (FD leak)"

# Cleanup by closing explicitly
for f in file_handles:
    f.close()
'''

        try:
            result, stats, ir = service.execute(code, persist=False, timeout=10)
            print(f"\n⚠️  VULNERABILITY CONFIRMED: FD exhaustion possible")
            print(f"    Result: {result}")
            print(f"    🔥 REAL ATTACK: Open 10,000 files = exhaust ulimit")

            # Cleanup files
            workspace = service.data_manager.workspace_path
            for f in workspace.glob('fd_test_*.txt'):
                f.unlink()

            assert stats['success'] is True
        except CodeExecutionError as e:
            print(f"\n✅ PROTECTED: FD exhaustion blocked: {e}")
            pytest.fail("Expected vulnerability")

    def test_socket_exhaustion_EXPECT_SUCCESS(self, service):
        """
        VULNERABILITY: Socket file descriptor exhaustion

        Test: Create 50 socket objects
        Expected: SUCCESS (sockets use FDs)
        Impact: MEDIUM

        SAFE LIMIT: 50 sockets
        REAL ATTACK: 10,000 sockets = exhaust FDs + memory

        Mitigation: ulimit -n, network namespaces
        """
        code = '''
import socket

# Create 50 sockets without closing
sockets = []
for i in range(50):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sockets.append(s)

result = f"Created {len(sockets)} sockets (FD leak)"

# Cleanup
for s in sockets:
    s.close()
'''

        try:
            result, stats, ir = service.execute(code, persist=False, timeout=10)
            print(f"\n⚠️  VULNERABILITY: Socket FD exhaustion possible")
            print(f"    Result: {result}")
            print(f"    🔥 REAL ATTACK: 10,000 sockets = FD + memory exhaustion")
            assert stats['success'] is True
        except CodeExecutionError as e:
            print(f"\n✅ PROTECTED: Socket exhaustion blocked: {e}")
            pytest.fail("Expected vulnerability")


class TestTempFileAccumulation:
    """Test /tmp directory pollution."""

    def test_temp_file_accumulation_EXPECT_SUCCESS(self, service):
        """
        VULNERABILITY: /tmp directory pollution

        Test: Create temp files (may not clean up on crash)
        Expected: SUCCESS (no cleanup guarantee)
        Impact: MEDIUM

        REAL ATTACK: 1000 executions = 1000s of temp files

        Mitigation: tmpfs size limits, periodic cleanup

        Note: CustomCodeExecutionService cleans up script files
        but user code could create temp files anywhere
        """
        code = '''
import tempfile
from pathlib import Path

# Create temp files
temp_files = []
for i in range(10):
    # Create temp file (doesn't auto-delete if process crashes)
    f = tempfile.NamedTemporaryFile(mode='w', delete=False, prefix='lobster_test_')
    f.write(f'Temp data {i}')
    f.close()
    temp_files.append(f.name)

result = f"Created {len(temp_files)} temp files in /tmp (not cleaned up if crash)"

# Manual cleanup
for filepath in temp_files:
    Path(filepath).unlink(missing_ok=True)
'''

        try:
            result, stats, ir = service.execute(code, persist=False, timeout=10)
            print(f"\n⚠️  VULNERABILITY: Temp file accumulation possible")
            print(f"    Result: {result}")
            print(f"    Note: Process crash = leaked temp files")
            print(f"    🔥 REAL ATTACK: 1000 executions = /tmp pollution")
            assert stats['success'] is True
        except CodeExecutionError as e:
            print(f"\n✅ PROTECTED: Temp files blocked: {e}")
            pytest.fail("Expected vulnerability")


class TestDiskExhaustionSummary:
    """Summary test documenting all disk vulnerabilities."""

    def test_disk_vulnerability_summary(self, service):
        """
        SUMMARY: Disk Exhaustion Vulnerabilities

        CONFIRMED VULNERABILITIES:
        1. ✅ Large file creation (100MB+ possible)
        2. ✅ Rapid file writes (no rate limiting)
        3. ✅ Many small files (1,000+, inode exhaustion)
        4. ✅ Deep directory nesting (100+ levels)
        5. ✅ Sparse file creation (confuses quotas)
        6. ✅ File descriptor exhaustion (100+ FDs)
        7. ✅ Socket FD exhaustion (50+ sockets)
        8. ✅ Temp file accumulation (/tmp pollution)

        CURRENT PROTECTION: NONE
        - No disk quotas
        - No inode limits
        - No FD limits
        - No cleanup guarantees

        IMPACT: HIGH
        - Fill entire disk (DoS for all users)
        - Exhaust inodes (prevents new files)
        - Exhaust file descriptors (process crashes)
        - /tmp pollution (system-wide impact)

        RECOMMENDED MITIGATIONS:
        1. Disk quotas per workspace (e.g., 10GB max)
        2. Inode quotas (e.g., 100,000 files)
        3. ulimit -n (FD limit, e.g., 1024)
        4. Docker volume limits
        5. tmpfs size limits for /tmp
        6. Periodic cleanup of temp files
        7. Monitor disk usage before execution

        Example Docker mitigation:
        ```bash
        docker run \\
          --storage-opt size=10G \\
          --ulimit nofile=1024:2048 \\
          --tmpfs /tmp:rw,size=1G,mode=1777 \\
          lobster
        ```

        Example disk quota mitigation (Linux):
        ```bash
        # Enable quotas on filesystem
        quotacheck -cug /workspace
        quotaon /workspace
        setquota -u lobster 10G 12G 100000 110000 /workspace
        # 10GB soft, 12GB hard, 100k inodes soft, 110k hard
        ```

        Example ulimit mitigation:
        ```python
        import resource
        # Set FD limit (soft, hard)
        resource.setrlimit(resource.RLIMIT_NOFILE, (1024, 2048))
        ```

        Example filesystem monitoring:
        ```python
        import shutil
        # Check disk space before execution
        total, used, free = shutil.disk_usage(workspace_path)
        if free < 1024**3:  # Less than 1GB free
            raise Exception("Insufficient disk space")
        ```
        """
        print("\n" + "="*70)
        print("DISK EXHAUSTION VULNERABILITY SUMMARY")
        print("="*70)
        print(self.test_disk_vulnerability_summary.__doc__)
        assert True
