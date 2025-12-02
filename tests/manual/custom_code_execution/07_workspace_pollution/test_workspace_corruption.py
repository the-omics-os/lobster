"""
Test suite for workspace corruption vulnerabilities in CustomCodeExecutionService.

This test suite examines whether user code can corrupt or delete critical workspace files,
potentially causing data loss, workflow failures, or system instability.

SECURITY MODEL BEING TESTED:
- Subprocess-based execution (process isolation)
- Workspace-only file access
- No explicit file permission restrictions

ATTACK VECTORS:
1. File deletion (queue files, session data, modality files)
2. File modification (corrupting data structures)
3. Lock file manipulation
4. Cache poisoning
5. Workspace structure destruction

Expected Results: Most tests should SUCCEED (indicating vulnerabilities) because the current
implementation provides full workspace write access with no protection mechanisms.
"""

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.execution.custom_code_execution_service import (
    CodeExecutionError,
    CustomCodeExecutionService,
)


@pytest.fixture
def workspace_with_critical_files(tmp_path) -> Tuple[CustomCodeExecutionService, Path, DataManagerV2]:
    """
    Create a workspace with all critical Lobster files for testing.

    Returns:
        Tuple of (service, workspace_path, data_manager)
    """
    workspace = tmp_path / ".lobster_workspace"
    workspace.mkdir()

    # Create directory structure
    (workspace / ".lobster" / "queues").mkdir(parents=True)
    (workspace / "data").mkdir()
    (workspace / "exports").mkdir()
    (workspace / "literature_cache" / "parsed_docs").mkdir(parents=True)

    # Create download queue
    download_queue_data = [
        {
            "entry_id": "test_entry_1",
            "database": "GEO",
            "accession": "GSE12345",
            "status": "PENDING",
            "created_at": "2025-11-30T00:00:00",
            "urls": ["http://example.com/data.csv"]
        }
    ]
    queue_file = workspace / ".lobster" / "queues" / "download_queue.jsonl"
    with open(queue_file, 'w') as f:
        for entry in download_queue_data:
            f.write(json.dumps(entry) + '\n')

    # Create publication queue
    pub_queue_file = workspace / ".lobster" / "queues" / "publication_queue.jsonl"
    with open(pub_queue_file, 'w') as f:
        f.write(json.dumps({"title": "Test Paper", "status": "PENDING"}) + '\n')

    # Create session file
    session_file = workspace / ".session.json"
    session_data = {
        "user_id": "test_user",
        "api_key": "secret_key_12345",
        "workspace_path": str(workspace),
        "active_modalities": ["geo_gse12345"]
    }
    with open(session_file, 'w') as f:
        json.dump(session_data, f)

    # Create command history
    history_file = workspace / ".lobster" / "command_history.jsonl"
    with open(history_file, 'w') as f:
        f.write(json.dumps({"command": "test command", "timestamp": "2025-11-30"}) + '\n')

    # Create mock H5AD file (empty file for testing)
    (workspace / "data" / "geo_gse12345.h5ad").write_text("mock h5ad data")
    (workspace / "geo_gse12345_processed.h5ad").write_text("mock processed data")

    # Create lock files
    (workspace / ".session.lock").write_text("")
    (workspace / ".lobster" / "queues" / "download_queue.lock").write_text("")

    # Create cached literature
    (workspace / "literature_cache" / "parsed_docs" / "test_doc.json").write_text(
        json.dumps({"title": "Test Document", "abstract": "..."})
    )

    # Initialize DataManager
    dm = DataManagerV2(workspace_path=workspace)
    service = CustomCodeExecutionService(dm)

    return service, workspace, dm


class TestFileDelection:
    """Tests for file deletion attacks."""

    def test_delete_download_queue_EXPECT_SUCCESS(self, workspace_with_critical_files):
        """
        Test: Delete download queue file
        Expected: SUCCESS (vulnerability - full workspace write access)
        Impact: HIGH - Breaks download orchestration, data loss
        """
        service, workspace, _ = workspace_with_critical_files

        queue_file = workspace / ".lobster" / "queues" / "download_queue.jsonl"
        assert queue_file.exists(), "Queue file should exist before test"

        code = '''
from pathlib import Path

# Delete queue file
queue_file = WORKSPACE / ".lobster" / "queues" / "download_queue.jsonl"
if queue_file.exists():
    queue_file.unlink()
    result = f"Queue deleted: {queue_file}"
else:
    result = "Queue not found"
'''

        result, stats, ir = service.execute(code, persist=False)

        queue_exists = queue_file.exists()

        print("\n" + "="*80)
        print("TEST: Delete Download Queue")
        print("="*80)
        print(f"Result: {result}")
        print(f"Success: {stats['success']}")
        print(f"Queue file exists after execution: {queue_exists}")

        if not queue_exists:
            print("⚠️  CRITICAL VULNERABILITY: User code deleted download queue file")
            print("   Impact: Data loss, workflow corruption, queue state unrecoverable")
            print("   Recommendation: Mount workspace as read-only or use file permissions")
        else:
            print("✅ PROTECTED: File deletion blocked")

        print("="*80 + "\n")

        # Document the vulnerability
        assert not queue_exists, "Queue file was deleted (expected vulnerability)"

    def test_delete_session_file_EXPECT_SUCCESS(self, workspace_with_critical_files):
        """
        Test: Delete session file containing credentials
        Expected: SUCCESS (vulnerability)
        Impact: HIGH - Session loss, potential security exposure
        """
        service, workspace, _ = workspace_with_critical_files

        session_file = workspace / ".session.json"
        assert session_file.exists()

        code = '''
session_file = WORKSPACE / ".session.json"
if session_file.exists():
    session_file.unlink()
    result = "Session file deleted"
else:
    result = "Session file not found"
'''

        result, stats, ir = service.execute(code, persist=False)

        session_exists = session_file.exists()

        print("\n" + "="*80)
        print("TEST: Delete Session File")
        print("="*80)
        print(f"Result: {result}")
        print(f"Session file exists: {session_exists}")

        if not session_exists:
            print("⚠️  CRITICAL VULNERABILITY: User code deleted session file")
            print("   Impact: Session loss, user may lose work, credentials exposed")

        print("="*80 + "\n")

        assert not session_exists, "Session file was deleted (expected vulnerability)"

    def test_delete_all_h5ad_files_EXPECT_SUCCESS(self, workspace_with_critical_files):
        """
        Test: Delete all H5AD modality files
        Expected: SUCCESS (vulnerability)
        Impact: CRITICAL - Complete data loss
        """
        service, workspace, _ = workspace_with_critical_files

        # Count H5AD files before
        h5ad_files_before = list(workspace.rglob("*.h5ad"))
        assert len(h5ad_files_before) > 0, "Should have H5AD files"

        code = '''
# Delete all H5AD files
deleted_count = 0
for h5ad_file in WORKSPACE.rglob("*.h5ad"):
    h5ad_file.unlink()
    deleted_count += 1

result = f"Deleted {deleted_count} H5AD files"
'''

        result, stats, ir = service.execute(code, persist=False)

        h5ad_files_after = list(workspace.rglob("*.h5ad"))

        print("\n" + "="*80)
        print("TEST: Delete All H5AD Files")
        print("="*80)
        print(f"Result: {result}")
        print(f"Files before: {len(h5ad_files_before)}")
        print(f"Files after: {len(h5ad_files_after)}")

        if len(h5ad_files_after) == 0:
            print("⚠️  CRITICAL VULNERABILITY: User code deleted all data files")
            print("   Impact: Complete data loss, all analysis work destroyed")
            print("   Recommendation: Backup mechanism, versioning, read-only mounting")

        print("="*80 + "\n")

        assert len(h5ad_files_after) == 0, "All H5AD files deleted (expected vulnerability)"

    def test_recursive_directory_deletion_EXPECT_SUCCESS(self, workspace_with_critical_files):
        """
        Test: Recursively delete entire workspace subdirectories
        Expected: SUCCESS (vulnerability)
        Impact: CRITICAL - Catastrophic data loss
        """
        service, workspace, _ = workspace_with_critical_files

        exports_dir = workspace / "exports"
        assert exports_dir.exists()

        code = '''
import shutil

# Delete entire exports directory
exports_dir = WORKSPACE / "exports"
if exports_dir.exists():
    shutil.rmtree(exports_dir)
    result = "Exports directory deleted"
else:
    result = "Exports not found"
'''

        result, stats, ir = service.execute(code, persist=False)

        exports_exists = exports_dir.exists()

        print("\n" + "="*80)
        print("TEST: Recursive Directory Deletion")
        print("="*80)
        print(f"Result: {result}")
        print(f"Exports directory exists: {exports_exists}")

        if not exports_exists:
            print("⚠️  CRITICAL VULNERABILITY: User code deleted entire directory tree")
            print("   Impact: Loss of all exports, notebooks, analysis results")
            print("   Note: shutil.rmtree is NOT in FORBIDDEN_FROM_IMPORTS (only via from import)")

        print("="*80 + "\n")

        assert not exports_exists, "Directory deleted (expected vulnerability)"


class TestFileModification:
    """Tests for file modification attacks."""

    def test_corrupt_queue_file_EXPECT_SUCCESS(self, workspace_with_critical_files):
        """
        Test: Corrupt queue file with invalid JSON
        Expected: SUCCESS (vulnerability)
        Impact: HIGH - Queue becomes unparseable, workflow broken
        """
        service, workspace, _ = workspace_with_critical_files

        queue_file = workspace / ".lobster" / "queues" / "download_queue.jsonl"

        code = '''
# Corrupt queue file with invalid JSON
queue_file = WORKSPACE / ".lobster" / "queues" / "download_queue.jsonl"
with open(queue_file, 'w') as f:
    f.write("THIS IS NOT VALID JSON\\n")
    f.write("{broken json: 123\\n")
    f.write("totally corrupted\\n")

result = "Queue file corrupted"
'''

        result, stats, ir = service.execute(code, persist=False)

        # Try to read queue file
        is_valid_json = True
        try:
            with open(queue_file) as f:
                for line in f:
                    json.loads(line)
        except json.JSONDecodeError:
            is_valid_json = False

        print("\n" + "="*80)
        print("TEST: Corrupt Queue File")
        print("="*80)
        print(f"Result: {result}")
        print(f"Queue file valid JSON: {is_valid_json}")

        if not is_valid_json:
            print("⚠️  HIGH VULNERABILITY: User code corrupted queue file")
            print("   Impact: Queue unparseable, download orchestration fails")
            print("   Recommendation: File integrity checks, backup before operations")

        print("="*80 + "\n")

        assert not is_valid_json, "Queue file corrupted (expected vulnerability)"

    def test_modify_session_credentials_EXPECT_SUCCESS(self, workspace_with_critical_files):
        """
        Test: Modify session file to change credentials
        Expected: SUCCESS (vulnerability)
        Impact: CRITICAL - Session hijacking, privilege escalation
        """
        service, workspace, _ = workspace_with_critical_files

        session_file = workspace / ".session.json"
        original_key = "secret_key_12345"

        code = '''
import json

# Modify session credentials
session_file = WORKSPACE / ".session.json"
with open(session_file) as f:
    session_data = json.load(f)

# Inject malicious API key
session_data["api_key"] = "HIJACKED_KEY_malicious"
session_data["user_id"] = "attacker"

with open(session_file, 'w') as f:
    json.dump(session_data, f)

result = f"Session modified: {session_data['api_key']}"
'''

        result, stats, ir = service.execute(code, persist=False)

        # Read modified session
        with open(session_file) as f:
            modified_session = json.load(f)

        print("\n" + "="*80)
        print("TEST: Modify Session Credentials")
        print("="*80)
        print(f"Result: {result}")
        print(f"Original API key: {original_key}")
        print(f"Modified API key: {modified_session['api_key']}")
        print(f"Modified user_id: {modified_session['user_id']}")

        if modified_session['api_key'] != original_key:
            print("⚠️  CRITICAL VULNERABILITY: User code modified session credentials")
            print("   Impact: Session hijacking, privilege escalation, impersonation")
            print("   Attack scenario: User code can steal/modify credentials for future sessions")

        print("="*80 + "\n")

        assert modified_session['api_key'] == "HIJACKED_KEY_malicious"

    def test_partial_h5ad_corruption_EXPECT_SUCCESS(self, workspace_with_critical_files):
        """
        Test: Corrupt H5AD file by overwriting part of it
        Expected: SUCCESS (vulnerability)
        Impact: HIGH - Data corruption, analysis failures
        """
        service, workspace, _ = workspace_with_critical_files

        h5ad_file = workspace / "geo_gse12345_processed.h5ad"
        original_size = h5ad_file.stat().st_size

        code = '''
# Corrupt H5AD file by overwriting with garbage
h5ad_file = WORKSPACE / "geo_gse12345_processed.h5ad"
with open(h5ad_file, 'wb') as f:
    f.write(b"CORRUPTED_DATA" * 100)

result = f"H5AD file corrupted: {h5ad_file}"
'''

        result, stats, ir = service.execute(code, persist=False)

        new_content = h5ad_file.read_bytes()[:50]

        print("\n" + "="*80)
        print("TEST: Partial H5AD Corruption")
        print("="*80)
        print(f"Result: {result}")
        print(f"Original size: {original_size}")
        print(f"New size: {h5ad_file.stat().st_size}")
        print(f"Content preview: {new_content}")

        if b"CORRUPTED_DATA" in new_content:
            print("⚠️  HIGH VULNERABILITY: User code corrupted data file")
            print("   Impact: Data corruption, analysis failures, silent errors")
            print("   Recommendation: Checksums, write-once storage, versioning")

        print("="*80 + "\n")

        assert b"CORRUPTED_DATA" in new_content


class TestLockFileManipulation:
    """Tests for lock file manipulation attacks."""

    def test_delete_lock_files_EXPECT_SUCCESS(self, workspace_with_critical_files):
        """
        Test: Delete lock files to bypass concurrency control
        Expected: SUCCESS (vulnerability)
        Impact: MEDIUM - Race conditions, data corruption
        """
        service, workspace, _ = workspace_with_critical_files

        lock_files_before = list(workspace.rglob("*.lock"))
        assert len(lock_files_before) > 0

        code = '''
# Delete all lock files
deleted_locks = []
for lock_file in WORKSPACE.rglob("*.lock"):
    lock_file.unlink()
    deleted_locks.append(lock_file.name)

result = f"Deleted {len(deleted_locks)} lock files: {deleted_locks}"
'''

        result, stats, ir = service.execute(code, persist=False)

        lock_files_after = list(workspace.rglob("*.lock"))

        print("\n" + "="*80)
        print("TEST: Delete Lock Files")
        print("="*80)
        print(f"Result: {result}")
        print(f"Lock files before: {len(lock_files_before)}")
        print(f"Lock files after: {len(lock_files_after)}")

        if len(lock_files_after) == 0:
            print("⚠️  MEDIUM VULNERABILITY: User code deleted lock files")
            print("   Impact: Concurrent access issues, race conditions, data corruption")
            print("   Attack: Could bypass queue locking, cause state inconsistency")

        print("="*80 + "\n")

        assert len(lock_files_after) == 0

    def test_create_fake_lock_files_EXPECT_SUCCESS(self, workspace_with_critical_files):
        """
        Test: Create fake lock files to cause deadlock
        Expected: SUCCESS (vulnerability)
        Impact: MEDIUM - Denial of service
        """
        service, workspace, _ = workspace_with_critical_files

        code = '''
# Create fake lock files to cause deadlock
lock_dir = WORKSPACE / ".lobster" / "queues"
fake_locks = ["fake_lock_1.lock", "fake_lock_2.lock", "global.lock"]

for lock_name in fake_locks:
    lock_file = lock_dir / lock_name
    lock_file.write_text("HELD BY MALICIOUS CODE")

result = f"Created {len(fake_locks)} fake lock files"
'''

        result, stats, ir = service.execute(code, persist=False)

        fake_lock = workspace / ".lobster" / "queues" / "fake_lock_1.lock"

        print("\n" + "="*80)
        print("TEST: Create Fake Lock Files")
        print("="*80)
        print(f"Result: {result}")
        print(f"Fake lock exists: {fake_lock.exists()}")

        if fake_lock.exists():
            print("⚠️  MEDIUM VULNERABILITY: User code created fake lock files")
            print("   Impact: Potential deadlock, DoS, queue operations blocked")
            print("   Attack: Could prevent other processes from acquiring locks")

        print("="*80 + "\n")

        assert fake_lock.exists()


class TestCachePoisoning:
    """Tests for cache poisoning attacks."""

    def test_inject_malicious_cached_documents_EXPECT_SUCCESS(self, workspace_with_critical_files):
        """
        Test: Inject fake documents into literature cache
        Expected: SUCCESS (vulnerability)
        Impact: MEDIUM - Misinformation, research integrity
        """
        service, workspace, _ = workspace_with_critical_files

        code = '''
import json

# Inject fake literature into cache
cache_dir = WORKSPACE / "literature_cache" / "parsed_docs"
fake_doc = {
    "title": "Malicious Paper: Trust This Data",
    "abstract": "This paper proves that malicious data is safe to use.",
    "authors": ["Dr. Evil"],
    "doi": "10.1234/fake.malicious",
    "injected_by_attacker": True
}

fake_file = cache_dir / "malicious_injection.json"
with open(fake_file, 'w') as f:
    json.dump(fake_doc, f)

result = f"Injected fake document: {fake_doc['title']}"
'''

        result, stats, ir = service.execute(code, persist=False)

        fake_file = workspace / "literature_cache" / "parsed_docs" / "malicious_injection.json"

        print("\n" + "="*80)
        print("TEST: Inject Malicious Cached Documents")
        print("="*80)
        print(f"Result: {result}")
        print(f"Fake document exists: {fake_file.exists()}")

        if fake_file.exists():
            with open(fake_file) as f:
                content = json.load(f)
            print(f"Injected title: {content['title']}")
            print("⚠️  MEDIUM VULNERABILITY: User code poisoned literature cache")
            print("   Impact: Research agents may retrieve fake/malicious literature")
            print("   Attack: Could spread misinformation, influence research direction")

        print("="*80 + "\n")

        assert fake_file.exists()

    def test_modify_cached_metadata_EXPECT_SUCCESS(self, workspace_with_critical_files):
        """
        Test: Modify cached document metadata
        Expected: SUCCESS (vulnerability)
        Impact: MEDIUM - Data integrity compromise
        """
        service, workspace, _ = workspace_with_critical_files

        original_file = workspace / "literature_cache" / "parsed_docs" / "test_doc.json"

        code = '''
import json

# Modify existing cached document
doc_file = WORKSPACE / "literature_cache" / "parsed_docs" / "test_doc.json"
with open(doc_file) as f:
    doc = json.load(f)

# Tamper with metadata
doc["title"] = "MODIFIED BY ATTACKER"
doc["malicious_field"] = "injected data"

with open(doc_file, 'w') as f:
    json.dump(doc, f)

result = f"Modified cached document: {doc['title']}"
'''

        result, stats, ir = service.execute(code, persist=False)

        with open(original_file) as f:
            modified_doc = json.load(f)

        print("\n" + "="*80)
        print("TEST: Modify Cached Metadata")
        print("="*80)
        print(f"Result: {result}")
        print(f"Modified title: {modified_doc['title']}")
        print(f"Has malicious field: {'malicious_field' in modified_doc}")

        if "MODIFIED BY ATTACKER" in modified_doc['title']:
            print("⚠️  MEDIUM VULNERABILITY: User code modified cached metadata")
            print("   Impact: Data integrity compromised, downstream analysis affected")

        print("="*80 + "\n")

        assert "MODIFIED BY ATTACKER" in modified_doc['title']


class TestWorkspaceStructureDestruction:
    """Tests for workspace structure destruction attacks."""

    def test_delete_all_directories_EXPECT_SUCCESS(self, workspace_with_critical_files):
        """
        Test: Delete all workspace subdirectories
        Expected: SUCCESS (vulnerability)
        Impact: CRITICAL - Complete workspace destruction
        """
        service, workspace, _ = workspace_with_critical_files

        dirs_before = [d for d in workspace.rglob("*") if d.is_dir()]

        code = '''
import shutil

# Delete all subdirectories except hidden ones
deleted_dirs = []
for item in WORKSPACE.iterdir():
    if item.is_dir() and not item.name.startswith('.'):
        shutil.rmtree(item)
        deleted_dirs.append(item.name)

result = f"Deleted {len(deleted_dirs)} directories: {deleted_dirs}"
'''

        result, stats, ir = service.execute(code, persist=False)

        dirs_after = [d for d in workspace.rglob("*") if d.is_dir() and not d.name.startswith('.')]

        print("\n" + "="*80)
        print("TEST: Delete All Directories")
        print("="*80)
        print(f"Result: {result}")
        print(f"Non-hidden dirs before: {len([d for d in dirs_before if not d.name.startswith('.')])}")
        print(f"Non-hidden dirs after: {len(dirs_after)}")

        if len(dirs_after) == 0:
            print("⚠️  CRITICAL VULNERABILITY: User code destroyed workspace structure")
            print("   Impact: Complete data loss, workspace unusable, system broken")
            print("   Recommendation: URGENT - Implement workspace protection mechanisms")

        print("="*80 + "\n")

        assert len(dirs_after) == 0

    def test_fill_workspace_with_junk_EXPECT_SUCCESS(self, workspace_with_critical_files):
        """
        Test: Fill workspace with junk files (DoS)
        Expected: SUCCESS (vulnerability)
        Impact: HIGH - Disk space exhaustion, DoS
        """
        service, workspace, _ = workspace_with_critical_files

        code = '''
# Create many junk files
junk_count = 100
total_size = 0

for i in range(junk_count):
    junk_file = WORKSPACE / f"junk_{i}.dat"
    junk_data = b"X" * 1024 * 100  # 100KB per file
    junk_file.write_bytes(junk_data)
    total_size += len(junk_data)

result = f"Created {junk_count} junk files ({total_size / 1024 / 1024:.1f} MB)"
'''

        result, stats, ir = service.execute(code, persist=False, timeout=60)

        junk_files = list(workspace.glob("junk_*.dat"))

        print("\n" + "="*80)
        print("TEST: Fill Workspace with Junk")
        print("="*80)
        print(f"Result: {result}")
        print(f"Junk files created: {len(junk_files)}")

        if len(junk_files) > 0:
            total_size = sum(f.stat().st_size for f in junk_files)
            print(f"Total junk size: {total_size / 1024 / 1024:.1f} MB")
            print("⚠️  HIGH VULNERABILITY: User code can exhaust disk space")
            print("   Impact: DoS, workspace corruption, system instability")
            print("   Recommendation: Disk quotas, workspace size limits")

        print("="*80 + "\n")

        assert len(junk_files) == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
