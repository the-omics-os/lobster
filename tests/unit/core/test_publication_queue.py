"""
Unit tests for publication queue infrastructure.

Tests cover:
- Pydantic schema validation (valid/invalid entries)
- CRUD operations (add, get, update, list, remove, clear)
- Persistence (save/load from JSON Lines)
- Status filtering (list by PENDING/COMPLETED/FAILED)
- Error handling (entry not found, invalid status, file corruption)
- Backup functionality (verify backup files created)
- Thread safety (concurrent operations)
- Statistics and reporting
- RIS file parsing (single entry, batch, error handling)
- Schema type inference (microbiome, single_cell, proteomics)
- Identifier extraction (PMID, DOI, PMC)
"""

import json
import threading
import time
from datetime import datetime
from pathlib import Path

import pytest

from lobster.core.publication_queue import (
    EntryNotFoundError,
    PublicationQueue,
    PublicationQueueError,
)
from lobster.core.ris_parser import RISParseError, RISParser
from lobster.core.schemas.publication_queue import (
    ExtractionLevel,
    PublicationQueueEntry,
    PublicationStatus,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_queue_file(tmp_path):
    """Create temporary queue file path."""
    return tmp_path / "test_pub_queue.jsonl"


@pytest.fixture
def publication_queue(temp_queue_file):
    """Create PublicationQueue instance."""
    return PublicationQueue(temp_queue_file)


@pytest.fixture
def sample_entry():
    """Create sample publication queue entry."""
    return PublicationQueueEntry(
        entry_id="pub_queue_test001",
        pmid="35042229",
        doi="10.1038/s41586-022-04426-0",
        pmc_id="PMC8891176",
        title="Single-cell RNA sequencing reveals novel cell types in human brain",
        authors=["Smith J", "Jones A", "Williams B"],
        year=2022,
        journal="Nature",
        priority=5,
        status=PublicationStatus.PENDING,
        extraction_level=ExtractionLevel.METHODS,
        schema_type="single_cell",
    )


@pytest.fixture
def temp_ris_file(tmp_path):
    """Create temporary RIS file path."""
    return tmp_path / "test_publications.ris"


@pytest.fixture
def sample_ris_content():
    """Create sample RIS file content."""
    return """TY  - JOUR
TI  - Single-cell RNA sequencing reveals novel cell types in human brain
AU  - Smith, John
AU  - Jones, Alice
AU  - Williams, Bob
PY  - 2022
JO  - Nature
AB  - This study uses single-cell RNA sequencing to identify novel cell types.
DO  - 10.1038/s41586-022-04426-0
PMID- 35042229
PMC - PMC8891176
KW  - single-cell
KW  - RNA-seq
KW  - brain
ER  -

TY  - JOUR
TI  - Microbiome analysis of gut bacteria in health and disease
AU  - Brown, Charlie
PY  - 2023
JO  - Cell
AB  - Comprehensive 16S rRNA sequencing reveals microbiome changes.
DO  - 10.1016/j.cell.2023.01.001
PMID- 36789012
KW  - microbiome
KW  - 16S rRNA
KW  - gut bacteria
ER  -
"""


# =============================================================================
# Schema Validation Tests
# =============================================================================


class TestSchemaValidation:
    """Test Pydantic schema validation."""

    def test_publication_queue_entry_valid(self):
        """Test valid PublicationQueueEntry creation."""
        entry = PublicationQueueEntry(
            entry_id="pub_queue_001",
            pmid="12345678",
            title="Test publication",
            priority=5,
        )

        assert entry.entry_id == "pub_queue_001"
        assert entry.pmid == "12345678"
        assert entry.title == "Test publication"
        assert entry.priority == 5
        assert entry.status == PublicationStatus.PENDING
        assert isinstance(entry.created_at, datetime)
        assert isinstance(entry.updated_at, datetime)

    def test_publication_queue_entry_empty_entry_id(self):
        """Test PublicationQueueEntry with empty entry_id."""
        with pytest.raises(ValueError, match="cannot be empty"):
            PublicationQueueEntry(
                entry_id="",
                pmid="12345678",
            )

    def test_publication_queue_entry_priority_bounds(self):
        """Test PublicationQueueEntry priority validation."""
        # Valid priority
        entry = PublicationQueueEntry(
            entry_id="test_001",
            priority=1,
        )
        assert entry.priority == 1

        entry = PublicationQueueEntry(
            entry_id="test_002",
            priority=10,
        )
        assert entry.priority == 10

        # Invalid priority (too low)
        with pytest.raises(ValueError):
            PublicationQueueEntry(
                entry_id="test_003",
                priority=0,
            )

        # Invalid priority (too high)
        with pytest.raises(ValueError):
            PublicationQueueEntry(
                entry_id="test_004",
                priority=11,
            )

    def test_extraction_level_enum(self):
        """Test ExtractionLevel enum values."""
        assert ExtractionLevel.ABSTRACT == "abstract"
        assert ExtractionLevel.METHODS == "methods"
        assert ExtractionLevel.FULL_TEXT == "full_text"
        assert ExtractionLevel.IDENTIFIERS == "identifiers"

    def test_publication_status_enum(self):
        """Test PublicationStatus enum values."""
        assert PublicationStatus.PENDING == "pending"
        assert PublicationStatus.EXTRACTING == "extracting"
        assert PublicationStatus.METADATA_EXTRACTED == "metadata_extracted"
        assert PublicationStatus.COMPLETED == "completed"
        assert PublicationStatus.FAILED == "failed"

    def test_publication_queue_entry_update_status(self, sample_entry):
        """Test update_status method."""
        original_updated_at = sample_entry.updated_at

        # Wait a bit to ensure timestamp changes
        time.sleep(0.01)

        # Update status
        sample_entry.update_status(
            status=PublicationStatus.COMPLETED,
            cached_content_path="/workspace/literature/pub_001.json",
            processed_by="research_agent",
        )

        assert sample_entry.status == PublicationStatus.COMPLETED
        assert sample_entry.cached_content_path == "/workspace/literature/pub_001.json"
        assert sample_entry.processed_by == "research_agent"
        assert sample_entry.updated_at > original_updated_at

    def test_publication_queue_entry_update_with_error(self, sample_entry):
        """Test update_status with error logging."""
        sample_entry.update_status(
            status=PublicationStatus.FAILED,
            error="Test error message",
        )

        assert sample_entry.status == PublicationStatus.FAILED
        assert len(sample_entry.error_log) == 1
        assert "Test error message" in sample_entry.error_log[0]

    def test_publication_queue_entry_serialization(self, sample_entry):
        """Test to_dict and from_dict methods."""
        # Serialize
        data = sample_entry.to_dict()
        assert isinstance(data, dict)
        assert data["entry_id"] == sample_entry.entry_id
        assert data["pmid"] == sample_entry.pmid
        assert data["title"] == sample_entry.title

        # Deserialize
        restored_entry = PublicationQueueEntry.from_dict(data)
        assert restored_entry.entry_id == sample_entry.entry_id
        assert restored_entry.pmid == sample_entry.pmid
        assert restored_entry.title == sample_entry.title
        assert restored_entry.priority == sample_entry.priority


# =============================================================================
# CRUD Operations Tests
# =============================================================================


class TestCRUDOperations:
    """Test create, read, update, delete operations."""

    def test_add_entry_success(self, publication_queue, sample_entry):
        """Test successfully adding entry to queue."""
        entry_id = publication_queue.add_entry(sample_entry)

        assert entry_id == sample_entry.entry_id
        assert publication_queue.queue_file.exists()

        # Verify entry was written
        retrieved = publication_queue.get_entry(entry_id)
        assert retrieved.entry_id == sample_entry.entry_id
        assert retrieved.pmid == sample_entry.pmid

    def test_add_entry_duplicate(self, publication_queue, sample_entry):
        """Test adding duplicate entry raises error."""
        publication_queue.add_entry(sample_entry)

        with pytest.raises(PublicationQueueError, match="already exists"):
            publication_queue.add_entry(sample_entry)

    def test_get_entry_success(self, publication_queue, sample_entry):
        """Test retrieving entry by ID."""
        publication_queue.add_entry(sample_entry)

        retrieved = publication_queue.get_entry(sample_entry.entry_id)
        assert retrieved.entry_id == sample_entry.entry_id
        assert retrieved.pmid == sample_entry.pmid
        assert retrieved.title == sample_entry.title

    def test_get_entry_not_found(self, publication_queue):
        """Test retrieving non-existent entry raises error."""
        with pytest.raises(EntryNotFoundError, match="not found"):
            publication_queue.get_entry("non_existent_id")

    def test_update_status_success(self, publication_queue, sample_entry):
        """Test updating entry status."""
        publication_queue.add_entry(sample_entry)

        updated = publication_queue.update_status(
            entry_id=sample_entry.entry_id,
            status=PublicationStatus.EXTRACTING,
            processed_by="research_agent",
        )

        assert updated.status == PublicationStatus.EXTRACTING
        assert updated.processed_by == "research_agent"

        # Verify persistence
        retrieved = publication_queue.get_entry(sample_entry.entry_id)
        assert retrieved.status == PublicationStatus.EXTRACTING

    def test_update_status_not_found(self, publication_queue):
        """Test updating non-existent entry raises error."""
        with pytest.raises(EntryNotFoundError, match="not found"):
            publication_queue.update_status(
                entry_id="non_existent_id",
                status=PublicationStatus.COMPLETED,
            )

    def test_update_status_with_error(self, publication_queue, sample_entry):
        """Test updating status with error message."""
        publication_queue.add_entry(sample_entry)

        updated = publication_queue.update_status(
            entry_id=sample_entry.entry_id,
            status=PublicationStatus.FAILED,
            error="Extraction failed: PMC access denied",
        )

        assert updated.status == PublicationStatus.FAILED
        assert len(updated.error_log) == 1
        assert "PMC access denied" in updated.error_log[0]

    def test_update_with_extracted_identifiers(self, publication_queue, sample_entry):
        """Test updating entry with extracted dataset identifiers."""
        publication_queue.add_entry(sample_entry)

        extracted_ids = {
            "geo": ["GSE180759", "GSE180760"],
            "sra": ["SRP12345"],
            "bioproject": ["PRJNA12345"],
        }

        updated = publication_queue.update_status(
            entry_id=sample_entry.entry_id,
            status=PublicationStatus.METADATA_EXTRACTED,
            extracted_identifiers=extracted_ids,
            processed_by="research_agent",
        )

        assert updated.status == PublicationStatus.METADATA_EXTRACTED
        assert updated.extracted_identifiers == extracted_ids
        assert "GSE180759" in updated.extracted_identifiers["geo"]

    def test_list_entries_all(self, publication_queue):
        """Test listing all entries."""
        # Add multiple entries
        for i in range(3):
            entry = PublicationQueueEntry(
                entry_id=f"pub_queue_{i}",
                pmid=f"1234567{i}",
                title=f"Test publication {i}",
                priority=i + 1,
            )
            publication_queue.add_entry(entry)

        entries = publication_queue.list_entries()
        assert len(entries) == 3
        assert all(isinstance(e, PublicationQueueEntry) for e in entries)

    def test_list_entries_by_status(self, publication_queue):
        """Test listing entries filtered by status."""
        # Add entries with different statuses
        entry1 = PublicationQueueEntry(
            entry_id="entry_1",
            pmid="11111111",
            status=PublicationStatus.PENDING,
        )
        entry2 = PublicationQueueEntry(
            entry_id="entry_2",
            pmid="22222222",
            status=PublicationStatus.COMPLETED,
        )
        entry3 = PublicationQueueEntry(
            entry_id="entry_3",
            pmid="33333333",
            status=PublicationStatus.PENDING,
        )

        publication_queue.add_entry(entry1)
        publication_queue.add_entry(entry2)
        publication_queue.add_entry(entry3)

        # List pending only
        pending = publication_queue.list_entries(status=PublicationStatus.PENDING)
        assert len(pending) == 2
        assert all(e.status == PublicationStatus.PENDING for e in pending)

        # List completed only
        completed = publication_queue.list_entries(status=PublicationStatus.COMPLETED)
        assert len(completed) == 1
        assert completed[0].status == PublicationStatus.COMPLETED

    def test_remove_entry_success(self, publication_queue, sample_entry):
        """Test removing entry from queue."""
        publication_queue.add_entry(sample_entry)

        # Verify entry exists
        assert publication_queue.get_entry(sample_entry.entry_id)

        # Remove entry
        publication_queue.remove_entry(sample_entry.entry_id)

        # Verify entry is gone
        with pytest.raises(EntryNotFoundError):
            publication_queue.get_entry(sample_entry.entry_id)

    def test_remove_entry_not_found(self, publication_queue):
        """Test removing non-existent entry raises error."""
        with pytest.raises(EntryNotFoundError, match="not found"):
            publication_queue.remove_entry("non_existent_id")

    def test_clear_queue_success(self, publication_queue):
        """Test clearing all entries from queue."""
        # Add multiple entries
        for i in range(5):
            entry = PublicationQueueEntry(
                entry_id=f"entry_{i}",
                pmid=f"1234567{i}",
            )
            publication_queue.add_entry(entry)

        # Verify entries exist
        assert len(publication_queue.list_entries()) == 5

        # Clear queue
        cleared_count = publication_queue.clear_queue()
        assert cleared_count == 5

        # Verify queue is empty
        assert len(publication_queue.list_entries()) == 0

    def test_clear_queue_empty(self, publication_queue):
        """Test clearing empty queue."""
        cleared_count = publication_queue.clear_queue()
        assert cleared_count == 0


# =============================================================================
# Persistence Tests
# =============================================================================


class TestPersistence:
    """Test persistence and JSON Lines format."""

    def test_persistence_across_instances(self, temp_queue_file, sample_entry):
        """Test that entries persist across PublicationQueue instances."""
        # Create first instance and add entry
        queue1 = PublicationQueue(temp_queue_file)
        queue1.add_entry(sample_entry)

        # Create second instance and verify entry exists
        queue2 = PublicationQueue(temp_queue_file)
        retrieved = queue2.get_entry(sample_entry.entry_id)

        assert retrieved.entry_id == sample_entry.entry_id
        assert retrieved.pmid == sample_entry.pmid

    def test_json_lines_format(self, publication_queue, sample_entry):
        """Test that queue file uses JSON Lines format."""
        publication_queue.add_entry(sample_entry)

        # Read raw file content
        with open(publication_queue.queue_file, "r") as f:
            lines = f.readlines()

        assert len(lines) == 1
        assert lines[0].strip()  # Not empty

        # Verify it's valid JSON
        data = json.loads(lines[0])
        assert data["entry_id"] == sample_entry.entry_id

    def test_multiple_entries_json_lines(self, publication_queue):
        """Test multiple entries in JSON Lines format."""
        # Add multiple entries
        for i in range(3):
            entry = PublicationQueueEntry(
                entry_id=f"entry_{i}",
                pmid=f"1234567{i}",
            )
            publication_queue.add_entry(entry)

        # Read raw file
        with open(publication_queue.queue_file, "r") as f:
            lines = f.readlines()

        assert len(lines) == 3

        # Verify each line is valid JSON
        for line in lines:
            data = json.loads(line.strip())
            assert "entry_id" in data
            assert "pmid" in data

    def test_corrupted_line_handling(self, publication_queue, temp_queue_file):
        """Test handling of corrupted JSON lines."""
        # Write valid entry
        entry = PublicationQueueEntry(
            entry_id="valid_entry",
            pmid="12345678",
        )
        publication_queue.add_entry(entry)

        # Manually add corrupted line
        with open(temp_queue_file, "a") as f:
            f.write("this is not valid json\n")

        # Add another valid entry
        entry2 = PublicationQueueEntry(
            entry_id="valid_entry_2",
            pmid="87654321",
        )
        publication_queue.add_entry(entry2)

        # Should load valid entries, skip corrupted
        entries = publication_queue.list_entries()
        assert len(entries) == 2
        assert entries[0].entry_id == "valid_entry"
        assert entries[1].entry_id == "valid_entry_2"


# =============================================================================
# Backup Tests
# =============================================================================


class TestBackup:
    """Test backup functionality."""

    def test_backup_created_on_modification(self, publication_queue, sample_entry):
        """Test that backup is created before modification."""
        # Add initial entry
        publication_queue.add_entry(sample_entry)

        # Modify queue (should trigger backup)
        publication_queue.update_status(
            entry_id=sample_entry.entry_id,
            status=PublicationStatus.COMPLETED,
        )

        # Check backup directory
        backup_files = list(publication_queue.backup_dir.glob("publication_queue_backup_*.jsonl"))
        assert len(backup_files) >= 1

    def test_backup_preserves_content(self, publication_queue, sample_entry):
        """Test that backup contains correct content."""
        publication_queue.add_entry(sample_entry)

        # Trigger backup by updating
        publication_queue.update_status(
            entry_id=sample_entry.entry_id,
            status=PublicationStatus.EXTRACTING,
        )

        # Get most recent backup
        backup_files = sorted(publication_queue.backup_dir.glob("publication_queue_backup_*.jsonl"))
        assert len(backup_files) > 0

        latest_backup = backup_files[-1]

        # Verify backup content
        with open(latest_backup, "r") as f:
            lines = f.readlines()

        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["entry_id"] == sample_entry.entry_id


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Test thread-safe operations."""

    def test_concurrent_add_operations(self, publication_queue):
        """Test concurrent add operations are thread-safe."""
        num_threads = 10
        threads = []
        errors = []

        def add_entry(thread_id):
            try:
                entry = PublicationQueueEntry(
                    entry_id=f"thread_{thread_id}_entry",
                    pmid=f"1234567{thread_id}",
                    title=f"Thread {thread_id} publication",
                )
                publication_queue.add_entry(entry)
            except Exception as e:
                errors.append(e)

        # Create and start threads
        for i in range(num_threads):
            thread = threading.Thread(target=add_entry, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Verify no errors
        assert len(errors) == 0

        # Verify all entries added
        entries = publication_queue.list_entries()
        assert len(entries) == num_threads

    def test_concurrent_read_operations(self, publication_queue, sample_entry):
        """Test concurrent read operations are thread-safe."""
        publication_queue.add_entry(sample_entry)

        num_threads = 20
        threads = []
        results = []
        errors = []

        def read_entry():
            try:
                entry = publication_queue.get_entry(sample_entry.entry_id)
                results.append(entry)
            except Exception as e:
                errors.append(e)

        # Create and start threads
        for _ in range(num_threads):
            thread = threading.Thread(target=read_entry)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Verify no errors
        assert len(errors) == 0
        assert len(results) == num_threads

        # Verify all reads got correct data
        assert all(r.entry_id == sample_entry.entry_id for r in results)


# =============================================================================
# Statistics Tests
# =============================================================================


class TestStatistics:
    """Test statistics and reporting functionality."""

    def test_get_statistics_empty_queue(self, publication_queue):
        """Test statistics on empty queue."""
        stats = publication_queue.get_statistics()

        assert stats["total_entries"] == 0
        assert stats["by_status"]["pending"] == 0
        assert stats["by_status"]["completed"] == 0

    def test_get_statistics_with_entries(self, publication_queue):
        """Test statistics with multiple entries."""
        # Add entries with different statuses
        entries_data = [
            ("entry_1", "11111111", PublicationStatus.PENDING, "single_cell"),
            ("entry_2", "22222222", PublicationStatus.PENDING, "microbiome"),
            ("entry_3", "33333333", PublicationStatus.COMPLETED, "single_cell"),
            ("entry_4", "44444444", PublicationStatus.FAILED, "proteomics"),
        ]

        for entry_id, pmid, status, schema_type in entries_data:
            entry = PublicationQueueEntry(
                entry_id=entry_id,
                pmid=pmid,
                status=status,
                schema_type=schema_type,
            )
            publication_queue.add_entry(entry)

        stats = publication_queue.get_statistics()

        assert stats["total_entries"] == 4
        assert stats["by_status"]["pending"] == 2
        assert stats["by_status"]["completed"] == 1
        assert stats["by_status"]["failed"] == 1
        assert stats["by_status"]["extracting"] == 0

        assert stats["by_schema_type"]["single_cell"] == 2
        assert stats["by_schema_type"]["microbiome"] == 1
        assert stats["by_schema_type"]["proteomics"] == 1


# =============================================================================
# RIS Parser Tests
# =============================================================================


# Check if rispy is available
try:
    import rispy
    RISPY_AVAILABLE = True
except ImportError:
    RISPY_AVAILABLE = False

rispy_required = pytest.mark.skipif(
    not RISPY_AVAILABLE, reason="rispy library not installed"
)


class TestRISParser:
    """Test RIS file parsing functionality."""

    @rispy_required
    def test_parse_file_success(self, temp_ris_file, sample_ris_content):
        """Test successfully parsing RIS file."""
        # Write sample RIS content
        temp_ris_file.write_text(sample_ris_content)

        parser = RISParser()
        entries = parser.parse_file(temp_ris_file)

        assert len(entries) == 2
        assert all(isinstance(e, PublicationQueueEntry) for e in entries)

        # Check first entry
        assert entries[0].title == "Single-cell RNA sequencing reveals novel cell types in human brain"
        assert "Smith" in entries[0].authors[0]
        assert entries[0].year == 2022
        assert entries[0].pmid == "35042229"
        assert entries[0].doi == "10.1038/s41586-022-04426-0"

    def test_parse_file_not_found(self, tmp_path):
        """Test parsing non-existent file raises error."""
        parser = RISParser()
        non_existent_file = tmp_path / "does_not_exist.ris"

        with pytest.raises(RISParseError, match="File not found"):
            parser.parse_file(non_existent_file)

    def test_parse_file_wrong_extension(self, tmp_path):
        """Test parsing file with wrong extension raises error."""
        parser = RISParser()
        wrong_file = tmp_path / "test.csv"
        wrong_file.touch()

        with pytest.raises(RISParseError, match="must have .ris or .txt extension"):
            parser.parse_file(wrong_file)

    def test_schema_type_inference_single_cell(self):
        """Test schema type inference for single-cell publications."""
        parser = RISParser()

        ris_entry = {
            "title": "Single-cell RNA-seq analysis of human brain",
            "abstract": "Using scRNA-seq to analyze cell types",
            "keywords": ["single-cell", "RNA-seq"],
        }

        inferred = parser._infer_schema_type(ris_entry)
        assert inferred == "single_cell"

    def test_schema_type_inference_microbiome(self):
        """Test schema type inference for microbiome publications."""
        parser = RISParser()

        ris_entry = {
            "title": "16S rRNA sequencing of gut microbiome",
            "abstract": "Microbiome analysis reveals bacterial diversity",
            "keywords": ["microbiome", "16S rRNA"],
        }

        inferred = parser._infer_schema_type(ris_entry)
        assert inferred == "microbiome"

    def test_schema_type_inference_proteomics(self):
        """Test schema type inference for proteomics publications."""
        parser = RISParser()

        ris_entry = {
            "title": "Mass spectrometry-based proteomics analysis",
            "abstract": "LC-MS reveals protein expression changes",
            "keywords": ["proteomics", "mass spectrometry"],
        }

        inferred = parser._infer_schema_type(ris_entry)
        assert inferred == "proteomics"

    def test_pmid_extraction_from_notes(self):
        """Test PMID extraction from notes field."""
        parser = RISParser()

        ris_entry = {
            "notes": "PMID: 12345678",
        }

        pmid = parser._extract_pmid(ris_entry)
        assert pmid == "12345678"

    def test_pmc_extraction_from_notes(self):
        """Test PMC ID extraction from notes field."""
        parser = RISParser()

        ris_entry = {
            "notes": "PMC8891176",
        }

        pmc_id = parser._extract_pmc_id(ris_entry)
        assert pmc_id == "PMC8891176"

    def test_year_extraction_from_various_formats(self):
        """Test year extraction from different date formats."""
        parser = RISParser()

        # Format: YYYY
        assert parser._extract_year({"year": "2022"}) == 2022

        # Format: YYYY/MM/DD
        assert parser._extract_year({"year": "2022/01/15"}) == 2022

        # Format: YYYY-MM-DD
        assert parser._extract_year({"year": "2022-01-15"}) == 2022

        # Invalid format
        assert parser._extract_year({"year": "invalid"}) is None

    def test_author_extraction_single_author(self):
        """Test author extraction with single author."""
        parser = RISParser()

        authors = parser._extract_authors({"authors": "Smith, John"})
        assert authors == ["Smith, John"]

    def test_author_extraction_multiple_authors(self):
        """Test author extraction with multiple authors."""
        parser = RISParser()

        authors = parser._extract_authors({
            "authors": ["Smith, John", "Jones, Alice", "Williams, Bob"]
        })
        assert len(authors) == 3
        assert "Smith, John" in authors

    @rispy_required
    def test_get_statistics(self, temp_ris_file, sample_ris_content):
        """Test parser statistics reporting."""
        temp_ris_file.write_text(sample_ris_content)

        parser = RISParser()
        entries = parser.parse_file(temp_ris_file)

        stats = parser.get_statistics()
        assert stats["parsed"] == 2
        assert stats["skipped"] == 0
        assert len(stats["errors"]) == 0

    def test_to_publication_entry(self):
        """Test public API for converting RIS entry to queue entry."""
        parser = RISParser()

        ris_entry = {
            "title": "Test publication",
            "authors": ["Smith, John"],
            "year": "2023",
            "doi": "10.1234/test",
            "pmid": "12345678",
        }

        entry = parser.to_publication_entry(
            ris_entry,
            priority=3,
            schema_type="single_cell",
            extraction_level="full_text",
        )

        assert entry.title == "Test publication"
        assert entry.priority == 3
        assert entry.schema_type == "single_cell"
        assert entry.extraction_level == ExtractionLevel.FULL_TEXT


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_queue_operations(self, publication_queue):
        """Test operations on empty queue."""
        # List entries on empty queue
        entries = publication_queue.list_entries()
        assert entries == []

        # Clear empty queue
        cleared = publication_queue.clear_queue()
        assert cleared == 0

        # Get statistics on empty queue
        stats = publication_queue.get_statistics()
        assert stats["total_entries"] == 0

    def test_special_characters_in_fields(self, publication_queue):
        """Test handling of special characters."""
        entry = PublicationQueueEntry(
            entry_id="entry_with_special_chars",
            pmid="12345678",
            title="Test 'quotes' and \"double quotes\"",
        )

        publication_queue.add_entry(entry)
        retrieved = publication_queue.get_entry(entry.entry_id)

        assert "quotes" in retrieved.title

    def test_unicode_in_fields(self, publication_queue):
        """Test handling of Unicode characters."""
        entry = PublicationQueueEntry(
            entry_id="unicode_entry",
            pmid="12345678",
            title="研究データ (Japanese) 🧬",
            journal="Nature 自然",
        )

        publication_queue.add_entry(entry)
        retrieved = publication_queue.get_entry(entry.entry_id)

        assert "研究データ" in retrieved.title
        assert "🧬" in retrieved.title

    def test_large_error_log(self, publication_queue, sample_entry):
        """Test handling of large error logs."""
        publication_queue.add_entry(sample_entry)

        # Add many errors
        for i in range(100):
            publication_queue.update_status(
                entry_id=sample_entry.entry_id,
                status=PublicationStatus.FAILED,
                error=f"Error message {i}",
            )

        retrieved = publication_queue.get_entry(sample_entry.entry_id)
        assert len(retrieved.error_log) == 100

    def test_datetime_serialization(self, publication_queue, sample_entry):
        """Test proper datetime serialization."""
        publication_queue.add_entry(sample_entry)

        # Read raw JSON
        with open(publication_queue.queue_file, "r") as f:
            line = f.readline()

        data = json.loads(line)

        # Verify datetime fields are serialized as ISO format strings
        assert isinstance(data["created_at"], str)
        assert isinstance(data["updated_at"], str)

        # Verify they can be parsed back
        created_at = datetime.fromisoformat(data["created_at"])
        assert isinstance(created_at, datetime)
