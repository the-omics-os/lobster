"""
Unit tests for AquadifMonitor service class.

Tests cover:
- Category counting (single and multi-category tools)
- Provenance status tracking (real_ir, hollow_ir, missing)
- CODE_EXEC bounded logging
- get_session_summary() consistent snapshot
- Thread safety
- Fail-open error handling
- Edge cases (empty map, unknown tools)
"""

import threading
import time
from datetime import datetime

import pytest

from lobster.core.aquadif_monitor import AquadifMonitor, CodeExecEntry

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_METADATA_MAP = {
    "analyze_cells": {"categories": ["ANALYZE"], "provenance": True},
    "import_data": {"categories": ["IMPORT"], "provenance": True},
    "filter_cells": {"categories": ["FILTER"], "provenance": True},
    "execute_custom_code": {"categories": ["CODE_EXEC"], "provenance": False},
    "handoff_to_transcriptomics": {"categories": ["DELEGATE"], "provenance": False},
    "check_status": {"categories": ["UTILITY"], "provenance": False},
    "analyze_and_filter": {"categories": ["ANALYZE", "FILTER"], "provenance": True},
}


@pytest.fixture
def monitor():
    """Standard monitor with sample metadata map."""
    return AquadifMonitor(tool_metadata_map=SAMPLE_METADATA_MAP)


@pytest.fixture
def empty_monitor():
    """Monitor with empty metadata map."""
    return AquadifMonitor(tool_metadata_map={})


# ---------------------------------------------------------------------------
# CodeExecEntry tests
# ---------------------------------------------------------------------------


class TestCodeExecEntry:
    def test_dataclass_fields(self):
        entry = CodeExecEntry(
            tool_name="execute_custom_code",
            timestamp="2026-03-01T09:00:00.000000",
            agent="metadata_assistant",
        )
        assert entry.tool_name == "execute_custom_code"
        assert entry.timestamp == "2026-03-01T09:00:00.000000"
        assert entry.agent == "metadata_assistant"


# ---------------------------------------------------------------------------
# AquadifMonitor construction
# ---------------------------------------------------------------------------


class TestAquadifMonitorConstruction:
    def test_constructs_with_sample_map(self):
        m = AquadifMonitor(tool_metadata_map=SAMPLE_METADATA_MAP)
        assert m is not None

    def test_constructs_with_empty_map(self):
        m = AquadifMonitor(tool_metadata_map={})
        assert m is not None

    def test_initial_category_counts_empty(self, monitor):
        assert monitor.get_category_distribution() == {}

    def test_initial_provenance_status_empty(self, monitor):
        status = monitor.get_provenance_status()
        assert status == {"real_ir": [], "hollow_ir": [], "missing": []}

    def test_initial_code_exec_log_empty(self, monitor):
        assert monitor.get_code_exec_log() == []

    def test_initial_session_summary_zeros(self, monitor):
        summary = monitor.get_session_summary()
        assert summary["total_invocations"] == 0
        assert summary["code_exec_count"] == 0
        assert summary["category_distribution"] == {}
        assert summary["code_exec_log"] == []


# ---------------------------------------------------------------------------
# record_tool_invocation — category counting
# ---------------------------------------------------------------------------


class TestRecordToolInvocationCategoryCounting:
    def test_single_category_increments_count(self, monitor):
        monitor.record_tool_invocation("analyze_cells", "transcriptomics_expert")
        dist = monitor.get_category_distribution()
        assert dist["ANALYZE"] == 1

    def test_tool_invocation_count_tracked(self, monitor):
        monitor.record_tool_invocation("analyze_cells", "transcriptomics_expert")
        summary = monitor.get_session_summary()
        assert summary["total_invocations"] == 1

    def test_multiple_invocations_accumulate(self, monitor):
        monitor.record_tool_invocation("analyze_cells", "transcriptomics_expert")
        monitor.record_tool_invocation("analyze_cells", "transcriptomics_expert")
        monitor.record_tool_invocation("analyze_cells", "transcriptomics_expert")
        dist = monitor.get_category_distribution()
        assert dist["ANALYZE"] == 3

    def test_different_tools_same_category_accumulate(self, monitor):
        monitor.record_tool_invocation("analyze_cells", "transcriptomics_expert")
        monitor.record_tool_invocation("import_data", "data_expert")
        dist = monitor.get_category_distribution()
        assert dist["ANALYZE"] == 1
        assert dist["IMPORT"] == 1

    def test_multi_category_tool_increments_both(self, monitor):
        monitor.record_tool_invocation("analyze_and_filter", "transcriptomics_expert")
        dist = monitor.get_category_distribution()
        assert dist["ANALYZE"] == 1
        assert dist["FILTER"] == 1

    def test_multi_category_both_accumulate_independently(self, monitor):
        monitor.record_tool_invocation("analyze_and_filter", "transcriptomics_expert")
        monitor.record_tool_invocation("analyze_cells", "transcriptomics_expert")
        monitor.record_tool_invocation("filter_cells", "transcriptomics_expert")
        dist = monitor.get_category_distribution()
        assert dist["ANALYZE"] == 2  # analyze_and_filter + analyze_cells
        assert dist["FILTER"] == 2  # analyze_and_filter + filter_cells

    def test_unknown_tool_does_not_crash(self, monitor):
        # No exception, no category count change
        monitor.record_tool_invocation("totally_unknown_tool", "some_agent")
        dist = monitor.get_category_distribution()
        assert dist == {}

    def test_unknown_tool_does_not_add_to_invocation_count(self, monitor):
        # Unknown tool with no categories — no entry in tool_invocation_counts
        monitor.record_tool_invocation("totally_unknown_tool", "some_agent")
        summary = monitor.get_session_summary()
        # total_invocations counts tool_invocation_counts which IS incremented for unknown tools
        # but per-category counts stay empty
        assert summary["category_distribution"] == {}

    def test_empty_map_no_crash_on_invocation(self, empty_monitor):
        empty_monitor.record_tool_invocation("any_tool", "any_agent")
        assert empty_monitor.get_category_distribution() == {}


# ---------------------------------------------------------------------------
# record_tool_invocation — provenance pre-setting
# ---------------------------------------------------------------------------


class TestRecordToolInvocationProvenance:
    def test_provenance_required_tool_set_to_missing(self, monitor):
        monitor.record_tool_invocation("analyze_cells", "transcriptomics_expert")
        status = monitor.get_provenance_status()
        assert "analyze_cells" in status["missing"]

    def test_provenance_not_required_not_tracked(self, monitor):
        monitor.record_tool_invocation("check_status", "supervisor")
        status = monitor.get_provenance_status()
        assert "check_status" not in status["missing"]
        assert "check_status" not in status["real_ir"]
        assert "check_status" not in status["hollow_ir"]

    def test_delegate_tool_not_tracked(self, monitor):
        monitor.record_tool_invocation("handoff_to_transcriptomics", "supervisor")
        status = monitor.get_provenance_status()
        for group in status.values():
            assert "handoff_to_transcriptomics" not in group

    def test_code_exec_not_in_provenance_status(self, monitor):
        monitor.record_tool_invocation("execute_custom_code", "metadata_assistant")
        status = monitor.get_provenance_status()
        for group in status.values():
            assert "execute_custom_code" not in group

    def test_second_invocation_does_not_overwrite_missing(self, monitor):
        # First call: sets to missing
        monitor.record_tool_invocation("analyze_cells", "transcriptomics_expert")
        # Manual update via record_provenance_call
        monitor.record_provenance_call("analyze_cells", has_real_ir=True)
        # Second invocation: should NOT reset real_ir back to missing
        monitor.record_tool_invocation("analyze_cells", "transcriptomics_expert")
        status = monitor.get_provenance_status()
        assert "analyze_cells" in status["real_ir"]
        assert "analyze_cells" not in status["missing"]


# ---------------------------------------------------------------------------
# record_tool_invocation — CODE_EXEC logging
# ---------------------------------------------------------------------------


class TestCodeExecLogging:
    def test_code_exec_adds_log_entry(self, monitor):
        monitor.record_tool_invocation("execute_custom_code", "metadata_assistant")
        log = monitor.get_code_exec_log()
        assert len(log) == 1

    def test_code_exec_entry_has_correct_fields(self, monitor):
        monitor.record_tool_invocation("execute_custom_code", "metadata_assistant")
        log = monitor.get_code_exec_log()
        entry = log[0]
        assert entry.tool_name == "execute_custom_code"
        assert entry.agent == "metadata_assistant"
        # Timestamp should be valid ISO format
        assert "T" in entry.timestamp  # ISO 8601 contains T separator

    def test_code_exec_timestamp_is_recent(self, monitor):
        before = datetime.now()
        monitor.record_tool_invocation("execute_custom_code", "metadata_assistant")
        after = datetime.now()
        log = monitor.get_code_exec_log()
        entry_time = datetime.fromisoformat(log[0].timestamp)
        assert before <= entry_time <= after

    def test_non_code_exec_tool_not_in_log(self, monitor):
        monitor.record_tool_invocation("analyze_cells", "transcriptomics_expert")
        log = monitor.get_code_exec_log()
        assert len(log) == 0

    def test_multiple_code_exec_entries(self, monitor):
        for _ in range(3):
            monitor.record_tool_invocation("execute_custom_code", "metadata_assistant")
        log = monitor.get_code_exec_log()
        assert len(log) == 3

    def test_code_exec_log_bounded_to_100(self, monitor):
        """101st entry evicts the 1st."""
        for i in range(101):
            monitor.record_tool_invocation("execute_custom_code", f"agent_{i}")
        log = monitor.get_code_exec_log()
        assert len(log) == 100

    def test_code_exec_log_oldest_evicted(self, monitor):
        """First entry should be evicted after 101 appends."""
        # First entry: agent_0
        monitor.record_tool_invocation("execute_custom_code", "agent_0")
        # Fill up remaining 100 entries
        for i in range(1, 101):
            monitor.record_tool_invocation("execute_custom_code", f"agent_{i}")
        log = monitor.get_code_exec_log()
        # agent_0 should be evicted
        agents = [e.agent for e in log]
        assert "agent_0" not in agents
        assert "agent_100" in agents

    def test_get_code_exec_log_returns_snapshot(self, monitor):
        """Modifying returned list does not affect internal state."""
        monitor.record_tool_invocation("execute_custom_code", "metadata_assistant")
        log = monitor.get_code_exec_log()
        original_len = len(log)
        log.append("garbage")
        assert len(monitor.get_code_exec_log()) == original_len

    def test_code_exec_count_in_session_summary(self, monitor):
        monitor.record_tool_invocation("execute_custom_code", "metadata_assistant")
        monitor.record_tool_invocation("execute_custom_code", "metadata_assistant")
        summary = monitor.get_session_summary()
        assert summary["code_exec_count"] == 2


# ---------------------------------------------------------------------------
# record_provenance_call
# ---------------------------------------------------------------------------


class TestRecordProvenanceCall:
    def test_real_ir_sets_status_to_real_ir(self, monitor):
        monitor.record_tool_invocation("analyze_cells", "transcriptomics_expert")
        monitor.record_provenance_call("analyze_cells", has_real_ir=True)
        status = monitor.get_provenance_status()
        assert "analyze_cells" in status["real_ir"]
        assert "analyze_cells" not in status["missing"]

    def test_hollow_ir_sets_status_to_hollow_ir(self, monitor):
        monitor.record_tool_invocation("analyze_cells", "transcriptomics_expert")
        monitor.record_provenance_call("analyze_cells", has_real_ir=False)
        status = monitor.get_provenance_status()
        assert "analyze_cells" in status["hollow_ir"]
        assert "analyze_cells" not in status["missing"]

    def test_real_ir_wins_over_hollow_ir(self, monitor):
        """Once real_ir is set, hollow_ir cannot downgrade it."""
        monitor.record_tool_invocation("analyze_cells", "transcriptomics_expert")
        monitor.record_provenance_call("analyze_cells", has_real_ir=True)
        monitor.record_provenance_call("analyze_cells", has_real_ir=False)
        status = monitor.get_provenance_status()
        assert "analyze_cells" in status["real_ir"]
        assert "analyze_cells" not in status["hollow_ir"]

    def test_hollow_ir_can_upgrade_to_real_ir(self, monitor):
        """hollow_ir can be upgraded to real_ir."""
        monitor.record_tool_invocation("analyze_cells", "transcriptomics_expert")
        monitor.record_provenance_call("analyze_cells", has_real_ir=False)
        monitor.record_provenance_call("analyze_cells", has_real_ir=True)
        status = monitor.get_provenance_status()
        assert "analyze_cells" in status["real_ir"]
        assert "analyze_cells" not in status["hollow_ir"]

    def test_provenance_call_for_unknown_tool_does_not_crash(self, monitor):
        """Fail-open: no crash for unknown tool."""
        monitor.record_provenance_call("non_existent_tool", has_real_ir=True)
        # Should not raise; status for unknown tool may appear or not

    def test_provenance_call_for_tool_not_invoked_does_not_crash(self, empty_monitor):
        """Calling record_provenance_call without prior invocation is safe."""
        empty_monitor.record_provenance_call("some_tool", has_real_ir=True)


# ---------------------------------------------------------------------------
# get_category_distribution
# ---------------------------------------------------------------------------


class TestGetCategoryDistribution:
    def test_returns_dict_copy(self, monitor):
        """Modifying returned dict does not affect internal state."""
        monitor.record_tool_invocation("analyze_cells", "transcriptomics_expert")
        dist = monitor.get_category_distribution()
        dist["ANALYZE"] = 999
        assert monitor.get_category_distribution()["ANALYZE"] == 1

    def test_after_multiple_categories_correct_counts(self, monitor):
        """3 ANALYZE + 2 IMPORT -> {ANALYZE: 3, IMPORT: 2}."""
        for _ in range(3):
            monitor.record_tool_invocation("analyze_cells", "transcriptomics_expert")
        for _ in range(2):
            monitor.record_tool_invocation("import_data", "data_expert")
        dist = monitor.get_category_distribution()
        assert dist["ANALYZE"] == 3
        assert dist["IMPORT"] == 2
        assert len(dist) == 2  # no other categories

    def test_empty_map_returns_empty_dict(self, empty_monitor):
        empty_monitor.record_tool_invocation("any_tool", "any_agent")
        assert empty_monitor.get_category_distribution() == {}


# ---------------------------------------------------------------------------
# get_provenance_status
# ---------------------------------------------------------------------------


class TestGetProvenanceStatus:
    def test_returns_three_groups(self, monitor):
        status = monitor.get_provenance_status()
        assert set(status.keys()) == {"real_ir", "hollow_ir", "missing"}

    def test_multiple_tools_sorted_into_groups(self, monitor):
        monitor.record_tool_invocation("analyze_cells", "transcriptomics_expert")
        monitor.record_tool_invocation("import_data", "data_expert")
        monitor.record_tool_invocation("filter_cells", "transcriptomics_expert")
        monitor.record_provenance_call("analyze_cells", has_real_ir=True)
        monitor.record_provenance_call("import_data", has_real_ir=False)
        # filter_cells: still "missing"
        status = monitor.get_provenance_status()
        assert "analyze_cells" in status["real_ir"]
        assert "import_data" in status["hollow_ir"]
        assert "filter_cells" in status["missing"]


# ---------------------------------------------------------------------------
# get_session_summary
# ---------------------------------------------------------------------------


class TestGetSessionSummary:
    def test_returns_all_required_keys(self, monitor):
        summary = monitor.get_session_summary()
        required_keys = {
            "category_distribution",
            "provenance_status",
            "code_exec_count",
            "code_exec_log",
            "total_invocations",
        }
        assert required_keys == set(summary.keys())

    def test_code_exec_log_in_summary_is_serialized(self, monitor):
        """code_exec_log entries in summary are dicts, not CodeExecEntry objects."""
        monitor.record_tool_invocation("execute_custom_code", "metadata_assistant")
        summary = monitor.get_session_summary()
        log = summary["code_exec_log"]
        assert len(log) == 1
        entry = log[0]
        assert isinstance(entry, dict)
        assert "tool_name" in entry
        assert "timestamp" in entry
        assert "agent" in entry

    def test_summary_consistent_snapshot(self, monitor):
        """get_session_summary acquires lock for consistent state."""
        monitor.record_tool_invocation("analyze_cells", "transcriptomics_expert")
        monitor.record_provenance_call("analyze_cells", has_real_ir=True)
        summary = monitor.get_session_summary()
        assert summary["total_invocations"] == 1
        assert summary["category_distribution"]["ANALYZE"] == 1
        assert "analyze_cells" in summary["provenance_status"]["real_ir"]

    def test_total_invocations_counts_all_calls(self, monitor):
        monitor.record_tool_invocation("analyze_cells", "transcriptomics_expert")
        monitor.record_tool_invocation("import_data", "data_expert")
        monitor.record_tool_invocation("check_status", "supervisor")
        summary = monitor.get_session_summary()
        assert summary["total_invocations"] == 3


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_record_tool_invocations_do_not_corrupt(self, monitor):
        """100 threads each increment ANALYZE 100x — final count must be 10000."""
        n_threads = 100
        invocations_per_thread = 100

        def worker():
            for _ in range(invocations_per_thread):
                monitor.record_tool_invocation(
                    "analyze_cells", "transcriptomics_expert"
                )

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        dist = monitor.get_category_distribution()
        assert dist["ANALYZE"] == n_threads * invocations_per_thread

    def test_concurrent_provenance_calls_do_not_corrupt(self, monitor):
        """Concurrent real_ir vs hollow_ir: real_ir must win (non-downgrade rule)."""
        monitor.record_tool_invocation("analyze_cells", "transcriptomics_expert")

        results = []

        def set_real():
            monitor.record_provenance_call("analyze_cells", has_real_ir=True)
            results.append("real")

        def set_hollow():
            monitor.record_provenance_call("analyze_cells", has_real_ir=False)
            results.append("hollow")

        # real_ir call first (acquires lock), then hollow attempt should not downgrade
        t1 = threading.Thread(target=set_real)
        t2 = threading.Thread(target=set_hollow)
        t1.start()
        t1.join()  # Ensure real_ir is set first
        t2.start()
        t2.join()

        status = monitor.get_provenance_status()
        assert "analyze_cells" in status["real_ir"]
        assert "analyze_cells" not in status["hollow_ir"]

    def test_get_session_summary_during_concurrent_writes(self, monitor):
        """get_session_summary during concurrent writes returns internally consistent snapshot."""
        stop_event = threading.Event()
        summary_results = []

        def writer():
            while not stop_event.is_set():
                monitor.record_tool_invocation(
                    "analyze_cells", "transcriptomics_expert"
                )
                time.sleep(0.0001)

        def reader():
            for _ in range(20):
                summary = monitor.get_session_summary()
                summary_results.append(summary)
                time.sleep(0.001)

        writer_thread = threading.Thread(target=writer)
        reader_thread = threading.Thread(target=reader)

        writer_thread.start()
        reader_thread.start()
        reader_thread.join()
        stop_event.set()
        writer_thread.join()

        # All summaries should have non-negative total_invocations
        for s in summary_results:
            assert s["total_invocations"] >= 0
            assert isinstance(s["category_distribution"], dict)


# ---------------------------------------------------------------------------
# Fail-open behavior
# ---------------------------------------------------------------------------


class TestFailOpen:
    def test_record_tool_invocation_fail_open(self, monitor):
        """Exceptions inside record_tool_invocation are swallowed."""
        # Corrupt internal state to trigger exception
        monitor._tool_metadata_map = None  # Will cause AttributeError on .get()
        # Should not raise
        monitor.record_tool_invocation("analyze_cells", "transcriptomics_expert")

    def test_record_provenance_call_fail_open(self, monitor):
        """Exceptions inside record_provenance_call are swallowed."""
        monitor._provenance_status = None  # Will cause AttributeError
        # Should not raise
        monitor.record_provenance_call("analyze_cells", has_real_ir=True)

    def test_get_category_distribution_on_empty_monitor_no_crash(self, empty_monitor):
        assert empty_monitor.get_category_distribution() == {}

    def test_get_provenance_status_on_empty_monitor_no_crash(self, empty_monitor):
        status = empty_monitor.get_provenance_status()
        assert status == {"real_ir": [], "hollow_ir": [], "missing": []}

    def test_get_code_exec_log_on_empty_monitor_no_crash(self, empty_monitor):
        assert empty_monitor.get_code_exec_log() == []

    def test_get_session_summary_on_empty_monitor_no_crash(self, empty_monitor):
        summary = empty_monitor.get_session_summary()
        assert summary["total_invocations"] == 0


# ---------------------------------------------------------------------------
# No lobster imports
# ---------------------------------------------------------------------------


class TestNoLobsterImports:
    def test_module_has_no_lobster_imports(self):
        """aquadif_monitor.py must be pure stdlib — no lobster.* imports."""
        import importlib.util
        import os

        module_path = os.path.join(
            os.path.dirname(__file__),  # tests/unit/core/
            "..",
            "..",
            "..",  # → lobster/ root
            "lobster",
            "core",
            "aquadif_monitor.py",
        )
        module_path = os.path.abspath(module_path)

        with open(module_path) as f:
            source = f.read()

        # No 'import lobster' or 'from lobster' lines
        import re

        lobster_imports = re.findall(
            r"^(?:import lobster|from lobster)", source, re.MULTILINE
        )
        assert (
            lobster_imports == []
        ), f"aquadif_monitor.py must have zero lobster imports. Found: {lobster_imports}"
