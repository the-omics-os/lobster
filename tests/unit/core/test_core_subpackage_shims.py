"""Tests for core/ subpackage backward-compatibility shims.

Covers all 13 planned shim moves across Plans 01 and 02.
Plan 01 shims (6): governance/license_manager, governance/aquadif_monitor,
    queues/download_queue, queues/publication_queue, queues/queue_storage,
    runtime/workspace
Plan 02 shims (7): notebooks/executor, notebooks/exporter, notebooks/validator,
    provenance/analysis_ir, provenance/provenance, provenance/lineage,
    provenance/ir_coverage
"""

import importlib
import sys
import warnings

import pytest

# All 13 shim pairs: (old_path, new_path, expected_name)
SHIM_PAIRS = [
    ("lobster.core.download_queue", "lobster.core.queues.download_queue", "DownloadQueue"),
    ("lobster.core.publication_queue", "lobster.core.queues.publication_queue", "PublicationQueue"),
    ("lobster.core.queue_storage", "lobster.core.queues.queue_storage", "InterProcessFileLock"),
    ("lobster.core.notebook_executor", "lobster.core.notebooks.executor", "NotebookExecutor"),
    ("lobster.core.notebook_exporter", "lobster.core.notebooks.exporter", "NotebookExporter"),
    ("lobster.core.notebook_validator", "lobster.core.notebooks.validator", "NotebookValidator"),
    ("lobster.core.license_manager", "lobster.core.governance.license_manager", "get_current_tier"),
    ("lobster.core.aquadif_monitor", "lobster.core.governance.aquadif_monitor", "AquadifMonitor"),
    ("lobster.core.analysis_ir", "lobster.core.provenance.analysis_ir", "AnalysisStep"),
    ("lobster.core.provenance", "lobster.core.provenance.provenance", "ProvenanceTracker"),
    ("lobster.core.lineage", "lobster.core.provenance.lineage", "LineageMetadata"),
    ("lobster.core.ir_coverage", "lobster.core.provenance.ir_coverage", "IRCoverageAnalyzer"),
    ("lobster.core.workspace", "lobster.core.runtime.workspace", "resolve_workspace"),
]

def _shim_id(pair):
    """Generate readable test ID from shim pair."""
    return pair[0].rsplit(".", 1)[-1]


@pytest.mark.parametrize(
    "old_path, new_path, expected_name",
    SHIM_PAIRS,
    ids=[_shim_id(p) for p in SHIM_PAIRS],
)
class TestShimReexportsAndWarns:
    """Verify shims re-export names and emit DeprecationWarning."""

    def test_shim_reexports_and_warns(self, old_path, new_path, expected_name):
        """Importing via old path should warn and provide the expected name."""
        # Evict cached module so the warning fires fresh
        sys.modules.pop(old_path, None)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mod = importlib.import_module(old_path)
            # Access the expected name inside the catch block to capture
            # warnings from __getattr__ lazy shims (provenance/__init__.py)
            has_name = hasattr(mod, expected_name)

        # The expected name must be accessible
        assert has_name, (
            f"{old_path} shim missing expected name '{expected_name}'"
        )

        # At least one DeprecationWarning must mention the new path
        dep_warnings = [
            w for w in caught
            if issubclass(w.category, DeprecationWarning)
            and new_path.rsplit(".", 1)[0] in str(w.message)
        ]
        assert dep_warnings, (
            f"No DeprecationWarning mentioning '{new_path}' when importing '{old_path}'. "
            f"Caught warnings: {[str(w.message) for w in caught]}"
        )

    def test_isinstance_identity(self, old_path, new_path, expected_name):
        """Objects from old and new paths must be the same object (identity)."""
        # Suppress warnings for clean import
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            old_mod = importlib.import_module(old_path)
        new_mod = importlib.import_module(new_path)

        old_obj = getattr(old_mod, expected_name)
        new_obj = getattr(new_mod, expected_name)

        assert old_obj is new_obj, (
            f"{expected_name} from old path is not identical to new path object"
        )


# Subpackage existence tests
SUBPACKAGES = [
    "lobster.core.governance",    # Plan 01
    "lobster.core.queues",        # Plan 01
    "lobster.core.runtime",       # Plan 01
    "lobster.core.notebooks",     # Plan 02
    "lobster.core.provenance",    # Plan 02
]


@pytest.mark.parametrize(
    "subpackage",
    SUBPACKAGES,
    ids=[s.rsplit(".", 1)[-1] for s in SUBPACKAGES],
)
def test_subpackages_exist(subpackage):
    """Verify subpackage directories have __init__.py and are importable."""
    mod = importlib.import_module(subpackage)
    assert mod is not None
