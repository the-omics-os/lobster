"""Tests for core/ subpackage backward-compatibility shims.

Covers all 14 shim moves across Phase 06 Plans 01/02 and Phase 07 Plan 01.
Phase 06 Plan 01 shims (6): governance/license_manager, governance/aquadif_monitor,
    queues/download_queue, queues/publication_queue, queues/queue_storage,
    runtime/workspace
Phase 06 Plan 02 shims (7): notebooks/executor, notebooks/exporter, notebooks/validator,
    provenance/analysis_ir, provenance/provenance, provenance/lineage,
    provenance/ir_coverage
Phase 07 Plan 01 shims (1): runtime/data_manager (data_manager_v2)
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
    ("lobster.core.data_manager_v2", "lobster.core.runtime.data_manager", "DataManagerV2"),
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


# --------------------------------------------------------------------------- #
# Provenance multi-module __getattr__ shim tests (Phase 10)
# --------------------------------------------------------------------------- #

# Names that must be resolvable via provenance package __getattr__ shim
PROVENANCE_SHIM_NAMES = [
    ("AnalysisStep", "analysis_ir"),
    ("ParameterSpec", "analysis_ir"),
    ("LineageMetadata", "lineage"),
    ("IRCoverageAnalyzer", "ir_coverage"),
    ("ProvenanceTracker", "provenance"),
]


@pytest.mark.parametrize(
    "attr_name, expected_submod",
    PROVENANCE_SHIM_NAMES,
    ids=[name for name, _ in PROVENANCE_SHIM_NAMES],
)
class TestProvenanceMultiModuleShim:
    """Verify provenance __getattr__ resolves names from all 4 submodules."""

    def test_resolves_with_deprecation_warning(self, attr_name, expected_submod):
        """Accessing name via provenance package emits DeprecationWarning."""
        sys.modules.pop("lobster.core.provenance", None)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mod = importlib.import_module("lobster.core.provenance")
            obj = getattr(mod, attr_name)

        assert obj is not None, f"Failed to resolve '{attr_name}' from provenance shim"

        dep_warnings = [
            w for w in caught
            if issubclass(w.category, DeprecationWarning)
            and expected_submod in str(w.message)
        ]
        assert dep_warnings, (
            f"No DeprecationWarning mentioning '{expected_submod}' for '{attr_name}'. "
            f"Caught: {[str(w.message) for w in caught]}"
        )

    def test_identity_with_canonical_import(self, attr_name, expected_submod):
        """Object from shim must be identical to object from canonical path."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            shim_mod = importlib.import_module("lobster.core.provenance")
            shim_obj = getattr(shim_mod, attr_name)

        canonical_mod = importlib.import_module(
            f"lobster.core.provenance.{expected_submod}"
        )
        canonical_obj = getattr(canonical_mod, attr_name)

        assert shim_obj is canonical_obj, (
            f"'{attr_name}' from shim is not identical to canonical import"
        )


def test_provenance_shim_unknown_name_raises():
    """Accessing nonexistent name via provenance shim raises AttributeError."""
    mod = importlib.import_module("lobster.core.provenance")
    with pytest.raises(AttributeError, match="nonexistent_xyz_123"):
        getattr(mod, "nonexistent_xyz_123")


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


def test_no_deprecated_data_manager_imports_in_packages():
    """packages/ must not import from deprecated lobster.core.data_manager_v2."""
    import subprocess

    result = subprocess.run(
        [
            "grep",
            "-rn",
            "--include=*.py",
            "--exclude=data_manager_v2.py",
            "from lobster.core.data_manager_v2 import",
            "packages/",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, (
        f"Deprecated imports found in packages/:\n{result.stdout}"
    )
