"""
Tests for backward-compatibility shims at lobster.core.vector.

Verifies that importing from the old paths still works (via shims)
and emits DeprecationWarning directing users to lobster.services.vector.
"""

import warnings

import pytest


class TestBackwardCompatShims:
    """Verify shims at lobster.core.vector re-export correctly with warnings."""

    def test_shim_service_emits_deprecation_warning(self):
        """Importing from lobster.core.vector.service should warn."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from lobster.core.vector.service import (  # noqa: F401
                ONTOLOGY_COLLECTIONS,
                VectorSearchService,
            )

            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) >= 1
            assert "lobster.services.vector" in str(dep_warnings[0].message)

    def test_shim_config_emits_deprecation_warning(self):
        """Importing from lobster.core.vector.config should warn."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from lobster.core.vector.config import VectorSearchConfig  # noqa: F401

            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) >= 1
            assert "lobster.services.vector" in str(dep_warnings[0].message)

    def test_shim_chromadb_backend_emits_deprecation_warning(self):
        """Importing from lobster.core.vector.backends.chromadb_backend should warn."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from lobster.core.vector.backends.chromadb_backend import (  # noqa: F401
                ONTOLOGY_TARBALLS,
                ChromaDBBackend,
            )

            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) >= 1
            assert "lobster.services.vector" in str(dep_warnings[0].message)

    def test_shim_exports_match_real_module(self):
        """Shim exports should be the same objects as the real module exports."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from lobster.core.vector.service import (
                VectorSearchService as ShimService,
            )

        from lobster.services.vector.service import (
            VectorSearchService as RealService,
        )

        assert ShimService is RealService
