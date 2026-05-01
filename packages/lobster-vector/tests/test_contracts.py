"""Contract tests for lobster-vector package boundaries."""
import sys
import warnings

import pytest


class TestCanonicalImport:
    """FR-1: Package independence — canonical import path works."""

    def test_import_service(self):
        from lobster.vector import VectorSearchService
        assert VectorSearchService is not None

    def test_import_config(self):
        from lobster.vector import VectorSearchConfig
        assert VectorSearchConfig is not None

    def test_import_collections(self):
        from lobster.vector import ONTOLOGY_COLLECTIONS
        assert isinstance(ONTOLOGY_COLLECTIONS, dict)
        assert "mondo" in ONTOLOGY_COLLECTIONS

    def test_import_artifact(self):
        from lobster.vector import ArtifactMetadata, CollectionUnavailable
        assert ArtifactMetadata is not None
        assert CollectionUnavailable is not None


class TestCompatImport:
    """FR-3: Backward compatibility — old path works with warning."""

    def test_compat_warns(self):
        mods_to_clear = [k for k in sys.modules if k.startswith("lobster.services.vector")]
        for m in mods_to_clear:
            del sys.modules[m]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from lobster.services.vector import VectorSearchService  # noqa: F401
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1
            assert "lobster.vector" in str(deprecation_warnings[0].message)


class TestLazyLoading:
    """NFR-1: No heavy deps loaded on bare import."""

    def test_no_chromadb_on_import(self):
        mods_to_clear = [k for k in sys.modules if "lobster.vector" in k or "chromadb" in k]
        for m in mods_to_clear:
            sys.modules.pop(m, None)

        import lobster.vector  # noqa: F401
        assert "chromadb" not in sys.modules

    def test_no_torch_on_import(self):
        mods_to_clear = [k for k in sys.modules if "lobster.vector" in k or "torch" in k]
        for m in mods_to_clear:
            sys.modules.pop(m, None)

        import lobster.vector  # noqa: F401
        assert "torch" not in sys.modules
