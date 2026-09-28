"""Contract tests for lobster-vector package boundaries."""

import subprocess
import sys
import textwrap
import warnings

import pytest

SUBPACKAGE_ABCS = [
    ("backends", "BaseVectorBackend"),
    ("embeddings", "BaseEmbedder"),
    ("rerankers", "BaseReranker"),
]


def _run_in_fresh_process(code: str) -> str:
    """Run code in a clean interpreter and return its stdout.

    In-process sys.modules surgery cannot prove import-time behaviour: the three
    broken subpackage __init__.py files these tests guard against passed the whole
    in-process suite, because every test reached the submodules directly and never
    touched the subpackage exports.
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"fresh interpreter exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result.stdout


class TestSubpackageExports:
    """Each subpackage must bind every name it advertises in __all__.

    backends/ and embeddings/ once declared __all__ for a name they never imported,
    so `from lobster.vector.backends import *` raised AttributeError. rerankers/ had
    no __all__ at all, so the same wildcard bound nothing and failed silently.
    """

    @pytest.mark.parametrize("subpackage,name", SUBPACKAGE_ABCS)
    def test_direct_import_is_the_base_class(self, subpackage, name):
        out = _run_in_fresh_process(f"""
            from lobster.vector.{subpackage} import {name}
            from lobster.vector.{subpackage}.base import {name} as from_base
            assert {name} is from_base, "subpackage export is not the class from base.py"
            print("ok")
            """)
        assert "ok" in out

    @pytest.mark.parametrize("subpackage,name", SUBPACKAGE_ABCS)
    def test_wildcard_import_binds_the_abc(self, subpackage, name):
        out = _run_in_fresh_process(f"""
            from lobster.vector.{subpackage} import *
            assert "{name}" in dir(), "wildcard import did not bind {name}"
            print("ok")
            """)
        assert "ok" in out

    @pytest.mark.parametrize("subpackage,name", SUBPACKAGE_ABCS)
    def test_every_all_entry_resolves(self, subpackage, name):
        out = _run_in_fresh_process(f"""
            import importlib

            mod = importlib.import_module("lobster.vector.{subpackage}")
            exported = getattr(mod, "__all__", None)
            assert exported, "__all__ is missing or empty"
            unbound = [n for n in exported if not hasattr(mod, n)]
            assert not unbound, f"__all__ names never bound: {{unbound}}"
            print("ok")
            """)
        assert "ok" in out

    def test_subpackage_imports_stay_lazy(self):
        out = _run_in_fresh_process("""
            import sys

            import lobster.vector.backends  # noqa: F401
            import lobster.vector.embeddings  # noqa: F401
            import lobster.vector.rerankers  # noqa: F401

            heavy = [
                "chromadb",
                "faiss",
                "psycopg2",
                "torch",
                "sentence_transformers",
                "transformers",
                "cohere",
            ]
            loaded = [m for m in heavy if m in sys.modules]
            assert not loaded, f"binding the ABCs pulled in heavy deps: {loaded}"
            print("ok")
            """)
        assert "ok" in out


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
        mods_to_clear = [
            k for k in sys.modules if k.startswith("lobster.services.vector")
        ]
        for m in mods_to_clear:
            del sys.modules[m]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from lobster.services.vector import VectorSearchService  # noqa: F401

            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) >= 1
            assert "lobster.vector" in str(deprecation_warnings[0].message)


class TestLazyLoading:
    """NFR-1: No heavy deps loaded on bare import."""

    def test_no_chromadb_on_import(self):
        mods_to_clear = [
            k for k in sys.modules if "lobster.vector" in k or "chromadb" in k
        ]
        for m in mods_to_clear:
            sys.modules.pop(m, None)

        import lobster.vector  # noqa: F401

        assert "chromadb" not in sys.modules

    def test_no_torch_on_import(self):
        mods_to_clear = [
            k for k in sys.modules if "lobster.vector" in k or "torch" in k
        ]
        for m in mods_to_clear:
            sys.modules.pop(m, None)

        import lobster.vector  # noqa: F401

        assert "torch" not in sys.modules
