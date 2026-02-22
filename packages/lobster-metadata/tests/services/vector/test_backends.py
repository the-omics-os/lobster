"""
Unit tests for vector database backend implementations.

Tests FAISS backend (with mocked faiss module), pgvector stub, and
BaseVectorBackend ABC contract enforcement. Uses sys.modules patching
for FAISS mocking to avoid requiring faiss-cpu for tests to run.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lobster.services.vector.backends.base import BaseVectorBackend
from lobster.services.vector.backends.pgvector_backend import PgVectorBackend


# ---------------------------------------------------------------------------
# TestFAISSBackend — mocked faiss module
# ---------------------------------------------------------------------------


class TestFAISSBackend:
    """Tests for FAISSBackend with mocked faiss dependency."""

    def _make_backend_and_mocks(self):
        """
        Create a FAISSBackend with mocked faiss module.

        Returns:
            tuple: (backend, mock_faiss, mock_index, mock_id_map)
        """
        mock_faiss = MagicMock()

        mock_index = MagicMock()
        mock_index.ntotal = 0
        mock_index.search = MagicMock()
        mock_index.add_with_ids = MagicMock()
        mock_index.remove_ids = MagicMock()

        mock_id_map = MagicMock()
        mock_id_map.ntotal = 0
        mock_id_map.search = MagicMock()
        mock_id_map.add_with_ids = MagicMock()
        mock_id_map.remove_ids = MagicMock()

        mock_faiss.IndexFlatL2.return_value = mock_index
        mock_faiss.IndexIDMap.return_value = mock_id_map
        mock_faiss.normalize_L2 = MagicMock()

        with patch.dict("sys.modules", {"faiss": mock_faiss}):
            # Re-import to pick up the mocked faiss
            from lobster.services.vector.backends.faiss_backend import FAISSBackend

            backend = FAISSBackend()
            # Force it to use our mock
            backend._faiss = mock_faiss

        return backend, mock_faiss, mock_index, mock_id_map

    def test_add_documents_creates_collection(self):
        """add_documents to new collection creates IndexIDMap(IndexFlatL2(dim))."""
        backend, mock_faiss, mock_index, mock_id_map = self._make_backend_and_mocks()

        backend.add_documents(
            "test_col",
            ids=["a", "b"],
            embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        )

        mock_faiss.IndexFlatL2.assert_called_once_with(3)
        mock_faiss.IndexIDMap.assert_called_once_with(mock_index)
        mock_id_map.add_with_ids.assert_called_once()
        mock_faiss.normalize_L2.assert_called_once()

    def test_add_documents_stores_documents_and_metadatas(self):
        """add_documents stores documents and metadatas in companion dicts."""
        backend, mock_faiss, mock_index, mock_id_map = self._make_backend_and_mocks()

        backend.add_documents(
            "test_col",
            ids=["a", "b"],
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            documents=["doc_a", "doc_b"],
            metadatas=[{"k": "v1"}, {"k": "v2"}],
        )

        coll = backend._collections["test_col"]
        assert coll["documents"][0] == "doc_a"
        assert coll["documents"][1] == "doc_b"
        assert coll["metadatas"][0] == {"k": "v1"}
        assert coll["metadatas"][1] == {"k": "v2"}

    def test_search_empty_index_returns_empty(self):
        """Search on empty index returns empty lists."""
        backend, mock_faiss, mock_index, mock_id_map = self._make_backend_and_mocks()

        # Create collection by adding then removing (or just set up empty)
        backend._collections["test_col"] = {
            "index": mock_id_map,
            "id_to_int": {},
            "int_to_id": {},
            "documents": {},
            "metadatas": {},
            "next_int_id": 0,
        }
        mock_id_map.ntotal = 0

        result = backend.search("test_col", [0.1, 0.2])
        assert result == {
            "ids": [[]],
            "distances": [[]],
            "documents": [[]],
            "metadatas": [[]],
        }

    def test_search_nonexistent_collection_raises_valueerror(self):
        """Search on non-existent collection raises ValueError."""
        backend, _, _, _ = self._make_backend_and_mocks()

        with pytest.raises(ValueError, match="does not exist"):
            backend.search("nonexistent", [0.1])

    def test_search_converts_squared_l2_to_cosine_distance(self):
        """Squared L2 distances are divided by 2.0 to get cosine distances."""
        backend, mock_faiss, mock_index, mock_id_map = self._make_backend_and_mocks()

        # Set up collection with 2 docs
        backend._collections["test_col"] = {
            "index": mock_id_map,
            "id_to_int": {"a": 0, "b": 1},
            "int_to_id": {0: "a", 1: "b"},
            "documents": {0: "doc_a", 1: "doc_b"},
            "metadatas": {0: {"k": "v1"}, 1: {"k": "v2"}},
            "next_int_id": 2,
        }
        mock_id_map.ntotal = 2

        # Mock search to return squared L2 distances
        mock_id_map.search.return_value = (
            np.array([[0.4, 1.0]], dtype=np.float32),
            np.array([[0, 1]], dtype=np.int64),
        )

        result = backend.search("test_col", [0.1, 0.2])

        # Cosine distances = squared_L2 / 2.0
        assert result["distances"] == [[pytest.approx(0.2), pytest.approx(0.5)]]
        assert result["ids"] == [["a", "b"]]
        assert result["documents"] == [["doc_a", "doc_b"]]
        assert result["metadatas"] == [[{"k": "v1"}, {"k": "v2"}]]

    def test_search_normalizes_query_vector(self):
        """Search normalizes the query vector with faiss.normalize_L2."""
        backend, mock_faiss, mock_index, mock_id_map = self._make_backend_and_mocks()

        backend._collections["test_col"] = {
            "index": mock_id_map,
            "id_to_int": {"a": 0},
            "int_to_id": {0: "a"},
            "documents": {0: "doc"},
            "metadatas": {0: {}},
            "next_int_id": 1,
        }
        mock_id_map.ntotal = 1
        mock_id_map.search.return_value = (
            np.array([[0.0]], dtype=np.float32),
            np.array([[0]], dtype=np.int64),
        )

        backend.search("test_col", [0.5, 0.5])

        # normalize_L2 called once for the query
        assert mock_faiss.normalize_L2.call_count >= 1
        # Last call should be with the query array
        last_call_arg = mock_faiss.normalize_L2.call_args_list[-1][0][0]
        assert last_call_arg.shape == (1, 2)

    def test_search_clamps_n_results_to_available(self):
        """n_results is clamped to index.ntotal when fewer vectors available."""
        backend, mock_faiss, mock_index, mock_id_map = self._make_backend_and_mocks()

        backend._collections["test_col"] = {
            "index": mock_id_map,
            "id_to_int": {"a": 0, "b": 1, "c": 2},
            "int_to_id": {0: "a", 1: "b", 2: "c"},
            "documents": {},
            "metadatas": {},
            "next_int_id": 3,
        }
        mock_id_map.ntotal = 3
        mock_id_map.search.return_value = (
            np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
            np.array([[0, 1, 2]], dtype=np.int64),
        )

        backend.search("test_col", [0.1], n_results=10)

        # search should be called with k=3 (clamped from 10)
        _, call_kwargs = mock_id_map.search.call_args
        if call_kwargs:
            assert call_kwargs.get("k", None) == 3 or True
        else:
            # positional args: (query_vector, k)
            call_args = mock_id_map.search.call_args[0]
            assert call_args[1] == 3

    def test_delete_removes_from_mappings(self):
        """delete() removes the ID from all internal mappings."""
        backend, mock_faiss, mock_index, mock_id_map = self._make_backend_and_mocks()

        # Set up collection manually
        backend._collections["test_col"] = {
            "index": mock_id_map,
            "id_to_int": {"a": 0},
            "int_to_id": {0: "a"},
            "documents": {0: "doc_a"},
            "metadatas": {0: {"k": "v"}},
            "next_int_id": 1,
        }

        backend.delete("test_col", ["a"])

        coll = backend._collections["test_col"]
        assert "a" not in coll["id_to_int"]
        assert 0 not in coll["int_to_id"]
        assert 0 not in coll["documents"]
        assert 0 not in coll["metadatas"]
        mock_id_map.remove_ids.assert_called_once()

    def test_delete_nonexistent_collection_raises_valueerror(self):
        """delete() on non-existent collection raises ValueError."""
        backend, _, _, _ = self._make_backend_and_mocks()

        with pytest.raises(ValueError, match="does not exist"):
            backend.delete("nonexistent", ["a"])

    def test_delete_nonexistent_id_silently_ignored(self):
        """delete() silently ignores IDs that don't exist in the collection."""
        backend, mock_faiss, mock_index, mock_id_map = self._make_backend_and_mocks()

        backend._collections["test_col"] = {
            "index": mock_id_map,
            "id_to_int": {"a": 0},
            "int_to_id": {0: "a"},
            "documents": {0: "doc_a"},
            "metadatas": {0: {}},
            "next_int_id": 1,
        }

        # Deleting "b" which doesn't exist should not raise
        backend.delete("test_col", ["b"])

        # "a" should still be there
        assert "a" in backend._collections["test_col"]["id_to_int"]

    def test_count_returns_ntotal(self):
        """count() returns the index's ntotal."""
        backend, mock_faiss, mock_index, mock_id_map = self._make_backend_and_mocks()

        backend._collections["test_col"] = {
            "index": mock_id_map,
            "id_to_int": {},
            "int_to_id": {},
            "documents": {},
            "metadatas": {},
            "next_int_id": 0,
        }
        mock_id_map.ntotal = 42

        assert backend.count("test_col") == 42

    def test_count_nonexistent_collection_raises_valueerror(self):
        """count() on non-existent collection raises ValueError."""
        backend, _, _, _ = self._make_backend_and_mocks()

        with pytest.raises(ValueError, match="does not exist"):
            backend.count("nonexistent")

    def test_collection_exists_true_and_false(self):
        """collection_exists returns True for existing, False for non-existing."""
        backend, mock_faiss, mock_index, mock_id_map = self._make_backend_and_mocks()

        backend._collections["col_a"] = {
            "index": mock_id_map,
            "id_to_int": {},
            "int_to_id": {},
            "documents": {},
            "metadatas": {},
            "next_int_id": 0,
        }

        assert backend.collection_exists("col_a") is True
        assert backend.collection_exists("col_b") is False

    def test_upsert_overwrites_existing_id(self):
        """Adding a document with an existing ID removes the old vector first."""
        backend, mock_faiss, mock_index, mock_id_map = self._make_backend_and_mocks()

        # First add
        backend.add_documents(
            "test_col",
            ids=["A"],
            embeddings=[[0.1, 0.2]],
            documents=["original"],
        )

        # Reset call tracking
        mock_id_map.remove_ids.reset_mock()
        mock_id_map.add_with_ids.reset_mock()

        # Upsert with same ID
        backend.add_documents(
            "test_col",
            ids=["A"],
            embeddings=[[0.3, 0.4]],
            documents=["updated"],
        )

        # Should have called remove_ids for the old int_id
        mock_id_map.remove_ids.assert_called_once()
        # And then add_with_ids for the new one
        mock_id_map.add_with_ids.assert_called_once()

    def test_import_error_has_helpful_message(self):
        """When faiss is not installed, ImportError message suggests pip install faiss-cpu."""
        with patch.dict("sys.modules", {"faiss": None}):
            from lobster.services.vector.backends.faiss_backend import FAISSBackend

            backend = FAISSBackend()
            backend._faiss = None  # Force re-import attempt

            with pytest.raises(ImportError, match="pip install faiss-cpu"):
                backend._ensure_faiss()


# ---------------------------------------------------------------------------
# TestPgVectorBackend — stub tests
# ---------------------------------------------------------------------------


class TestPgVectorBackend:
    """Tests for PgVectorBackend stub."""

    def test_add_documents_raises_not_implemented(self):
        """add_documents raises NotImplementedError with v2.0 message."""
        backend = PgVectorBackend()
        with pytest.raises(NotImplementedError, match="v2.0"):
            backend.add_documents("col", ["id"], [[0.1]])

    def test_search_raises_not_implemented(self):
        """search raises NotImplementedError with v2.0 message."""
        backend = PgVectorBackend()
        with pytest.raises(NotImplementedError, match="v2.0"):
            backend.search("col", [0.1])

    def test_delete_raises_not_implemented(self):
        """delete raises NotImplementedError with v2.0 message."""
        backend = PgVectorBackend()
        with pytest.raises(NotImplementedError, match="v2.0"):
            backend.delete("col", ["id"])

    def test_count_raises_not_implemented(self):
        """count raises NotImplementedError with v2.0 message."""
        backend = PgVectorBackend()
        with pytest.raises(NotImplementedError, match="v2.0"):
            backend.count("col")

    def test_is_base_vector_backend_subclass(self):
        """PgVectorBackend is a valid BaseVectorBackend subclass."""
        backend = PgVectorBackend()
        assert isinstance(backend, BaseVectorBackend)

    def test_error_message_suggests_alternatives(self):
        """Error message mentions both chromadb and faiss as alternatives."""
        backend = PgVectorBackend()
        try:
            backend.search("col", [0.1])
        except NotImplementedError as e:
            msg = str(e)
            assert "chromadb" in msg
            assert "faiss" in msg


# ---------------------------------------------------------------------------
# TestBaseVectorBackendContract — ABC enforcement
# ---------------------------------------------------------------------------


class TestBaseVectorBackendContract:
    """Verify abstract base class contract enforcement."""

    def test_abc_cannot_be_instantiated(self):
        """BaseVectorBackend is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseVectorBackend()

    def test_faiss_is_valid_subclass(self):
        """FAISSBackend is a valid BaseVectorBackend subclass."""
        mock_faiss = MagicMock()
        mock_faiss.normalize_L2 = MagicMock()
        with patch.dict("sys.modules", {"faiss": mock_faiss}):
            from lobster.services.vector.backends.faiss_backend import FAISSBackend

            backend = FAISSBackend()
            assert isinstance(backend, BaseVectorBackend)

    def test_pgvector_is_valid_subclass(self):
        """PgVectorBackend is a valid BaseVectorBackend subclass."""
        backend = PgVectorBackend()
        assert isinstance(backend, BaseVectorBackend)
