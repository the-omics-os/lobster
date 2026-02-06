"""
Unit tests for vector search backends.

Tests ChromaDB and FAISS backend implementations.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

# Check for optional dependencies
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    import chromadb
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False


class TestBaseVectorBackend:
    """Tests for BaseVectorBackend interface."""

    def test_interface_contract(self):
        """Verify the abstract interface defines required methods."""
        from lobster.services.search.backends.base import BaseVectorBackend

        # Check abstract methods exist
        assert hasattr(BaseVectorBackend, "name")
        assert hasattr(BaseVectorBackend, "add_documents")
        assert hasattr(BaseVectorBackend, "search")
        assert hasattr(BaseVectorBackend, "delete")
        assert hasattr(BaseVectorBackend, "count")


@pytest.mark.skipif(not HAS_FAISS, reason="FAISS not installed (pip install lobster-ai[search])")
class TestFAISSBackend:
    """Tests for FAISSBackend."""

    @pytest.fixture
    def mock_faiss(self):
        """Mock FAISS import."""
        with patch.dict("sys.modules", {"faiss": MagicMock()}):
            yield

    def test_initialization(self):
        """Test backend initializes with dimension."""
        from lobster.services.search.backends.faiss_backend import FAISSBackend

        backend = FAISSBackend(dimension=384)

        assert backend.dimension == 384
        assert backend.name == "faiss:memory"
        assert backend._index is None  # Lazy initialization

    def test_add_documents(self):
        """Test adding documents to index."""
        from lobster.services.search.backends.faiss_backend import FAISSBackend

        # Mock FAISS
        with patch("faiss.IndexFlatL2") as mock_index_class:
            mock_index = MagicMock()
            mock_index.ntotal = 0
            mock_index_class.return_value = mock_index

            backend = FAISSBackend(dimension=4)

            ids = ["doc1", "doc2"]
            embeddings = [np.array([1, 0, 0, 0]), np.array([0, 1, 0, 0])]
            texts = ["First document", "Second document"]
            metadatas = [{"key": "value1"}, {"key": "value2"}]

            backend.add_documents(ids, embeddings, texts, metadatas)

            assert backend.count() == 2
            assert "doc1" in backend._documents
            assert "doc2" in backend._documents

    def test_add_documents_validation(self):
        """Test add_documents validates input lengths."""
        from lobster.services.search.backends.faiss_backend import FAISSBackend

        with patch("faiss.IndexFlatL2"):
            backend = FAISSBackend(dimension=4)

            with pytest.raises(ValueError) as exc_info:
                backend.add_documents(
                    ids=["doc1", "doc2"],
                    embeddings=[np.array([1, 0, 0, 0])],  # Wrong length
                    texts=["Text"],
                )

            assert "same length" in str(exc_info.value)

    def test_search(self):
        """Test searching documents."""
        from lobster.services.search.backends.faiss_backend import FAISSBackend

        # Mock FAISS
        with patch("faiss.IndexFlatL2") as mock_index_class:
            mock_index = MagicMock()
            mock_index.ntotal = 2
            mock_index.search.return_value = (
                np.array([[0.0, 0.5]]),  # distances
                np.array([[0, 1]]),  # indices
            )
            mock_index_class.return_value = mock_index

            backend = FAISSBackend(dimension=4)

            # Add documents first
            backend._documents = {
                "doc1": {"content": "First", "metadata": {}},
                "doc2": {"content": "Second", "metadata": {}},
            }
            backend._idx_to_id = {0: "doc1", 1: "doc2"}
            backend._id_to_idx = {"doc1": 0, "doc2": 1}
            backend._index = mock_index

            query = np.array([1, 0, 0, 0])
            results = backend.search(query, k=2)

            assert len(results) == 2
            assert results[0]["id"] == "doc1"
            assert results[1]["id"] == "doc2"
            assert "similarity_score" in results[0]

    def test_search_empty_index(self):
        """Test search on empty index returns empty list."""
        from lobster.services.search.backends.faiss_backend import FAISSBackend

        with patch("faiss.IndexFlatL2") as mock_index_class:
            mock_index = MagicMock()
            mock_index.ntotal = 0
            mock_index_class.return_value = mock_index

            backend = FAISSBackend(dimension=4)
            results = backend.search(np.array([1, 0, 0, 0]), k=5)

            assert results == []

    def test_delete(self):
        """Test deleting documents."""
        from lobster.services.search.backends.faiss_backend import FAISSBackend

        with patch("faiss.IndexFlatL2"):
            backend = FAISSBackend(dimension=4)
            backend._documents = {"doc1": {"content": "Test", "metadata": {}}}
            backend._id_to_idx = {"doc1": 0}
            backend._idx_to_id = {0: "doc1"}

            backend.delete(["doc1"])

            assert "doc1" not in backend._documents
            assert backend.count() == 0

    def test_clear(self):
        """Test clearing all documents."""
        from lobster.services.search.backends.faiss_backend import FAISSBackend

        with patch("faiss.IndexFlatL2"):
            backend = FAISSBackend(dimension=4)
            backend._documents = {"doc1": {}, "doc2": {}}
            backend._index = MagicMock()

            backend.clear()

            assert backend.count() == 0
            assert backend._index is None

    def test_add_document_convenience(self):
        """Test add_document convenience method."""
        from lobster.services.search.backends.faiss_backend import FAISSBackend

        with patch("faiss.IndexFlatL2") as mock_index_class:
            mock_index = MagicMock()
            mock_index.ntotal = 0
            mock_index_class.return_value = mock_index

            backend = FAISSBackend(dimension=4)
            backend.add_document(
                doc_id="doc1",
                text="Test document",
                embedding=np.array([1, 0, 0, 0]),
                metadata={"key": "value"},
            )

            assert backend.count() == 1


@pytest.mark.skipif(not HAS_CHROMADB, reason="ChromaDB not installed (pip install lobster-ai[search])")
class TestChromaBackend:
    """Tests for ChromaBackend."""

    def test_initialization(self):
        """Test backend initializes without loading."""
        from lobster.services.search.backends.chroma_backend import ChromaBackend

        backend = ChromaBackend(
            collection_name="test_collection",
            auto_download=False,
        )

        assert backend.collection_name == "test_collection"
        assert backend.name == "chroma:test_collection"
        assert backend._client is None  # Lazy initialization

    def test_initialization_custom_cache(self):
        """Test backend accepts custom cache directory."""
        from lobster.services.search.backends.chroma_backend import ChromaBackend

        backend = ChromaBackend(
            collection_name="test",
            cache_dir="/custom/path",
            auto_download=False,
        )

        assert str(backend.cache_dir) == "/custom/path"

    @patch("chromadb.PersistentClient")
    def test_add_documents(self, mock_client_class, tmp_path):
        """Test adding documents to collection."""
        from lobster.services.search.backends.chroma_backend import ChromaBackend

        # Setup mock
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client

        # Create collection directory
        collection_path = tmp_path / "test"
        collection_path.mkdir()

        backend = ChromaBackend(
            collection_name="test",
            cache_dir=str(tmp_path),
            auto_download=False,
        )

        ids = ["doc1", "doc2"]
        embeddings = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]
        texts = ["First", "Second"]

        backend.add_documents(ids, embeddings, texts)

        mock_collection.add.assert_called_once()

    @patch("chromadb.PersistentClient")
    def test_search(self, mock_client_class, tmp_path):
        """Test searching collection."""
        from lobster.services.search.backends.chroma_backend import ChromaBackend

        # Setup mock
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 2
        mock_collection.query.return_value = {
            "ids": [["doc1", "doc2"]],
            "documents": [["First", "Second"]],
            "metadatas": [[{"key": "v1"}, {"key": "v2"}]],
            "distances": [[0.1, 0.2]],
        }
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client

        # Create collection directory
        collection_path = tmp_path / "test"
        collection_path.mkdir()

        backend = ChromaBackend(
            collection_name="test",
            cache_dir=str(tmp_path),
            auto_download=False,
        )

        results = backend.search(np.array([0.1, 0.2]), k=2)

        assert len(results) == 2
        assert results[0]["id"] == "doc1"
        assert results[0]["content"] == "First"
        assert "similarity_score" in results[0]

    def test_missing_collection_no_autodownload(self, tmp_path):
        """Test raises error when collection missing and auto_download=False."""
        from lobster.services.search.backends.chroma_backend import ChromaBackend

        backend = ChromaBackend(
            collection_name="missing",
            cache_dir=str(tmp_path),
            auto_download=False,
        )

        with pytest.raises(FileNotFoundError) as exc_info:
            backend._ensure_downloaded()

        assert "not found" in str(exc_info.value)

    @patch("chromadb.PersistentClient")
    def test_delete(self, mock_client_class, tmp_path):
        """Test deleting documents."""
        from lobster.services.search.backends.chroma_backend import ChromaBackend

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client

        collection_path = tmp_path / "test"
        collection_path.mkdir()

        backend = ChromaBackend(
            collection_name="test",
            cache_dir=str(tmp_path),
            auto_download=False,
        )

        backend.delete(["doc1", "doc2"])

        mock_collection.delete.assert_called_once_with(ids=["doc1", "doc2"])
