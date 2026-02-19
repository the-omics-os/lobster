"""
Unit tests for reranker implementations.

Tests CrossEncoderReranker and CohereReranker with mocked dependencies
(no torch, no Cohere API needed). Validates lazy loading, graceful
degradation, score normalization, and the MockReranker helper used by
service integration tests.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from lobster.core.vector.rerankers.base import BaseReranker, normalize_scores


# ---------------------------------------------------------------------------
# MockReranker (shared test helper)
# ---------------------------------------------------------------------------


class MockReranker(BaseReranker):
    """
    Test helper reranker that reverses document order.

    Produces known scores for deterministic assertions. Exposes
    _last_query and _last_documents for service integration tests.

    Args:
        reverse: If True (default), reverse the document order to
            simulate reranking. If False, preserve original order.
    """

    def __init__(self, reverse: bool = True) -> None:
        self._reverse = reverse
        self._last_query: str | None = None
        self._last_documents: list[str] | None = None

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        self._last_query = query
        self._last_documents = list(documents)

        indices = list(range(len(documents)))
        if self._reverse:
            indices = indices[::-1]

        results = []
        for rank, idx in enumerate(indices):
            # Descending scores: 1.0, 0.9, 0.8, ...
            score = max(0.0, 1.0 - rank * 0.1)
            results.append(
                {
                    "corpus_id": idx,
                    "score": score,
                    "text": documents[idx],
                }
            )

        if top_k is not None:
            results = results[:top_k]

        return results


# ---------------------------------------------------------------------------
# TestBaseRerankerContract
# ---------------------------------------------------------------------------


class TestBaseRerankerContract:
    """Verify abstract base class contract and normalize_scores helper."""

    def test_abc_cannot_be_instantiated(self):
        """BaseReranker is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseReranker()

    def test_normalize_scores_basic(self):
        """normalize_scores maps [8.6, 3.4, -1.2] to [1.0, ~0.47, 0.0]."""
        results = [
            {"corpus_id": 0, "score": 8.6, "text": "a"},
            {"corpus_id": 1, "score": 3.4, "text": "b"},
            {"corpus_id": 2, "score": -1.2, "text": "c"},
        ]
        normalized = normalize_scores(results)

        assert normalized[0]["score"] == 1.0
        assert normalized[2]["score"] == 0.0
        # Middle: (3.4 - (-1.2)) / (8.6 - (-1.2)) = 4.6 / 9.8 = 0.4694
        assert abs(normalized[1]["score"] - 0.4694) < 0.001

    def test_normalize_scores_equal_values(self):
        """All same score -> all 1.0."""
        results = [
            {"corpus_id": 0, "score": 5.0, "text": "a"},
            {"corpus_id": 1, "score": 5.0, "text": "b"},
            {"corpus_id": 2, "score": 5.0, "text": "c"},
        ]
        normalized = normalize_scores(results)
        for r in normalized:
            assert r["score"] == 1.0

    def test_normalize_scores_empty_list(self):
        """Empty input -> empty output."""
        assert normalize_scores([]) == []

    def test_normalize_scores_single_item(self):
        """Single item -> score 1.0 (equal min and max)."""
        results = [{"corpus_id": 0, "score": 3.7, "text": "only"}]
        normalized = normalize_scores(results)
        assert normalized[0]["score"] == 1.0


# ---------------------------------------------------------------------------
# TestCrossEncoderReranker
# ---------------------------------------------------------------------------


class TestCrossEncoderReranker:
    """Tests for CrossEncoderReranker with mocked sentence-transformers."""

    def _make_reranker(self):
        """Create a CrossEncoderReranker instance (lazy import)."""
        from lobster.core.vector.rerankers.cross_encoder_reranker import (
            CrossEncoderReranker,
        )

        return CrossEncoderReranker()

    def test_model_not_loaded_on_init(self):
        """CrossEncoderReranker should NOT load model on construction."""
        reranker = self._make_reranker()
        assert reranker._model is None

    def test_lazy_loading_on_first_rerank(self):
        """First rerank() call should trigger model loading."""
        reranker = self._make_reranker()

        mock_model = MagicMock()
        mock_model.rank.return_value = [
            {"corpus_id": 0, "score": 8.6, "text": "doc1"},
            {"corpus_id": 1, "score": 3.4, "text": "doc2"},
        ]

        mock_ce_cls = MagicMock(return_value=mock_model)

        with patch.dict(
            "sys.modules",
            {
                "sentence_transformers": MagicMock(CrossEncoder=mock_ce_cls),
            },
        ):
            reranker._load_model()
            mock_ce_cls.assert_called_once_with(
                "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )

    def test_model_loaded_once(self):
        """Multiple rerank() calls should load model only once."""
        reranker = self._make_reranker()

        mock_model = MagicMock()
        mock_model.rank.return_value = [
            {"corpus_id": 0, "score": 5.0, "text": "a"},
            {"corpus_id": 1, "score": 3.0, "text": "b"},
        ]
        reranker._model = mock_model

        reranker.rerank("q", ["a", "b"])
        reranker.rerank("q", ["c", "d"])
        assert mock_model.rank.call_count == 2

    def test_rerank_returns_sorted_results(self):
        """rerank() output should match contract: corpus_id, score, text keys."""
        reranker = self._make_reranker()

        mock_model = MagicMock()
        mock_model.rank.return_value = [
            {"corpus_id": 1, "score": 9.2, "text": "best"},
            {"corpus_id": 0, "score": 3.1, "text": "okay"},
        ]
        reranker._model = mock_model

        results = reranker.rerank("query", ["okay", "best"])
        assert len(results) == 2
        assert results[0]["corpus_id"] == 1
        assert results[0]["score"] == 9.2
        assert results[0]["text"] == "best"
        assert results[1]["corpus_id"] == 0

    def test_rerank_top_k_passed_to_model(self):
        """top_k should be forwarded to model.rank()."""
        reranker = self._make_reranker()

        mock_model = MagicMock()
        mock_model.rank.return_value = [
            {"corpus_id": 0, "score": 5.0, "text": "a"},
        ]
        reranker._model = mock_model

        reranker.rerank("q", ["a", "b", "c"], top_k=3)
        _, kwargs = mock_model.rank.call_args
        assert kwargs.get("top_k") == 3

    def test_rerank_empty_documents(self):
        """Empty documents list -> empty result, no model loaded."""
        reranker = self._make_reranker()
        results = reranker.rerank("q", [])
        assert results == []
        assert reranker._model is None

    def test_rerank_single_document(self):
        """Single document -> returned as-is with score=1.0, no model loaded."""
        reranker = self._make_reranker()
        results = reranker.rerank("q", ["only_doc"])
        assert len(results) == 1
        assert results[0]["corpus_id"] == 0
        assert results[0]["score"] == 1.0
        assert results[0]["text"] == "only_doc"
        assert reranker._model is None

    def test_import_error_has_helpful_message(self):
        """ImportError should mention pip install and vector-search."""
        reranker = self._make_reranker()
        reranker._model = None

        original_import = __import__

        def mock_import(name, *args, **kwargs):
            if name == "sentence_transformers":
                raise ImportError("No module named 'sentence_transformers'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="pip install"):
                reranker._load_model()


# ---------------------------------------------------------------------------
# TestCohereReranker
# ---------------------------------------------------------------------------


class TestCohereReranker:
    """Tests for CohereReranker with mocked Cohere client."""

    def _make_reranker(self):
        """Create a CohereReranker instance."""
        from lobster.core.vector.rerankers.cohere_reranker import CohereReranker

        return CohereReranker()

    def test_no_api_key_logs_warning_returns_original_order(
        self, monkeypatch, caplog
    ):
        """Missing API key: logs warning, returns original order."""
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        monkeypatch.delenv("CO_API_KEY", raising=False)

        reranker = self._make_reranker()

        import logging

        with caplog.at_level(logging.WARNING):
            results = reranker.rerank("q", ["a", "b", "c"])

        assert len(results) == 3
        # Original order preserved
        assert results[0]["text"] == "a"
        assert results[1]["text"] == "b"
        assert results[2]["text"] == "c"
        assert "API key not found" in caplog.text

    def test_no_api_key_returns_synthetic_scores(self, monkeypatch):
        """Degraded path returns descending synthetic scores."""
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        monkeypatch.delenv("CO_API_KEY", raising=False)

        reranker = self._make_reranker()
        results = reranker.rerank("q", ["a", "b", "c"])

        # Scores should be 1.0, 0.99, 0.98
        assert results[0]["score"] == 1.0
        assert results[1]["score"] == 0.99
        assert results[2]["score"] == 0.98

    def test_api_key_present_calls_client(self, monkeypatch):
        """With API key, Cohere client should be initialized and called."""
        monkeypatch.setenv("COHERE_API_KEY", "test-key-123")

        # Mock the cohere module
        mock_result_0 = MagicMock()
        mock_result_0.index = 1
        mock_result_0.relevance_score = 0.95
        mock_result_1 = MagicMock()
        mock_result_1.index = 0
        mock_result_1.relevance_score = 0.72

        mock_response = MagicMock()
        mock_response.results = [mock_result_0, mock_result_1]

        mock_client = MagicMock()
        mock_client.rerank.return_value = mock_response

        mock_cohere = MagicMock()
        mock_cohere.ClientV2.return_value = mock_client

        with patch.dict("sys.modules", {"cohere": mock_cohere}):
            reranker = self._make_reranker()
            results = reranker.rerank("q", ["a", "b"])

        mock_client.rerank.assert_called_once()

    def test_cohere_response_format(self, monkeypatch):
        """Verify output format matches contract (corpus_id, score, text)."""
        monkeypatch.setenv("COHERE_API_KEY", "test-key")

        mock_result_0 = MagicMock()
        mock_result_0.index = 1
        mock_result_0.relevance_score = 0.95
        mock_result_1 = MagicMock()
        mock_result_1.index = 0
        mock_result_1.relevance_score = 0.72

        mock_response = MagicMock()
        mock_response.results = [mock_result_0, mock_result_1]

        mock_client = MagicMock()
        mock_client.rerank.return_value = mock_response

        mock_cohere = MagicMock()
        mock_cohere.ClientV2.return_value = mock_client

        with patch.dict("sys.modules", {"cohere": mock_cohere}):
            reranker = self._make_reranker()
            results = reranker.rerank("q", ["first", "second"])

        assert len(results) == 2
        assert results[0]["corpus_id"] == 1
        assert results[0]["score"] == 0.95
        assert results[0]["text"] == "second"
        assert results[1]["corpus_id"] == 0
        assert results[1]["score"] == 0.72
        assert results[1]["text"] == "first"

    def test_cohere_top_k_passed_as_top_n(self, monkeypatch):
        """top_k should be forwarded to client.rerank() as top_n."""
        monkeypatch.setenv("COHERE_API_KEY", "test-key")

        mock_response = MagicMock()
        mock_response.results = []
        mock_client = MagicMock()
        mock_client.rerank.return_value = mock_response
        mock_cohere = MagicMock()
        mock_cohere.ClientV2.return_value = mock_client

        with patch.dict("sys.modules", {"cohere": mock_cohere}):
            reranker = self._make_reranker()
            reranker.rerank("q", ["a", "b", "c"], top_k=2)

        _, kwargs = mock_client.rerank.call_args
        assert kwargs.get("top_n") == 2

    def test_cohere_model_name_default(self, monkeypatch):
        """Default model should be rerank-v4.0-pro."""
        monkeypatch.setenv("COHERE_API_KEY", "test-key")
        monkeypatch.delenv("COHERE_RERANK_MODEL", raising=False)

        mock_response = MagicMock()
        mock_response.results = []
        mock_client = MagicMock()
        mock_client.rerank.return_value = mock_response
        mock_cohere = MagicMock()
        mock_cohere.ClientV2.return_value = mock_client

        with patch.dict("sys.modules", {"cohere": mock_cohere}):
            reranker = self._make_reranker()
            reranker.rerank("q", ["a", "b"])

        _, kwargs = mock_client.rerank.call_args
        assert kwargs.get("model") == "rerank-v4.0-pro"

    def test_cohere_model_env_override(self, monkeypatch):
        """COHERE_RERANK_MODEL env var should override default model."""
        monkeypatch.setenv("COHERE_API_KEY", "test-key")
        monkeypatch.setenv("COHERE_RERANK_MODEL", "rerank-v3.5-custom")

        mock_response = MagicMock()
        mock_response.results = []
        mock_client = MagicMock()
        mock_client.rerank.return_value = mock_response
        mock_cohere = MagicMock()
        mock_cohere.ClientV2.return_value = mock_client

        with patch.dict("sys.modules", {"cohere": mock_cohere}):
            reranker = self._make_reranker()
            reranker.rerank("q", ["a", "b"])

        _, kwargs = mock_client.rerank.call_args
        assert kwargs.get("model") == "rerank-v3.5-custom"

    def test_cohere_import_error_degrades(self, monkeypatch, caplog):
        """Missing cohere package: logs warning and degrades gracefully."""
        monkeypatch.setenv("COHERE_API_KEY", "test-key")

        # Remove cohere from sys.modules if present
        original_import = __import__

        def mock_import(name, *args, **kwargs):
            if name == "cohere":
                raise ImportError("No module named 'cohere'")
            return original_import(name, *args, **kwargs)

        import logging

        with caplog.at_level(logging.WARNING):
            with patch("builtins.__import__", side_effect=mock_import):
                reranker = self._make_reranker()
                results = reranker.rerank("q", ["a", "b", "c"])

        assert len(results) == 3
        # Should degrade to original order with synthetic scores
        assert results[0]["text"] == "a"
        assert "not installed" in caplog.text

    def test_available_checked_once(self, monkeypatch):
        """_init_client() logic should only run once (cached _available flag)."""
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        monkeypatch.delenv("CO_API_KEY", raising=False)

        reranker = self._make_reranker()
        assert reranker._available is None  # Not checked yet

        reranker.rerank("q", ["a", "b"])
        assert reranker._available is False  # Checked: no key

        # Second call should not re-check
        reranker.rerank("q", ["c", "d"])
        assert reranker._available is False  # Still cached

    def test_empty_documents(self):
        """Empty document list -> empty result returned."""
        reranker = self._make_reranker()
        results = reranker.rerank("q", [])
        assert results == []

    def test_single_document(self):
        """Single document -> returned as-is with score=1.0."""
        reranker = self._make_reranker()
        results = reranker.rerank("q", ["only_doc"])
        assert len(results) == 1
        assert results[0]["corpus_id"] == 0
        assert results[0]["score"] == 1.0
        assert results[0]["text"] == "only_doc"


# ---------------------------------------------------------------------------
# TestMockReranker
# ---------------------------------------------------------------------------


class TestMockReranker:
    """Tests for the MockReranker test helper itself."""

    def test_mock_reranker_reverses_order(self):
        """MockReranker(reverse=True) should reverse document order."""
        reranker = MockReranker(reverse=True)
        results = reranker.rerank("q", ["first", "second", "third"])

        assert len(results) == 3
        # Reversed: third, second, first
        assert results[0]["text"] == "third"
        assert results[1]["text"] == "second"
        assert results[2]["text"] == "first"
        # Scores descending
        assert results[0]["score"] > results[1]["score"]
        assert results[1]["score"] > results[2]["score"]

    def test_mock_reranker_respects_top_k(self):
        """MockReranker should truncate to top_k."""
        reranker = MockReranker(reverse=True)
        results = reranker.rerank("q", ["a", "b", "c", "d"], top_k=2)
        assert len(results) == 2

    def test_mock_reranker_tracks_last_query(self):
        """MockReranker should expose _last_query and _last_documents."""
        reranker = MockReranker()
        reranker.rerank("my query", ["doc1", "doc2"])
        assert reranker._last_query == "my query"
        assert reranker._last_documents == ["doc1", "doc2"]

    def test_mock_reranker_preserves_order_when_reverse_false(self):
        """MockReranker(reverse=False) should preserve original order."""
        reranker = MockReranker(reverse=False)
        results = reranker.rerank("q", ["a", "b", "c"])
        assert results[0]["text"] == "a"
        assert results[1]["text"] == "b"
        assert results[2]["text"] == "c"
