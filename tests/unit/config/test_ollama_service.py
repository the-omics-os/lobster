"""
Unit tests for OllamaService.

Tests cover:
- Model listing and discovery
- Model validation
- Model info retrieval
- Server availability checking
- Model selection heuristics
"""

from unittest.mock import MagicMock, patch

import pytest

from lobster.config.ollama_service import OllamaModelInfo, OllamaService


class TestOllamaModelInfo:
    """Test OllamaModelInfo dataclass."""

    def test_size_human_bytes(self):
        """Test human-readable size formatting for bytes."""
        model = OllamaModelInfo(
            name="test:model", size_bytes=512, modified_at="2024-01-01"
        )
        assert model.size_human == "512.0B"

    def test_size_human_gb(self):
        """Test human-readable size formatting for gigabytes."""
        model = OllamaModelInfo(
            name="test:model",
            size_bytes=40_000_000_000,  # ~37GB
            modified_at="2024-01-01",
        )
        assert "GB" in model.size_human
        assert model.size_human.startswith("37")

    def test_description_full(self):
        """Test description with all metadata."""
        model = OllamaModelInfo(
            name="llama3:70b-instruct",
            size_bytes=40_000_000_000,
            modified_at="2024-01-01",
            family="llama",
            parameter_size="70B",
        )

        description = model.description
        assert "GB" in description
        assert "70B params" in description
        assert "(llama family)" in description

    def test_description_minimal(self):
        """Test description with only size."""
        model = OllamaModelInfo(
            name="test:model", size_bytes=1_000_000_000, modified_at="2024-01-01"
        )

        description = model.description
        assert "GB" in description or "MB" in description
        assert "params" not in description
        assert "family" not in description


class TestOllamaService:
    """Test OllamaService functionality."""

    @patch("requests.get")
    def test_is_available_success(self, mock_get):
        """Test successful server availability check."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        assert OllamaService.is_available() is True
        mock_get.assert_called_once()

    @patch("requests.get")
    def test_is_available_failure(self, mock_get):
        """Test server availability check when server is down."""
        mock_get.side_effect = Exception("Connection refused")

        assert OllamaService.is_available() is False

    @patch("requests.get")
    def test_list_models_success(self, mock_get):
        """Test successful model listing."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {
                    "name": "llama3:8b-instruct",
                    "size": 4_700_000_000,
                    "modified_at": "2024-01-01",
                    "digest": "abc123",
                    "details": {"family": "llama", "parameter_size": "8B"},
                },
                {
                    "name": "mixtral:8x7b-instruct",
                    "size": 26_000_000_000,
                    "modified_at": "2024-01-02",
                    "digest": "def456",
                    "details": {"family": "mixtral", "parameter_size": "47B"},
                },
            ]
        }
        mock_get.return_value = mock_response

        models = OllamaService.list_models()

        assert len(models) == 2
        # Should be sorted by size (largest first)
        assert models[0].name == "mixtral:8x7b-instruct"
        assert models[1].name == "llama3:8b-instruct"

    @patch("requests.get")
    def test_list_models_empty(self, mock_get):
        """Test model listing when no models installed."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": []}
        mock_get.return_value = mock_response

        models = OllamaService.list_models()

        assert len(models) == 0

    @patch("requests.get")
    def test_list_models_api_error(self, mock_get):
        """Test model listing when API returns error."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        models = OllamaService.list_models()

        assert len(models) == 0

    @patch("lobster.config.ollama_service.OllamaService.list_models")
    def test_get_model_names(self, mock_list):
        """Test getting model names for tab completion."""
        mock_list.return_value = [
            OllamaModelInfo("llama3:70b", 40_000_000_000, "2024-01-01"),
            OllamaModelInfo("llama3:8b", 4_700_000_000, "2024-01-01"),
        ]

        names = OllamaService.get_model_names()

        assert names == ["llama3:70b", "llama3:8b"]

    @patch("lobster.config.ollama_service.OllamaService.get_model_names")
    def test_validate_model_exists(self, mock_names):
        """Test model validation when model exists."""
        mock_names.return_value = ["llama3:8b-instruct", "mixtral:8x7b-instruct"]

        assert OllamaService.validate_model("llama3:8b-instruct") is True

    @patch("lobster.config.ollama_service.OllamaService.get_model_names")
    def test_validate_model_not_exists(self, mock_names):
        """Test model validation when model doesn't exist."""
        mock_names.return_value = ["llama3:8b-instruct"]

        assert OllamaService.validate_model("llama3:70b-instruct") is False

    @patch("lobster.config.ollama_service.OllamaService.list_models")
    def test_get_model_info_found(self, mock_list):
        """Test getting model info when model exists."""
        model_info = OllamaModelInfo(
            "llama3:8b-instruct", 4_700_000_000, "2024-01-01", family="llama"
        )
        mock_list.return_value = [model_info]

        result = OllamaService.get_model_info("llama3:8b-instruct")

        assert result is not None
        assert result.name == "llama3:8b-instruct"
        assert result.family == "llama"

    @patch("lobster.config.ollama_service.OllamaService.list_models")
    def test_get_model_info_not_found(self, mock_list):
        """Test getting model info when model doesn't exist."""
        mock_list.return_value = []

        result = OllamaService.get_model_info("nonexistent:model")

        assert result is None

    def test_suggest_model_for_agent_high_complexity(self):
        """Test model suggestion for high-complexity agents."""
        assert OllamaService.suggest_model_for_agent("supervisor") == "large"
        assert (
            OllamaService.suggest_model_for_agent("custom_feature_agent") == "large"
        )

    def test_suggest_model_for_agent_medium_complexity(self):
        """Test model suggestion for medium-complexity agents."""
        assert (
            OllamaService.suggest_model_for_agent("transcriptomics_expert")
            == "medium"
        )
        assert OllamaService.suggest_model_for_agent("proteomics_expert") == "medium"

    def test_suggest_model_for_agent_low_complexity(self):
        """Test model suggestion for low-complexity agents."""
        assert OllamaService.suggest_model_for_agent("data_expert") == "small"
        assert OllamaService.suggest_model_for_agent("unknown_agent") == "small"

    @patch("lobster.config.ollama_service.OllamaService.list_models")
    def test_select_best_model_large(self, mock_list):
        """Test selecting best model with 'large' preference."""
        mock_list.return_value = [
            OllamaModelInfo(
                "llama3:70b-instruct", 40_000_000_000, "2024-01-01", family="llama"
            ),
            OllamaModelInfo(
                "llama3:8b-instruct", 4_700_000_000, "2024-01-01", family="llama"
            ),
        ]

        model = OllamaService.select_best_model("large")

        assert model == "llama3:70b-instruct"

    @patch("lobster.config.ollama_service.OllamaService.list_models")
    def test_select_best_model_small(self, mock_list):
        """Test selecting best model with 'small' preference."""
        mock_list.return_value = [
            OllamaModelInfo(
                "llama3:70b-instruct", 40_000_000_000, "2024-01-01", family="llama"
            ),
            OllamaModelInfo(
                "llama3:8b-instruct", 4_700_000_000, "2024-01-01", family="llama"
            ),
        ]

        model = OllamaService.select_best_model("small")

        assert model == "llama3:8b-instruct"

    @patch("lobster.config.ollama_service.OllamaService.list_models")
    def test_select_best_model_auto(self, mock_list):
        """Test auto-selecting best model (prefers 70B)."""
        mock_list.return_value = [
            OllamaModelInfo(
                "mixtral:8x7b-instruct", 26_000_000_000, "2024-01-01", family="mixtral"
            ),
            OllamaModelInfo(
                "llama3:70b-instruct", 40_000_000_000, "2024-01-01", family="llama"
            ),
            OllamaModelInfo(
                "llama3:8b-instruct", 4_700_000_000, "2024-01-01", family="llama"
            ),
        ]

        model = OllamaService.select_best_model("auto")

        # Should prefer 70B model
        assert model == "llama3:70b-instruct"

    @patch("lobster.config.ollama_service.OllamaService.list_models")
    def test_select_best_model_no_models(self, mock_list):
        """Test selecting best model when no models available."""
        mock_list.return_value = []

        model = OllamaService.select_best_model()

        assert model is None
