# OpenRouter Provider Integration — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add OpenRouter as a first-class LLM provider, enabling access to 600+ models (Claude, GPT-4o, Llama, DeepSeek, Gemini, etc.) via a single API key.

**Architecture:** Standalone `OpenRouterProvider(ILLMProvider)` registered in `ProviderRegistry`. Uses `ChatOpenAI` with `base_url="https://openrouter.ai/api/v1"` — no new dependencies. Live model catalog fetched from OpenRouter API on first `list_models()` call, cached in-process, falls back to 20-model curated list on network failure.

**Tech Stack:** `langchain-openai` (already installed), `httpx` (already in LangChain's dependency tree), Pydantic, Typer/Rich (CLI).

**Design doc:** `.planning/openrouter-provider-design.md`

---

## Task 1: Register `openrouter` in VALID_PROVIDERS

This is the gate. All validation throughout the codebase checks `VALID_PROVIDERS`.

**Files:**
- Modify: `lobster/config/constants.py`
- Modify: `lobster/config/llm_factory.py`

**Step 1: Write the failing test**

In `tests/unit/config/test_constants.py` (create if missing, otherwise add):

```python
def test_openrouter_in_valid_providers():
    from lobster.config.constants import VALID_PROVIDERS
    assert "openrouter" in VALID_PROVIDERS

def test_openrouter_in_display_names():
    from lobster.config.constants import PROVIDER_DISPLAY_NAMES
    assert "openrouter" in PROVIDER_DISPLAY_NAMES
    assert "OpenRouter" in PROVIDER_DISPLAY_NAMES["openrouter"]
```

**Step 2: Run to verify it fails**

```bash
cd /Users/tyo/Omics-OS/lobster && python -m pytest tests/unit/config/test_constants.py -k "test_openrouter" -v
```
Expected: FAIL — `AssertionError` (openrouter not in list)

**Step 3: Implement**

In `lobster/config/constants.py`, add `"openrouter"` to both lists:

```python
VALID_PROVIDERS: Final[List[str]] = [
    "anthropic",
    "bedrock",
    "ollama",
    "gemini",
    "azure",
    "openai",
    "openrouter",   # ← add this
]

PROVIDER_DISPLAY_NAMES: Final[dict] = {
    "anthropic": "Anthropic Direct API",
    "bedrock": "AWS Bedrock",
    "ollama": "Ollama (Local)",
    "gemini": "Google Gemini",
    "azure": "Azure AI",
    "openai": "OpenAI",
    "openrouter": "OpenRouter (600+ models)",   # ← add this
}
```

In `lobster/config/llm_factory.py`, add to `LLMProvider` enum (backward compat only):

```python
class LLMProvider(Enum):
    ANTHROPIC_DIRECT = "anthropic"
    BEDROCK_ANTHROPIC = "bedrock"
    OLLAMA = "ollama"
    GEMINI = "gemini"
    AZURE = "azure"
    OPENAI = "openai"
    OPENROUTER = "openrouter"   # ← add this
```

**Step 4: Run to verify it passes**

```bash
python -m pytest tests/unit/config/test_constants.py -k "test_openrouter" -v
```
Expected: PASS

**Step 5: Verify nothing broke**

```bash
python -m pytest tests/unit/config/ -v --tb=short -q
```
Expected: All existing tests still pass.

**Step 6: Commit**

```bash
git add lobster/config/constants.py lobster/config/llm_factory.py tests/unit/config/test_constants.py
git commit -m "feat(openrouter): register openrouter in VALID_PROVIDERS and LLMProvider enum"
```

---

## Task 2: Create `OpenRouterProvider`

The core implementation. Follows the exact same pattern as `openai_provider.py`.

**Files:**
- Create: `lobster/config/providers/openrouter_provider.py`
- Create: `tests/unit/config/test_openrouter_provider.py`

**Step 1: Write the failing tests**

Create `tests/unit/config/test_openrouter_provider.py`:

```python
"""Tests for OpenRouter provider implementation."""
import os
from unittest.mock import MagicMock, patch

import pytest

from lobster.config.providers.openrouter_provider import OpenRouterProvider


@pytest.fixture(autouse=True)
def clear_model_cache():
    """Reset the class-level model cache before each test."""
    OpenRouterProvider._models_cache = None
    yield
    OpenRouterProvider._models_cache = None


@pytest.fixture
def provider():
    return OpenRouterProvider()


class TestIsConfigured:
    def test_configured_with_key(self, provider, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")
        assert provider.is_configured() is True

    def test_not_configured_without_key(self, provider, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        assert provider.is_configured() is False

    def test_not_configured_empty_key(self, provider, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "   ")
        assert provider.is_configured() is False


class TestIsAvailable:
    def test_available_when_configured(self, provider, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")
        assert provider.is_available() is True

    def test_not_available_when_not_configured(self, provider, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        assert provider.is_available() is False


class TestProviderIdentity:
    def test_name(self, provider):
        assert provider.name == "openrouter"

    def test_display_name(self, provider):
        assert "OpenRouter" in provider.display_name

    def test_default_model(self, provider):
        model = provider.get_default_model()
        assert "/" in model  # OpenRouter models use provider/model-name format
        assert model == "anthropic/claude-sonnet-4-5"


class TestListModels:
    def test_list_models_uses_fallback_on_network_error(self, provider, monkeypatch):
        """When network fails, returns curated fallback catalog."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

        with patch("httpx.get") as mock_get:
            mock_get.side_effect = Exception("Network error")
            models = provider.list_models()

        assert len(models) >= 10  # Fallback has at least 10 models
        names = [m.name for m in models]
        assert "anthropic/claude-sonnet-4-5" in names
        assert "openai/gpt-4o" in names

    def test_list_models_parses_live_catalog(self, provider, monkeypatch):
        """When API succeeds, returns parsed live catalog."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "anthropic/claude-3-5-sonnet",
                    "name": "Claude 3.5 Sonnet",
                    "description": "Anthropic's fastest model",
                    "context_length": 200000,
                    "pricing": {"prompt": "0.000003", "completion": "0.000015"},
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response):
            models = provider.list_models()

        assert len(models) == 1
        assert models[0].name == "anthropic/claude-3-5-sonnet"

    def test_list_models_caches_result(self, provider, monkeypatch):
        """Second call uses cache, does not re-fetch."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [
            {"id": "openai/gpt-4o", "name": "GPT-4o", "description": "",
             "context_length": 128000, "pricing": {"prompt": "0.0000025", "completion": "0.000010"}}
        ]}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response) as mock_get:
            provider.list_models()
            provider.list_models()

        assert mock_get.call_count == 1  # Only called once despite two list_models() calls

    def test_list_models_fallback_when_httpx_missing(self, provider, monkeypatch):
        """Falls back gracefully when httpx is not installed."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

        with patch.dict("sys.modules", {"httpx": None}):
            models = provider.list_models()

        assert len(models) >= 10


class TestValidateModel:
    def test_validate_with_cache_known_model(self, provider, monkeypatch):
        """Returns True for model in populated cache."""
        OpenRouterProvider._models_cache = [
            MagicMock(name="anthropic/claude-sonnet-4-5"),
        ]
        # Need to use proper ModelInfo-like object
        from lobster.config.providers.base_provider import ModelInfo
        OpenRouterProvider._models_cache = [
            ModelInfo(
                name="anthropic/claude-sonnet-4-5",
                display_name="Claude Sonnet 4.5",
                description="",
                provider="openrouter",
            )
        ]
        assert provider.validate_model("anthropic/claude-sonnet-4-5") is True

    def test_validate_with_cache_unknown_model(self, provider):
        """Returns False for model NOT in populated cache."""
        from lobster.config.providers.base_provider import ModelInfo
        OpenRouterProvider._models_cache = [
            ModelInfo(
                name="openai/gpt-4o",
                display_name="GPT-4o",
                description="",
                provider="openrouter",
            )
        ]
        assert provider.validate_model("made-up/model") is False

    def test_validate_without_cache_accepts_nonempty(self, provider):
        """Without cache, accepts any non-empty model ID (passthrough)."""
        assert OpenRouterProvider._models_cache is None
        assert provider.validate_model("any/model-id") is True

    def test_validate_rejects_empty_string(self, provider):
        assert provider.validate_model("") is False


class TestCreateChatModel:
    def test_create_chat_model_success(self, provider, monkeypatch):
        """Creates ChatOpenAI with correct base_url and headers."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")

        with patch("langchain_openai.ChatOpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            llm = provider.create_chat_model("anthropic/claude-sonnet-4-5")

        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["base_url"] == "https://openrouter.ai/api/v1"
        assert "HTTP-Referer" in call_kwargs["default_headers"]
        assert "lobsterbio.com" in call_kwargs["default_headers"]["HTTP-Referer"]
        assert call_kwargs["default_headers"]["X-Title"] == "Lobster AI"
        assert call_kwargs["model"] == "anthropic/claude-sonnet-4-5"

    def test_create_chat_model_no_api_key_raises(self, provider, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            provider.create_chat_model("anthropic/claude-sonnet-4-5")

    def test_create_chat_model_passes_temperature_max_tokens(self, provider, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")

        with patch("langchain_openai.ChatOpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            provider.create_chat_model("openai/gpt-4o", temperature=0.5, max_tokens=2048)

        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 2048
```

**Step 2: Run to verify they fail**

```bash
python -m pytest tests/unit/config/test_openrouter_provider.py -v 2>&1 | head -20
```
Expected: ImportError — `openrouter_provider` module doesn't exist yet.

**Step 3: Write `openrouter_provider.py`**

Create `lobster/config/providers/openrouter_provider.py`:

```python
"""
OpenRouter provider implementation.

OpenRouter is a unified API gateway for 600+ LLM models across providers.
Uses the OpenAI API protocol — no new LangChain packages required.

API key setup: https://openrouter.ai/keys
Model catalog:  https://openrouter.ai/models

Architecture:
    - Implements ILLMProvider interface
    - Uses ChatOpenAI with base_url override (OpenAI-compatible API)
    - Lazy live catalog fetched from openrouter.ai/api/v1/models on first list_models() call
    - Class-level in-process cache — fetched once, never re-fetched
    - Curated fallback catalog used when network is unavailable
    - Hardcoded Lobster branding headers for OpenRouter leaderboard

Model naming convention: "provider/model-name" (e.g., "anthropic/claude-sonnet-4-5")

Example:
    >>> provider = OpenRouterProvider()
    >>> if provider.is_configured():
    ...     models = provider.list_models()  # fetches live catalog
    ...     llm = provider.create_chat_model("anthropic/claude-sonnet-4-5")
"""

import logging
import os
from typing import Any, ClassVar, List, Optional

from lobster.config.providers.base_provider import ILLMProvider, ModelInfo

logger = logging.getLogger(__name__)

# OpenRouter API constants
_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_HEADERS = {
    "HTTP-Referer": "https://lobsterbio.com",
    "X-Title": "Lobster AI",
}
_ENV_VAR = "OPENROUTER_API_KEY"
_DEFAULT_MODEL = "anthropic/claude-sonnet-4-5"
_CATALOG_TIMEOUT = 5.0  # seconds


# Curated fallback catalog — representative models from major families.
# Used when live catalog fetch fails or httpx is unavailable.
_FALLBACK_MODELS: List[ModelInfo] = [
    ModelInfo(
        name="anthropic/claude-sonnet-4-5",
        display_name="Claude Sonnet 4.5",
        description="Anthropic's latest Sonnet — fast, capable, excellent for complex tasks",
        provider="openrouter",
        context_window=200000,
        is_default=True,
        input_cost_per_million=3.00,
        output_cost_per_million=15.00,
    ),
    ModelInfo(
        name="anthropic/claude-3-5-haiku",
        display_name="Claude 3.5 Haiku",
        description="Anthropic's fastest model — ideal for lightweight tasks",
        provider="openrouter",
        context_window=200000,
        is_default=False,
        input_cost_per_million=0.80,
        output_cost_per_million=4.00,
    ),
    ModelInfo(
        name="openai/gpt-4o",
        display_name="GPT-4o",
        description="OpenAI's flagship multimodal model",
        provider="openrouter",
        context_window=128000,
        is_default=False,
        input_cost_per_million=2.50,
        output_cost_per_million=10.00,
    ),
    ModelInfo(
        name="openai/gpt-4o-mini",
        display_name="GPT-4o Mini",
        description="Fast and affordable GPT-4 variant",
        provider="openrouter",
        context_window=128000,
        is_default=False,
        input_cost_per_million=0.15,
        output_cost_per_million=0.60,
    ),
    ModelInfo(
        name="openai/o1",
        display_name="o1 (Reasoning)",
        description="Advanced reasoning model for complex problem-solving",
        provider="openrouter",
        context_window=200000,
        is_default=False,
        input_cost_per_million=15.00,
        output_cost_per_million=60.00,
    ),
    ModelInfo(
        name="openai/o3-mini",
        display_name="o3 Mini",
        description="Efficient compact reasoning model",
        provider="openrouter",
        context_window=200000,
        is_default=False,
        input_cost_per_million=1.10,
        output_cost_per_million=4.40,
    ),
    ModelInfo(
        name="meta-llama/llama-3.3-70b-instruct",
        display_name="Llama 3.3 70B Instruct",
        description="Meta's most capable open-source model",
        provider="openrouter",
        context_window=128000,
        is_default=False,
        input_cost_per_million=0.35,
        output_cost_per_million=0.40,
    ),
    ModelInfo(
        name="meta-llama/llama-3.1-8b-instruct",
        display_name="Llama 3.1 8B Instruct",
        description="Compact, fast open-source model",
        provider="openrouter",
        context_window=131072,
        is_default=False,
        input_cost_per_million=0.06,
        output_cost_per_million=0.06,
    ),
    ModelInfo(
        name="deepseek/deepseek-r1",
        display_name="DeepSeek R1",
        description="Strong reasoning model, open-source",
        provider="openrouter",
        context_window=164000,
        is_default=False,
        input_cost_per_million=0.55,
        output_cost_per_million=2.19,
    ),
    ModelInfo(
        name="deepseek/deepseek-chat-v3-0324",
        display_name="DeepSeek Chat V3",
        description="DeepSeek's top conversational model",
        provider="openrouter",
        context_window=164000,
        is_default=False,
        input_cost_per_million=0.27,
        output_cost_per_million=1.10,
    ),
    ModelInfo(
        name="google/gemini-2.0-flash-001",
        display_name="Gemini 2.0 Flash",
        description="Google's fastest multimodal model",
        provider="openrouter",
        context_window=1000000,
        is_default=False,
        input_cost_per_million=0.10,
        output_cost_per_million=0.40,
    ),
    ModelInfo(
        name="google/gemini-2.5-pro-preview",
        display_name="Gemini 2.5 Pro Preview",
        description="Google's most capable model with 1M context",
        provider="openrouter",
        context_window=1000000,
        is_default=False,
        input_cost_per_million=1.25,
        output_cost_per_million=10.00,
    ),
    ModelInfo(
        name="mistralai/mistral-large",
        display_name="Mistral Large",
        description="Mistral's flagship model for complex tasks",
        provider="openrouter",
        context_window=128000,
        is_default=False,
        input_cost_per_million=2.00,
        output_cost_per_million=6.00,
    ),
    ModelInfo(
        name="mistralai/mistral-nemo",
        display_name="Mistral Nemo",
        description="Compact Mistral model, great cost/quality ratio",
        provider="openrouter",
        context_window=128000,
        is_default=False,
        input_cost_per_million=0.13,
        output_cost_per_million=0.13,
    ),
    ModelInfo(
        name="qwen/qwen-2.5-72b-instruct",
        display_name="Qwen 2.5 72B Instruct",
        description="Alibaba's top open model, strong at coding and science",
        provider="openrouter",
        context_window=131072,
        is_default=False,
        input_cost_per_million=0.35,
        output_cost_per_million=0.40,
    ),
    ModelInfo(
        name="x-ai/grok-2-1212",
        display_name="Grok 2",
        description="xAI's flagship model",
        provider="openrouter",
        context_window=131072,
        is_default=False,
        input_cost_per_million=2.00,
        output_cost_per_million=10.00,
    ),
    ModelInfo(
        name="cohere/command-r-plus",
        display_name="Command R+",
        description="Cohere's enterprise-grade RAG model",
        provider="openrouter",
        context_window=128000,
        is_default=False,
        input_cost_per_million=2.50,
        output_cost_per_million=10.00,
    ),
    ModelInfo(
        name="microsoft/phi-4",
        display_name="Phi-4",
        description="Microsoft's compact 14B reasoning model",
        provider="openrouter",
        context_window=16384,
        is_default=False,
        input_cost_per_million=0.07,
        output_cost_per_million=0.14,
    ),
    ModelInfo(
        name="nvidia/llama-3.1-nemotron-70b-instruct",
        display_name="Nemotron 70B",
        description="NVIDIA-tuned Llama with superior instruction following",
        provider="openrouter",
        context_window=131072,
        is_default=False,
        input_cost_per_million=0.35,
        output_cost_per_million=0.40,
    ),
    ModelInfo(
        name="amazon/nova-pro-v1",
        display_name="Amazon Nova Pro",
        description="Amazon's multimodal frontier model",
        provider="openrouter",
        context_window=300000,
        is_default=False,
        input_cost_per_million=0.80,
        output_cost_per_million=3.20,
    ),
]


class OpenRouterProvider(ILLMProvider):
    """
    OpenRouter provider — unified API gateway to 600+ LLM models.

    OpenRouter accepts requests in OpenAI API format and routes them to the
    appropriate upstream model. A single OPENROUTER_API_KEY gives access to
    models from Anthropic, OpenAI, Meta, Google, Mistral, DeepSeek, and more.

    Model names use the "provider/model-name" format:
        - "anthropic/claude-sonnet-4-5"
        - "openai/gpt-4o"
        - "meta-llama/llama-3.3-70b-instruct"

    Features:
        - Live model catalog fetched from openrouter.ai/api/v1/models (lazy, cached)
        - 20-model curated fallback catalog for offline use
        - Hardcoded Lobster AI branding headers (https://openrouter.ai/leaderboard)
        - No new dependencies — uses ChatOpenAI with base_url override

    Usage:
        >>> provider = OpenRouterProvider()
        >>> if not provider.is_configured():
        ...     print("Set OPENROUTER_API_KEY in .env")
        >>> models = provider.list_models()  # fetches live catalog on first call
        >>> llm = provider.create_chat_model("anthropic/claude-sonnet-4-5")
    """

    # Class-level in-process cache for live model catalog
    _models_cache: ClassVar[Optional[List[ModelInfo]]] = None

    @property
    def name(self) -> str:
        return "openrouter"

    @property
    def display_name(self) -> str:
        return "OpenRouter (600+ models)"

    def is_configured(self) -> bool:
        """
        Check if OPENROUTER_API_KEY is present and non-empty.

        Returns:
            bool: True if OPENROUTER_API_KEY is set and non-empty

        Example:
            >>> provider = OpenRouterProvider()
            >>> if not provider.is_configured():
            ...     print("Set OPENROUTER_API_KEY=sk-or-... in .env")
        """
        api_key = os.environ.get(_ENV_VAR)
        return bool(api_key and api_key.strip())

    def is_available(self) -> bool:
        """
        Check if OpenRouter is accessible.

        For cloud providers, availability equals configuration — we don't
        ping the API to avoid latency. Invalid keys surface at inference time.

        Returns:
            bool: True if OPENROUTER_API_KEY is configured
        """
        return self.is_configured()

    def get_default_model(self) -> str:
        """
        Get the recommended default model.

        Returns:
            str: "anthropic/claude-sonnet-4-5"

        Example:
            >>> model_id = provider.get_default_model()
            >>> llm = provider.create_chat_model(model_id)
        """
        return _DEFAULT_MODEL

    def list_models(self) -> List[ModelInfo]:
        """
        List available models from OpenRouter.

        On first call, fetches the live catalog from openrouter.ai/api/v1/models
        and caches it in the class for the process lifetime. On network failure
        or if httpx is unavailable, returns the curated fallback catalog.

        Returns:
            List[ModelInfo]: Available models (live or fallback)

        Example:
            >>> for model in provider.list_models():
            ...     print(f"{model.name}: ${model.input_cost_per_million}/M")
        """
        if OpenRouterProvider._models_cache is not None:
            return OpenRouterProvider._models_cache

        try:
            import httpx

            api_key = os.environ.get(_ENV_VAR, "")
            response = httpx.get(
                f"{_BASE_URL}/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=_CATALOG_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json().get("data", [])
            models = [self._parse_model(m) for m in data if m.get("id")]
            OpenRouterProvider._models_cache = models
            logger.debug(f"Fetched {len(models)} models from OpenRouter API")
            return models

        except ImportError:
            logger.debug("httpx not available, using OpenRouter fallback catalog")
        except Exception as e:
            logger.debug(f"OpenRouter model fetch failed ({e}), using fallback catalog")

        return list(_FALLBACK_MODELS)

    def _parse_model(self, data: dict) -> ModelInfo:
        """Parse a single model entry from the OpenRouter API response."""
        pricing = data.get("pricing", {})
        # OpenRouter pricing is per-token strings (e.g., "0.000003")
        # Convert to per-million for ModelInfo consistency
        input_cost = None
        output_cost = None
        try:
            prompt_price = float(pricing.get("prompt", 0))
            completion_price = float(pricing.get("completion", 0))
            if prompt_price > 0:
                input_cost = prompt_price * 1_000_000
            if completion_price > 0:
                output_cost = completion_price * 1_000_000
        except (ValueError, TypeError):
            pass

        return ModelInfo(
            name=data["id"],
            display_name=data.get("name", data["id"]),
            description=data.get("description", ""),
            provider="openrouter",
            context_window=data.get("context_length"),
            is_default=data["id"] == _DEFAULT_MODEL,
            input_cost_per_million=input_cost,
            output_cost_per_million=output_cost,
        )

    def validate_model(self, model_id: str) -> bool:
        """
        Check if a model ID is valid.

        If the live catalog has been fetched, checks against it.
        Without cache, accepts any non-empty string — OpenRouter validates
        at inference time.

        Args:
            model_id: Model identifier (e.g., "anthropic/claude-sonnet-4-5")

        Returns:
            bool: True if valid (or cache not loaded)

        Example:
            >>> if provider.validate_model("anthropic/claude-sonnet-4-5"):
            ...     llm = provider.create_chat_model("anthropic/claude-sonnet-4-5")
        """
        if not model_id:
            return False
        if OpenRouterProvider._models_cache is not None:
            return model_id in [m.name for m in OpenRouterProvider._models_cache]
        # Passthrough: let OpenRouter validate at inference time
        logger.debug(f"OpenRouter model '{model_id}' not validated against catalog (cache not loaded)")
        return True

    def create_chat_model(
        self,
        model_id: str,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> Any:
        """
        Create a ChatOpenAI instance routed through OpenRouter.

        Uses the OpenAI-compatible API with OpenRouter's base URL and
        hardcoded Lobster AI branding headers.

        Args:
            model_id: Model identifier in "provider/model-name" format
            temperature: Sampling temperature (0.0-2.0, default 1.0)
            max_tokens: Maximum tokens in response (default 4096)
            **kwargs: Additional ChatOpenAI parameters

        Returns:
            ChatOpenAI: Configured LangChain chat model pointing to OpenRouter

        Raises:
            ImportError: If langchain-openai not installed
            ValueError: If OPENROUTER_API_KEY not set

        Example:
            >>> llm = provider.create_chat_model(
            ...     "meta-llama/llama-3.3-70b-instruct",
            ...     temperature=0.7
            ... )
        """
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "langchain-openai package not installed. "
                "Install with: uv pip install langchain-openai"
            )

        api_key = kwargs.pop("api_key", None) or os.environ.get(_ENV_VAR)
        if not api_key:
            raise ValueError(
                f"{_ENV_VAR} not found in environment. "
                f"Set it with: export {_ENV_VAR}=sk-or-..."
            )

        return ChatOpenAI(
            model=model_id,
            api_key=api_key,
            base_url=_BASE_URL,
            default_headers=_DEFAULT_HEADERS,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    def get_configuration_help(self) -> str:
        return (
            "Configure OpenRouter:\n\n"
            "1. Get API key from: https://openrouter.ai/keys\n"
            "2. Set environment variable:\n"
            "   export OPENROUTER_API_KEY=sk-or-...\n\n"
            "Or add to .env file:\n"
            "   OPENROUTER_API_KEY=sk-or-...\n\n"
            f"Default model: {_DEFAULT_MODEL}\n"
            "Model format: provider/model-name (e.g., anthropic/claude-sonnet-4-5)\n"
            "Browse 600+ models at: https://openrouter.ai/models\n"
        )


# Auto-register provider with registry
from lobster.config.providers.registry import ProviderRegistry

ProviderRegistry.register(OpenRouterProvider())
```

**Step 4: Run to verify tests pass**

```bash
python -m pytest tests/unit/config/test_openrouter_provider.py -v
```
Expected: All tests PASS.

**Step 5: Commit**

```bash
git add lobster/config/providers/openrouter_provider.py tests/unit/config/test_openrouter_provider.py
git commit -m "feat(openrouter): implement OpenRouterProvider with lazy live catalog"
```

---

## Task 3: Wire into ProviderRegistry and `__init__.py`

**Files:**
- Modify: `lobster/config/providers/registry.py:175-182`
- Modify: `lobster/config/providers/__init__.py`

**Step 1: Add test for registry discovery**

Add to `tests/unit/config/test_openrouter_provider.py`:

```python
class TestRegistryIntegration:
    def test_openrouter_in_provider_registry(self):
        """OpenRouterProvider is discoverable via ProviderRegistry."""
        from lobster.config.providers.registry import ProviderRegistry
        ProviderRegistry.reset()
        ProviderRegistry._ensure_initialized()
        assert ProviderRegistry.is_registered("openrouter")

    def test_get_provider_returns_openrouter(self):
        from lobster.config.providers import get_provider
        provider = get_provider("openrouter")
        assert provider is not None
        assert provider.name == "openrouter"
```

**Step 2: Run to verify it fails**

```bash
python -m pytest tests/unit/config/test_openrouter_provider.py::TestRegistryIntegration -v
```
Expected: FAIL — openrouter not in registry

**Step 3: Update `registry.py`**

In `lobster/config/providers/registry.py`, add to `_provider_specs` list (line ~181):

```python
_provider_specs = [
    ("lobster.config.providers.anthropic_provider", "AnthropicProvider"),
    ("lobster.config.providers.bedrock_provider", "BedrockProvider"),
    ("lobster.config.providers.ollama_provider", "OllamaProvider"),
    ("lobster.config.providers.gemini_provider", "GeminiProvider"),
    ("lobster.config.providers.azure_provider", "AzureProvider"),
    ("lobster.config.providers.openai_provider", "OpenAIProvider"),
    ("lobster.config.providers.openrouter_provider", "OpenRouterProvider"),  # ← add
]
```

**Step 4: Update `__init__.py`**

In `lobster/config/providers/__init__.py`, add OpenRouterProvider export:

```python
from lobster.config.providers.base_provider import ILLMProvider, ModelInfo
from lobster.config.providers.registry import ProviderRegistry, get_provider
from lobster.config.providers.openrouter_provider import OpenRouterProvider  # ← add

__all__ = [
    "ILLMProvider",
    "ModelInfo",
    "ProviderRegistry",
    "get_provider",
    "OpenRouterProvider",   # ← add
]
```

**Step 5: Run to verify tests pass**

```bash
python -m pytest tests/unit/config/test_openrouter_provider.py -v
```
Expected: All tests PASS.

**Step 6: Commit**

```bash
git add lobster/config/providers/registry.py lobster/config/providers/__init__.py
git commit -m "feat(openrouter): wire OpenRouterProvider into ProviderRegistry"
```

---

## Task 4: Add config schema fields

Add `openrouter_default_model` to `GlobalProviderConfig` and `openrouter_model` to `WorkspaceProviderConfig`. These follow the field naming convention so `base_config.get_model_for_provider("openrouter")` auto-resolves them.

**Files:**
- Modify: `lobster/config/global_config.py`
- Modify: `lobster/config/workspace_config.py`

**Step 1: Write tests**

Add to a new file `tests/unit/config/test_openrouter_config.py`:

```python
"""Tests for OpenRouter config schema fields."""


def test_global_config_has_openrouter_model_field():
    from lobster.config.global_config import GlobalProviderConfig
    config = GlobalProviderConfig(openrouter_default_model="anthropic/claude-sonnet-4-5")
    assert config.openrouter_default_model == "anthropic/claude-sonnet-4-5"


def test_global_config_get_model_for_openrouter():
    """base_config.get_model_for_provider() auto-resolves openrouter_default_model."""
    from lobster.config.global_config import GlobalProviderConfig
    config = GlobalProviderConfig(openrouter_default_model="openai/gpt-4o")
    assert config.get_model_for_provider("openrouter") == "openai/gpt-4o"


def test_global_config_openrouter_model_defaults_none():
    from lobster.config.global_config import GlobalProviderConfig
    config = GlobalProviderConfig()
    assert config.openrouter_default_model is None


def test_workspace_config_has_openrouter_model_field():
    from lobster.config.workspace_config import WorkspaceProviderConfig
    config = WorkspaceProviderConfig(openrouter_model="meta-llama/llama-3.3-70b-instruct")
    assert config.openrouter_model == "meta-llama/llama-3.3-70b-instruct"


def test_workspace_config_get_model_for_openrouter():
    """base_config.get_model_for_provider() auto-resolves openrouter_model."""
    from lobster.config.workspace_config import WorkspaceProviderConfig
    config = WorkspaceProviderConfig(openrouter_model="openai/gpt-4o")
    assert config.get_model_for_provider("openrouter") == "openai/gpt-4o"


def test_workspace_config_accepts_openrouter_provider():
    from lobster.config.workspace_config import WorkspaceProviderConfig
    config = WorkspaceProviderConfig(global_provider="openrouter")
    assert config.global_provider == "openrouter"


def test_global_config_roundtrip_json(tmp_path):
    """OpenRouter config field survives JSON serialization."""
    from lobster.config.global_config import GlobalProviderConfig
    config = GlobalProviderConfig(
        default_provider="openrouter",
        openrouter_default_model="deepseek/deepseek-r1"
    )
    # Serialize to JSON and reload
    json_str = config.model_dump_json()
    reloaded = GlobalProviderConfig.model_validate_json(json_str)
    assert reloaded.default_provider == "openrouter"
    assert reloaded.openrouter_default_model == "deepseek/deepseek-r1"
```

**Step 2: Run to verify they fail**

```bash
python -m pytest tests/unit/config/test_openrouter_config.py -v 2>&1 | head -20
```
Expected: FAIL — `openrouter_default_model` field not found

**Step 3: Add field to `GlobalProviderConfig`**

In `lobster/config/global_config.py`, add after `azure_default_model`:

```python
    openrouter_default_model: Optional[str] = Field(
        None,
        description="Default OpenRouter model (e.g., 'anthropic/claude-sonnet-4-5', 'openai/gpt-4o')",
    )
```

Also update the `reset()` method to include `self.openrouter_default_model = None`.

**Step 4: Add field to `WorkspaceProviderConfig`**

In `lobster/config/workspace_config.py`, add after `openai_model`:

```python
    openrouter_model: Optional[str] = Field(
        None,
        description="OpenRouter model (e.g., 'anthropic/claude-sonnet-4-5', 'openai/gpt-4o')",
    )
```

Also update docstring `global_provider` field description to include `"openrouter"` in the valid values list.

**Step 5: Run to verify tests pass**

```bash
python -m pytest tests/unit/config/test_openrouter_config.py -v
```
Expected: All tests PASS.

**Step 6: Verify no existing tests broke**

```bash
python -m pytest tests/unit/config/ -v --tb=short -q
```
Expected: All pass.

**Step 7: Commit**

```bash
git add lobster/config/global_config.py lobster/config/workspace_config.py tests/unit/config/test_openrouter_config.py
git commit -m "feat(openrouter): add openrouter model config fields to global and workspace configs"
```

---

## Task 5: Add `create_openrouter_config` to `provider_setup.py`

**Files:**
- Modify: `lobster/config/provider_setup.py`

**Step 1: Write tests**

Add to `tests/unit/config/test_openrouter_config.py`:

```python
def test_create_openrouter_config_valid():
    from lobster.config.provider_setup import create_openrouter_config
    result = create_openrouter_config("sk-or-test-key-123")
    assert result.success is True
    assert result.provider_type == "openrouter"
    assert result.env_vars["OPENROUTER_API_KEY"] == "sk-or-test-key-123"
    assert result.env_vars["LOBSTER_LLM_PROVIDER"] == "openrouter"


def test_create_openrouter_config_strips_whitespace():
    from lobster.config.provider_setup import create_openrouter_config
    result = create_openrouter_config("  sk-or-test  ")
    assert result.env_vars["OPENROUTER_API_KEY"] == "sk-or-test"


def test_create_openrouter_config_empty_key_fails():
    from lobster.config.provider_setup import create_openrouter_config
    result = create_openrouter_config("")
    assert result.success is False
    assert result.message is not None


def test_create_openrouter_config_whitespace_only_fails():
    from lobster.config.provider_setup import create_openrouter_config
    result = create_openrouter_config("   ")
    assert result.success is False
```

**Step 2: Run to verify they fail**

```bash
python -m pytest tests/unit/config/test_openrouter_config.py -k "create_openrouter" -v
```
Expected: FAIL — `create_openrouter_config` not found

**Step 3: Implement**

In `lobster/config/provider_setup.py`, add after `create_openai_config()`:

```python
def create_openrouter_config(api_key: str) -> ProviderConfig:
    """
    Create configuration for OpenRouter.

    OpenRouter gives access to 600+ models via a single API key.
    Browse models at: https://openrouter.ai/models

    Args:
        api_key: OpenRouter API key (format: sk-or-...)

    Returns:
        ProviderConfig with environment variables
    """
    if not api_key or not api_key.strip():
        return ProviderConfig(
            provider_type="openrouter",
            env_vars={},
            success=False,
            message="API key cannot be empty",
        )

    return ProviderConfig(
        provider_type="openrouter",
        env_vars={
            "LOBSTER_LLM_PROVIDER": "openrouter",
            "OPENROUTER_API_KEY": api_key.strip(),
        },
        success=True,
    )
```

**Step 4: Run to verify tests pass**

```bash
python -m pytest tests/unit/config/test_openrouter_config.py -k "create_openrouter" -v
```
Expected: All PASS.

**Step 5: Commit**

```bash
git add lobster/config/provider_setup.py tests/unit/config/test_openrouter_config.py
git commit -m "feat(openrouter): add create_openrouter_config helper in provider_setup"
```

---

## Task 6: Add OpenRouter to `lobster init` CLI

The most involved task — adds OpenRouter as option 7 in the interactive provider selection menu.

**Files:**
- Modify: `lobster/cli.py`

There are 6 locations in `cli.py` that need changes. Apply each precisely:

**Location 1 — `_PROVIDER_PACKAGES` dict (~line 4476)**

Add `"openrouter": "langchain-openai"` (same package as OpenAI, no new dep):

```python
_PROVIDER_PACKAGES = {
    "anthropic": "langchain-anthropic",
    "bedrock": "langchain-aws",
    "ollama": "langchain-ollama",
    "gemini": "langchain-google-genai",
    "azure": "langchain-azure-ai",
    "openai": "langchain-openai",
    "openrouter": "langchain-openai",   # ← add
}
```

**Location 2 — `_PROVIDER_IMPORT_NAMES` dict (~line 4486)**

Add `"openrouter": "langchain_openai"`:

```python
_PROVIDER_IMPORT_NAMES = {
    "anthropic": "langchain_anthropic",
    "bedrock": "langchain_aws",
    "ollama": "langchain_ollama",
    "gemini": "langchain_google_genai",
    "azure": "langchain_azure_ai",
    "openai": "langchain_openai",
    "openrouter": "langchain_openai",   # ← add
}
```

**Location 3 — Interactive menu print block (~line 5529)**

After the `console.print("  [cyan]6[/cyan] - OpenAI ...")` line, add:

```python
console.print(
    "  [cyan]7[/cyan] - OpenRouter - 600+ models via one API key (Claude, GPT-4o, Llama, DeepSeek, ...)"
)
```

**Location 4 — `Prompt.ask` choices (~line 5532)**

Extend choices and default:

```python
provider = Prompt.ask(
    "[bold white]Choose provider[/bold white]",
    choices=["1", "2", "3", "4", "5", "6", "7"],   # ← add "7"
    default="1",
)
```

**Location 5 — `provider_map` dict (~line 5539)**

Add entry 7:

```python
provider_map = {
    "1": "anthropic",
    "2": "bedrock",
    "3": "ollama",
    "4": "gemini",
    "5": "azure",
    "6": "openai",
    "7": "openrouter",   # ← add
}
```

**Location 6 — OpenRouter setup block**

After the `elif provider == "6":` block (ends around line 5790), add:

```python
elif provider == "7":
    # OpenRouter setup
    console.print("\n[bold white]🔀 OpenRouter Configuration[/bold white]")
    console.print(
        "OpenRouter gives you access to 600+ models via a single API key.\n"
        "Browse models at: [link]https://openrouter.ai/models[/link]\n"
    )
    console.print(
        "Get your API key from: [link]https://openrouter.ai/keys[/link]\n"
    )

    api_key = Prompt.ask(
        "[bold white]Enter your OpenRouter API key[/bold white]", password=True
    )

    if not api_key.strip():
        console.print("[red]❌ API key cannot be empty[/red]")
        raise typer.Exit(1)

    config = provider_setup.create_openrouter_config(api_key)
    if config.success:
        for key, value in config.env_vars.items():
            env_lines.append(f"{key}={value}")
        config_dict["provider"] = "openrouter"
        console.print("[green]✓ OpenRouter provider configured[/green]")

        # Optional: prompt for default model
        console.print(
            "\n[dim]Default model: anthropic/claude-sonnet-4-5[/dim]"
        )
        console.print(
            "[dim]You can override per-workspace in provider_config.json[/dim]"
        )
        custom_model = Prompt.ask(
            "[bold white]Default model (press Enter to use default)[/bold white]",
            default="anthropic/claude-sonnet-4-5",
        )
        if custom_model.strip() and custom_model.strip() != "anthropic/claude-sonnet-4-5":
            config_dict["model"] = custom_model.strip()
            console.print(f"[green]✓ Default model set to: {custom_model.strip()}[/green]")
    else:
        console.print(f"[red]❌ Configuration failed: {config.message}[/red]")
        raise typer.Exit(1)
```

**Location 7 — No-profile providers condition (~line 5839)**

Extend to include "7":

```python
elif provider in ["3", "4", "5", "6", "7"]:  # Ollama, Gemini, Azure, OpenAI, or OpenRouter
    console.print(
        "[dim]ℹ️  Note: Profile configuration not applicable for this provider[/dim]"
    )
```

**Step 1: Verify CLI changes work (smoke test)**

```bash
cd /Users/tyo/Omics-OS/lobster && python -c "
from lobster.cli import app
print('CLI imports OK')
"
```
Expected: No ImportError, prints "CLI imports OK"

**Step 2: Verify non-interactive init accepts openrouter**

```bash
python -c "
from lobster.config.constants import VALID_PROVIDERS
from lobster.config.provider_setup import create_openrouter_config
assert 'openrouter' in VALID_PROVIDERS
c = create_openrouter_config('test-key')
assert c.success
print('Non-interactive path OK')
"
```
Expected: Prints "Non-interactive path OK"

**Step 3: Commit**

```bash
git add lobster/cli.py
git commit -m "feat(openrouter): add OpenRouter as option 7 in lobster init interactive flow"
```

---

## Task 7: ConfigResolver end-to-end validation

Verify the full resolution chain works: `ConfigResolver → ProviderRegistry → OpenRouterProvider → ChatOpenAI`.

**Files:**
- Modify: `tests/unit/config/test_openrouter_config.py` (add end-to-end tests)

**Step 1: Add end-to-end tests**

```python
def test_config_resolver_accepts_openrouter(tmp_path, monkeypatch):
    """ConfigResolver resolves 'openrouter' from env var without error."""
    monkeypatch.setenv("LOBSTER_LLM_PROVIDER", "openrouter")
    from lobster.core.config_resolver import ConfigResolver
    ConfigResolver.reset_instance()
    resolver = ConfigResolver.get_instance(tmp_path)
    provider, source = resolver.resolve_provider()
    assert provider == "openrouter"
    assert "environment" in source


def test_llm_factory_creates_openrouter_model(monkeypatch):
    """LLMFactory.create_llm() can create an OpenRouter model (mocked)."""
    from unittest.mock import patch, MagicMock
    monkeypatch.setenv("LOBSTER_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    from lobster.core.config_resolver import ConfigResolver
    from lobster.config.providers.registry import ProviderRegistry
    ConfigResolver.reset_instance()
    ProviderRegistry.reset()

    with patch("langchain_openai.ChatOpenAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        from lobster.config.llm_factory import LLMFactory
        llm = LLMFactory.create_llm(
            model_config={"temperature": 0.7, "max_tokens": 4096},
            agent_name="supervisor",
        )

    assert mock_cls.called
    call_kwargs = mock_cls.call_args[1]
    assert call_kwargs["base_url"] == "https://openrouter.ai/api/v1"
```

**Step 2: Run**

```bash
python -m pytest tests/unit/config/test_openrouter_config.py -k "config_resolver or llm_factory" -v
```
Expected: PASS

**Step 3: Run the full config test suite**

```bash
python -m pytest tests/unit/config/ -v --tb=short -q
```
Expected: All pass.

**Step 4: Commit**

```bash
git add tests/unit/config/test_openrouter_config.py
git commit -m "test(openrouter): add end-to-end ConfigResolver + LLMFactory integration tests"
```

---

## Task 8: Full test suite pass + final verification

**Step 1: Run all unit tests**

```bash
python -m pytest tests/unit/ -v --tb=short -q 2>&1 | tail -20
```
Expected: All pass.

**Step 2: Run AQUADIF contract tests** (ensure no new tools were added without metadata — unlikely but safety check)

```bash
python -m pytest -m contract -v --tb=short -q 2>&1 | tail -10
```
Expected: All pass.

**Step 3: Smoke test provider discovery**

```bash
python -c "
from lobster.config.providers.registry import ProviderRegistry
ProviderRegistry.reset()
ProviderRegistry._ensure_initialized()
providers = ProviderRegistry.get_provider_names()
print('Registered providers:', providers)
assert 'openrouter' in providers, 'openrouter missing from registry!'
print('✓ openrouter in registry')

provider = ProviderRegistry.get('openrouter')
print('Provider name:', provider.name)
print('Display name:', provider.display_name)
print('Default model:', provider.get_default_model())
print('Is configured:', provider.is_configured())
print('Fallback models (first 5):', [m.name for m in provider.list_models()[:5]])
"
```
Expected: Output shows openrouter provider with correct attributes and fallback models.

**Step 4: Final commit**

```bash
git add -A
git status  # verify nothing unexpected
git commit -m "feat(openrouter): complete OpenRouter provider integration

- OpenRouterProvider with lazy live catalog fetch + 20-model fallback
- Registers in ProviderRegistry, VALID_PROVIDERS, LLMProvider enum
- openrouter_default_model / openrouter_model config fields
- create_openrouter_config() helper
- Option 7 in lobster init interactive flow
- 40+ unit tests covering all paths"
```

---

## Summary of All Changed Files

| File | Type | Change |
|------|------|--------|
| `lobster/config/constants.py` | Modify | Add `"openrouter"` to VALID_PROVIDERS + PROVIDER_DISPLAY_NAMES |
| `lobster/config/llm_factory.py` | Modify | Add `OPENROUTER = "openrouter"` to LLMProvider enum |
| `lobster/config/providers/openrouter_provider.py` | **New** | Full ILLMProvider implementation |
| `lobster/config/providers/registry.py` | Modify | Add openrouter_provider to `_provider_specs` |
| `lobster/config/providers/__init__.py` | Modify | Export OpenRouterProvider |
| `lobster/config/provider_setup.py` | Modify | Add `create_openrouter_config()` |
| `lobster/config/global_config.py` | Modify | Add `openrouter_default_model` field + reset() |
| `lobster/config/workspace_config.py` | Modify | Add `openrouter_model` field |
| `lobster/cli.py` | Modify | 7 locations: menu, choices, map, setup block, profile condition |
| `tests/unit/config/test_openrouter_provider.py` | **New** | ~40 tests for OpenRouterProvider |
| `tests/unit/config/test_openrouter_config.py` | **New** | ~12 tests for config schema + e2e |
| `tests/unit/config/test_constants.py` | Modify | Add openrouter assertions |
