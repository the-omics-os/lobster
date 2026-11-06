"""Direct Ollama Cloud inference and lazy hosted model discovery."""

import logging
import os
import threading
from dataclasses import replace
from typing import Any, ClassVar, Optional

from lobster.config.providers.base_provider import ILLMProvider, ModelInfo

logger = logging.getLogger(__name__)
_BASE_URL = "https://ollama.com/v1"
_DEFAULT_MODEL = "glm-5.2"
# USD / million tokens, standard input/output rates verified 2026-11-05.
# https://ollama.com/pricing ; cached-input/off-peak discounts are not estimated.
_PRICING = {
    "glm-5.2": (1.40, 4.40),
    "gpt-oss:120b": (0.15, 0.60),
    "gpt-oss:20b": (0.07, 0.30),
}
_CONTEXT = {"glm-5.2": 1_000_000, "gpt-oss:120b": 131_072, "gpt-oss:20b": 131_072}


class OllamaCloudProvider(ILLMProvider):
    """Ollama's hosted service; no local daemon or model pull required."""

    _default_context_window = 8192
    _models_cache: ClassVar[Optional[list[ModelInfo]]] = None
    _cache_lock: ClassVar[threading.Lock] = threading.Lock()

    @property
    def name(self) -> str:
        return "ollama-cloud"

    @property
    def display_name(self) -> str:
        return "Ollama Cloud"

    def is_configured(self) -> bool:
        return bool(os.environ.get("OLLAMA_API_KEY", "").strip())

    def is_available(self) -> bool:
        return self.is_configured()

    def check_dependencies(self) -> None:
        try:
            import langchain_openai  # noqa: F401
        except ImportError as exc:
            from lobster.core.component_registry import get_install_command

            raise ImportError(
                f"Install Ollama Cloud support: {get_install_command('openai', is_extra=True)}"
            ) from exc

    def get_default_model(self) -> str:
        return _DEFAULT_MODEL

    @staticmethod
    def _model_info(name: str, context: Optional[int] = None) -> ModelInfo:
        prices = _PRICING.get(name, (None, None))
        return ModelInfo(
            name=name,
            display_name=name,
            description="Hosted completion and tool-calling model"
            + (" (pricing unknown)" if prices[0] is None else ""),
            provider="ollama-cloud",
            context_window=context or _CONTEXT.get(name, 8192),
            is_default=name == _DEFAULT_MODEL,
            input_cost_per_million=prices[0],
            output_cost_per_million=prices[1],
        )

    @classmethod
    def static_models(cls) -> list[ModelInfo]:
        """Pricing initialization must not trigger discovery or network requests."""
        return [cls._model_info(name) for name in _PRICING]

    def _fetch_models(self) -> list[ModelInfo]:
        import requests

        response = requests.get(
            "https://ollama.com/api/tags", timeout=5, allow_redirects=False
        )
        if response.status_code != 200:
            raise ValueError("Hosted catalog unavailable")
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
            raise ValueError("Invalid hosted catalog")
        names = set()
        for item in payload["models"]:
            if not isinstance(item, dict):
                continue
            name, caps = item.get("name"), item.get("capabilities")
            if (
                isinstance(name, str)
                and name.strip()
                and len(name) <= 256
                and not any(c.isspace() or ord(c) < 32 for c in name)
                and isinstance(caps, list)
                and "completion" in caps
                and "tools" in caps
            ):
                names.add(name)
        if not names:
            raise ValueError("No usable hosted models")
        contexts = {}
        try:
            metadata = requests.get(
                f"{_BASE_URL}/models", timeout=5, allow_redirects=False
            )
            if metadata.status_code == 200:
                data = metadata.json()
                rows = data.get("models", []) if isinstance(data, dict) else []
                if isinstance(rows, list):
                    for item in rows:
                        if not isinstance(item, dict):
                            continue
                        name, ctx = item.get("slug"), item.get("context_window")
                        if (
                            isinstance(name, str)
                            and type(ctx) is int
                            and 0 < ctx <= 10_000_000
                        ):
                            contexts[name] = ctx
        except (requests.RequestException, ValueError):
            logger.debug("Ollama Cloud context metadata unavailable")
        return [
            self._model_info(name, contexts.get(name))
            for name in sorted(names, key=lambda n: (n != _DEFAULT_MODEL, n))
        ]

    def list_models(self) -> list[ModelInfo]:
        with self._cache_lock:
            if self.__class__._models_cache is None:
                try:
                    models = self._fetch_models()
                except Exception:
                    logger.warning(
                        "Ollama Cloud discovery unavailable; using fallback catalog"
                    )
                    models = [self._model_info(name) for name in _PRICING]
                self.__class__._models_cache = models
            return [replace(model) for model in self.__class__._models_cache]

    def create_chat_model(
        self,
        model_id: str,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> Any:
        self.check_dependencies()
        from langchain_openai import ChatOpenAI

        api_key = kwargs.pop("api_key", None) or os.environ.get("OLLAMA_API_KEY", "")
        if not api_key.strip():
            raise ValueError("Set OLLAMA_API_KEY to use Ollama Cloud")
        return ChatOpenAI(
            model=model_id,
            api_key=api_key.strip(),
            base_url=_BASE_URL,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    def get_configuration_help(self) -> str:
        return "Get an API key at https://ollama.com/settings/keys and set OLLAMA_API_KEY. Default model: glm-5.2. No local Ollama installation required."


from lobster.config.providers.registry import ProviderRegistry

ProviderRegistry.register(OllamaCloudProvider())
