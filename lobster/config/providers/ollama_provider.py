"""
Ollama provider for local LLM inference.

This provider enables Lobster to use locally-hosted Ollama models,
providing privacy, zero cost, and offline capability. It dynamically
discovers available models and auto-selects the best one based on size
and capability heuristics.

Example:
    >>> from lobster.config.providers import get_provider
    >>>
    >>> provider = get_provider("ollama")
    >>> if provider and provider.is_available():
    ...     models = provider.list_models()
    ...     llm = provider.create_chat_model(provider.get_default_model())
"""

import logging
import os
import re
import threading
from typing import Any, List, Optional

from lobster.config.providers.base_provider import ILLMProvider, ModelInfo

logger = logging.getLogger(__name__)


class OllamaProvider(ILLMProvider):
    """
    Provider for local Ollama models.

    Ollama enables running large language models locally with GPU acceleration.
    This provider dynamically discovers installed models and auto-selects the
    best available model based on size and capability heuristics.

    Key Features:
        - Dynamic model discovery via HTTP API
        - Smart model selection (prefers larger instruct models)
        - Zero-cost inference with local GPU
        - Privacy-focused (no data sent to cloud)
        - Offline capability

    Configuration:
        - OLLAMA_BASE_URL: Ollama server URL (default: "http://localhost:11434")
        - OLLAMA_DEFAULT_MODEL: Explicit model override (bypasses auto-selection)
        - OLLAMA_NUM_CTX: Explicit context window override (bypasses auto-detection)

    Auto-Selection Priority:
        1. OLLAMA_DEFAULT_MODEL environment variable (explicit override)
        2. Best available model by heuristic (70B > 8x7B > 13B > 8B)
        3. Fallback default: "gpt-oss:20b"

    Example:
        >>> provider = OllamaProvider()
        >>> if provider.is_available():
        ...     # List models
        ...     for model in provider.list_models():
        ...         print(f"{model.name}: {model.description}")
        ...
        ...     # Create LLM with auto-selected model
        ...     llm = provider.create_chat_model(
        ...         provider.get_default_model(),
        ...         temperature=0.7
        ...     )
    """

    def __init__(self):
        """Initialize Ollama provider with default configuration."""
        self._base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

    @property
    def name(self) -> str:
        """
        Provider identifier.

        Returns:
            str: Always returns "ollama"
        """
        return "ollama"

    @property
    def display_name(self) -> str:
        """
        Human-friendly provider name.

        Returns:
            str: Display name for UI/CLI
        """
        return "Ollama (Local)"

    def check_dependencies(self) -> None:
        try:
            import langchain_ollama  # noqa: F401
        except ImportError:
            from lobster.core.component_registry import get_install_command

            cmd = get_install_command("ollama", is_extra=True)
            raise ImportError(
                f"langchain-ollama package not installed. Install with: {cmd}"
            )

    def is_configured(self) -> bool:
        """
        Check if Ollama configuration is present.

        For Ollama, we consider it configured if either:
        1. OLLAMA_BASE_URL environment variable is set, OR
        2. Default localhost endpoint exists (always True)

        Returns:
            bool: Always True (Ollama uses default localhost if not configured)
        """
        # Ollama is considered "configured" if base URL is set or defaults to localhost
        return True

    def is_available(self) -> bool:
        """
        Check if Ollama server is accessible (health check).

        Performs an HTTP GET to /api/tags endpoint with 2-second timeout.
        This verifies the Ollama server is running and responsive.

        Returns:
            bool: True if Ollama server responds with 200 OK

        Example:
            >>> provider = OllamaProvider()
            >>> if provider.is_available():
            ...     print("Ollama is running")
            ... else:
            ...     print("Start Ollama: ollama serve")
        """
        try:
            import requests

            response = requests.get(f"{self._base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Ollama server not accessible at {self._base_url}: {e}")
            return False

    def _fetch_models(self) -> List[ModelInfo]:
        """
        Fetch models from Ollama without marking defaults.

        This is the low-level fetcher used by both ``list_models`` and
        ``get_default_model``.  Keeping it separate breaks the mutual
        recursion that previously existed between those two methods.
        """
        try:
            import requests

            response = requests.get(f"{self._base_url}/api/tags", timeout=5)

            if response.status_code != 200:
                logger.warning(
                    f"Ollama API returned status {response.status_code} at {self._base_url}"
                )
                return []

            data = response.json()
            models = data.get("models", [])

            model_infos = []
            for model in models:
                model_name = model.get("name", "unknown")
                size_bytes = model.get("size", 0)
                details = model.get("details", {})

                model_info = ModelInfo(
                    name=model_name,
                    display_name=self._format_display_name(model_name),
                    description=self._generate_description(
                        size_bytes,
                        details.get("parameter_size"),
                        details.get("family"),
                    ),
                    provider="ollama",
                    context_window=self._estimate_context_window(model_name),
                    is_default=False,
                    input_cost_per_million=0.0,
                    output_cost_per_million=0.0,
                )
                model_infos.append(model_info)

            model_infos.sort(
                key=lambda m: self._extract_size_bytes_from_description(m.description),
                reverse=True,
            )
            return model_infos

        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []

    def list_models(self) -> List[ModelInfo]:
        """
        List all available models from Ollama instance.

        Queries the /api/tags endpoint to dynamically discover installed models.
        Models are sorted by size (largest first) for better UX.

        Returns:
            List[ModelInfo]: Available models with metadata (name, size, parameters)

        Example:
            >>> provider = OllamaProvider()
            >>> for model in provider.list_models():
            ...     print(f"{model.name}: {model.description}")
            ...     # Output: "llama3:70b-instruct: 40.5GB - 70B params"
        """
        model_infos = self._fetch_models()

        # Mark default model using the SAME list (single fetch, no divergence)
        if model_infos:
            default_model_name = self._pick_default(model_infos)
            for model in model_infos:
                if model.name == default_model_name:
                    model.is_default = True
                    break

        return model_infos

    def _pick_default(self, models: List[ModelInfo]) -> str:
        """Select the best model from a pre-fetched list by heuristic."""
        instruct_models = [
            m
            for m in models
            if "instruct" in m.name.lower() or "chat" in m.name.lower()
        ]
        if not instruct_models:
            instruct_models = models

        best = max(instruct_models, key=lambda m: self._score_model(m.name))
        return best.name

    def get_default_model(self) -> str:
        """
        Get the recommended default model for this provider.

        Selection strategy (in priority order):
        1. OLLAMA_DEFAULT_MODEL environment variable (explicit override)
        2. Best available model by heuristic (70B > 8x7B > 13B > 8B)
        3. Fallback default: "gpt-oss:20b"

        Returns:
            str: Default model identifier

        Example:
            >>> provider = OllamaProvider()
            >>> model = provider.get_default_model()
            >>> print(model)  # "llama3:70b-instruct"
        """
        # 1. Check environment variable first
        env_model = os.environ.get("OLLAMA_DEFAULT_MODEL")
        if env_model:
            logger.debug(f"Using OLLAMA_DEFAULT_MODEL: {env_model}")
            return env_model

        # 2. Auto-select best available model (single fetch)
        models = self._fetch_models()

        if not models:
            logger.warning("No Ollama models detected, using default: gpt-oss:20b")
            return "gpt-oss:20b"

        best_name = self._pick_default(models)
        logger.info(
            f"Auto-selected Ollama model: {best_name} from {len(models)} available models"
        )
        return best_name

    def create_chat_model(
        self,
        model_id: str,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> Any:
        """
        Create a LangChain ChatOllama instance.

        Args:
            model_id: Ollama model identifier (e.g., "llama3:70b-instruct")
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens in response (ignored by Ollama)
            **kwargs: Additional parameters for ChatOllama

        Returns:
            ChatOllama: LangChain chat model instance

        Raises:
            ImportError: If langchain-ollama package not installed

        Example:
            >>> provider = OllamaProvider()
            >>> llm = provider.create_chat_model(
            ...     "llama3:70b-instruct",
            ...     temperature=0.7
            ... )
        """
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            from lobster.core.component_registry import get_install_command

            cmd = get_install_command("ollama", is_extra=True)
            raise ImportError(
                f"langchain-ollama package not installed. " f"Install with: {cmd}"
            )

        # Build parameters for ChatOllama
        ollama_params = {
            "model": model_id,
            "temperature": temperature,
            **kwargs,
        }

        # Add base URL if custom endpoint
        if self._base_url != "http://localhost:11434":
            ollama_params["base_url"] = self._base_url

        # Set keep_alive to prevent Ollama from evicting the model mid-session.
        # Ollama's default is 5 minutes, which causes re-loads during longer
        # conversations. Default 30m covers typical analysis sessions.
        # Configurable via OLLAMA_KEEP_ALIVE env var.
        keep_alive = os.environ.get("OLLAMA_KEEP_ALIVE", "30m")
        ollama_params["keep_alive"] = keep_alive

        # Set num_ctx to ensure the model sees the full supervisor prompt + tools.
        # Priority: OLLAMA_NUM_CTX env var > /api/show model metadata > 8192 floor.
        # Without this, Ollama's low default (2048-4096) silently truncates the
        # prompt, causing the model to lose its instructions and tool definitions.
        num_ctx = self._resolve_num_ctx(model_id)
        ollama_params["num_ctx"] = num_ctx

        logger.debug(
            f"Creating ChatOllama with model '{model_id}', "
            f"num_ctx={num_ctx}, keep_alive={keep_alive}"
        )
        return ChatOllama(**ollama_params)

    def preload_model(self, model_id: str, keep_alive: str = "30m") -> None:
        """
        Preload a model into Ollama's VRAM by sending an empty generate request.

        Ollama loads model weights on first inference. This method triggers that
        loading eagerly so the model is warm when the user sends their first query.
        Intended to be called in a background daemon thread during CLI startup.

        Args:
            model_id: Ollama model identifier (e.g., "qwen3:8b")
            keep_alive: How long to keep the model resident after loading
                        (default: "30m"). Supports Ollama duration syntax.
        """
        try:
            import requests

            resp = requests.post(
                f"{self._base_url}/api/generate",
                json={"model": model_id, "prompt": "", "keep_alive": keep_alive},
                timeout=120,
            )
            if resp.status_code == 200:
                logger.debug(f"Preloaded model '{model_id}' (keep_alive={keep_alive})")
            else:
                logger.debug(f"Preload returned {resp.status_code} for '{model_id}'")
        except Exception as e:
            logger.debug(f"Preload failed for '{model_id}': {e}")

    def preload_model_async(self, model_id: str, keep_alive: str = "30m") -> None:
        """
        Fire preload_model() in a background daemon thread.

        Call this during CLI startup to overlap model loading with client
        initialization. The thread is daemonic so it won't block process exit.

        Args:
            model_id: Ollama model identifier
            keep_alive: How long to keep the model resident
        """
        thread = threading.Thread(
            target=self.preload_model,
            args=(model_id, keep_alive),
            daemon=True,
        )
        thread.start()
        logger.debug(f"Started background preload for '{model_id}'")

    def validate_model(self, model_id: str) -> bool:
        """
        Check if a model exists locally in Ollama.

        Args:
            model_id: Model identifier to validate

        Returns:
            bool: True if model is installed

        Example:
            >>> provider = OllamaProvider()
            >>> if provider.validate_model("llama3:70b-instruct"):
            ...     print("Model is ready")
            ... else:
            ...     print("Pull model: ollama pull llama3:70b-instruct")
        """
        available_models = [m.name for m in self.list_models()]
        return model_id in available_models

    def get_configuration_help(self) -> str:
        """
        Get help text for configuring Ollama.

        Returns:
            str: Configuration instructions with quickstart commands
        """
        return (
            "Ollama (Local) Configuration:\n\n"
            "1. Install Ollama: https://ollama.ai/download\n"
            "2. Start Ollama: ollama serve\n"
            "3. Pull a model: ollama pull llama3:70b-instruct\n\n"
            "Environment Variables:\n"
            "  OLLAMA_BASE_URL: Server URL (default: http://localhost:11434)\n"
            "  OLLAMA_DEFAULT_MODEL: Model name (default: auto-select best)\n"
            "  OLLAMA_NUM_CTX: Context window override (default: auto-detect from model)\n\n"
            "Current Configuration:\n"
            f"  Base URL: {self._base_url}\n"
            f"  Server Available: {self.is_available()}\n"
            f"  Default Model: {self.get_default_model()}"
        )

    # ---- Private Helper Methods ----

    def _format_display_name(self, model_name: str) -> str:
        """
        Format model name for display.

        Args:
            model_name: Raw model name (e.g., "llama3:70b-instruct")

        Returns:
            str: Human-friendly name (e.g., "Llama 3 70B Instruct")
        """
        # Replace colons/hyphens with spaces, title case
        display = model_name.replace(":", " ").replace("-", " ")
        return display.title()

    def _generate_description(
        self,
        size_bytes: int,
        parameter_size: Optional[str],
        family: Optional[str],
    ) -> str:
        """
        Generate model description from metadata.

        Args:
            size_bytes: Model size in bytes
            parameter_size: Parameter count (e.g., "70B")
            family: Model family (e.g., "llama")

        Returns:
            str: Formatted description (e.g., "40.5GB - 70B params (llama family)")
        """
        parts = [self._format_size(size_bytes)]

        if parameter_size:
            parts.append(f"{parameter_size} params")

        if family:
            parts.append(f"({family} family)")

        return " - ".join(parts)

    def _format_size(self, size_bytes: int) -> str:
        """
        Format bytes as human-readable size.

        Args:
            size_bytes: Size in bytes

        Returns:
            str: Formatted size (e.g., "40.5GB")
        """
        size = float(size_bytes)
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024.0:
                return f"{size:.1f}{unit}"
            size /= 1024.0
        return f"{size:.1f}PB"

    def _extract_size_bytes_from_description(self, description: str) -> float:
        """
        Extract size in bytes from description for sorting.

        Args:
            description: Model description with size

        Returns:
            float: Size in bytes (0 if not found)
        """
        # Match patterns like "40.5GB", "4.7MB"
        match = re.search(r"(\d+\.?\d*)(GB|MB|KB|B)", description)
        if not match:
            return 0.0

        value = float(match.group(1))
        unit = match.group(2)

        # Convert to bytes
        multipliers = {
            "B": 1,
            "KB": 1024,
            "MB": 1024**2,
            "GB": 1024**3,
            "TB": 1024**4,
        }
        return value * multipliers.get(unit, 1)

    # ---- Context Window Resolution ----

    _MIN_NUM_CTX = 8192

    def _resolve_num_ctx(self, model_id: str) -> int:
        """
        Determine the optimal num_ctx for a model.

        Resolution order:
        1. OLLAMA_NUM_CTX environment variable (power-user override)
        2. Model's native context_length from /api/show metadata
        3. Minimum floor (8192)

        This ensures the full supervisor prompt + tool schemas (~6-7K tokens)
        always fit, while respecting each model's trained capacity. Ollama
        handles VRAM management automatically — if the context requires more
        VRAM than available, it offloads layers to CPU.

        Args:
            model_id: Ollama model identifier

        Returns:
            int: Context window size to pass as num_ctx
        """
        # 1. Explicit env var override
        env_ctx = os.environ.get("OLLAMA_NUM_CTX")
        if env_ctx:
            try:
                val = int(env_ctx)
                logger.debug(f"Using OLLAMA_NUM_CTX={val} from environment")
                return val
            except ValueError:
                logger.warning(
                    f"Invalid OLLAMA_NUM_CTX='{env_ctx}', falling back to auto-detection"
                )

        # 2. Query model's native context length from Ollama
        model_ctx = self._get_model_context_length(model_id)
        if model_ctx > 0:
            result = max(model_ctx, self._MIN_NUM_CTX)
            logger.debug(
                f"Model '{model_id}' native context_length={model_ctx}, "
                f"using num_ctx={result}"
            )
            return result

        # 3. Minimum floor fallback
        logger.debug(
            f"Could not detect context length for '{model_id}', "
            f"using minimum num_ctx={self._MIN_NUM_CTX}"
        )
        return self._MIN_NUM_CTX

    def _get_model_context_length(self, model_id: str) -> int:
        """
        Query Ollama's /api/show endpoint for the model's trained context length.

        The response contains model_info with architecture-specific keys like
        "qwen3.context_length" or "llama.context_length". This is the model
        author's specification — the most reliable source of truth.

        Args:
            model_id: Ollama model identifier

        Returns:
            int: Trained context length, or 0 if detection fails
        """
        try:
            import requests

            resp = requests.post(
                f"{self._base_url}/api/show",
                json={"model": model_id},
                timeout=5,
            )
            if resp.status_code != 200:
                logger.debug(f"/api/show returned {resp.status_code} for '{model_id}'")
                return 0

            data = resp.json()
            model_info = data.get("model_info", {})

            # Find <architecture>.context_length key
            for key, value in model_info.items():
                if key.endswith(".context_length"):
                    ctx_len = int(value)
                    logger.debug(f"Detected {key}={ctx_len} for '{model_id}'")
                    return ctx_len

            logger.debug(f"No context_length key found in model_info for '{model_id}'")
            return 0

        except Exception as e:
            logger.debug(f"Failed to query /api/show for '{model_id}': {e}")
            return 0

    def _estimate_context_window(self, model_name: str) -> Optional[int]:
        """
        Get context window size for a model.

        Attempts live detection via /api/show first, falls back to
        name-based heuristics for offline estimation.

        Args:
            model_name: Model identifier

        Returns:
            Optional[int]: Context window size (tokens) or None
        """
        # Try live detection first
        live_ctx = self._get_model_context_length(model_name)
        if live_ctx > 0:
            return live_ctx

        # Fallback: name-based heuristics
        model_lower = model_name.lower()

        if "llama3" in model_lower or "llama-3" in model_lower:
            return 8192
        elif "mixtral" in model_lower:
            return 32768
        elif "gpt-oss" in model_lower:
            return 8192
        else:
            return None

    def _score_model(self, model_name: str) -> int:
        """
        Score model by quality heuristic (higher = better).

        Scoring factors:
        - Instruct/chat models: +100 points
        - Parameter size: +1 point per billion parameters
        - Model version: +10 points per version number

        Args:
            model_name: Model identifier

        Returns:
            int: Quality score
        """
        score = 0
        model_lower = model_name.lower()

        # Prefer instruct/chat models
        if "instruct" in model_lower or "chat" in model_lower:
            score += 100

        # Size-based scoring (extract parameter count)
        # Examples: "llama3:8b", "gpt-oss:20b", "mixtral:8x7b"
        size_match = re.search(r"(\d+)(?:x)?(\d+)?b", model_lower)
        if size_match:
            main_size = int(size_match.group(1))
            multiplier = int(size_match.group(2)) if size_match.group(2) else 1
            total_params = main_size * multiplier
            score += total_params  # Larger models score higher

        # Prefer newer versions (llama3 > llama2)
        version_match = re.search(r"(\d+)", model_lower)
        if version_match:
            version = int(version_match.group(1))
            score += version * 10

        return score


# Auto-register with ProviderRegistry on import
from lobster.config.providers.registry import ProviderRegistry

ProviderRegistry.register(OllamaProvider())
