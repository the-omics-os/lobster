"""
Azure AI provider implementation.

Provides access to models in Azure AI Foundry including OpenAI GPT-4o,
DeepSeek R1, Cohere, Phi, and Mistral via customer-managed Azure resources.
Enables enterprise customers (e.g., Biognosys) to use their Azure credentials.

Architecture:
    - Implements ILLMProvider interface
    - Uses AzureAIChatCompletionsModel from langchain-azure-ai
    - Requires AZURE_AI_CREDENTIAL + AZURE_AI_ENDPOINT

Reference: https://docs.langchain.com → Azure AI section
GitHub: https://github.com/langchain-ai/langchain-azure

Example:
    >>> from lobster.config.providers.azure_provider import AzureProvider
    >>> provider = AzureProvider()
    >>> if provider.is_configured():
    ...     llm = provider.create_chat_model("gpt-4o")
"""

import logging
import os
from typing import Any, List

from lobster.config.providers.base_provider import ILLMProvider, ModelInfo

logger = logging.getLogger(__name__)


class AzureProvider(ILLMProvider):
    """
    Azure AI provider for enterprise deployments.

    Enables customers to use Azure AI Foundry models with Lobster.
    Supports GPT-4o, DeepSeek R1, Cohere, Phi, Mistral, and more.
    Requires Azure-specific configuration (endpoint + API key/credential).

    Environment Variables:
        AZURE_AI_ENDPOINT: Azure AI Foundry endpoint URL
        AZURE_AI_CREDENTIAL: Azure API key or credential
        AZURE_AI_API_VERSION: API version (default: 2024-05-01-preview)

    Note:
        Unlike pure Azure OpenAI, Azure AI Foundry supports multiple
        model providers. Model names are standard (e.g., "gpt-4o")
        rather than custom deployment names.

    Usage:
        >>> provider = AzureProvider()
        >>> if provider.is_configured():
        ...     llm = provider.create_chat_model("gpt-4o")
    """

    # Azure AI Foundry supported models
    # Source: Azure AI Model Catalog (January 2025)
    MODELS = [
        ModelInfo(
            name="gpt-4o",
            display_name="GPT-4o (Azure AI)",
            description="OpenAI GPT-4o via Azure AI Foundry - recommended",
            provider="azure",
            context_window=128000,
            is_default=True,
            input_cost_per_million=5.00,
            output_cost_per_million=15.00,
        ),
        ModelInfo(
            name="gpt-4-turbo",
            display_name="GPT-4 Turbo (Azure AI)",
            description="OpenAI GPT-4 Turbo via Azure AI Foundry",
            provider="azure",
            context_window=128000,
            is_default=False,
            input_cost_per_million=10.00,
            output_cost_per_million=30.00,
        ),
        ModelInfo(
            name="gpt-35-turbo",
            display_name="GPT-3.5 Turbo (Azure AI)",
            description="OpenAI GPT-3.5 Turbo - fast and economical",
            provider="azure",
            context_window=16385,
            is_default=False,
            input_cost_per_million=0.50,
            output_cost_per_million=1.50,
        ),
        ModelInfo(
            name="deepseek-r1",
            display_name="DeepSeek R1 (Azure AI)",
            description="DeepSeek R1 reasoning model via Azure AI Foundry",
            provider="azure",
            context_window=128000,
            is_default=False,
            input_cost_per_million=0.55,
            output_cost_per_million=2.19,
        ),
        ModelInfo(
            name="cohere-command-r-plus",
            display_name="Cohere Command R+ (Azure AI)",
            description="Cohere Command R+ via Azure AI Foundry",
            provider="azure",
            context_window=128000,
            is_default=False,
            input_cost_per_million=3.00,
            output_cost_per_million=15.00,
        ),
        ModelInfo(
            name="phi-4",
            display_name="Phi-4 (Azure AI)",
            description="Microsoft Phi-4 small language model",
            provider="azure",
            context_window=16384,
            is_default=False,
            input_cost_per_million=0.07,
            output_cost_per_million=0.14,
        ),
        ModelInfo(
            name="mistral-large",
            display_name="Mistral Large (Azure AI)",
            description="Mistral Large via Azure AI Foundry",
            provider="azure",
            context_window=128000,
            is_default=False,
            input_cost_per_million=4.00,
            output_cost_per_million=12.00,
        ),
    ]

    @property
    def name(self) -> str:
        """
        Provider identifier.

        Returns:
            str: "azure"
        """
        return "azure"

    @property
    def display_name(self) -> str:
        """
        Human-friendly provider name.

        Returns:
            str: "Azure AI"
        """
        return "Azure AI"

    def is_configured(self) -> bool:
        """
        Check if Azure AI credentials are present.

        Requires BOTH endpoint and credential to be set.
        Supports both new (AZURE_AI_*) and legacy (AZURE_OPENAI_*) env vars.

        Returns:
            bool: True if required configuration is present

        Example:
            >>> provider = AzureProvider()
            >>> if not provider.is_configured():
            ...     print("Set AZURE_AI_CREDENTIAL and AZURE_AI_ENDPOINT in .env")
        """
        # Check new langchain-azure-ai env vars first
        credential = os.environ.get("AZURE_AI_CREDENTIAL")
        endpoint = os.environ.get("AZURE_AI_ENDPOINT")

        # Fall back to legacy Azure OpenAI env vars for backwards compatibility
        if not credential:
            credential = os.environ.get("AZURE_OPENAI_API_KEY")
        if not endpoint:
            endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")

        return bool(credential and credential.strip() and endpoint and endpoint.strip())

    def is_available(self) -> bool:
        """
        Check if Azure AI is accessible.

        For cloud providers like Azure, availability equals configuration
        (no local service to health-check). Actual credential validation
        happens at first API call.

        Returns:
            bool: True if configured (cloud always available if configured)

        Note:
            Unlike Ollama, we don't ping the API here to avoid unnecessary
            latency and API charges. Invalid credentials fail gracefully at runtime.
        """
        return self.is_configured()

    def list_models(self) -> List[ModelInfo]:
        """
        List Azure AI Foundry supported models.

        Returns static catalog of models available through Azure AI Foundry
        including OpenAI GPT, DeepSeek, Cohere, Phi, and Mistral models.
        Models are ordered by capability (most capable first).

        Returns:
            List[ModelInfo]: Available models with pricing information

        Example:
            >>> provider = AzureProvider()
            >>> for model in provider.list_models():
            ...     print(f"{model.display_name}: ${model.input_cost_per_million}/M tokens")
        """
        return self.MODELS.copy()

    def get_default_model(self) -> str:
        """
        Get the recommended default model.

        Returns:
            str: "gpt-4o" (GPT-4o is the recommended default for Azure AI)

        Example:
            >>> provider = AzureProvider()
            >>> model_id = provider.get_default_model()
            >>> llm = provider.create_chat_model(model_id)
        """
        for model in self.MODELS:
            if model.is_default:
                return model.name
        return self.MODELS[0].name  # Fallback to first model

    def validate_model(self, model_id: str) -> bool:
        """
        Check if a model ID is valid for Azure AI.

        Azure AI Foundry uses standard model names. We validate against
        known models but allow unknown names for future compatibility.

        Args:
            model_id: Model identifier (e.g., "gpt-4o", "deepseek-r1")

        Returns:
            bool: True if model exists or is unknown (allows future models)

        Example:
            >>> provider = AzureProvider()
            >>> if provider.validate_model("gpt-4o"):
            ...     llm = provider.create_chat_model("gpt-4o")
        """
        known_models = {m.name for m in self.MODELS}
        if model_id in known_models:
            return True
        # Allow unknown models (Azure may add new ones)
        logger.warning(
            f"Model '{model_id}' not in known Azure AI models. "
            "Proceeding anyway (may work if deployed in your Azure account)."
        )
        return True

    def create_chat_model(
        self,
        model_id: str,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> Any:
        """
        Create an AzureAIChatCompletionsModel instance.

        Instantiates a LangChain AzureAIChatCompletionsModel with the specified
        parameters. Uses AZURE_AI_CREDENTIAL and AZURE_AI_ENDPOINT from environment.

        Args:
            model_id: Model name (e.g., "gpt-4o", "deepseek-r1")
            temperature: Sampling temperature (0.0-2.0, default 1.0)
            max_tokens: Maximum tokens in response (default 4096)
            **kwargs: Additional AzureAIChatCompletionsModel parameters

        Returns:
            AzureAIChatCompletionsModel: Configured LangChain chat model

        Raises:
            ImportError: If langchain-azure-ai not installed
            ValueError: If credentials not configured

        Environment Variables:
            AZURE_AI_ENDPOINT: Required - Azure AI Foundry endpoint
            AZURE_AI_CREDENTIAL: Required - Azure API key
            AZURE_AI_API_VERSION: Optional (default: 2024-05-01-preview)

        Example:
            >>> provider = AzureProvider()
            >>> llm = provider.create_chat_model(
            ...     "gpt-4o",
            ...     temperature=0.7
            ... )
            >>> response = llm.invoke("Hello!")

        Notes:
            - Reads AZURE_AI_CREDENTIAL from environment by default
            - Override with credential kwarg if needed
            - Also supports legacy AZURE_OPENAI_* env vars
        """
        try:
            from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
        except ImportError:
            raise ImportError(
                "langchain-azure-ai package not installed. "
                "Install with: pip install lobster-ai[azure]"
            )

        # Get configuration (support both new and legacy env vars)
        credential = kwargs.pop("credential", None) or os.environ.get(
            "AZURE_AI_CREDENTIAL"
        )
        if not credential:
            credential = os.environ.get("AZURE_OPENAI_API_KEY")

        endpoint = kwargs.pop("endpoint", None) or os.environ.get("AZURE_AI_ENDPOINT")
        if not endpoint:
            endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")

        api_version = kwargs.pop("api_version", None) or os.environ.get(
            "AZURE_AI_API_VERSION", "2024-05-01-preview"
        )

        if not credential:
            raise ValueError(
                "Azure AI credential not found. "
                "Set with: export AZURE_AI_CREDENTIAL=your-key\n"
                "Or legacy: export AZURE_OPENAI_API_KEY=your-key"
            )

        if not endpoint:
            raise ValueError(
                "Azure AI endpoint not found. "
                "Set with: export AZURE_AI_ENDPOINT=https://your-endpoint\n"
                "Or legacy: export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/"
            )

        logger.debug(
            f"Creating AzureAIChatCompletionsModel: model={model_id}, "
            f"endpoint={endpoint[:50]}..."
        )

        # Build model kwargs
        model_kwargs = {
            "model_name": model_id,
            "api_version": api_version,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "endpoint": endpoint,
            "credential": credential,
        }

        # Merge any additional kwargs
        model_kwargs.update(kwargs)

        return AzureAIChatCompletionsModel(**model_kwargs)

    def get_configuration_help(self) -> str:
        """
        Get help text for configuring Azure AI provider.

        Returns:
            str: Configuration instructions for the user

        Example:
            >>> provider = AzureProvider()
            >>> if not provider.is_configured():
            ...     print(provider.get_configuration_help())
        """
        return (
            "Configure Azure AI:\n\n"
            "1. Access Azure AI Foundry in Azure Portal\n"
            "2. Deploy a model (e.g., gpt-4o, deepseek-r1) to your project\n"
            "3. Get endpoint URL and API key from deployment details\n"
            "4. Set environment variables:\n\n"
            "   export AZURE_AI_ENDPOINT=https://your-endpoint.inference.ai.azure.com/\n"
            "   export AZURE_AI_CREDENTIAL=your-api-key\n"
            "   export AZURE_AI_API_VERSION=2024-05-01-preview  # Optional\n\n"
            "Or add to .env file:\n"
            "   AZURE_AI_ENDPOINT=https://...\n"
            "   AZURE_AI_CREDENTIAL=...\n\n"
            "Legacy Azure OpenAI env vars also supported:\n"
            "   AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY\n\n"
            "Supported models: gpt-4o, gpt-4-turbo, gpt-35-turbo,\n"
            "                  deepseek-r1, cohere-command-r-plus,\n"
            "                  phi-4, mistral-large\n\n"
            f"Default model: {self.get_default_model()}\n"
            f"Available models: {', '.join(self.get_model_names())}"
        )


# Auto-register provider with registry
from lobster.config.providers.registry import ProviderRegistry

ProviderRegistry.register(AzureProvider())
