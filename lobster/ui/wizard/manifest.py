"""WizardManifest — single source of truth for init wizard data.

Consumed by:
- Go TUI wizard (via JSON over stdin)
- questionary fallback (Python)
- non-interactive mode (CLI flags)

Each ProviderDef declares `model_selection` to fix the Bedrock double-prompt bug:
- "explicit"  → user picks model from provider's catalog
- "profile"   → user picks profile (model controlled by profile) — Anthropic, Bedrock
- "local"     → user picks from locally detected models — Ollama
- "managed"   → skip model selection entirely — Omics-OS
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class CredentialField:
    """A single credential the user must provide for a provider."""

    key: str  # env var name, e.g. "ANTHROPIC_API_KEY"
    label: str  # display label, e.g. "API Key"
    secret: bool = True  # mask input
    required: bool = True
    env_var: str | None = None  # if different from key
    help_url: str | None = None


@dataclass
class ModelDef:
    """A model available from a provider."""

    name: str  # model ID, e.g. "claude-sonnet-4-20250514"
    display_name: str
    description: str = ""
    is_default: bool = False
    context_window: int = 0
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0


@dataclass
class ProfileDef:
    """A profile (model tier) for profile-based providers."""

    name: str  # e.g. "development"
    display_name: str  # e.g. "Development"
    description: str  # e.g. "Sonnet 4 - fastest, most affordable"
    is_default: bool = False


@dataclass
class AgentPackageDef:
    """An installable agent package."""

    package_name: str  # e.g. "lobster-research"
    description: str
    agents: list[str] = field(default_factory=list)
    published: bool = True
    experimental: bool = False


@dataclass
class OllamaStatus:
    """Result of Ollama detection."""

    available: bool = False
    models: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class AuthMethodDef:
    """An authentication method available for a provider."""

    type: str  # "oauth" or "api_key"
    label: str  # e.g. "Login with Claude account (Pro/Max)"
    is_default: bool = False


@dataclass
class ProviderDef:
    """A provider with its credentials, models/profiles, and model selection mode."""

    name: str  # e.g. "anthropic"
    display_name: str
    description: str = ""
    model_selection: Literal["explicit", "profile", "local", "managed"] = (
        "explicit"
    )
    credentials: list[CredentialField] = field(default_factory=list)
    models: list[ModelDef] = field(default_factory=list)
    profiles: list[ProfileDef] = field(default_factory=list)
    auth_methods: list[AuthMethodDef] = field(default_factory=list)


@dataclass
class WizardManifest:
    """Complete init wizard manifest — single source of truth.

    JSON-serializable for transport to Go TUI or other consumers.
    """

    providers: list[ProviderDef] = field(default_factory=list)
    agent_packages: list[AgentPackageDef] = field(default_factory=list)
    ollama_status: OllamaStatus = field(default_factory=OllamaStatus)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, data: str) -> WizardManifest:
        raw = json.loads(data)
        return cls(
            providers=[
                ProviderDef(
                    **{
                        **p,
                        "credentials": [
                            CredentialField(**c) for c in p.get("credentials", [])
                        ],
                        "models": [ModelDef(**m) for m in p.get("models", [])],
                        "profiles": [ProfileDef(**pr) for pr in p.get("profiles", [])],
                        "auth_methods": [
                            AuthMethodDef(**am) for am in p.get("auth_methods", [])
                        ],
                    }
                )
                for p in raw.get("providers", [])
            ],
            agent_packages=[
                AgentPackageDef(**ap) for ap in raw.get("agent_packages", [])
            ],
            ollama_status=OllamaStatus(**raw.get("ollama_status", {})),
        )


# -- Provider metadata (model_selection + credentials + display names) --

_MODEL_SELECTION: dict[str, Literal["explicit", "profile", "local", "managed"]] = {
    "anthropic": "profile",
    "bedrock": "profile",
    "ollama": "local",
    "gemini": "explicit",
    "azure": "explicit",
    "openai": "explicit",
    "openrouter": "explicit",
    "omics-os": "managed",
}

_PROVIDER_DISPLAY: dict[str, tuple[str, str]] = {
    "anthropic": ("Anthropic", "Claude models via Anthropic API"),
    "bedrock": ("AWS Bedrock", "Claude models via AWS Bedrock"),
    "ollama": ("Ollama", "Local models via Ollama"),
    "gemini": ("Google Gemini", "Gemini models via Google AI"),
    "azure": ("Azure AI", "Models via Azure AI"),
    "openai": ("OpenAI", "GPT models via OpenAI API"),
    "openrouter": ("OpenRouter", "Multi-provider model proxy"),
    "omics-os": ("Omics-OS Cloud", "Managed models via Omics-OS"),
}

_PROVIDER_CREDENTIALS: dict[str, list[CredentialField]] = {
    "anthropic": [
        CredentialField(key="ANTHROPIC_API_KEY", label="Anthropic API Key"),
    ],
    "bedrock": [
        CredentialField(
            key="AWS_ACCESS_KEY_ID", label="AWS Access Key ID", secret=False
        ),
        CredentialField(key="AWS_SECRET_ACCESS_KEY", label="AWS Secret Access Key"),
        CredentialField(
            key="AWS_DEFAULT_REGION",
            label="AWS Region",
            secret=False,
            required=False,
        ),
    ],
    "ollama": [],  # No credentials needed
    "gemini": [
        CredentialField(key="GOOGLE_API_KEY", label="Google API Key"),
    ],
    "azure": [
        CredentialField(key="AZURE_OPENAI_API_KEY", label="Azure OpenAI API Key"),
        CredentialField(
            key="AZURE_OPENAI_ENDPOINT",
            label="Azure OpenAI Endpoint",
            secret=False,
        ),
    ],
    "openai": [
        CredentialField(key="OPENAI_API_KEY", label="OpenAI API Key"),
    ],
    "openrouter": [
        CredentialField(key="OPENROUTER_API_KEY", label="OpenRouter API Key"),
    ],
    "omics-os": [
        CredentialField(key="OMICS_OS_API_KEY", label="Omics-OS API Key"),
    ],
}

_PROVIDER_AUTH_METHODS: dict[str, list[AuthMethodDef]] = {
    "anthropic": [
        AuthMethodDef(
            type="oauth",
            label="Login with Claude account (Pro/Max)",
            is_default=True,
        ),
        AuthMethodDef(type="api_key", label="Paste API key"),
    ],
    # Other providers: api_key only (no explicit auth_methods needed)
}

_PROFILE_DESCRIPTIONS: dict[str, tuple[str, str, bool]] = {
    "development": ("Development", "Sonnet 4 — fastest, most affordable", False),
    "production": (
        "Production",
        "Sonnet 4 + Sonnet 4.5 supervisor [recommended]",
        True,
    ),
    "performance": ("Performance", "Sonnet 4.5 — highest quality", False),
    "max": ("Max", "Opus 4.5 supervisor — most capable, most expensive", False),
    "hybrid": ("Hybrid", "Mixed model tiers per agent", False),
}


def build_init_manifest(detect_ollama: bool = True) -> WizardManifest:
    """Build a WizardManifest from authoritative sources.

    Reads VALID_PROVIDERS, provider KNOWN_MODELS, VALID_PROFILES,
    and AVAILABLE_AGENT_PACKAGES to construct the single manifest.
    """
    from lobster.cli_internal.commands.heavy.init_commands import (
        AVAILABLE_AGENT_PACKAGES,
    )
    from lobster.config.constants import VALID_PROFILES, VALID_PROVIDERS
    from lobster.config.providers.registry import ProviderRegistry

    # Build provider definitions
    providers: list[ProviderDef] = []
    for name in VALID_PROVIDERS:
        display, desc = _PROVIDER_DISPLAY.get(name, (name.title(), ""))
        model_sel = _MODEL_SELECTION.get(name, "explicit")
        creds = _PROVIDER_CREDENTIALS.get(name, [])

        # Get models from provider's KNOWN_MODELS
        models: list[ModelDef] = []
        if model_sel == "explicit":
            provider = ProviderRegistry.get(name)
            if provider and hasattr(provider, "KNOWN_MODELS"):
                for mi in provider.KNOWN_MODELS:
                    models.append(
                        ModelDef(
                            name=mi.name,
                            display_name=mi.display_name,
                            description=getattr(mi, "description", ""),
                            is_default=getattr(mi, "is_default", False),
                            context_window=getattr(mi, "context_window", 0),
                            input_cost_per_million=getattr(
                                mi, "input_cost_per_million", 0.0
                            ),
                            output_cost_per_million=getattr(
                                mi, "output_cost_per_million", 0.0
                            ),
                        )
                    )

        # Get profiles for profile-based providers
        profiles: list[ProfileDef] = []
        if model_sel == "profile":
            for pname in VALID_PROFILES:
                if pname in _PROFILE_DESCRIPTIONS:
                    pdisp, pdesc, pdefault = _PROFILE_DESCRIPTIONS[pname]
                    profiles.append(
                        ProfileDef(
                            name=pname,
                            display_name=pdisp,
                            description=pdesc,
                            is_default=pdefault,
                        )
                    )

        auth_methods = _PROVIDER_AUTH_METHODS.get(name, [])

        providers.append(
            ProviderDef(
                name=name,
                display_name=display,
                description=desc,
                model_selection=model_sel,
                credentials=creds,
                models=models,
                profiles=profiles,
                auth_methods=auth_methods,
            )
        )

    # Build agent packages
    agent_packages: list[AgentPackageDef] = [
        AgentPackageDef(
            package_name=pkg[0],
            description=pkg[1],
            agents=list(pkg[2]),
            published=pkg[3],
            experimental=pkg[4],
        )
        for pkg in AVAILABLE_AGENT_PACKAGES
    ]

    # Detect Ollama
    ollama = OllamaStatus()
    if detect_ollama:
        try:
            from lobster.config.providers.ollama_provider import OllamaProvider

            op = OllamaProvider()
            if op.is_available():
                ollama.available = True
                known = op.get_available_models()
                ollama.models = [m.name for m in known]
        except Exception as e:
            ollama.error = str(e)
            logger.debug(f"Ollama detection failed: {e}")

    return WizardManifest(
        providers=providers,
        agent_packages=agent_packages,
        ollama_status=ollama,
    )
