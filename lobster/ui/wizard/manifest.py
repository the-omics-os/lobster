"""WizardManifest — single source of truth for init wizard data.

Consumed by:
- Ink TUI wizard (via JSON over stdin)
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
from dataclasses import asdict, dataclass, field
from typing import Literal


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


@dataclass
class WizardManifest:
    """Complete init wizard manifest — single source of truth.

    JSON-serializable for transport to Ink TUI or other consumers.
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
                    }
                )
                for p in raw.get("providers", [])
            ],
            agent_packages=[
                AgentPackageDef(**ap) for ap in raw.get("agent_packages", [])
            ],
            ollama_status=OllamaStatus(**raw.get("ollama_status", {})),
        )
