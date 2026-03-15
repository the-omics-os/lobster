"""Questionary-based init wizard — Python-native improved UX.

Canonical question order (matches Go TUI wizard):

  1. Provider selection  (unified list — Omics-OS is just another provider)
  2. Credentials         (provider-specific API keys)
  3. Model selection     (curated + "Other" for all cloud providers)
  4. Profile selection   (Anthropic / Bedrock only)
  5. Agent packages      (multi-select, after user knows their world)
  6. Optional keys       (NCBI, Cloud key — skip Cloud if Omics-OS)
  7. Smart Standardization (if selected agents benefit)

Returns the exact same JSON dict as the Go TUI wizard so that
``apply_tui_init_result()`` handles both paths without branching.

If ``questionary`` is not installed the function raises ``ImportError`` and
the caller falls back to the classic Rich flow in ``init_impl()``.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public contract
# ---------------------------------------------------------------------------

# Same field names the Go wizard returns.
_EMPTY_RESULT: dict = {
    "provider": "",
    "api_key": "",
    "api_key_secondary": "",
    "profile": "",
    "agents": [],
    "ncbi_key": "",
    "cloud_key": "",
    "ollama_model": "",
    "model_id": "",
    "smart_standardization_enabled": False,
    "smart_standardization_openai_key": "",
    "cancelled": False,
}

_CUSTOM_MODEL_SENTINEL = "__custom_model__"
_CUSTOM_OLLAMA_MODEL = "__custom_ollama_model__"
_DEFAULT_OLLAMA_MODEL = "gpt-oss:20b"

# ---------------------------------------------------------------------------
# Provider + model catalogs
# ---------------------------------------------------------------------------

_PROVIDER_CHOICES = [
    {
        "name": "Omics-OS Cloud          — managed, login via browser",
        "value": "omics-os",
    },
    {
        "name": "Claude API (Anthropic)  — direct access to Claude models",
        "value": "anthropic",
    },
    {"name": "AWS Bedrock             — production, enterprise", "value": "bedrock"},
    {
        "name": "Ollama (local)          — privacy, zero cost, offline",
        "value": "ollama",
    },
    {"name": "Google Gemini           — Gemini models + thinking", "value": "gemini"},
    {
        "name": "Azure AI                — enterprise Azure deployments",
        "value": "azure",
    },
    {"name": "OpenAI                  — GPT-4o, o-series reasoning", "value": "openai"},
    {
        "name": "OpenRouter              — 600+ models via one API key",
        "value": "openrouter",
    },
]

# Curated model catalogs per provider.  Keep in sync with
# lobster-tui/internal/initwizard/wizard.go → providerModels.
# Every list ends with _CUSTOM_MODEL_SENTINEL so users are never blocked
# by a stale catalog.
_PROVIDER_MODELS: dict[str, list[tuple[str, str, bool]]] = {
    # (model_id, description, is_default)
    "anthropic": [
        (
            "claude-sonnet-4-20250514",
            "Claude Sonnet 4 — balanced speed & capability",
            True,
        ),
        (
            "claude-opus-4-20250514",
            "Claude Opus 4 — most capable, complex reasoning",
            False,
        ),
        (
            "claude-3-5-sonnet-20241022",
            "Claude 3.5 Sonnet — previous generation",
            False,
        ),
        (
            "claude-3-5-haiku-20241022",
            "Claude 3.5 Haiku — fastest, high throughput",
            False,
        ),
    ],
    "bedrock": [
        (
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "Claude Sonnet 4.5 — highest quality Sonnet",
            True,
        ),
        (
            "us.anthropic.claude-sonnet-4-20250514-v1:0",
            "Claude Sonnet 4 — balanced quality & speed",
            False,
        ),
        (
            "global.anthropic.claude-opus-4-5-20251101-v1:0",
            "Claude Opus 4.5 — most capable",
            False,
        ),
    ],
    "openai": [
        ("gpt-4o", "GPT-4o — most capable GPT model", True),
        ("gpt-4o-mini", "GPT-4o Mini — fast and affordable", False),
        ("o3-mini", "o3 Mini — compact reasoning model", False),
    ],
    "gemini": [
        (
            "gemini-3-pro-preview",
            "Gemini 3 Pro — best balance of speed & capability",
            True,
        ),
        (
            "gemini-3-flash-preview",
            "Gemini 3 Flash — fastest, free tier available",
            False,
        ),
    ],
    "azure": [
        ("gpt-4o", "GPT-4o via Azure AI Foundry", True),
        ("deepseek-r1", "DeepSeek R1 reasoning model", False),
        ("phi-4", "Microsoft Phi-4 small language model", False),
    ],
    "openrouter": [
        ("anthropic/claude-sonnet-4-5", "Claude Sonnet 4.5 via OpenRouter", True),
        ("openai/gpt-4o", "GPT-4o via OpenRouter", False),
        ("google/gemini-3-pro-preview", "Gemini 3 Pro via OpenRouter", False),
    ],
}

_PROFILE_CHOICES = [
    {
        "name": "development   (Sonnet 4 — fastest, most affordable)",
        "value": "development",
    },
    {
        "name": "production    (Sonnet 4 + Sonnet 4.5 supervisor) [recommended]",
        "value": "production",
    },
    {"name": "performance   (Sonnet 4.5 — highest quality)", "value": "performance"},
    {
        "name": "max           (Opus 4.5 supervisor — most capable, most expensive)",
        "value": "max",
    },
]

# Providers that support the profile selection step
_PROFILE_PROVIDERS = {"anthropic", "bedrock"}


# ---------------------------------------------------------------------------
# Choice builders
# ---------------------------------------------------------------------------


def _build_agent_choices(questionary) -> list:
    from lobster.cli_internal.commands.heavy.init_commands import (
        _DEFAULT_TUI_AGENT_PACKAGES,
        _get_tui_agent_package_choices,
    )

    choices = []
    defaults = set(_DEFAULT_TUI_AGENT_PACKAGES)
    for pkg_name, description, agents in _get_tui_agent_package_choices():
        agent_count = len(agents)
        agent_label = "agent" if agent_count == 1 else "agents"
        label = f"{pkg_name:<26} {description} ({agent_count} {agent_label})"
        choices.append(
            questionary.Choice(
                label,
                value=pkg_name,
                checked=pkg_name in defaults,
            )
        )
    return choices


def _build_model_choices(questionary, provider: str) -> tuple[list, str]:
    """Build model selection choices for a given provider.

    Returns (choices, default_value).  Always includes an "Other" escape
    hatch so users are never blocked by a stale catalog.
    """
    models = _PROVIDER_MODELS.get(provider, [])
    if not models:
        return [], ""

    choices = []
    default_value = ""
    for model_id, description, is_default in models:
        label = f"{model_id:<48} {description}"
        choices.append(questionary.Choice(label, value=model_id))
        if is_default:
            default_value = model_id

    choices.append(
        questionary.Choice(
            "Other model                                      Enter a custom model ID",
            value=_CUSTOM_MODEL_SENTINEL,
        )
    )

    if not default_value:
        default_value = choices[0].value

    return choices, default_value


def _build_ollama_prompt(status) -> str:
    if not status.installed:
        return (
            "Choose an Ollama model. Ollama is not installed yet; install via "
            "https://ollama.com/download or `curl -fsSL https://ollama.com/install.sh | sh`."
        )
    if not status.running:
        return "Choose an Ollama model. Ollama is installed but not running; start it with `ollama serve`."
    if status.models:
        return "Choose an Ollama model. Detected local models are listed first."
    return "Choose an Ollama model. No local models were detected yet, so curated options are shown."


def _build_ollama_choices(questionary, status) -> tuple[list, str]:
    curated = [
        ("qwen3:8b", "Qwen 3 8B"),
        ("qwen3:14b", "Qwen 3 14B"),
        ("gpt-oss:20b", "GPT OSS 20B"),
        ("qwen3:30b-a3b", "Qwen 3 30B A3B"),
    ]

    choices = []
    seen = set()
    default_value = _DEFAULT_OLLAMA_MODEL

    for model_name in status.models:
        if model_name in seen:
            continue
        choices.append(
            questionary.Choice(
                f"{model_name:<24} detected locally",
                value=model_name,
            )
        )
        seen.add(model_name)

    for model_name, label in curated:
        if model_name in seen:
            continue
        choices.append(
            questionary.Choice(
                f"{model_name:<24} {label}",
                value=model_name,
            )
        )
        seen.add(model_name)

    choices.append(
        questionary.Choice(
            "Other model               Type a custom model name",
            value=_CUSTOM_OLLAMA_MODEL,
        )
    )

    if default_value not in seen:
        default_value = choices[0].value

    return choices, default_value


# ---------------------------------------------------------------------------
# Cancellation helper
# ---------------------------------------------------------------------------


def _cancelled(result: dict) -> dict:
    result["cancelled"] = True
    return result


# ---------------------------------------------------------------------------
# Main wizard
# ---------------------------------------------------------------------------


def run_questionary_init() -> dict:
    """Run questionary-based init wizard.

    Returns the same dict structure as the Go TUI wizard.

    Raises:
        ImportError: when ``questionary`` is not installed.
        KeyboardInterrupt: when the user presses Ctrl-C (callers should treat
            this as cancellation and check ``result["cancelled"]``).
    """
    import questionary  # intentional — caller catches ImportError

    from lobster.cli_internal.commands.heavy.init_commands import (
        _get_smart_standardization_beneficiaries,
        _normalize_selected_agents,
    )
    from lobster.config import provider_setup

    result = dict(_EMPTY_RESULT)

    try:
        # ================================================================== #
        # Step 1 — Provider selection (unified list)                          #
        # ================================================================== #
        provider = questionary.select(
            "Select your LLM provider:",
            choices=[
                questionary.Choice(c["name"], value=c["value"])
                for c in _PROVIDER_CHOICES
            ],
            default="omics-os",
        ).ask()

        if provider is None:
            return _cancelled(result)

        result["provider"] = provider

        # ================================================================== #
        # Step 2 — Credentials (provider-specific)                            #
        # ================================================================== #
        if provider == "omics-os":
            try:
                from lobster.cli_internal.commands.light.cloud_commands import (
                    attempt_login_for_init,
                )

                success = attempt_login_for_init()
            except Exception:
                success = False

            if not success:
                key = questionary.password(
                    "Enter your Omics-OS API key (from app.omics-os.com/settings/api-keys):"
                ).ask()
                if key is None:
                    return _cancelled(result)
                result["api_key"] = key.strip()

        elif provider == "anthropic":
            key = questionary.password(
                "Enter your Claude API key (https://console.anthropic.com):"
            ).ask()
            if key is None:
                return _cancelled(result)
            result["api_key"] = key.strip()

        elif provider == "bedrock":
            access = questionary.password("Enter your AWS access key:").ask()
            if access is None:
                return _cancelled(result)
            secret = questionary.password("Enter your AWS secret key:").ask()
            if secret is None:
                return _cancelled(result)
            result["api_key"] = access.strip()
            result["api_key_secondary"] = secret.strip()

        elif provider == "ollama":
            status = provider_setup.get_ollama_status()
            ollama_choices, default_model = _build_ollama_choices(questionary, status)
            model = questionary.select(
                _build_ollama_prompt(status),
                choices=ollama_choices,
                default=default_model,
            ).ask()
            if model is None:
                return _cancelled(result)
            if model == _CUSTOM_OLLAMA_MODEL:
                custom_model = questionary.text(
                    "Enter your Ollama model name:",
                    default=_DEFAULT_OLLAMA_MODEL,
                ).ask()
                if custom_model is None:
                    return _cancelled(result)
                result["ollama_model"] = custom_model.strip()
            else:
                result["ollama_model"] = model.strip()

        elif provider == "gemini":
            key = questionary.password(
                "Enter your Google API key (https://aistudio.google.com/apikey):"
            ).ask()
            if key is None:
                return _cancelled(result)
            result["api_key"] = key.strip()

        elif provider == "azure":
            endpoint = questionary.text(
                "Enter your Azure AI endpoint URL:",
                default="https://your-project.inference.ai.azure.com/",
            ).ask()
            if endpoint is None:
                return _cancelled(result)
            credential = questionary.password("Enter your Azure API credential:").ask()
            if credential is None:
                return _cancelled(result)
            result["api_key"] = credential.strip()
            result["api_key_secondary"] = endpoint.strip()

        elif provider == "openai":
            key = questionary.password(
                "Enter your OpenAI API key (https://platform.openai.com/api-keys):"
            ).ask()
            if key is None:
                return _cancelled(result)
            result["api_key"] = key.strip()

        elif provider == "openrouter":
            key = questionary.password(
                "Enter your OpenRouter API key (https://openrouter.ai/keys):"
            ).ask()
            if key is None:
                return _cancelled(result)
            result["api_key"] = key.strip()

        # ================================================================== #
        # Step 3 — Model selection (skip for Ollama and Omics-OS)             #
        # ================================================================== #
        if provider not in ("ollama", "omics-os"):
            model_choices, default_model = _build_model_choices(questionary, provider)
            if model_choices:
                selected_model = questionary.select(
                    "Select default model (you can change this later):",
                    choices=model_choices,
                    default=default_model,
                ).ask()
                if selected_model is None:
                    return _cancelled(result)

                if selected_model == _CUSTOM_MODEL_SENTINEL:
                    custom_id = questionary.text(
                        "Enter your model ID:",
                        default=default_model,
                    ).ask()
                    if custom_id is None:
                        return _cancelled(result)
                    result["model_id"] = custom_id.strip()
                else:
                    result["model_id"] = selected_model

        # ================================================================== #
        # Step 4 — Profile (Anthropic / Bedrock only)                         #
        # ================================================================== #
        if provider in _PROFILE_PROVIDERS:
            profile = questionary.select(
                "Select agent configuration profile:",
                choices=[
                    questionary.Choice(c["name"], value=c["value"])
                    for c in _PROFILE_CHOICES
                ],
                default="production",
            ).ask()

            if profile is None:
                return _cancelled(result)

            result["profile"] = profile

        # ================================================================== #
        # Step 5 — Agent packages                                             #
        # ================================================================== #
        selected_pkgs = questionary.checkbox(
            "Select agent packages to install (Space to toggle, Enter to confirm):",
            choices=_build_agent_choices(questionary),
        ).ask()

        if selected_pkgs is None:
            return _cancelled(result)

        result["agents"] = _normalize_selected_agents(selected_pkgs)

        # ================================================================== #
        # Step 6 — Optional NCBI key                                          #
        # ================================================================== #
        add_ncbi = questionary.confirm(
            "Add an NCBI API key? (enhances literature search, optional)",
            default=False,
        ).ask()

        if add_ncbi is None:
            return _cancelled(result)

        if add_ncbi:
            ncbi = questionary.password("Enter your NCBI API key:").ask()
            if ncbi is None:
                return _cancelled(result)
            result["ncbi_key"] = ncbi.strip()

        # ================================================================== #
        # Step 7 — Optional Cloud key (skip when already via Omics-OS Cloud)  #
        # ================================================================== #
        if provider != "omics-os":
            add_cloud = questionary.confirm(
                "Add an Omics-OS Cloud API key? (enables premium tier, optional)",
                default=False,
            ).ask()

            if add_cloud is None:
                return _cancelled(result)

            if add_cloud:
                ckey = questionary.password("Enter your Omics-OS Cloud API key:").ask()
                if ckey is None:
                    return _cancelled(result)
                result["cloud_key"] = ckey.strip()

        # ================================================================== #
        # Step 8 — Smart Standardization / vector search                      #
        # ================================================================== #
        beneficiaries = _get_smart_standardization_beneficiaries(result["agents"])
        smart_std_prompt = "Enable Smart Standardization / vector search? (OpenAI embeddings + ontology matching, optional)"
        if beneficiaries:
            smart_std_prompt = (
                "Enable Smart Standardization / vector search? Helpful for "
                + ", ".join(beneficiaries)
            )

        enable_smart_std = questionary.confirm(
            smart_std_prompt,
            default=bool(beneficiaries),
        ).ask()

        if enable_smart_std is None:
            return _cancelled(result)

        if enable_smart_std:
            result["smart_standardization_enabled"] = True
            existing_openai_key = os.environ.get("OPENAI_API_KEY", "").strip()

            if provider == "openai":
                result["smart_standardization_openai_key"] = result["api_key"]
            elif existing_openai_key:
                result["smart_standardization_openai_key"] = existing_openai_key
            else:
                embedding_key = questionary.password(
                    "Enter your OpenAI API key for embeddings:",
                ).ask()
                if embedding_key is None:
                    return _cancelled(result)
                result["smart_standardization_openai_key"] = embedding_key.strip()

    except KeyboardInterrupt:
        result["cancelled"] = True

    return result
