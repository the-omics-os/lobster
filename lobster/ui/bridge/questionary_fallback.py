"""Questionary-based init wizard — Python-native improved UX.

This is the ``--ui classic`` path, upgraded from numbered ``Prompt.ask()``
prompts to arrow-key navigation via ``questionary``.  It returns the exact
same JSON dict as the Go TUI wizard so that ``apply_tui_init_result()``
handles both paths without any branching.

If ``questionary`` is not installed the function raises ``ImportError`` and
the caller falls back to the classic Rich ``Prompt.ask`` flow already
implemented in ``init_impl()``.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public contract
# ---------------------------------------------------------------------------

# Same field names the Go wizard returns.
_EMPTY_RESULT: dict = {
    "provider":           "",
    "api_key":            "",
    "api_key_secondary":  "",
    "profile":            "",
    "agents":             [],
    "ncbi_key":           "",
    "cloud_key":          "",
    "ollama_model":       "",
    "smart_standardization_enabled": False,
    "smart_standardization_openai_key": "",
    "cancelled":          False,
}

_CUSTOM_OLLAMA_MODEL = "__custom_ollama_model__"
_DEFAULT_OLLAMA_MODEL = "gpt-oss:20b"

_PROVIDER_CHOICES = [
    {"name": "Omics-OS Cloud          — managed Bedrock, login via browser", "value": "omics-os"},
    {"name": "Claude API (Anthropic)  — quick testing, development",    "value": "anthropic"},
    {"name": "AWS Bedrock             — production, enterprise",         "value": "bedrock"},
    {"name": "Ollama (local)          — privacy, zero cost, offline",    "value": "ollama"},
    {"name": "Google Gemini           — latest models + thinking",       "value": "gemini"},
    {"name": "Azure AI                — enterprise Azure deployments",   "value": "azure"},
    {"name": "OpenAI                  — GPT-4o, o1 reasoning models",   "value": "openai"},
    {"name": "OpenRouter              — 600+ models via one API key",    "value": "openrouter"},
]

_PROFILE_CHOICES = [
    {"name": "development   (Sonnet 4 — fastest, most affordable)",               "value": "development"},
    {"name": "production    (Sonnet 4 + Sonnet 4.5 supervisor) [recommended]",    "value": "production"},
    {"name": "performance   (Sonnet 4.5 — highest quality)",                      "value": "performance"},
    {"name": "max           (Opus 4.5 supervisor — most capable, most expensive)", "value": "max"},
]

# Providers that support the profile selection step
_PROFILE_PROVIDERS = {"anthropic", "bedrock"}


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


def _build_ollama_prompt(status) -> str:
    if not status.installed:
        return (
            "Choose an Ollama model. Ollama is not installed yet; install via "
            "https://ollama.com/download or `curl -fsSL https://ollama.com/install.sh | sh`."
        )
    if not status.running:
        return (
            "Choose an Ollama model. Ollama is installed but not running; start it with `ollama serve`."
        )
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
        questionary.Choice("Other model               Type a custom model name", value=_CUSTOM_OLLAMA_MODEL)
    )

    if default_value not in seen:
        default_value = choices[0].value

    return choices, default_value


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
        # ------------------------------------------------------------------ #
        # Step 1 — Agent package selection                                    #
        # ------------------------------------------------------------------ #
        selected_pkgs = questionary.checkbox(
            "Select agent packages to install (Space to toggle, Enter to confirm):",
            choices=_build_agent_choices(questionary),
        ).ask()

        if selected_pkgs is None:  # user pressed Ctrl-C
            result["cancelled"] = True
            return result

        result["agents"] = _normalize_selected_agents(selected_pkgs)

        # ------------------------------------------------------------------ #
        # Step 1.5 — Omics-OS Cloud pre-question                             #
        # ------------------------------------------------------------------ #
        use_cloud = questionary.select(
            "How would you like to connect to an LLM?",
            choices=[
                questionary.Choice(
                    "Omics-OS Cloud   — managed Bedrock, login via browser",
                    value="omics-os",
                ),
                questionary.Choice(
                    "Bring your own   — Anthropic, OpenAI, Ollama, etc.",
                    value="byok",
                ),
            ],
            default="omics-os",
        ).ask()

        if use_cloud is None:
            result["cancelled"] = True
            return result

        if use_cloud == "omics-os":
            result["provider"] = "omics-os"
            # Try browser login
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
                    result["cancelled"] = True
                    return result
                result["api_key"] = key.strip()
            # Skip directly to NCBI step (step 5)
            provider = "omics-os"
        else:
            # -------------------------------------------------------------- #
            # Step 2 — Provider selection                                     #
            # -------------------------------------------------------------- #
            provider = questionary.select(
                "Select your LLM provider:",
                choices=[
                    questionary.Choice(c["name"], value=c["value"])
                    for c in _PROVIDER_CHOICES
                    if c["value"] != "omics-os"
                ],
                default="anthropic",
            ).ask()

            if provider is None:
                result["cancelled"] = True
                return result

            result["provider"] = provider

        # ------------------------------------------------------------------ #
        # Step 3 — API key(s)                                                 #
        # ------------------------------------------------------------------ #
        if provider == "omics-os":
            pass  # Already handled above

        elif provider == "anthropic":
            key = questionary.password(
                "Enter your Claude API key (https://console.anthropic.com):"
            ).ask()
            if key is None:
                result["cancelled"] = True
                return result
            result["api_key"] = key.strip()

        elif provider == "bedrock":
            access = questionary.password("Enter your AWS access key:").ask()
            if access is None:
                result["cancelled"] = True
                return result
            secret = questionary.password("Enter your AWS secret key:").ask()
            if secret is None:
                result["cancelled"] = True
                return result
            result["api_key"]           = access.strip()
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
                result["cancelled"] = True
                return result
            if model == _CUSTOM_OLLAMA_MODEL:
                custom_model = questionary.text(
                    "Enter your Ollama model name:",
                    default=_DEFAULT_OLLAMA_MODEL,
                ).ask()
                if custom_model is None:
                    result["cancelled"] = True
                    return result
                result["ollama_model"] = custom_model.strip()
            else:
                result["ollama_model"] = model.strip()

        elif provider == "gemini":
            key = questionary.password(
                "Enter your Google API key (https://aistudio.google.com/apikey):"
            ).ask()
            if key is None:
                result["cancelled"] = True
                return result
            result["api_key"] = key.strip()

        elif provider == "azure":
            endpoint = questionary.text(
                "Enter your Azure AI endpoint URL:",
                default="https://your-project.inference.ai.azure.com/",
            ).ask()
            if endpoint is None:
                result["cancelled"] = True
                return result
            credential = questionary.password(
                "Enter your Azure API credential:"
            ).ask()
            if credential is None:
                result["cancelled"] = True
                return result
            result["api_key"]           = credential.strip()
            result["api_key_secondary"] = endpoint.strip()

        elif provider == "openai":
            key = questionary.password(
                "Enter your OpenAI API key (https://platform.openai.com/api-keys):"
            ).ask()
            if key is None:
                result["cancelled"] = True
                return result
            result["api_key"] = key.strip()

        elif provider == "openrouter":
            key = questionary.password(
                "Enter your OpenRouter API key (https://openrouter.ai/keys):"
            ).ask()
            if key is None:
                result["cancelled"] = True
                return result
            result["api_key"] = key.strip()

        # ------------------------------------------------------------------ #
        # Step 4 — Profile (Anthropic / Bedrock only)                        #
        # ------------------------------------------------------------------ #
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
                result["cancelled"] = True
                return result

            result["profile"] = profile

        # ------------------------------------------------------------------ #
        # Step 5 — Optional NCBI key                                         #
        # ------------------------------------------------------------------ #
        add_ncbi = questionary.confirm(
            "Add an NCBI API key? (enhances literature search, optional)",
            default=False,
        ).ask()

        if add_ncbi is None:
            result["cancelled"] = True
            return result

        if add_ncbi:
            ncbi = questionary.password("Enter your NCBI API key:").ask()
            if ncbi is None:
                result["cancelled"] = True
                return result
            result["ncbi_key"] = ncbi.strip()

        # ------------------------------------------------------------------ #
        # Step 6 — Optional Cloud key (skip when already via Omics-OS Cloud) #
        # ------------------------------------------------------------------ #
        if provider != "omics-os":
            add_cloud = questionary.confirm(
                "Add an Omics-OS Cloud API key? (enables premium tier, optional)",
                default=False,
            ).ask()

            if add_cloud is None:
                result["cancelled"] = True
                return result

            if add_cloud:
                ckey = questionary.password("Enter your Lobster Cloud API key:").ask()
                if ckey is None:
                    result["cancelled"] = True
                    return result
                result["cloud_key"] = ckey.strip()

        # ------------------------------------------------------------------ #
        # Step 7 — Smart Standardization / vector search                     #
        # ------------------------------------------------------------------ #
        beneficiaries = _get_smart_standardization_beneficiaries(result["agents"])
        smart_std_prompt = (
            "Enable Smart Standardization / vector search? (OpenAI embeddings + ontology matching, optional)"
        )
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
            result["cancelled"] = True
            return result

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
                    result["cancelled"] = True
                    return result
                result["smart_standardization_openai_key"] = embedding_key.strip()

    except KeyboardInterrupt:
        result["cancelled"] = True

    return result
