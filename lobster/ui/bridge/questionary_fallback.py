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
    "cancelled":          False,
}

# Matches AVAILABLE_AGENT_PACKAGES in init_commands.py.
_AGENT_CHOICES = [
    {"name": "lobster-research            Literature search & data discovery",         "value": "lobster-research"},
    {"name": "lobster-transcriptomics     Single-cell & bulk RNA-seq analysis",        "value": "lobster-transcriptomics"},
    {"name": "lobster-visualization       Data visualization & plotting",              "value": "lobster-visualization"},
    {"name": "lobster-genomics            Genomics/DNA analysis (VCF, GWAS, variants)", "value": "lobster-genomics"},
    {"name": "lobster-proteomics          Mass spec & affinity proteomics",            "value": "lobster-proteomics"},
    {"name": "lobster-metabolomics        LC-MS/GC-MS/NMR metabolomics",               "value": "lobster-metabolomics"},
    {"name": "lobster-ml                  Machine learning & survival analysis",        "value": "lobster-ml"},
    {"name": "lobster-drug-discovery      Drug discovery & cheminformatics",            "value": "lobster-drug-discovery"},
]

_PROVIDER_CHOICES = [
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


def run_questionary_init() -> dict:
    """Run questionary-based init wizard.

    Returns the same dict structure as the Go TUI wizard.

    Raises:
        ImportError: when ``questionary`` is not installed.
        KeyboardInterrupt: when the user presses Ctrl-C (callers should treat
            this as cancellation and check ``result["cancelled"]``).
    """
    import questionary  # intentional — caller catches ImportError

    result = dict(_EMPTY_RESULT)

    try:
        # ------------------------------------------------------------------ #
        # Step 1 — Agent package selection                                    #
        # ------------------------------------------------------------------ #
        selected_pkgs = questionary.checkbox(
            "Select agent packages to install (Space to toggle, Enter to confirm):",
            choices=[
                questionary.Choice(c["name"], value=c["value"], checked=False)
                for c in _AGENT_CHOICES
            ],
        ).ask()

        if selected_pkgs is None:  # user pressed Ctrl-C
            result["cancelled"] = True
            return result

        result["agents"] = selected_pkgs

        # ------------------------------------------------------------------ #
        # Step 2 — Provider selection                                         #
        # ------------------------------------------------------------------ #
        provider = questionary.select(
            "Select your LLM provider:",
            choices=[
                questionary.Choice(c["name"], value=c["value"])
                for c in _PROVIDER_CHOICES
            ],
            default=_PROVIDER_CHOICES[0]["value"],
        ).ask()

        if provider is None:
            result["cancelled"] = True
            return result

        result["provider"] = provider

        # ------------------------------------------------------------------ #
        # Step 3 — API key(s)                                                 #
        # ------------------------------------------------------------------ #
        if provider == "anthropic":
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
            # No API key needed — optionally ask for model name
            model = questionary.text(
                "Ollama model name (leave blank for default llama3:8b-instruct):",
                default="",
            ).ask()
            if model is None:
                result["cancelled"] = True
                return result
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
        # Step 6 — Optional Cloud key                                        #
        # ------------------------------------------------------------------ #
        add_cloud = questionary.confirm(
            "Add a Lobster Cloud API key? (enables premium tier, optional)",
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

    except KeyboardInterrupt:
        result["cancelled"] = True

    return result
