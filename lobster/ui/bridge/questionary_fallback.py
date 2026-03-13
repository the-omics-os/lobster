"""Questionary-based init wizard — Python-native improved UX.

Canonical question order (matches Go TUI wizard):

  1. Provider selection  (from manifest.providers)
  2. Credentials         (from selected_provider.credentials)
  3. Model selection     (routed by selected_provider.model_selection)
  4. Profile selection   (if model_selection == "profile")
  5. Agent packages      (from manifest.agent_packages)
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

from lobster.ui.wizard.manifest import WizardManifest

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


# ---------------------------------------------------------------------------
# Cancellation helper
# ---------------------------------------------------------------------------


def _cancelled(result: dict) -> dict:
    result["cancelled"] = True
    return result


# ---------------------------------------------------------------------------
# Main wizard
# ---------------------------------------------------------------------------


def run_questionary_init(manifest: WizardManifest) -> dict:
    """Run questionary-based init wizard using WizardManifest data.

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

    result = dict(_EMPTY_RESULT)

    try:
        # ================================================================== #
        # Step 1 — Provider selection (from manifest)                         #
        # ================================================================== #
        provider_choices = [
            questionary.Choice(f"{p.display_name:<24} {p.description}", value=p.name)
            for p in manifest.providers
        ]
        provider = questionary.select(
            "Select your LLM provider:",
            choices=provider_choices,
            default="omics-os",
        ).ask()

        if provider is None:
            return _cancelled(result)

        result["provider"] = provider

        # Find the selected provider definition
        provider_def = next((p for p in manifest.providers if p.name == provider), None)
        if provider_def is None:
            return _cancelled(result)

        # ================================================================== #
        # Step 2 — Credentials (from provider_def.credentials)               #
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

        elif provider == "ollama":
            # Ollama has no credentials — handled via model selection below
            pass

        else:
            # Generic credential collection from manifest
            for i, cred in enumerate(provider_def.credentials):
                if not cred.required:
                    continue
                prompt = f"Enter your {cred.label}:"
                if cred.secret:
                    value = questionary.password(prompt).ask()
                else:
                    value = questionary.text(prompt).ask()
                if value is None:
                    return _cancelled(result)
                # Map to result dict keys
                if i == 0:
                    result["api_key"] = value.strip()
                elif i == 1:
                    result["api_key_secondary"] = value.strip()

        # ================================================================== #
        # Step 3 — Model selection (routed by model_selection)               #
        # ================================================================== #
        mode = provider_def.model_selection

        if mode == "local":
            # Ollama-style local model selection
            ollama_choices = []
            default_model = ""

            for m in manifest.ollama_status.models:
                ollama_choices.append(
                    questionary.Choice(f"{m:<24} detected locally", value=m)
                )
                if not default_model:
                    default_model = m

            ollama_choices.append(
                questionary.Choice(
                    "Other model               Type a custom model name",
                    value=_CUSTOM_OLLAMA_MODEL,
                )
            )
            if not default_model and ollama_choices:
                default_model = ollama_choices[0].value

            model = questionary.select(
                "Choose an Ollama model:",
                choices=ollama_choices,
                default=default_model or None,
            ).ask()
            if model is None:
                return _cancelled(result)

            if model == _CUSTOM_OLLAMA_MODEL:
                custom_model = questionary.text("Enter your Ollama model name:").ask()
                if custom_model is None:
                    return _cancelled(result)
                result["ollama_model"] = custom_model.strip()
            else:
                result["ollama_model"] = model.strip()

        elif mode == "explicit" and provider_def.models:
            model_choices = []
            default_model = ""
            for m in provider_def.models:
                desc = f" {m.description}" if m.description else ""
                label = f"{m.name:<48} {m.display_name}{desc}"
                model_choices.append(questionary.Choice(label, value=m.name))
                if m.is_default:
                    default_model = m.name

            model_choices.append(
                questionary.Choice(
                    "Other model                                      Enter a custom model ID",
                    value=_CUSTOM_MODEL_SENTINEL,
                )
            )

            if not default_model:
                default_model = model_choices[0].value

            selected_model = questionary.select(
                "Select default model (you can change this later):",
                choices=model_choices,
                default=default_model,
            ).ask()
            if selected_model is None:
                return _cancelled(result)

            if selected_model == _CUSTOM_MODEL_SENTINEL:
                custom_id = questionary.text(
                    "Enter your model ID:", default=default_model
                ).ask()
                if custom_id is None:
                    return _cancelled(result)
                result["model_id"] = custom_id.strip()
            else:
                result["model_id"] = selected_model

        # mode == "managed" → skip model selection (Omics-OS)

        # ================================================================== #
        # Step 4 — Profile (if model_selection == "profile")                 #
        # ================================================================== #
        if mode == "profile" and provider_def.profiles:
            profile_choices = [
                questionary.Choice(
                    f"{p.display_name:<14} ({p.description})", value=p.name
                )
                for p in provider_def.profiles
            ]
            default_profile = next(
                (p.name for p in provider_def.profiles if p.is_default),
                "production",
            )

            profile = questionary.select(
                "Select agent configuration profile:",
                choices=profile_choices,
                default=default_profile,
            ).ask()

            if profile is None:
                return _cancelled(result)

            result["profile"] = profile

        # ================================================================== #
        # Step 5 — Agent packages (from manifest)                            #
        # ================================================================== #
        agent_choices = []
        for pkg in manifest.agent_packages:
            exp_tag = " [experimental]" if pkg.experimental else ""
            agents_str = (
                f"({len(pkg.agents)} agent{'s' if len(pkg.agents) != 1 else ''})"
            )
            label = f"{pkg.package_name}{exp_tag:<26} {pkg.description} {agents_str}"
            checked = pkg.published and not pkg.experimental
            agent_choices.append(
                questionary.Choice(label, value=pkg.package_name, checked=checked)
            )

        selected_pkgs = questionary.checkbox(
            "Select agent packages to install (Space to toggle, Enter to confirm):",
            choices=agent_choices,
        ).ask()

        if selected_pkgs is None:
            return _cancelled(result)

        result["agents"] = _normalize_selected_agents(selected_pkgs)

        # ================================================================== #
        # Step 6 — Optional NCBI key                                         #
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
        # Step 7 — Optional Cloud key (skip when already via Omics-OS Cloud) #
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
        # Step 8 — Smart Standardization / vector search                     #
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
