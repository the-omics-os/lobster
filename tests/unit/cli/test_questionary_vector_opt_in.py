"""Actual fallback branch via synthetic dialog; NOT an installed UI acceptance."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from lobster.ui.bridge.questionary_fallback import run_questionary_init
from lobster.ui.wizard.manifest import AgentPackageDef, ProviderDef, WizardManifest


@pytest.mark.parametrize("agents", [[], ["lobster-metadata"]])
@pytest.mark.parametrize("explicit_opt_in", [False, True])
def test_questionary_vector_is_default_no_with_explicit_opt_in(
    monkeypatch, agents, explicit_opt_in
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    questionary = ModuleType("questionary")
    questionary.Choice = lambda title, **kwargs: SimpleNamespace(title=title, **kwargs)
    questionary.select = lambda *args, **kwargs: SimpleNamespace(ask=lambda: "omics-os")
    questionary.checkbox = lambda *args, **kwargs: SimpleNamespace(ask=lambda: agents)
    confirms = []
    passwords = []

    def confirm(prompt, *, default):
        confirms.append((prompt, default))
        if "Smart Standardization" in prompt:
            # Simulate Enter (the real declared default) when not opting in.
            answer = True if explicit_opt_in else default
        else:
            answer = False
        return SimpleNamespace(ask=lambda: answer)

    def password(prompt, **kwargs):
        passwords.append(prompt)
        value = (
            "sk-synthetic-embedding-key"
            if "embeddings" in prompt
            else "omx_synthetic_provider_key"
        )
        return SimpleNamespace(ask=lambda: value)

    questionary.confirm = confirm
    questionary.password = password
    monkeypatch.setitem(sys.modules, "questionary", questionary)
    manifest = WizardManifest(
        providers=[
            ProviderDef(
                name="omics-os",
                display_name="Omics-OS Cloud",
                model_selection="managed",
            )
        ],
        agent_packages=[
            AgentPackageDef(
                package_name="lobster-metadata",
                description="Metadata",
                agents=["metadata_assistant"],
            )
        ],
    )
    result = run_questionary_init(manifest)
    smart_confirm = [
        (prompt, default)
        for prompt, default in confirms
        if "Smart Standardization" in prompt
    ]
    assert len(smart_confirm) == 1
    assert smart_confirm[0][1] is False
    if agents:
        assert "Helpful for" in smart_confirm[0][0]  # useful agent context retained
    assert result["agents"] == (["metadata_assistant"] if agents else [])
    assert result["cancelled"] is False
    embedding_prompts = [prompt for prompt in passwords if "embeddings" in prompt]
    if explicit_opt_in:
        assert result["smart_standardization_enabled"] is True
        assert (
            result["smart_standardization_openai_key"] == "sk-synthetic-embedding-key"
        )
        assert len(embedding_prompts) == 1
    else:
        assert result["smart_standardization_enabled"] is False
        assert not result["smart_standardization_openai_key"]
        assert embedding_prompts == []
