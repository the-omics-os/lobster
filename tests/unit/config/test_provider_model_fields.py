"""Every provider Lobster accepts must have somewhere to store its model.

The model field name is built at run time from the provider name
(``openai`` -> ``openai_default_model``), so a provider added to
VALID_PROVIDERS without a matching field only fails when a user picks it.
"""

import pytest

from lobster.config.constants import VALID_PROVIDERS
from lobster.config.global_config import GlobalProviderConfig
from lobster.config.workspace_config import WorkspaceProviderConfig

# Providers that deliberately have no model setting: `lobster init` refuses
# --model for them before writing anything (see test_init_project.py).
NO_MODEL_SETTING = {"omics-os"}
CONFIGS = [GlobalProviderConfig, WorkspaceProviderConfig]


@pytest.mark.parametrize(
    "provider", [p for p in VALID_PROVIDERS if p not in NO_MODEL_SETTING]
)
@pytest.mark.parametrize("config_cls", CONFIGS)
def test_every_valid_provider_can_store_a_model(config_cls, provider):
    config = config_cls()
    config.set_model_for_provider(provider, "some-model")
    assert config.get_model_for_provider(provider) == "some-model"


@pytest.mark.parametrize("provider", sorted(NO_MODEL_SETTING))
@pytest.mark.parametrize("config_cls", CONFIGS)
def test_providers_without_a_model_setting_have_no_field(config_cls, provider):
    config = config_cls()
    assert config.get_model_for_provider(provider) is None
    with pytest.raises(AttributeError):
        config.set_model_for_provider(provider, "some-model")
