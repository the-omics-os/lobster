"""Every provider Lobster accepts must have somewhere to store its model.

The model field name is built at run time from the provider name
(``openai`` -> ``openai_default_model``), so a provider added to
VALID_PROVIDERS without a matching field only fails when a user picks it.
"""

import pytest

from lobster.config.constants import VALID_PROVIDERS
from lobster.config.global_config import GlobalProviderConfig
from lobster.config.workspace_config import WorkspaceProviderConfig


@pytest.mark.parametrize("provider", VALID_PROVIDERS)
@pytest.mark.parametrize("config_cls", [GlobalProviderConfig, WorkspaceProviderConfig])
def test_every_valid_provider_can_store_a_model(config_cls, provider):
    config = config_cls()
    config.set_model_for_provider(provider, "some-model")
    assert config.get_model_for_provider(provider) == "some-model"
