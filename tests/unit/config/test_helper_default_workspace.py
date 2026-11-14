"""Owner tracking must not change an unbound helper's workspace policy."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lobster.config.inference_context import INFERENCE_BINDING_VERSION, InferenceContext
from lobster.config.inference_runtime import capture_inference_binding
from lobster.core.exceptions import InferenceRefused


# The assistant module binds ``create_llm`` at first import; under the automatic provider
# mock that name would be a Mock for every later test in the process.
@pytest.mark.no_auto_config
def test_unbound_assistant_uses_original_workspace_resolution_and_retains_guard(
    tmp_path, monkeypatch
):
    module = pytest.importorskip(
        "lobster.agents.data_expert.assistant",
        reason="Assistant implementation requires lobster-research",
    )
    from lobster.core import workspace

    ordinary = tmp_path / "ordinary"
    manager = SimpleNamespace(workspace_path=tmp_path / "owner")
    resolve = Mock(return_value=ordinary)
    create = Mock(return_value=Mock())
    monkeypatch.setattr(workspace, "resolve_workspace", resolve)
    monkeypatch.setattr(module, "create_llm", create)
    monkeypatch.setattr(
        module,
        "get_settings",
        lambda: SimpleNamespace(get_agent_llm_params=lambda name: {}),
    )
    assistant = module.DataExpertAssistant(data_manager=manager)
    assert assistant.llm is create.return_value
    resolve.assert_called_once_with(explicit_path=None, create=False)
    create.assert_called_once_with("data_expert_assistant", {}, workspace_path=ordinary)

    class Factory:
        capability_version = INFERENCE_BINDING_VERSION

        def create(self, *, component, context):
            raise AssertionError("Stale unbound helper must not construct a new model")

    context = InferenceContext(
        provider="bedrock",
        model_id="synthetic",
        binding_id="run",
        binding_version=INFERENCE_BINDING_VERSION,
        context_window=200000,
        max_output_tokens=8192,
    )
    capture_inference_binding(manager, context, Factory())
    with pytest.raises(InferenceRefused, match="context"):
        _ = assistant.llm
    assert create.call_count == 1
