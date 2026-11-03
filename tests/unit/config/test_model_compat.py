"""Model compatibility for Claude 5.x, Fable and GPT-6.x.

Covers thinking-preset translation per model family, the default
temperature (unset = model default), the new catalog entries and defaults,
and that no request field rejected by these models is sent.
"""

from unittest.mock import Mock, patch

import pytest

from lobster.config.agent_defaults import THINKING_PRESETS, get_agent_params
from lobster.config.thinking import (
    effort_for_budget,
    translate_thinking,
    uses_adaptive_thinking,
)

# The factory tests need the real LLMFactory, not the unit-suite mock.
pytestmark = pytest.mark.no_auto_config


def _preset(name):
    return THINKING_PRESETS[name].to_dict()


class TestEffortMapping:
    @pytest.mark.parametrize(
        "preset,effort",
        [
            ("light", "low"),
            ("standard", "medium"),
            ("extended", "high"),
            ("deep", "max"),
        ],
    )
    def test_presets_map_to_effort(self, preset, effort):
        budget = _preset(preset)["thinking"]["budget_tokens"]
        assert effort_for_budget(budget) == effort


class TestAdaptiveDetection:
    @pytest.mark.parametrize(
        "model",
        [
            "claude-sonnet-5-5",
            "global.anthropic.claude-haiku-5-5",
            "us.anthropic.claude-opus-5-5",
            "claude-fable-5-1",
            "global.anthropic.claude-sonnet-6-0",
        ],
    )
    def test_claude_5_and_later(self, model):
        assert uses_adaptive_thinking(model)

    @pytest.mark.parametrize(
        "model",
        [
            "claude-sonnet-4-20250514",
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "claude-3-5-sonnet-20241022",
            "global.anthropic.claude-opus-4-8",
            "gpt-6.1-sol",
        ],
    )
    def test_older_and_non_claude(self, model):
        assert not uses_adaptive_thinking(model)


class TestTranslateThinking:
    def test_anthropic_claude_5_uses_adaptive_and_effort(self):
        out = translate_thinking("anthropic", "claude-sonnet-5-5", _preset("standard"))
        assert out == {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "medium"},
        }

    @pytest.mark.parametrize("provider", ["bedrock", "omics-os"])
    def test_bedrock_routes_nest_fields(self, provider):
        out = translate_thinking(
            provider, "global.anthropic.claude-sonnet-5-5", _preset("deep")
        )
        assert "thinking" not in out
        assert out["additional_model_request_fields"] == {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "max"},
        }

    def test_older_claude_keeps_budget(self):
        fields = _preset("extended")
        out = translate_thinking("anthropic", "claude-sonnet-4-20250514", fields)
        assert out["thinking"] == {"type": "enabled", "budget_tokens": 5000}

    def test_older_claude_on_bedrock_nests_budget(self):
        out = translate_thinking(
            "bedrock",
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            _preset("light"),
        )
        assert out == {
            "additional_model_request_fields": {
                "thinking": {"type": "enabled", "budget_tokens": 1000}
            }
        }

    def test_existing_additional_fields_are_kept(self):
        fields = {
            **_preset("light"),
            "additional_model_request_fields": {"top_k": 5},
        }
        out = translate_thinking("bedrock", "global.anthropic.claude-haiku-5-5", fields)
        assert out["additional_model_request_fields"]["top_k"] == 5
        assert out["additional_model_request_fields"]["output_config"] == {
            "effort": "low"
        }

    def test_existing_output_config_is_kept_direct(self):
        fields = {**_preset("standard"), "output_config": {"format": {"type": "x"}}}
        out = translate_thinking("anthropic", "claude-opus-5-5", fields)
        assert out["output_config"] == {"format": {"type": "x"}, "effort": "medium"}

    def test_existing_output_config_is_kept_nested(self):
        fields = {
            **_preset("light"),
            "additional_model_request_fields": {"output_config": {"format": "x"}},
        }
        out = translate_thinking("bedrock", "global.anthropic.claude-opus-5-5", fields)
        assert out["additional_model_request_fields"]["output_config"] == {
            "format": "x",
            "effort": "low",
        }

    @pytest.mark.parametrize(
        "preset,effort",
        [("light", "low"), ("standard", "medium"), ("deep", "high")],
    )
    def test_openai_uses_reasoning_effort(self, preset, effort):
        out = translate_thinking("openai", "gpt-6.1-sol", _preset(preset))
        assert out == {"reasoning": {"effort": effort}}

    def test_gpt_on_bedrock_sends_nothing(self, caplog):
        out = translate_thinking(
            "bedrock", "global.openai.gpt-6.1-sol", _preset("standard")
        )
        assert out == {}
        assert "Thinking preset ignored" in caplog.text

    def test_other_provider_drops_preset_with_warning(self, caplog):
        out = translate_thinking("ollama", "llama3", _preset("standard"))
        assert out == {}
        assert "Thinking preset ignored" in caplog.text

    def test_ignored_preset_warns_once_per_route(self, caplog):
        for _ in range(3):
            translate_thinking("ollama", "warn-once-model", _preset("light"))
        assert caplog.text.count("warn-once-model") == 1

    def test_no_preset_passes_through(self):
        fields = {"foo": 1}
        assert translate_thinking("openai", "gpt-6.1-sol", fields) == fields

    def test_non_enabled_thinking_passes_through(self):
        fields = {"thinking": {"type": "adaptive"}}
        assert translate_thinking("anthropic", "claude-sonnet-5-5", fields) == fields


class TestFactoryWiring:
    @patch("lobster.core.config_resolver.ConfigResolver")
    @patch("lobster.config.providers.get_provider")
    def test_factory_translates_and_omits_temperature(
        self, mock_get_provider, mock_config_resolver
    ):
        from lobster.config.llm_factory import LLMFactory

        resolver = Mock()
        resolver.resolve_provider.return_value = ("bedrock", "test")
        resolver.resolve_model.return_value = (
            "global.anthropic.claude-sonnet-5-5",
            "test",
        )
        mock_config_resolver.get_instance.return_value = resolver
        provider = Mock()
        provider.create_chat_model = Mock(return_value=Mock())
        mock_get_provider.return_value = provider

        LLMFactory.create_llm(
            model_config={"additional_model_request_fields": _preset("standard")},
            agent_name="supervisor",
        )

        kwargs = provider.create_chat_model.call_args[1]
        assert "temperature" not in kwargs
        assert "thinking" not in kwargs
        assert kwargs["additional_model_request_fields"] == {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "medium"},
        }

    @patch("lobster.core.config_resolver.ConfigResolver")
    @patch("lobster.config.providers.get_provider")
    def test_factory_forwards_explicit_temperature(
        self, mock_get_provider, mock_config_resolver
    ):
        from lobster.config.llm_factory import LLMFactory

        resolver = Mock()
        resolver.resolve_provider.return_value = ("anthropic", "test")
        resolver.resolve_model.return_value = ("claude-sonnet-4-20250514", "test")
        mock_config_resolver.get_instance.return_value = resolver
        provider = Mock()
        provider.create_chat_model = Mock(return_value=Mock())
        mock_get_provider.return_value = provider

        LLMFactory.create_llm(
            model_config={"temperature": 0.3}, agent_name="supervisor"
        )
        assert provider.create_chat_model.call_args[1]["temperature"] == 0.3


class TestDefaultTemperature:
    def test_agent_params_have_no_temperature_by_default(self, monkeypatch):
        monkeypatch.delenv("LOBSTER_SUPERVISOR_TEMPERATURE", raising=False)
        assert "temperature" not in get_agent_params("supervisor")

    def test_bedrock_builder_omits_unset_temperature(self):
        pytest.importorskip("langchain_aws")
        from lobster.config.providers.bedrock_builder import build_bedrock_converse

        with patch("langchain_aws.ChatBedrockConverse") as chat:
            build_bedrock_converse(model_id="global.anthropic.claude-sonnet-5-5")
        assert "temperature" not in chat.call_args[1]

    def test_openai_omits_unset_temperature(self, monkeypatch):
        pytest.importorskip("langchain_openai")
        from lobster.config.providers.openai_provider import OpenAIProvider

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        with patch("langchain_openai.ChatOpenAI") as chat:
            OpenAIProvider().create_chat_model("gpt-6.1-sol")
        kwargs = chat.call_args[1]
        assert "temperature" not in kwargs
        assert "max_tokens" not in kwargs

    def test_openai_has_no_model_prefix_list(self):
        from lobster.config.providers import openai_provider

        assert not hasattr(openai_provider.OpenAIProvider, "_is_reasoning_model")


class TestCatalog:
    def test_defaults_are_newest_models(self):
        from lobster.config.providers.anthropic_provider import AnthropicProvider
        from lobster.config.providers.bedrock_provider import BedrockProvider
        from lobster.config.providers.openai_provider import OpenAIProvider

        assert AnthropicProvider().get_default_model() == "claude-sonnet-5-5"
        assert (
            BedrockProvider().get_default_model()
            == "global.anthropic.claude-sonnet-5-5"
        )
        assert OpenAIProvider().get_default_model() == "gpt-6.1-sol"

    def test_omics_os_default_waits_for_gateway(self):
        # The gateway admits 5.5 only after its catalog update; keep 4.5 until then.
        from lobster.config.providers.omics_os_provider import OmicsOSProvider

        provider = OmicsOSProvider()
        assert (
            provider.get_default_model()
            == "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
        )
        names = [m.name for m in provider.list_models()]
        assert "global.anthropic.claude-sonnet-5-5" in names

    @pytest.mark.parametrize(
        "provider_cls,model,context,price_in,price_out",
        [
            ("anthropic", "claude-sonnet-5-5", 1_000_000, 2.0, 10.0),
            ("anthropic", "claude-haiku-5-5", 1_000_000, 0.10, 0.50),
            ("anthropic", "claude-opus-5-5", 1_000_000, 4.0, 20.0),
            ("bedrock", "global.anthropic.claude-sonnet-5-5", 1_000_000, 2.0, 10.0),
            ("bedrock", "us.anthropic.claude-haiku-5-5", 1_000_000, 0.11, 0.55),
            ("bedrock", "global.openai.gpt-6.1-sol", 1_050_000, 2.0, 10.0),
            ("openai", "gpt-6.1-sol", 272_000, 2.0, 10.0),
            ("openai", "gpt-6-luna", 272_000, 0.10, 0.50),
        ],
    )
    def test_new_models_are_priced_with_context(
        self, provider_cls, model, context, price_in, price_out
    ):
        from lobster.config.providers import get_provider

        info = {m.name: m for m in get_provider(provider_cls).list_models()}[model]
        assert info.context_window == context
        assert info.input_cost_per_million == pytest.approx(price_in)
        assert info.output_cost_per_million == pytest.approx(price_out)

    def test_one_default_per_provider(self):
        from lobster.config.providers import get_provider

        for name in ("anthropic", "bedrock", "openai", "omics-os"):
            models = get_provider(name).list_models()
            assert sum(m.is_default for m in models) == 1, name


class TestStatusBarNames:
    @pytest.mark.parametrize(
        "model,provider,label",
        [
            ("global.anthropic.claude-opus-5-5", "bedrock", "opus 5.5"),
            ("claude-haiku-5-5", "anthropic", "haiku 5.5"),
            ("us.anthropic.claude-sonnet-4-5-20250929-v1:0", "bedrock", "sonnet 4.5"),
            ("claude-sonnet-4-20250514", "anthropic", "sonnet 4.0"),
            ("global.anthropic.claude-opus-4-8", "bedrock", "opus 4.8"),
            ("claude-3-5-haiku-20241022", "anthropic", "haiku 3.5"),
        ],
    )
    def test_labels_follow_model_version(self, model, provider, label):
        pytest.importorskip("textual")
        from lobster.ui.widgets.status_bar import get_friendly_model_name

        assert get_friendly_model_name(model, provider) == label


class TestStructuredOutputToolChoice:
    """Sonnet/Opus 5.5 reject a forced tool_choice; structured output must not force one.

    The raised langchain-aws / langchain-anthropic floors provide this; the test
    guards the component mapper's ``with_structured_output`` path.
    """

    @staticmethod
    def _bound_tool_choice(runnable):
        first = getattr(runnable, "first", runnable)
        return getattr(first, "kwargs", {}).get("tool_choice")

    @pytest.mark.parametrize(
        "model", ["global.anthropic.claude-sonnet-5-5", "us.anthropic.claude-opus-5-5"]
    )
    def test_bedrock_converse_does_not_force_tool(self, model, monkeypatch):
        pytest.importorskip("langchain_aws")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "placeholder")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "placeholder")
        from langchain_aws import ChatBedrockConverse

        from lobster.services.interaction.component_mapper import AskUserContext

        llm = ChatBedrockConverse(model_id=model, region_name="us-east-1")
        runnable = llm.with_structured_output(AskUserContext)
        assert self._bound_tool_choice(runnable) is None

    @pytest.mark.parametrize("model", ["claude-sonnet-5-5", "claude-opus-5-5"])
    def test_anthropic_does_not_force_tool(self, model):
        pytest.importorskip("langchain_anthropic")
        from langchain_anthropic import ChatAnthropic

        from lobster.services.interaction.component_mapper import AskUserContext

        runnable = ChatAnthropic(model=model, api_key="test").with_structured_output(
            AskUserContext
        )
        assert self._bound_tool_choice(runnable) is None
