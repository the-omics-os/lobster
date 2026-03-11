from lobster.cli_internal.commands.heavy import init_commands
from lobster.config.workspace_agent_config import WorkspaceAgentConfig
from lobster.ui.bridge.init_adapter import apply_tui_init_result


def test_check_and_prompt_install_packages_expands_package_ids(monkeypatch, tmp_path):
    installed_agents = [{}]
    installed_packages = []

    def _fake_get_installed_agents():
        if installed_agents:
            return installed_agents.pop(0)
        raise AssertionError("init install path should not rediscover agents in-process")

    def _fake_install(package_name):
        installed_packages.append(package_name)
        return True, "ok"

    monkeypatch.setattr(init_commands, "_get_installed_agents", _fake_get_installed_agents)
    monkeypatch.setattr(
        "lobster.core.uv_tool_env.is_uv_tool_env",
        lambda: False,
    )
    monkeypatch.setattr(
        "lobster.cli_internal.commands.heavy.init_commands.Confirm.ask",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        "lobster.cli_internal.commands.light.agent_commands._uv_pip_install",
        _fake_install,
    )

    selected_agents = init_commands._check_and_prompt_install_packages(
        ["lobster-research"],
        tmp_path / ".lobster_workspace",
    )

    assert installed_packages == ["lobster-research"]
    assert selected_agents == ["research_agent", "data_expert_agent"]


def test_check_and_prompt_install_packages_returns_only_successful_package_agents(
    monkeypatch, tmp_path
):
    installed_packages = []

    monkeypatch.setattr(init_commands, "_get_installed_agents", lambda: {})
    monkeypatch.setattr(
        "lobster.core.uv_tool_env.is_uv_tool_env",
        lambda: False,
    )
    monkeypatch.setattr(
        "lobster.cli_internal.commands.heavy.init_commands.Confirm.ask",
        lambda *args, **kwargs: True,
    )

    def _fake_install(package_name):
        installed_packages.append(package_name)
        if package_name == "lobster-research":
            return True, "ok"
        return False, "boom"

    monkeypatch.setattr(
        "lobster.cli_internal.commands.light.agent_commands._uv_pip_install",
        _fake_install,
    )

    selected_agents = init_commands._check_and_prompt_install_packages(
        ["lobster-research", "lobster-transcriptomics"],
        tmp_path / ".lobster_workspace",
    )

    assert installed_packages == ["lobster-research", "lobster-transcriptomics"]
    assert selected_agents == ["research_agent", "data_expert_agent"]


def test_check_and_prompt_install_packages_skips_confirm_in_noninteractive_mode(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(init_commands, "_get_installed_agents", lambda: {})
    monkeypatch.setattr(
        "lobster.core.uv_tool_env.is_uv_tool_env",
        lambda: False,
    )

    def _unexpected_confirm(*args, **kwargs):
        raise AssertionError("Confirm.ask should not run in non-interactive mode")

    def _unexpected_install(*args, **kwargs):
        raise AssertionError("Installer should not run without auto-install")

    monkeypatch.setattr(
        "lobster.cli_internal.commands.heavy.init_commands.Confirm.ask",
        _unexpected_confirm,
    )
    monkeypatch.setattr(
        "lobster.cli_internal.commands.light.agent_commands._uv_pip_install",
        _unexpected_install,
    )

    selected_agents = init_commands._check_and_prompt_install_packages(
        ["lobster-research"],
        tmp_path / ".lobster_workspace",
        prompt_for_install=False,
        auto_install_missing=False,
    )

    assert selected_agents == []


def test_noninteractive_preset_selection_filters_unpublished_agents(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "lobster.config.agent_presets.is_valid_preset",
        lambda preset_name: preset_name == "scrna-full",
    )
    monkeypatch.setattr(
        "lobster.config.agent_presets.expand_preset",
        lambda preset_name: [
            "research_agent",
            "metadata_assistant",
            "annotation_expert",
        ],
    )
    monkeypatch.setattr(
        init_commands,
        "AVAILABLE_AGENT_PACKAGES",
        [
            ("lobster-research", "Research", ["research_agent"], True, False),
            ("lobster-metadata", "Metadata", ["metadata_assistant"], False, True),
            (
                "lobster-transcriptomics",
                "Transcriptomics",
                ["annotation_expert"],
                True,
                False,
            ),
        ],
    )

    selected_agents, preset_name = init_commands._perform_agent_selection_non_interactive(
        agents_flag="",
        preset_flag="scrna-full",
        auto_agents_flag=False,
        agents_description_flag="",
        workspace_path=tmp_path / ".lobster_workspace",
    )

    assert selected_agents == ["research_agent", "annotation_expert"]
    assert preset_name == "scrna-full"


def test_prompt_smart_standardization_skips_when_backend_unavailable(monkeypatch):
    monkeypatch.setattr(
        init_commands,
        "_is_vector_search_backend_available",
        lambda: False,
    )

    def _unexpected_prompt(*args, **kwargs):
        raise AssertionError("Prompt.ask should not run when backend is unavailable")

    monkeypatch.setattr(
        "lobster.cli_internal.commands.heavy.init_commands.Prompt.ask",
        _unexpected_prompt,
    )

    env_lines = init_commands._prompt_smart_standardization(
        selected_agents=["annotation_expert"]
    )

    assert env_lines == []


def test_postprocess_tui_init_result_normalizes_agents_before_apply(monkeypatch, tmp_path):
    captured = {}

    def _fake_apply(result, workspace_path, env_path, global_config=False):
        captured["result"] = result
        captured["workspace_path"] = workspace_path
        captured["env_path"] = env_path
        captured["global_config"] = global_config

    monkeypatch.setattr(
        init_commands,
        "_check_and_prompt_install_packages",
        lambda selected_agents, workspace_path: selected_agents,
    )
    monkeypatch.setattr(
        "lobster.ui.bridge.init_adapter.apply_tui_init_result",
        _fake_apply,
    )
    monkeypatch.setattr(
        "lobster.core.uv_tool_env.is_uv_tool_env",
        lambda: False,
    )
    monkeypatch.setattr(init_commands, "_ensure_provider_installed", lambda provider: True)
    monkeypatch.setattr(
        init_commands,
        "_install_smart_standardization_dependencies",
        lambda: captured.setdefault("smart_std_installed", True),
    )

    result = init_commands._postprocess_tui_init_result(
        {
            "provider": "anthropic",
            "api_key": "sk-ant-test",
            "profile": "production",
            "agents": ["lobster-research"],
            "smart_standardization_enabled": True,
        },
        workspace_path=tmp_path / ".lobster_workspace",
        env_path=tmp_path / ".env",
        global_config=False,
        skip_extras=False,
    )

    assert result["agents"] == ["research_agent", "data_expert_agent"]
    assert captured["result"]["agents"] == ["research_agent", "data_expert_agent"]
    assert captured["smart_std_installed"] is True


def test_apply_tui_init_result_writes_ollama_and_smart_standardization_config(tmp_path):
    workspace_path = tmp_path / ".lobster_workspace"
    env_path = tmp_path / ".env"

    apply_tui_init_result(
        {
            "provider": "ollama",
            "api_key": "",
            "api_key_secondary": "",
            "profile": "",
            "agents": ["lobster-research"],
            "ncbi_key": "",
            "cloud_key": "",
            "ollama_model": "gpt-oss:20b",
            "smart_standardization_enabled": True,
            "smart_standardization_openai_key": "sk-embed-test",
            "cancelled": False,
        },
        workspace_path=workspace_path,
        env_path=env_path,
        global_config=False,
    )

    env_text = env_path.read_text(encoding="utf-8")
    assert "LOBSTER_LLM_PROVIDER=ollama" in env_text
    assert "OLLAMA_DEFAULT_MODEL=gpt-oss:20b" in env_text
    assert "OPENAI_API_KEY=sk-embed-test" in env_text
    assert "LOBSTER_EMBEDDING_PROVIDER=openai" in env_text

    config = WorkspaceAgentConfig.load(workspace_path)
    assert config.enabled_agents == ["research_agent", "data_expert_agent"]
