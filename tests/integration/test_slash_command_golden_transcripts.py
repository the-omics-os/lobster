from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from lobster.cli_internal.commands.output_adapter import ProtocolOutputAdapter
import lobster.cli_internal.commands.heavy.slash_commands as slash_commands


pytestmark = pytest.mark.integration

_GOLDEN_DIR = Path(__file__).resolve().parents[1] / "golden" / "slash_commands"
_UPDATE_GOLDENS = os.getenv("LOBSTER_UPDATE_GOLDENS", "").lower() in {"1", "true", "yes"}


class _DummyDataManager:
    def __init__(self, workspace_path: Path):
        self.workspace_path = workspace_path
        self.modalities = {"rna": object(), "atac": object()}
        self.available_datasets = {
            "geo_gse12345_rna": {
                "size_mb": 12.5,
                "shape": (1024, 2048),
                "modified": "2026-03-01T10:00:00",
                "path": str(workspace_path / "data" / "geo_gse12345_rna.h5ad"),
                "type": "h5ad",
            },
            "geo_gse67890_atac_clustered": {
                "size_mb": 8.3,
                "shape": (512, 1024),
                "modified": "2026-02-20T08:30:00",
                "path": str(workspace_path / "data" / "geo_gse67890_atac_clustered.h5ad"),
                "type": "h5ad",
            },
        }
        self.download_queue = _DummyDownloadQueue()

    def get_available_datasets(self, force_refresh: bool = False):
        _ = force_refresh
        return self.available_datasets

    def get_workspace_status(self):
        return {
            "workspace_path": str(self.workspace_path),
            "modalities_loaded": len(self.modalities),
            "provenance_enabled": True,
            "mudata_available": True,
            "directories": {
                "data": "/__lobster_test__/data",
                "exports": "/__lobster_test__/exports",
                "metadata": "/__lobster_test__/metadata",
                "notebooks": "/__lobster_test__/notebooks",
                "queues": "/__lobster_test__/queues",
            },
        }

    def list_notebooks(self):
        return [
            {
                "name": "RNA QC Workflow",
                "filename": "rna_qc_workflow.ipynb",
                "created_by": "research_agent",
                "created_at": "2026-03-01T12:00:00",
                "n_steps": 7,
                "size_kb": 43.2,
                "path": str(self.workspace_path / "notebooks" / "rna_qc_workflow.ipynb"),
            },
            {
                "name": "ATAC Cluster Analysis",
                "filename": "atac_cluster_analysis.ipynb",
                "created_by": "data_expert",
                "created_at": "2026-03-02T09:15:00",
                "n_steps": 5,
                "size_kb": 27.8,
                "path": str(self.workspace_path / "notebooks" / "atac_cluster_analysis.ipynb"),
            },
        ]


class _DummyDownloadQueue:
    def get_statistics(self):
        return {
            "total_entries": 3,
            "by_status": {
                "pending": 2,
                "completed": 1,
            },
            "by_database": {
                "geo": 2,
                "sra": 1,
            },
        }


class _DummyPublicationQueue:
    def get_statistics(self):
        return {
            "total_entries": 2,
            "by_status": {
                "pending": 1,
                "completed": 1,
            },
            "by_extraction_level": {
                "methods": 2,
            },
            "identifiers_extracted": 4,
        }


class _DummyClient:
    def __init__(self, workspace_path: Path):
        self.session_id = "sess_123"
        self.workspace_path = workspace_path
        self.data_manager = _DummyDataManager(workspace_path)
        self.publication_queue = _DummyPublicationQueue()
        self.provider_override = "openai"
        self.model_override = "gpt-4.1"

    def get_token_usage(self):
        return {
            "session_id": "sess_123",
            "total_input_tokens": 123,
            "total_output_tokens": 456,
            "total_tokens": 579,
            "total_cost_usd": 0.0123,
            "by_agent": {
                "research_agent": {
                    "input_tokens": 100,
                    "output_tokens": 200,
                    "total_tokens": 300,
                    "cost_usd": 0.005,
                    "invocation_count": 2,
                },
                "data_expert": {
                    "input_tokens": 23,
                    "output_tokens": 256,
                    "total_tokens": 279,
                    "cost_usd": 0.0073,
                    "invocation_count": 1,
                },
            },
        }


def _apply_deterministic_family_mocks(monkeypatch):
    class _FakeMetadataOverviewService:
        def __init__(self, data_manager):
            self.data_manager = data_manager

        def get_publication_queue_summary(self, status_filter=None):
            _ = status_filter
            return {
                "total": 3,
                "status_breakdown": {
                    "pending": 2,
                    "metadata_extracted": 1,
                },
                "identifier_coverage": {
                    "pmid": {"count": 2, "pct": 66.7},
                    "doi": {"count": 1, "pct": 33.3},
                },
                "extracted_datasets": {
                    "geo": 2,
                    "sra": 1,
                },
                "workspace_ready": 1,
                "recent_errors": [],
            }

    class _Provider:
        def __init__(self, name: str, display_name: str):
            self.name = name
            self.display_name = display_name

    import lobster.config.llm_factory as llm_factory
    import lobster.config.providers as providers
    import lobster.services.metadata.metadata_overview_service as metadata_overview_service

    monkeypatch.setattr(
        metadata_overview_service,
        "MetadataOverviewService",
        _FakeMetadataOverviewService,
    )
    monkeypatch.setattr(
        llm_factory.LLMFactory,
        "get_available_providers",
        staticmethod(lambda: ["openai", "ollama"]),
    )
    monkeypatch.setattr(
        llm_factory.LLMFactory,
        "get_current_provider",
        staticmethod(lambda: "openai"),
    )
    monkeypatch.setattr(
        providers.ProviderRegistry,
        "get_all",
        staticmethod(
            lambda: [
                _Provider("openai", "OpenAI"),
                _Provider("ollama", "Ollama"),
                _Provider("anthropic", "Anthropic"),
            ]
        ),
    )


def _apply_config_show_mocks(
    monkeypatch: pytest.MonkeyPatch,
    workspace_path: Path,
    global_config_dir: Path,
) -> None:
    workspace_path.mkdir(parents=True, exist_ok=True)
    (workspace_path / "provider_config.json").write_text("{}", encoding="utf-8")
    global_config_dir.mkdir(parents=True, exist_ok=True)
    (global_config_dir / "providers.json").write_text("{}", encoding="utf-8")

    class _FakeResolver:
        def __init__(self, workspace_path: Path | None = None):
            self.workspace_path = workspace_path

        def resolve_provider(self, runtime_override=None):
            _ = runtime_override
            return ("ollama", "workspace config")

        def resolve_profile(self):
            return ("production", "workspace config")

        def resolve_model(self, agent_name, runtime_override=None, provider=None):
            _ = runtime_override, provider
            models = {
                "supervisor": "llama3.2:latest",
                "research_agent": "llama3.2:latest",
            }
            return (models[agent_name], "workspace config")

    class _FakeSettings:
        def get_agent_llm_params(self, agent_name):
            _ = agent_name
            return {}

    class _FakeProvider:
        def get_default_model(self):
            return "llama3.2:latest"

    import lobster.config.agent_registry as agent_registry
    import lobster.config.global_config as global_config
    import lobster.config.providers as providers
    import lobster.config.settings as settings
    import lobster.config.subscription_tiers as subscription_tiers
    import lobster.config.workspace_agent_config as workspace_agent_config
    import lobster.config.workspace_config as workspace_config
    import lobster.core.component_registry as component_registry
    import lobster.core.config_resolver as config_resolver
    import lobster.core.license_manager as license_manager

    monkeypatch.setattr(global_config, "CONFIG_DIR", global_config_dir)
    monkeypatch.setattr(config_resolver, "ConfigResolver", _FakeResolver)
    monkeypatch.setattr(settings, "get_settings", lambda: _FakeSettings())
    monkeypatch.setattr(providers, "get_provider", lambda provider_name: _FakeProvider())
    monkeypatch.setattr(
        workspace_config.WorkspaceProviderConfig,
        "exists",
        staticmethod(lambda path: Path(path) == workspace_path),
    )
    monkeypatch.setattr(license_manager, "get_current_tier", lambda: "free")
    monkeypatch.setattr(subscription_tiers, "is_agent_available", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        agent_registry,
        "AGENT_REGISTRY",
        {
            "supervisor": SimpleNamespace(
                display_name="Supervisor",
                child_agents=[],
            ),
            "research_agent": SimpleNamespace(
                display_name="Research Agent",
                child_agents=[],
            ),
        },
    )
    monkeypatch.setattr(agent_registry, "get_valid_handoffs", lambda: {"supervisor": set()})
    monkeypatch.setattr(
        component_registry.component_registry,
        "list_agents",
        lambda: {
            "research_agent": SimpleNamespace(
                display_name="Research Agent",
                tier_requirement="free",
            ),
            "supervisor": SimpleNamespace(
                display_name="Supervisor",
                tier_requirement="free",
            ),
        },
    )
    monkeypatch.setattr(
        workspace_agent_config.WorkspaceAgentConfig,
        "load",
        staticmethod(lambda _path: SimpleNamespace(preset=None, enabled_agents=[])),
    )


def _apply_config_model_mocks(
    monkeypatch: pytest.MonkeyPatch,
    workspace_path: Path,
) -> None:
    workspace_path.mkdir(parents=True, exist_ok=True)

    class _FakeResolver:
        def __init__(self, workspace_path: Path | None = None):
            self.workspace_path = workspace_path

        def resolve_provider(self, runtime_override=None):
            _ = runtime_override
            return ("ollama", "workspace config")

        def resolve_model(self, agent_name=None, runtime_override=None, provider=None):
            _ = agent_name, runtime_override, provider
            return ("llama3.2:latest", "workspace config (ollama model)")

    class _FakeModel:
        def __init__(
            self,
            name: str,
            display_name: str,
            description: str,
            is_default: bool = False,
        ):
            self.name = name
            self.display_name = display_name
            self.description = description
            self.is_default = is_default

    class _FakeProvider:
        display_name = "Ollama"

        def is_available(self):
            return True

        def list_models(self):
            return [
                _FakeModel(
                    "llama3.2:latest",
                    "Llama 3.2 Latest",
                    "Balanced default local model for everyday chat and analysis.",
                    is_default=True,
                ),
                _FakeModel(
                    "qwen2.5-coder:14b",
                    "Qwen 2.5 Coder 14B",
                    "Code-focused local model with stronger instruction following for long edits.",
                ),
            ]

        def get_default_model(self):
            return "llama3.2:latest"

    import lobster.config.providers as providers
    import lobster.core.config_resolver as config_resolver

    monkeypatch.setattr(config_resolver, "ConfigResolver", _FakeResolver)
    monkeypatch.setattr(providers, "get_provider", lambda _provider_name: _FakeProvider())


@pytest.mark.parametrize(
    ("command", "golden_name"),
    [
        ("/help", "help.json"),
        ("/session", "session.json"),
        ("/status", "status.json"),
        ("/tokens", "tokens.json"),
        ("/config", "config_show.json"),
        ("/config model", "config_model.json"),
        ("/workspace list", "workspace_list.json"),
        ("/queue", "queue.json"),
        ("/metadata publications", "metadata_publications.json"),
        ("/config provider", "config_provider.json"),
        ("/pipeline", "pipeline_list.json"),
    ],
)
def test_slash_command_protocol_golden_transcripts(
    command: str,
    golden_name: str,
    monkeypatch,
):
    try:
        import lobster.core.governance.license_manager as license_manager

        monkeypatch.setattr(license_manager, "get_current_tier", lambda: "free")
    except Exception:
        pass

    if command == "/config":
        workspace_path = Path("/tmp/lobster_config_show_ws")
        _apply_config_show_mocks(
            monkeypatch,
            workspace_path=workspace_path,
            global_config_dir=Path("/tmp/lobster_config_show_global"),
        )
    elif command == "/config model":
        workspace_path = Path("/tmp/lobster_config_model_ws")
        _apply_config_model_mocks(
            monkeypatch,
            workspace_path=workspace_path,
        )
    else:
        workspace_path = Path("/tmp/lobster_ws")
        _apply_deterministic_family_mocks(monkeypatch)

    client = _DummyClient(workspace_path)
    if command == "/config model":
        client.provider_override = None
        client.model_override = None
    events = []
    output = ProtocolOutputAdapter(lambda msg_type, payload: events.append({"type": msg_type, "payload": payload}))

    summary = slash_commands._execute_command(
        command,
        client,
        original_command=command,
        output=output,
    )

    actual = {"summary": summary, "events": events}
    golden_path = _GOLDEN_DIR / golden_name

    if _UPDATE_GOLDENS:
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        pytest.skip(f"updated golden file: {golden_path}")

    expected = json.loads(golden_path.read_text(encoding="utf-8"))
    assert actual == expected
