from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from lobster.cli_internal.commands.output_adapter import (
    ProtocolOutputAdapter,
    kv_block,
    list_block,
    section_block,
    table_block,
)
import lobster.cli_internal.commands.heavy.slash_commands as slash_commands


pytestmark = pytest.mark.integration

_GOLDEN_DIR = Path(__file__).resolve().parents[1] / "golden" / "slash_commands"
_UPDATE_GOLDENS = os.getenv("LOBSTER_UPDATE_GOLDENS", "").lower() in {"1", "true", "yes"}


class _DummyDataManager:
    def __init__(self, workspace_path: Path):
        self.workspace_path = workspace_path
        (workspace_path / "data").mkdir(parents=True, exist_ok=True)
        (workspace_path / "exports").mkdir(parents=True, exist_ok=True)
        (workspace_path / "queues").mkdir(parents=True, exist_ok=True)
        (workspace_path / "metadata").mkdir(parents=True, exist_ok=True)
        (workspace_path / "metadata" / "exports").mkdir(parents=True, exist_ok=True)
        (workspace_path / "notebooks").mkdir(parents=True, exist_ok=True)
        (workspace_path / "demo.txt").write_text("hello\n", encoding="utf-8")
        (workspace_path / "notes.txt").write_text("alpha\nbeta\n", encoding="utf-8")
        (workspace_path / "publications.ris").write_text(
            "TY  - JOUR\nER  -\n", encoding="utf-8"
        )
        (workspace_path / "import_queue.jsonl").write_text(
            '{"title":"Paper A"}\n{"title":"Paper B"}\n',
            encoding="utf-8",
        )
        (workspace_path / "metadata" / "study_metadata.json").write_text(
            '{"study":"demo"}\n',
            encoding="utf-8",
        )
        (workspace_path / "metadata" / "exports" / "legacy_summary.csv").write_text(
            "sample,count\nA,10\n",
            encoding="utf-8",
        )
        (workspace_path / "exports" / "analysis_summary.csv").write_text(
            "sample,count\nA,10\n",
            encoding="utf-8",
        )
        (workspace_path / "notebooks" / "rna_qc_workflow.ipynb").write_text(
            '{"cells":[],"metadata":{"lobster":{"created_by":"research_agent","created_at":"2026-03-01T12:00:00","lobster_version":"1.1.2","dependencies":{"scanpy":"1.10.0","numpy":"2.1.0"}}},"nbformat":4,"nbformat_minor":5}',
            encoding="utf-8",
        )
        (workspace_path / "notebooks" / "atac_cluster_analysis.ipynb").write_text(
            '{"cells":[],"metadata":{"lobster":{"created_by":"data_expert","created_at":"2026-03-02T09:15:00","lobster_version":"1.1.2","dependencies":{"muon":"0.1.6"}}},"nbformat":4,"nbformat_minor":5}',
            encoding="utf-8",
        )
        (workspace_path / "data" / "geo_gse12345_rna.h5ad").write_bytes(b"H5AD")
        (workspace_path / "data" / "geo_gse67890_atac_clustered.h5ad").write_bytes(b"H5AD")
        self.modalities = {
            "rna": self._make_modality(
                n_obs=1024,
                n_vars=2048,
                obs_columns=["cell_type", "batch", "condition"],
                var_columns=["gene_name", "feature_type"],
                layers=["counts"],
                obsm=["X_pca", "X_umap"],
                uns=["neighbors", "rank_genes_groups"],
            ),
            "atac": self._make_modality(
                n_obs=512,
                n_vars=1024,
                obs_columns=["cluster", "sample"],
                var_columns=["peak_id"],
                layers=[],
                obsm=["X_lsi"],
                uns=["peaks"],
            ),
        }
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
        self.metadata_store = {
            "GSE12345": {
                "metadata": {
                    "title": "Pancreatic atlas",
                    "samples": {"sample_a": {}, "sample_b": {}},
                },
                "validation": {"predicted_data_type": "single_cell_rna"},
                "fetch_timestamp": "2026-03-01T12:00:00Z",
            }
        }
        self.current_metadata = {
            "dataset_id": "GSE12345",
            "summary": {"platform": "10x", "species": "human"},
            "tags": ["tumor", "atlas"],
        }

    @staticmethod
    def _make_modality(
        *,
        n_obs: int,
        n_vars: int,
        obs_columns: list[str],
        var_columns: list[str],
        layers: list[str],
        obsm: list[str],
        uns: list[str],
    ):
        class _AxisFrame:
            def __init__(self, columns):
                self.columns = columns

        return SimpleNamespace(
            n_obs=n_obs,
            n_vars=n_vars,
            obs=_AxisFrame(obs_columns),
            var=_AxisFrame(var_columns),
            layers={name: object() for name in layers},
            obsm={name: object() for name in obsm},
            varm={},
            uns={name: object() for name in uns},
        )

    def get_available_datasets(self, force_refresh: bool = False):
        _ = force_refresh
        return self.available_datasets

    def list_modalities(self):
        return list(self.modalities.keys())

    def log_tool_usage(self, **_kwargs):
        return None

    def get_modality(self, modality_name: str):
        return self.modalities[modality_name]

    def list_workspace_files(self):
        return {
            "data": [
                {
                    "name": "geo_gse12345_rna.h5ad",
                    "size": 12_800,
                    "modified": 1_709_285_400,
                    "path": str(self.workspace_path / "data" / "geo_gse12345_rna.h5ad"),
                }
            ],
            "exports": [
                {
                    "name": "analysis_summary.csv",
                    "size": 3_072,
                    "modified": 1_709_199_000,
                    "path": str(self.workspace_path / "exports" / "analysis_summary.csv"),
                }
            ],
        }

    def auto_save_state(self, force: bool = False):
        _ = force
        return ["rna_saved.h5ad", "Skipped atac_saved.h5ad"]

    def restore_session(self, pattern: str = "recent"):
        if pattern == "recent":
            return {
                "restored": ["geo_gse12345_rna", "geo_gse67890_atac_clustered"],
                "skipped": ["already_loaded_reference"],
                "total_size_mb": 20.8,
            }
        return {
            "restored": [],
            "skipped": [],
            "total_size_mb": 0.0,
        }

    def load_dataset(self, dataset_name: str):
        return dataset_name in self.available_datasets

    def get_workspace_status(self):
        return {
            "workspace_path": str(self.workspace_path),
            "modalities_loaded": len(self.modalities),
            "provenance_enabled": True,
            "mudata_available": True,
            "modality_names": ["rna", "atac"],
            "registered_backends": ["h5ad", "mudata"],
            "registered_adapters": [
                "transcriptomics",
                "chromatin",
                "multimodal",
                "proteomics",
                "metadata",
                "plots",
            ],
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

    def export_notebook(self, name: str, description: str = ""):
        _ = description
        path = self.workspace_path / "notebooks" / f"{name}.ipynb"
        path.write_text(
            '{"cells":[],"metadata":{"lobster":{"created_by":"research_agent","created_at":"2026-03-06T10:00:00","lobster_version":"1.1.2"}},"nbformat":4,"nbformat_minor":5}',
            encoding="utf-8",
        )
        return path

    def run_notebook(self, notebook_name: str, input_modality: str, dry_run: bool = False):
        _ = input_modality
        if dry_run:
            return {
                "validation": SimpleNamespace(
                    has_errors=False,
                    errors=[],
                    has_warnings=False,
                    warnings=[],
                ),
                "steps_to_execute": 4,
                "estimated_duration_minutes": 3,
            }
        return {
            "status": "success",
            "output_notebook": str(
                self.workspace_path
                / "notebooks"
                / f"{Path(notebook_name).stem}_executed.ipynb"
            ),
            "execution_time": 12.5,
        }


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

    def list_entries(self):
        return [
            SimpleNamespace(
                dataset_id="GSE12345",
                database="geo",
                status=SimpleNamespace(value="pending"),
                recommended_strategy=SimpleNamespace(strategy_name="fastq"),
                priority=3,
            ),
            SimpleNamespace(
                dataset_id="SRP000001",
                database="sra",
                status=SimpleNamespace(value="completed"),
                recommended_strategy=SimpleNamespace(strategy_name="sra"),
                priority=1,
            ),
            SimpleNamespace(
                dataset_id="GSE77777",
                database="geo",
                status=SimpleNamespace(value="pending"),
                recommended_strategy=None,
                priority=5,
            ),
        ]

    def clear_queue(self):
        return None


class _DummyPublicationQueue:
    def __init__(self, workspace_path: Path):
        self.queue_file = workspace_path / "queues" / "publication_queue.jsonl"
        self.queue_file.write_text(
            '{"title":"Paper A"}\n{"title":"Paper B"}\n',
            encoding="utf-8",
        )

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

    def list_entries(self):
        return [
            SimpleNamespace(
                title="Single-cell atlas of pancreatic cancer",
                year=2024,
                status=SimpleNamespace(value="pending"),
                pmid="123456",
                doi=None,
            ),
            SimpleNamespace(
                title="ATAC landscape in tumor microenvironment",
                year=2023,
                status=SimpleNamespace(value="completed"),
                pmid=None,
                doi="10.1000/example",
            ),
        ]

    def clear_queue(self):
        return None

    def import_entries(self, entries, skip_duplicates=True):
        _ = skip_duplicates
        return {
            "imported": len(entries),
            "skipped": 0,
            "errors": 0,
        }


class _DummyClient:
    def __init__(self, workspace_path: Path):
        self.session_id = "sess_123"
        self.workspace_path = workspace_path
        self.data_manager = _DummyDataManager(workspace_path)
        self.publication_queue = _DummyPublicationQueue(workspace_path)
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

    def get_status(self):
        return {
            "session_id": self.session_id,
            "message_count": 7,
            "workspace": str(self.workspace_path),
            "has_data": True,
            "data_summary": {
                "shape": "2 modalities",
                "memory_usage": "20 MB",
            },
        }

    def detect_file_type(self, file_path: Path):
        suffix = file_path.suffix.lower()
        if suffix in {".txt", ".md", ".py", ".json", ".yaml", ".yml", ".csv", ".tsv", ".log"}:
            return {"description": "Text file", "category": "text", "binary": False}
        if suffix == ".h5ad":
            return {
                "description": "AnnData H5AD",
                "category": "bioinformatics",
                "binary": True,
            }
        return {"description": "Binary file", "category": "unknown", "binary": True}

    def locate_file(self, filename: str):
        candidates = [
            self.workspace_path / filename,
            self.workspace_path / "data" / filename,
            self.workspace_path / "exports" / filename,
        ]
        for candidate in candidates:
            if candidate.exists():
                return {"found": True, "path": candidate}
        return {
            "found": False,
            "error": f"File not found: {filename}",
            "searched_paths": [str(path) for path in candidates],
        }

    def load_data_file(self, file_path: str):
        path = Path(file_path)
        return {
            "success": True,
            "modality_name": path.stem,
            "data_shape": (128, 64),
        }

    def load_publication_list(
        self,
        *,
        file_path: str,
        priority: int,
        schema_type: str,
        extraction_level: str,
    ):
        _ = (file_path, priority, schema_type, extraction_level)
        return {
            "added_count": 2,
            "duplicate_count": 1,
            "file_duplicates": 1,
            "queue_duplicates": 0,
            "skipped_count": 0,
        }


def _apply_deterministic_family_mocks(monkeypatch):
    class _FakeMetadataOverviewService:
        def __init__(self, data_manager):
            self.data_manager = data_manager

        def get_quick_overview(self):
            return {
                "publication_queue": {
                    "total": 3,
                    "status_breakdown": {
                        "pending": 2,
                        "metadata_extracted": 1,
                    },
                    "workspace_ready": 1,
                    "extracted_datasets": 2,
                },
                "samples": {
                    "total_samples": 1200,
                    "bioproject_count": 4,
                    "has_aggregated": True,
                    "filtered_samples": 900,
                    "retention_rate": 75.0,
                    "disease_coverage": 62.5,
                },
                "workspace": {
                    "metadata_files": 3,
                    "export_files": 2,
                    "in_memory_entries": 1,
                    "total_size_mb": 4.5,
                },
                "next_steps": ["/metadata publications", "/metadata workspace"],
                "has_deprecated": True,
            }

        def get_sample_statistics(self):
            return {
                "total_samples": 1200,
                "bioproject_count": 4,
                "has_aggregated": True,
                "filtered_samples": 900,
                "retention_rate": 75.0,
                "disease_coverage": 62.5,
                "filter_criteria": "disease != control",
                "filter_breakdown": {
                    "disease_filter": {"retained": 900, "total": 1200},
                },
            }

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

        def get_workspace_inventory(self):
            return {
                "metadata_store_count": 3,
                "metadata_store_categories": {"geo": 2, "sra": 1},
                "workspace_files": {"study": 2, "samples": 1},
                "workspace_files_total": 3,
                "total_size_mb": 2.5,
                "exports_total": 2,
                "exports": [
                    {
                        "name": "analysis_summary.csv",
                        "size_kb": 3.0,
                        "modified": "2026-03-02 10:00",
                    },
                    {
                        "name": "sample_table.tsv",
                        "size_kb": 1.2,
                        "modified": "2026-03-01 09:30",
                    },
                ],
                "deprecated_warnings": [
                    "Found files in deprecated metadata/exports/ location."
                ],
            }

        def get_export_summary(self):
            return {
                "total_count": 2,
                "categories": {"analysis_results": 1, "sample_data": 1},
                "files": [
                    {
                        "name": "analysis_summary.csv",
                        "size_kb": 3.0,
                        "modified": "2026-03-02 10:00",
                    },
                    {
                        "name": "sample_table.tsv",
                        "size_kb": 1.2,
                        "modified": "2026-03-01 09:30",
                    },
                ],
                "usage_hints": {
                    "list": "/files",
                    "access": "workspace/exports/analysis_summary.csv",
                    "cli": "/metadata exports",
                },
            }

    class _Provider:
        def __init__(self, name: str, display_name: str):
            self.name = name
            self.display_name = display_name

    class _FakeModalityManagementService:
        def __init__(self, data_manager):
            self.data_manager = data_manager

        def remove_modality(self, modality_name: str):
            if modality_name not in self.data_manager.modalities:
                return False, {}, {}
            self.data_manager.modalities.pop(modality_name, None)
            return (
                True,
                {
                    "removed_modality": modality_name,
                    "shape": {"n_obs": 1024, "n_vars": 2048},
                },
                {"op": "remove", "modality": modality_name},
            )

    import lobster.config.llm_factory as llm_factory
    import lobster.config.agent_defaults as agent_defaults
    import lobster.config.providers as providers
    import lobster.services.metadata.metadata_overview_service as metadata_overview_service
    import lobster.services.data_management.modality_management_service as modality_management_service
    import lobster.utils as lobster_utils

    monkeypatch.setattr(agent_defaults, "get_current_profile", lambda: "semantic")
    monkeypatch.setattr(
        metadata_overview_service,
        "MetadataOverviewService",
        _FakeMetadataOverviewService,
    )
    monkeypatch.setattr(
        modality_management_service,
        "ModalityManagementService",
        _FakeModalityManagementService,
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
    monkeypatch.setattr(
        slash_commands,
        "build_status_blocks",
        lambda: [
            section_block(title="Lobster Status"),
            kv_block(
                [
                    ("Initialization", "Configured"),
                    ("Provider", "openai"),
                    ("Config File", "/tmp/lobster_ws/.env"),
                ],
                title="Initialization",
            ),
            kv_block(
                [("Subscription Tier", "Free"), ("Source", "default")],
                title="Subscription",
            ),
            table_block(
                title="Installed Packages",
                columns=[
                    {"name": "Package"},
                    {"name": "Version"},
                    {"name": "Status"},
                ],
                rows=[["lobster-ai", "1.1.2", "Installed"]],
            ),
            table_block(
                title="Optional Capabilities",
                columns=[
                    {"name": "Status"},
                    {"name": "Capability"},
                    {"name": "Details"},
                ],
                rows=[
                    ["available", "Semantic Search", "Vector backend available"],
                    ["available", "Document Intelligence", "docling"],
                ],
            ),
            list_block(
                ["data_expert_agent", "research_agent"],
                title="Available Agents (2)",
            ),
            list_block(
                ["community_support", "local_only"],
                title="Enabled Features",
            ),
        ],
    )
    monkeypatch.setattr(
        lobster_utils,
        "open_path",
        lambda path: (True, f"Opened {Path(path).name}"),
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
        ("/workspace", "workspace.json"),
        ("/workspace list", "workspace_list.json"),
        ("/workspace info 1", "workspace_info.json"),
        ("/workspace load 1", "workspace_load.json"),
        ("/workspace load recent", "workspace_load_recent.json"),
        ("/workspace load archived", "workspace_load_archived.json"),
        ("/workspace remove rna", "workspace_remove.json"),
        ("/files", "files.json"),
        ("/read notes.txt", "read_notes.json"),
        ("/read data/*.h5ad", "read_data_glob.json"),
        ("/read missing.txt", "read_missing_file.json"),
        ("/read missing/*.txt", "read_missing_glob.json"),
        ("/open demo.txt", "open_demo.json"),
        ("/open missing.txt", "open_missing.json"),
        ("/open ../outside.txt", "open_traversal.json"),
        ("/save", "save.json"),
        ("/restore", "restore.json"),
        ("/queue", "queue.json"),
        ("/queue list", "queue_list.json"),
        ("/queue list download", "queue_list_download.json"),
        ("/queue load publications.ris", "queue_load.json"),
        ("/queue clear all", "queue_clear_all.json"),
        ("/queue export saved_queue", "queue_export.json"),
        ("/queue import import_queue.jsonl", "queue_import.json"),
        ("/metadata", "metadata_overview.json"),
        ("/metadata publications", "metadata_publications.json"),
        ("/metadata samples", "metadata_samples.json"),
        ("/metadata workspace", "metadata_workspace.json"),
        ("/metadata exports", "metadata_exports.json"),
        ("/metadata list", "metadata_list.json"),
        ("/metadata clear", "metadata_clear.json"),
        ("/metadata clear exports", "metadata_clear_exports.json"),
        ("/metadata clear all", "metadata_clear_all.json"),
        ("/config provider", "config_provider.json"),
        ("/status-panel", "status_panel.json"),
        ("/workspace-info", "workspace_info_panel.json"),
        ("/analysis-dash", "analysis_dash.json"),
        ("/progress", "progress_panel.json"),
        ("/pipeline", "pipeline_list.json"),
        ("/pipeline export analysis_workflow Demo notebook", "pipeline_export.json"),
        ("/pipeline run rna_qc_workflow.ipynb rna", "pipeline_run.json"),
        ("/pipeline info", "pipeline_info.json"),
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
    monkeypatch.setattr(slash_commands, "current_directory", workspace_path)
    if command == "/queue import import_queue.jsonl":
        import lobster.core.schemas.publication_queue as publication_queue_schema

        monkeypatch.setattr(
            publication_queue_schema.PublicationQueueEntry,
            "from_dict",
            staticmethod(
                lambda data: SimpleNamespace(
                    title=data.get("title", "unknown"),
                    workspace_metadata_keys=[],
                )
            ),
        )
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
