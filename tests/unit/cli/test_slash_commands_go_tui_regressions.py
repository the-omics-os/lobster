from pathlib import Path
from types import SimpleNamespace

import pytest

import lobster.cli_internal.commands.heavy.display_helpers as display_helpers
import lobster.cli_internal.commands.heavy.slash_commands as slash_commands
from lobster.cli_internal.commands.output_adapter import (
    hint_block,
    kv_block,
    list_block,
    section_block,
    table_block,
)


class _DummyOutput:
    def __init__(self):
        self.messages = []

    def print(self, message, style=None):
        self.messages.append((style, message))

    def print_table(self, table_data):
        self.messages.append(("table", table_data))

    def confirm(self, question):
        self.messages.append(("confirm", question))
        return False

    def prompt(self, question, default=""):
        self.messages.append(("prompt", question))
        return default

    def print_code_block(self, code, language="python"):
        self.messages.append(("code", language, code))


class ProtocolOutputAdapter(_DummyOutput):
    def render_blocks(self, blocks):
        for block in blocks:
            data = block.data
            if block.kind == "section":
                if data.get("title"):
                    self.print(data["title"], style=data.get("style"))
                if data.get("body"):
                    self.print(data["body"], style=data.get("style"))
            elif block.kind == "kv":
                self.print_table(
                    {
                        "title": data.get("title"),
                        "columns": [
                            {"name": data.get("key_label", "Field")},
                            {"name": data.get("value_label", "Value")},
                        ],
                        "rows": data.get("rows", []),
                    }
                )
            elif block.kind == "table":
                self.print_table(
                    {
                        "title": data.get("title"),
                        "columns": data.get("columns", []),
                        "rows": data.get("rows", []),
                        "width": data.get("width"),
                    }
                )
            elif block.kind == "list":
                if data.get("title"):
                    self.print(data["title"])
                ordered = bool(data.get("ordered"))
                lines = []
                for index, item in enumerate(data.get("items", []), start=1):
                    prefix = f"{index}." if ordered else "-"
                    lines.append(f"{prefix} {item}")
                if lines:
                    self.print("\n".join(lines))
            elif block.kind == "code":
                if data.get("title"):
                    self.print(data["title"])
                self.print_code_block(
                    data.get("code", ""),
                    language=data.get("language", "python"),
                )
            elif block.kind == "alert":
                self.print(data.get("message", ""), style=data.get("level", "info"))
            elif block.kind == "hint":
                self.print(data.get("message", ""), style="dim")
            else:
                raise ValueError(f"Unsupported block kind: {block.kind}")


class ConfirmingProtocolOutputAdapter(ProtocolOutputAdapter):
    def confirm(self, question):
        self.messages.append(("confirm", question))
        return True


class _DummyClient:
    def __init__(self, workspace_path: Path):
        self.workspace_path = workspace_path
        self.session_id = "test_session"
        self.provider_override = None
        self.model_override = None
        (workspace_path / "data").mkdir(parents=True, exist_ok=True)
        (workspace_path / "exports").mkdir(parents=True, exist_ok=True)
        (workspace_path / "metadata").mkdir(parents=True, exist_ok=True)
        (workspace_path / "metadata" / "exports").mkdir(parents=True, exist_ok=True)
        (workspace_path / "notebooks").mkdir(parents=True, exist_ok=True)
        (workspace_path / "demo.txt").write_text("hello", encoding="utf-8")
        (workspace_path / "notes.txt").write_text("alpha\nbeta\n", encoding="utf-8")
        (workspace_path / "metadata" / "study_metadata.json").write_text(
            '{"study":"demo"}',
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
        (workspace_path / "notebooks" / "analysis_workflow.ipynb").write_text(
            '{"cells":[],"metadata":{"lobster":{"created_by":"research_agent","created_at":"2026-03-01T12:00:00","lobster_version":"1.1.2","dependencies":{"scanpy":"1.10.0","numpy":"2.1.0"}}},"nbformat":4,"nbformat_minor":5}',
            encoding="utf-8",
        )
        (workspace_path / "data" / "rna.h5ad").write_bytes(b"H5AD")
        available_datasets = {
            "geo_gse12345_rna": {
                "size_mb": 12.5,
                "shape": (1024, 2048),
                "modified": "2026-03-01T10:00:00",
                "path": str(workspace_path / "data" / "geo_gse12345_rna.h5ad"),
                "type": "h5ad",
            }
        }
        self.data_manager = type(
            "DM",
            (),
            {
                "workspace_path": workspace_path,
                "modalities": {},
                "available_datasets": available_datasets,
                "get_available_datasets": lambda _self, force_refresh=False: available_datasets,
                "get_workspace_status": lambda _self: {
                    "workspace_path": str(workspace_path),
                    "modalities_loaded": 0,
                    "provenance_enabled": True,
                    "mudata_available": False,
                    "directories": {
                        "data": str(workspace_path / "data"),
                        "exports": str(workspace_path / "exports"),
                    },
                    "registered_backends": ["h5ad"],
                    "registered_adapters": ["transcriptomics"],
                    "modality_names": [],
                },
                "list_workspace_files": lambda _self: {
                    "data": [
                        {
                            "name": "rna.h5ad",
                            "size": 2048,
                            "modified": 1700000000,
                            "path": str(workspace_path / "data" / "rna.h5ad"),
                        }
                    ],
                    "exports": [],
                },
                "metadata_store": {
                    "GSE12345": {
                        "metadata": {
                            "title": "Pancreatic atlas",
                            "samples": {"sample_a": {}, "sample_b": {}},
                        },
                        "validation": {
                            "predicted_data_type": "single_cell_rna",
                        },
                        "fetch_timestamp": "2026-03-01T12:00:00Z",
                    }
                },
                "current_metadata": {
                    "dataset_id": "GSE12345",
                    "summary": {"platform": "10x", "species": "human"},
                    "tags": ["tumor", "atlas"],
                },
                "list_modalities": lambda _self: [],
                "list_notebooks": lambda _self: [
                    {
                        "name": "RNA QC Workflow",
                        "filename": "analysis_workflow.ipynb",
                        "created_by": "research_agent",
                        "created_at": "2026-03-01T12:00:00",
                        "n_steps": 7,
                        "size_kb": 43.2,
                        "path": str(workspace_path / "notebooks" / "analysis_workflow.ipynb"),
                    }
                ],
                "export_notebook": lambda _self, name, description="": workspace_path
                / "notebooks"
                / f"{name}.ipynb",
                "run_notebook": lambda _self, notebook_name, input_modality, dry_run=False: {
                    "validation": SimpleNamespace(
                        has_errors=False,
                        errors=[],
                        has_warnings=False,
                        warnings=[],
                    ),
                    "steps_to_execute": 4,
                    "estimated_duration_minutes": 3,
                }
                if dry_run
                else {
                    "status": "success",
                    "output_notebook": str(workspace_path / "notebooks" / f"{Path(notebook_name).stem}_executed.ipynb"),
                    "execution_time": 12.5,
                },
                "log_tool_usage": lambda _self, **_kwargs: None,
                "load_dataset": lambda _self, _name: True,
                "restore_session": lambda _self, pattern="recent": {
                    "restored": ["geo_gse12345_rna"] if pattern == "recent" else [],
                    "skipped": [],
                    "total_size_mb": 12.5,
                },
                "auto_save_state": lambda _self, force=False: [],
            },
        )()
        self.data_manager.modalities = {
            "rna": SimpleNamespace(n_obs=100, n_vars=200),
        }
        self.data_manager.download_queue = type(
            "DownloadQueue",
            (),
            {
                "get_statistics": lambda _self: {
                    "total_entries": 3,
                    "by_status": {"pending": 2, "completed": 1},
                    "by_database": {"geo": 2, "sra": 1},
                },
                "list_entries": lambda _self: [
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
                ],
                "clear_queue": lambda _self: None,
            },
        )()
        self.publication_queue = type(
            "PublicationQueue",
            (),
            {
                "get_statistics": lambda _self: {
                    "total_entries": 2,
                    "by_status": {"pending": 1, "completed": 1},
                    "by_extraction_level": {"methods": 2},
                    "identifiers_extracted": 4,
                },
                "list_entries": lambda _self: [
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
                ],
                "clear_queue": lambda _self: None,
            },
        )()
        self.publication_queue.queue_file = workspace_path / "queues" / "publication_queue.jsonl"
        self.publication_queue.queue_file.parent.mkdir(parents=True, exist_ok=True)
        self.publication_queue.queue_file.write_text(
            '{"title":"demo"}\n', encoding="utf-8"
        )

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


def test_open_command_executes_and_returns_summary(tmp_path, monkeypatch):
    output = _DummyOutput()
    client = _DummyClient(tmp_path)
    target = tmp_path / "demo.txt"

    monkeypatch.setattr(slash_commands, "current_directory", tmp_path)

    called = {}

    def _fake_open_path(path):
        called["path"] = path
        return True, f"Opened {path.name}"

    import lobster.utils as lobster_utils

    monkeypatch.setattr(lobster_utils, "open_path", _fake_open_path)

    summary = slash_commands._execute_command(
        "/open demo.txt",
        client,
        original_command="/open demo.txt",
        output=output,
    )

    assert summary == "Opened demo.txt"
    assert called["path"] == target
    assert ("success", "Opened demo.txt") in output.messages


def test_open_command_protocol_mode_renders_structured_output(tmp_path, monkeypatch):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)
    target = tmp_path / "demo.txt"
    monkeypatch.setattr(slash_commands, "current_directory", tmp_path)

    def _fake_open_path(path):
        assert path == target
        return True, f"Opened {path.name}"

    import lobster.utils as lobster_utils

    monkeypatch.setattr(lobster_utils, "open_path", _fake_open_path)

    summary = slash_commands._execute_command(
        "/open demo.txt",
        client,
        original_command="/open demo.txt",
        output=output,
    )

    assert summary == "Opened demo.txt"
    assert output.messages == [
        ("success", "Opened demo.txt"),
        (None, f"Path: {target}"),
    ]


def test_open_command_protocol_mode_handles_missing_argument(tmp_path, monkeypatch):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)
    monkeypatch.setattr(slash_commands, "current_directory", tmp_path)

    summary = slash_commands._execute_command(
        "/open",
        client,
        original_command="/open",
        output=output,
    )

    assert summary == "No file or folder specified for /open command"
    assert output.messages == [
        ("error", "/open: missing file or folder argument"),
        (None, "Usage: /open <file_or_folder>"),
    ]


def test_open_command_protocol_mode_handles_missing_target(tmp_path, monkeypatch):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)
    monkeypatch.setattr(slash_commands, "current_directory", tmp_path)

    summary = slash_commands._execute_command(
        "/open missing.txt",
        client,
        original_command="/open missing.txt",
        output=output,
    )

    assert summary is None
    assert output.messages == [
        ("error", "/open: 'missing.txt': No such file or folder"),
        (None, f"Path: {tmp_path / 'missing.txt'}"),
    ]


def test_open_command_protocol_mode_rejects_path_traversal(tmp_path, monkeypatch):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)
    monkeypatch.setattr(slash_commands, "current_directory", tmp_path)

    summary = slash_commands._execute_command(
        "/open ../outside.txt",
        client,
        original_command="/open ../outside.txt",
        output=output,
    )

    assert summary is None
    assert output.messages == [
        (
            "error",
            "/open security error: Path traversal detected: '../outside.txt' escapes allowed directories",
        )
    ]


def test_restore_command_delegates_to_restore_handler(tmp_path, monkeypatch):
    output = _DummyOutput()
    client = _DummyClient(tmp_path)

    seen = {}

    def _fake_restore(_client, _output, pattern="recent"):
        seen["pattern"] = pattern
        return f"restored:{pattern}"

    monkeypatch.setattr(slash_commands, "_command_restore", _fake_restore)

    summary = slash_commands._execute_command(
        "/restore all",
        client,
        original_command="/restore all",
        output=output,
    )

    assert summary == "restored:all"
    assert seen["pattern"] == "all"


def test_files_command_protocol_mode_uses_shared_output_path(tmp_path):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)

    summary = slash_commands._execute_command(
        "/files",
        client,
        original_command="/files",
        output=output,
    )

    assert summary == "Listed 1 workspace files"
    assert len(output.messages) == 1
    kind, table = output.messages[0]
    assert kind == "table"
    assert table["title"] == "Data Files"
    assert table["columns"] == [
        {"name": "Name", "style": "bold white"},
        {"name": "Size", "style": "grey74"},
        {"name": "Modified", "style": "grey50"},
        {"name": "Path", "style": "dim grey50"},
    ]
    assert table["rows"][0][0] == "rna.h5ad"
    assert table["rows"][0][1] == "2.0 KB"
    assert table["rows"][0][3] == "data"


def test_read_command_without_argument_emits_usage_through_output_adapter(tmp_path):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)

    summary = slash_commands._execute_command(
        "/read",
        client,
        original_command="/read",
        output=output,
    )

    assert summary is None
    assert output.messages == [
        (None, "/read views file contents without loading data into memory."),
        (None, "Usage: /read <filename>"),
        ("dim", "Examples: /read my_data.h5ad, /read config.yaml, /read data/*.csv"),
        ("dim", "To load data for analysis, use /workspace load <name>."),
    ]


def test_read_command_protocol_mode_renders_text_file_contents(tmp_path, monkeypatch):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)
    monkeypatch.setattr(slash_commands, "current_directory", tmp_path)

    summary = slash_commands._execute_command(
        "/read notes.txt",
        client,
        original_command="/read notes.txt",
        output=output,
    )

    assert summary == "Displayed text file 'notes.txt' (Text file, 2 lines)"
    assert output.messages == [
        (
            "table",
            {
                "title": "File",
                "columns": [{"name": "Field"}, {"name": "Value"}],
                "rows": [
                    ["Name", "notes.txt"],
                    ["Path", str(tmp_path / "notes.txt")],
                    ["Type", "Text file"],
                ],
            },
        ),
        (None, "Contents: notes.txt"),
        ("code", "text", "alpha\nbeta\n"),
    ]


def test_read_command_protocol_mode_truncates_large_text_preview(tmp_path, monkeypatch):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)
    monkeypatch.setattr(slash_commands, "current_directory", tmp_path)
    long_file = tmp_path / "guide.md"
    long_content = "".join(f"line {idx:03d}\n" for idx in range(130))
    long_file.write_text(long_content, encoding="utf-8")

    summary = slash_commands._execute_command(
        "/read guide.md",
        client,
        original_command="/read guide.md",
        output=output,
    )

    assert summary == "Displayed text file 'guide.md' (Text file, 130 lines)"
    assert output.messages[0] == (
        "table",
        {
            "title": "File",
            "columns": [{"name": "Field"}, {"name": "Value"}],
            "rows": [
                ["Name", "guide.md"],
                ["Path", str(long_file)],
                ["Type", "Text file"],
            ],
        },
    )
    assert output.messages[1] == (None, "Contents: guide.md")
    preview = output.messages[2]
    assert preview[0] == "code"
    assert preview[1] == "text"
    assert "line 119" in preview[2]
    assert "line 120" not in preview[2]
    assert output.messages[3] == (
        "dim",
        "Preview truncated for chat. Use `/open guide.md` for the full file.",
    )


def test_read_command_protocol_mode_handles_glob_miss_with_structured_output(tmp_path, monkeypatch):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)
    monkeypatch.setattr(slash_commands, "current_directory", tmp_path)

    summary = slash_commands._execute_command(
        "/read missing/*.txt",
        client,
        original_command="/read missing/*.txt",
        output=output,
    )

    assert summary is None
    assert output.messages == [
        ("error", "No files found matching pattern: missing/*.txt"),
        (None, f"Searched in: {tmp_path}"),
    ]


def test_read_command_protocol_mode_renders_binary_file_guidance(tmp_path, monkeypatch):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)
    monkeypatch.setattr(slash_commands, "current_directory", tmp_path)

    summary = slash_commands._execute_command(
        "/read rna.h5ad",
        client,
        original_command="/read rna.h5ad",
        output=output,
    )

    assert (
        summary
        == "Inspected file 'rna.h5ad' (AnnData H5AD, 4 bytes) - use /workspace load to load data files"
    )
    assert output.messages == [
        (
            "table",
            {
                "title": "File",
                "columns": [{"name": "Field"}, {"name": "Value"}],
                "rows": [
                    ["Name", "rna.h5ad"],
                    ["Path", str(tmp_path / "data" / "rna.h5ad")],
                    ["Type", "AnnData H5AD"],
                ],
            },
        ),
        (
            "table",
            {
                "title": "File Info",
                "columns": [{"name": "Field"}, {"name": "Value"}],
                "rows": [
                    ["Name", "rna.h5ad"],
                    ["Path", str(tmp_path / "data" / "rna.h5ad")],
                    ["Type", "AnnData H5AD"],
                    ["Category", "bioinformatics"],
                    ["Size", "4 bytes"],
                ],
            },
        ),
        (None, "This is a bioinformatics data file (AnnData H5AD)."),
        ("dim", "To load it into the workspace: /workspace load rna.h5ad"),
    ]


def test_read_command_protocol_mode_reports_missing_file_search_paths(tmp_path, monkeypatch):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)
    monkeypatch.setattr(slash_commands, "current_directory", tmp_path)

    summary = slash_commands._execute_command(
        "/read missing.txt",
        client,
        original_command="/read missing.txt",
        output=output,
    )

    assert summary == "File 'missing.txt' not found"
    assert output.messages == [
        ("error", "File not found: missing.txt"),
        (None, "Searched in:"),
        (
            None,
            "\n".join(
                [
                    f"- {tmp_path / 'missing.txt'}",
                    f"- {tmp_path / 'data' / 'missing.txt'}",
                    f"- {tmp_path / 'exports' / 'missing.txt'}",
                ]
            ),
        ),
    ]


def test_workspace_status_protocol_mode_renders_structured_blocks(tmp_path):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)

    summary = slash_commands._execute_command(
        "/workspace",
        client,
        original_command="/workspace",
        output=output,
    )

    assert summary == "Displayed workspace status and information"
    assert output.messages[0] == (None, "Workspace Status")
    assert output.messages[1] == (
        "table",
        {
            "title": "Summary",
            "columns": [{"name": "Field"}, {"name": "Value"}],
            "rows": [
                ["Workspace", str(tmp_path)],
                ["Datasets Available", "1"],
                ["Modalities Loaded", "0"],
                ["Provenance", "Enabled"],
                ["MuData", "Not installed"],
            ],
            "width": None,
        },
    )
    assert output.messages[2][0] == "table"
    assert output.messages[2][1]["title"] == "Directories"
    assert output.messages[2][1]["columns"] == [
        {"name": "Directory", "style": "bold white"},
        {"name": "Files", "style": "cyan", "justify": "right", "width": 8},
        {"name": "Size", "style": "green", "justify": "right", "width": 10},
    ]
    assert output.messages[3] == ("dim", "No modalities currently loaded")
    assert output.messages[4] == (
        "dim",
        "Use `/workspace list` for dataset inventory and `/workspace info <#>` for file details.",
    )


def test_workspace_status_protocol_mode_uses_compact_active_modality_list(tmp_path):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)
    modality_names = [f"modality_{idx}" for idx in range(5)]

    client.data_manager.get_workspace_status = lambda: {
        "workspace_path": str(tmp_path),
        "modalities_loaded": len(modality_names),
        "provenance_enabled": True,
        "mudata_available": False,
        "directories": {
            "data": str(tmp_path / "data"),
            "exports": str(tmp_path / "exports"),
        },
        "registered_backends": ["h5ad"],
        "registered_adapters": ["transcriptomics"],
        "modality_names": modality_names,
    }
    client.data_manager.list_modalities = lambda: modality_names
    client.data_manager.get_modality = lambda _name: SimpleNamespace(
        n_obs=128,
        n_vars=64,
        obs=SimpleNamespace(columns=["cell_type", "batch"]),
        var=SimpleNamespace(columns=["gene_name"]),
        layers={"counts": object()},
        obsm={"X_umap": object()},
        varm={},
        uns={"neighbors": object()},
    )

    summary = slash_commands._execute_command(
        "/workspace",
        client,
        original_command="/workspace",
        output=output,
    )

    assert summary == "Displayed workspace status and information"
    assert (None, "Active Modalities") in output.messages
    assert any(
        message[0] is None and "modality_0" in message[1] and "modality_4" in message[1]
        for message in output.messages
    )
    assert all(
        not (
            message[0] == "table"
            and message[1].get("title") in modality_names
        )
        for message in output.messages
    )
    assert (
        "dim",
        "Use `/workspace list` for dataset inventory and `/workspace info <#>` for file details.",
    ) in output.messages


def test_workspace_info_protocol_mode_renders_dataset_table(tmp_path):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)

    summary = slash_commands._execute_command(
        "/workspace info 1",
        client,
        original_command="/workspace info 1",
        output=output,
    )

    assert summary == "Displayed details for 1 dataset(s)"
    assert output.messages == [
        (
            "table",
            {
                "title": "Dataset: geo_gse12345_rna",
                "columns": [
                    {"name": "Property", "style": "bold cyan"},
                    {"name": "Value", "style": "white"},
                ],
                "rows": [
                    ["Name", "geo_gse12345_rna"],
                    ["Status", "Not Loaded"],
                    ["Path", str(tmp_path / "data" / "geo_gse12345_rna.h5ad")],
                    ["Size", "12.50 MB"],
                    ["Shape", "1,024 observations × 2,048 variables"],
                    ["Type", "h5ad"],
                    ["Modified", "2026-03-01T10:00:00"],
                ],
                "width": None,
            },
        )
    ]


def test_workspace_load_protocol_mode_renders_loaded_dataset(tmp_path):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)

    summary = slash_commands._execute_command(
        "/workspace load 1",
        client,
        original_command="/workspace load 1",
        output=output,
    )

    assert summary == "Loaded dataset from workspace"
    assert output.messages == [
        (None, "Loading dataset: geo_gse12345_rna..."),
        (None, "Loaded dataset: geo_gse12345_rna (12.5 MB)"),
    ]


def test_data_command_protocol_mode_compacts_long_modality_labels(tmp_path):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)
    long_name = "geo_gse247686_transcriptomics_single_cell_autosave"

    client.data_manager.has_data = lambda: True
    client.data_manager.get_data_summary = lambda: {
        "status": "2 modalities loaded",
        "total_obs": 999,
        "total_vars": 888,
        "modalities": {
            long_name: {
                "shape": (30197, 10),
                "data_type": "ndarray",
                "memory_usage": "1.15 MB (dense)",
                "is_sparse": False,
            },
            "rna_reference": {
                "shape": (128, 64),
                "data_type": "csr_matrix",
                "memory_usage": "0.12 MB (sparse)",
                "is_sparse": True,
            },
        },
    }
    client.data_manager.current_metadata = {}

    summary = slash_commands._execute_command(
        "/data",
        client,
        original_command="/data",
        output=output,
    )

    assert summary == "Displayed data summary (2 modalities)"
    assert output.messages[0] == (
        "table",
        {
            "title": "🦞 Current Data Summary",
            "columns": [
                {"name": "Property", "style": "bold grey93"},
                {"name": "Value", "style": "white"},
            ],
            "rows": [
                ["Status", "2 modalities loaded"],
                ["Total Shape", "999 × 888"],
            ],
        },
    )
    assert output.messages[1] == (None, "Individual Modality Details")
    detail_rows = output.messages[2][1]["rows"]
    assert detail_rows[0][0] != long_name
    assert "..." in detail_rows[0][0]
    assert detail_rows[0][2] == "1.15 MB"
    assert output.messages[3] == (
        "dim",
        "Use `/describe <modality>` for a full modality inspection.",
    )


def test_workspace_load_protocol_mode_renders_file_load(tmp_path, monkeypatch):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)
    monkeypatch.setattr(slash_commands, "current_directory", tmp_path)

    summary = slash_commands._execute_command(
        "/workspace load demo.txt",
        client,
        original_command="/workspace load demo.txt",
        output=output,
    )

    assert summary == "Loaded file 'demo.txt' as modality 'demo'"
    assert output.messages == [
        (None, "Loading file into workspace: demo.txt"),
        (None, "Loaded 'demo' (128 × 64)"),
    ]


def test_workspace_load_protocol_mode_renders_pattern_restore_with_skips(tmp_path):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)
    client.data_manager.restore_session = lambda pattern="recent": {
        "restored": ["geo_gse12345_rna"],
        "skipped": ["already_loaded_reference"],
        "total_size_mb": 12.5,
    }

    summary = slash_commands._execute_command(
        "/workspace load recent",
        client,
        original_command="/workspace load recent",
        output=output,
    )

    assert summary == "Loaded 1 datasets from workspace"
    assert output.messages == [
        (None, "Loading workspace datasets (pattern: recent)..."),
        (None, "Loaded 1 datasets (12.5 MB)"),
        (None, "Loaded Datasets"),
        (None, "- geo_gse12345_rna"),
        ("dim", "Skipped 1 dataset(s) that were already loaded or unavailable."),
        (None, "Skipped Datasets"),
        (None, "- already_loaded_reference"),
    ]


def test_workspace_load_protocol_mode_renders_skipped_only_restore(tmp_path):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)
    client.data_manager.restore_session = lambda pattern="recent": {
        "restored": [],
        "skipped": ["already_loaded_reference"],
        "total_size_mb": 0.0,
    }

    summary = slash_commands._execute_command(
        "/workspace load recent",
        client,
        original_command="/workspace load recent",
        output=output,
    )

    assert summary is None
    assert output.messages == [
        (None, "Loading workspace datasets (pattern: recent)..."),
        ("warning", "No datasets loaded"),
        ("dim", "Skipped 1 dataset(s) that were already loaded or unavailable."),
        (None, "Skipped Datasets"),
        (None, "- already_loaded_reference"),
    ]


def test_workspace_load_protocol_mode_renders_empty_restore(tmp_path):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)
    client.data_manager.restore_session = lambda pattern="recent": {
        "restored": [],
        "skipped": [],
        "total_size_mb": 0.0,
    }

    summary = slash_commands._execute_command(
        "/workspace load archived",
        client,
        original_command="/workspace load archived",
        output=output,
    )

    assert summary is None
    assert output.messages == [
        (None, "Loading workspace datasets (pattern: archived)..."),
        ("warning", "No datasets loaded"),
    ]


def test_queue_status_protocol_mode_renders_structured_output(tmp_path):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)

    summary = slash_commands._execute_command(
        "/queue",
        client,
        original_command="/queue",
        output=output,
    )

    assert summary == "Queue status: 3 downloads, 2 publications (5 total)"
    assert output.messages[0] == (None, "Queue Status")
    assert output.messages[1] == (None, "Download Queue")
    assert output.messages[2] == (
        "table",
        {
            "title": "Status Breakdown",
            "columns": [{"name": "Status"}, {"name": "Count"}],
            "rows": [["○ Pending", "2"], ["✓ Completed", "1"], ["Total", "3"]],
            "width": None,
        },
    )
    assert output.messages[3] == ("dim", "Databases: geo: 2, sra: 1")
    assert output.messages[4] == (None, "Publication Queue")


def test_queue_list_protocol_mode_renders_publication_table(tmp_path):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)

    summary = slash_commands._execute_command(
        "/queue list",
        client,
        original_command="/queue list",
        output=output,
    )

    assert summary == "Listed 2 of 2 publication queue items"
    assert output.messages == [
        (None, "Publication Queue (2 of 2 shown)"),
        (
            "table",
            {
                "title": None,
                "columns": [
                    {"name": "#", "width": 4},
                    {"name": "Title"},
                    {"name": "Year", "width": 6},
                    {"name": "Status"},
                    {"name": "PMID/DOI"},
                ],
                "rows": [
                    [
                        "1",
                        "Single-cell atlas of pancreatic cancer",
                        "2024",
                        "○ pending",
                        "123456",
                    ],
                    [
                        "2",
                        "ATAC landscape in tumor microenvironment",
                        "2023",
                        "✓ completed",
                        "10.1000/example",
                    ],
                ],
                "width": None,
            },
        ),
    ]


def test_queue_list_download_protocol_mode_renders_download_table(tmp_path):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)

    summary = slash_commands._execute_command(
        "/queue list download",
        client,
        original_command="/queue list download",
        output=output,
    )

    assert summary == "Listed 3 of 3 download queue items"
    assert output.messages == [
        (None, "Download Queue (3 of 3 shown)"),
        (
            "table",
            {
                "title": None,
                "columns": [
                    {"name": "#", "width": 4},
                    {"name": "Accession"},
                    {"name": "Database"},
                    {"name": "Status"},
                    {"name": "Strategy"},
                    {"name": "Priority"},
                ],
                "rows": [
                    ["1", "GSE12345", "geo", "○ pending", "fastq", "3"],
                    ["2", "SRP000001", "sra", "✓ completed", "sra", "1"],
                    ["3", "GSE77777", "geo", "○ pending", "N/A", "5"],
                ],
                "width": None,
            },
        ),
        ("dim", "Summary: ○ pending: 2 | ✓ completed: 1"),
    ]


def test_queue_unknown_subcommand_protocol_mode_uses_structured_output(tmp_path):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)

    summary = slash_commands._execute_command(
        "/queue nope",
        client,
        original_command="/queue nope",
        output=output,
    )

    assert summary is None
    assert output.messages == [
        ("warning", "Unknown queue subcommand: nope"),
        (None, "Available: load, list, clear, export, import"),
    ]


def test_queue_load_protocol_mode_renders_structured_output(tmp_path, monkeypatch):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)
    ris_path = tmp_path / "publications.ris"
    ris_path.write_text("TY  - JOUR\nER  -\n", encoding="utf-8")
    monkeypatch.setattr(slash_commands, "current_directory", tmp_path)

    summary = slash_commands._execute_command(
        "/queue load publications.ris",
        client,
        original_command="/queue load publications.ris",
        output=output,
    )

    assert (
        summary
        == "Loaded 2 publications into queue from publications.ris. Awaiting user intent."
    )
    assert output.messages == [
        (None, "Loading into queue: publications.ris"),
        (None, "Loaded 2 items into queue"),
        ("dim", "Deduplicated: 1 duplicates in file"),
        (None, "Next Steps"),
        (
            None,
            "- Extract methods and parameters\n- Search for related datasets (GEO)\n- Build citation network\n- Custom analysis (describe your intent)",
        ),
    ]


def test_queue_load_protocol_mode_surfaces_unsupported_file_type(tmp_path, monkeypatch):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)
    bib_path = tmp_path / "publications.bib"
    bib_path.write_text("@article{demo}\n", encoding="utf-8")
    monkeypatch.setattr(slash_commands, "current_directory", tmp_path)

    summary = slash_commands._execute_command(
        "/queue load publications.bib",
        client,
        original_command="/queue load publications.bib",
        output=output,
    )

    assert summary is None
    assert output.messages == [
        (
            "warning",
            "BibTeX (.bib) support coming soon. Convert to .ris format or wait for future release.",
        )
    ]


def test_queue_clear_all_protocol_mode_confirms_then_cancels(tmp_path):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)

    summary = slash_commands._execute_command(
        "/queue clear all",
        client,
        original_command="/queue clear all",
        output=output,
    )

    assert summary is None
    assert output.messages == [
        (
            "table",
            {
                "title": "About to Clear",
                "columns": [{"name": "Field"}, {"name": "Value"}],
                "rows": [
                    ["Publication queue", "2 items"],
                    ["Download queue", "3 items"],
                    ["Total", "5 items"],
                ],
            },
        ),
        ("confirm", "Clear all 5 items from both queues?"),
        (None, "Operation cancelled"),
    ]


def test_queue_clear_download_protocol_mode_confirms_and_succeeds(tmp_path):
    output = ConfirmingProtocolOutputAdapter()
    client = _DummyClient(tmp_path)
    cleared = []
    client.data_manager.download_queue.clear_queue = lambda: cleared.append("download")

    summary = slash_commands._execute_command(
        "/queue clear download",
        client,
        original_command="/queue clear download",
        output=output,
    )

    assert summary == "Cleared 3 items from download queue"
    assert cleared == ["download"]
    assert output.messages == [
        ("confirm", "Clear all 3 items from download queue?"),
        ("success", "Cleared 3 items from download queue"),
    ]


def test_queue_export_protocol_mode_renders_structured_output(tmp_path):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)

    summary = slash_commands._execute_command(
        "/queue export saved_queue",
        client,
        original_command="/queue export saved_queue",
        output=output,
    )

    expected_path = tmp_path / "saved_queue.jsonl"
    assert summary == "Exported 2 queue items to workspace as 'saved_queue'"
    assert expected_path.exists()
    assert output.messages == [
        (None, "Exporting queue to workspace as 'saved_queue'..."),
        ("success", f"Exported 2 items to: {expected_path}"),
    ]


def test_queue_export_protocol_mode_warns_when_queue_is_empty(tmp_path):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)
    client.publication_queue.get_statistics = lambda: {
        "total_entries": 0,
        "by_status": {},
        "by_extraction_level": {},
        "identifiers_extracted": 0,
    }

    summary = slash_commands._execute_command(
        "/queue export empty_queue",
        client,
        original_command="/queue export empty_queue",
        output=output,
    )

    assert summary is None
    assert output.messages == [("warning", "Queue is empty, nothing to export")]


def test_queue_import_protocol_mode_renders_structured_output(tmp_path, monkeypatch):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)
    import_path = tmp_path / "import_queue.jsonl"
    import_path.write_text('{"title":"Paper A"}\n{"title":"Paper B"}\n', encoding="utf-8")
    monkeypatch.setattr(slash_commands, "current_directory", tmp_path)

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
    client.publication_queue.import_entries = (
        lambda entries, skip_duplicates=True: {
            "imported": len(entries),
            "skipped": 0,
            "errors": 0,
        }
    )

    summary = slash_commands._execute_command(
        "/queue import import_queue.jsonl",
        client,
        original_command="/queue import import_queue.jsonl",
        output=output,
    )

    assert summary == "Queue import complete: imported 2"
    assert output.messages == [
        (None, "Importing queue from: import_queue.jsonl"),
        ("success", "Imported 2 entries into queue"),
    ]


def test_queue_import_protocol_mode_rejects_non_jsonl_files(tmp_path, monkeypatch):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)
    ris_path = tmp_path / "publications.ris"
    ris_path.write_text("TY  - JOUR\nER  -\n", encoding="utf-8")
    monkeypatch.setattr(slash_commands, "current_directory", tmp_path)

    summary = slash_commands._execute_command(
        "/queue import publications.ris",
        client,
        original_command="/queue import publications.ris",
        output=output,
    )

    assert summary is None
    assert output.messages == [
        ("error", "Unsupported file type: .ris"),
        (
            None,
            "The /queue import command only accepts .jsonl files (exported via /queue export).",
        ),
        ("dim", "To load .ris files, use: /queue load <file.ris>"),
    ]


def test_metadata_overview_protocol_mode_renders_structured_output(tmp_path, monkeypatch):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)

    class _FakeMetadataOverviewService:
        def __init__(self, data_manager):
            self.data_manager = data_manager

        def get_quick_overview(self):
            return {
                "publication_queue": {
                    "total": 3,
                    "status_breakdown": {"pending": 2, "metadata_extracted": 1},
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

    import lobster.services.metadata.metadata_overview_service as metadata_overview_service

    monkeypatch.setattr(
        metadata_overview_service,
        "MetadataOverviewService",
        _FakeMetadataOverviewService,
    )

    summary = slash_commands._execute_command(
        "/metadata",
        client,
        original_command="/metadata",
        output=output,
    )

    assert summary == "Metadata overview: 3 publications, 1200 samples"
    assert output.messages[0] == (None, "Metadata Overview")
    assert output.messages[1] == (None, "Publication Queue")
    assert output.messages[2][0] == "table"
    assert output.messages[3] == (
        "dim",
        "Total: 3 | Workspace-ready: 1 | Extracted datasets: 2",
    )
    assert output.messages[4][0] == "table"
    assert output.messages[4][1]["title"] == "Sample Statistics"
    assert output.messages[5][0] == "table"
    assert output.messages[5][1]["title"] == "Workspace Files"
    assert output.messages[6] == (None, "Next Steps")
    assert output.messages[7] == (
        None,
        "- /metadata publications\n- /metadata workspace",
    )
    assert output.messages[8] == (
        "warning",
        "Found files in deprecated metadata/exports/ location. Use /metadata workspace for details.",
    )
    assert output.messages[9] == (
        "dim",
        "Commands: /metadata publications | samples | workspace | exports | clear",
    )


def test_metadata_publications_protocol_mode_renders_structured_output(tmp_path, monkeypatch):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)

    class _FakeMetadataOverviewService:
        def __init__(self, data_manager):
            self.data_manager = data_manager

        def get_publication_queue_summary(self, status_filter=None):
            _ = status_filter
            return {
                "total": 3,
                "status_breakdown": {"pending": 2, "metadata_extracted": 1},
                "identifier_coverage": {
                    "pmid": {"count": 2, "pct": 66.7},
                    "doi": {"count": 1, "pct": 33.3},
                },
                "extracted_datasets": {"geo": 2, "sra": 1},
                "workspace_ready": 1,
                "recent_errors": [],
            }

    import lobster.services.metadata.metadata_overview_service as metadata_overview_service

    monkeypatch.setattr(
        metadata_overview_service,
        "MetadataOverviewService",
        _FakeMetadataOverviewService,
    )

    summary = slash_commands._execute_command(
        "/metadata publications",
        client,
        original_command="/metadata publications",
        output=output,
    )

    assert summary == "Publication queue: 3 entries"
    assert output.messages[0] == (None, "Publication Queue (3 entries)")
    assert output.messages[1][0] == "table"
    assert output.messages[1][1]["title"] == "Status Breakdown"
    assert output.messages[2][0] == "table"
    assert output.messages[2][1]["title"] == "Identifier Coverage"
    assert output.messages[3][0] == "table"
    assert output.messages[3][1]["title"] == "Extracted Identifiers"
    assert output.messages[4] == (
        None,
        "Workspace Status: 1 entries with metadata files",
    )
    assert output.messages[5] == (
        "dim",
        "Tip: Filter by status with /metadata publications --status=<status>",
    )


def test_metadata_samples_protocol_mode_renders_structured_output(tmp_path, monkeypatch):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)

    class _FakeMetadataOverviewService:
        def __init__(self, data_manager):
            self.data_manager = data_manager

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

    import lobster.services.metadata.metadata_overview_service as metadata_overview_service

    monkeypatch.setattr(
        metadata_overview_service,
        "MetadataOverviewService",
        _FakeMetadataOverviewService,
    )

    summary = slash_commands._execute_command(
        "/metadata samples",
        client,
        original_command="/metadata samples",
        output=output,
    )

    assert summary == "Sample stats: 1,200 samples, 4 BioProjects"
    assert output.messages[0][0] == "table"
    assert output.messages[0][1]["title"] == "Sample Statistics"
    assert output.messages[1][0] == "table"
    assert output.messages[1][1]["title"] == "Aggregated Statistics"
    assert output.messages[2][0] == "table"
    assert output.messages[2][1]["title"] == "Filter Breakdown"
    assert output.messages[3] == (
        None,
        "Aggregated metadata available. Use /metadata exports to see export files.",
    )


def test_metadata_workspace_protocol_mode_renders_structured_output(tmp_path, monkeypatch):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)

    class _FakeMetadataOverviewService:
        def __init__(self, data_manager):
            self.data_manager = data_manager

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

    import lobster.services.metadata.metadata_overview_service as metadata_overview_service

    monkeypatch.setattr(
        metadata_overview_service,
        "MetadataOverviewService",
        _FakeMetadataOverviewService,
    )

    summary = slash_commands._execute_command(
        "/metadata workspace",
        client,
        original_command="/metadata workspace",
        output=output,
    )

    assert summary == "Workspace: 3 in-memory, 3 files"
    assert output.messages[0] == (None, "Workspace Inventory")
    assert output.messages[1][0] == "table"
    assert output.messages[1][1]["title"] == "In-Memory Store"
    assert output.messages[2][0] == "table"
    assert output.messages[2][1]["title"] == "Workspace Files"
    assert output.messages[3][0] == "table"
    assert output.messages[3][1]["title"] == "Export Files (2)"
    assert output.messages[4] == (
        "warning",
        "Found files in deprecated metadata/exports/ location.",
    )
    assert output.messages[5] == (
        "dim",
        "Consider migrating: mv workspace/metadata/exports/* workspace/exports/",
    )


def test_metadata_exports_protocol_mode_renders_structured_output(tmp_path, monkeypatch):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)

    class _FakeMetadataOverviewService:
        def __init__(self, data_manager):
            self.data_manager = data_manager

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

    import lobster.services.metadata.metadata_overview_service as metadata_overview_service

    monkeypatch.setattr(
        metadata_overview_service,
        "MetadataOverviewService",
        _FakeMetadataOverviewService,
    )

    summary = slash_commands._execute_command(
        "/metadata exports",
        client,
        original_command="/metadata exports",
        output=output,
    )

    assert summary == "Export files: 2 files"
    assert output.messages[0] == (None, "Export Files (2 files)")
    assert output.messages[1][0] == "table"
    assert output.messages[1][1]["title"] == "File Categories"
    assert output.messages[2][0] == "table"
    assert output.messages[2][1]["title"] == "Recent Files"
    assert output.messages[3][0] == "table"
    assert output.messages[3][1]["title"] == "Usage Tips"


def test_metadata_list_protocol_mode_renders_structured_output(tmp_path):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)

    summary = slash_commands._execute_command(
        "/metadata list",
        client,
        original_command="/metadata list",
        output=output,
    )

    assert summary == "Displayed metadata information (4 entries)"
    assert output.messages[0] == (None, "Metadata Information")
    assert output.messages[1][0] == "table"
    assert output.messages[1][1]["title"] == "Metadata Store"
    assert output.messages[2][0] == "table"
    assert output.messages[2][1]["title"] == "Current Data Metadata"
    assert output.messages[3][0] == "table"
    assert output.messages[3][1]["title"] == "Workspace Metadata Files"
    assert output.messages[4] == ("dim", f"Path: {tmp_path / 'metadata'}")
    assert output.messages[5][0] == "table"
    assert output.messages[5][1]["title"] == "Export Files"
    assert output.messages[6] == ("dim", f"Path: {tmp_path / 'exports'}")
    assert output.messages[7] == (
        "warning",
        f"Found 1 file(s) in old location: {tmp_path / 'metadata' / 'exports'}",
    )
    assert output.messages[8] == ("dim", "New exports go to: workspace/exports/")


def test_metadata_clear_protocol_mode_confirms_then_cancels(tmp_path):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)

    summary = slash_commands._execute_command(
        "/metadata clear",
        client,
        original_command="/metadata clear",
        output=output,
    )

    assert summary is None
    assert output.messages == [
        (
            "table",
            {
                "title": "About to Clear",
                "columns": [{"name": "Field"}, {"name": "Value"}],
                "rows": [
                    ["Memory (metadata_store)", "1 entries"],
                    ["Memory (current_metadata)", "3 entries"],
                    ["Disk (workspace/metadata/)", "1 files"],
                    ["Total", "5 items"],
                ],
            },
        ),
        ("confirm", "Clear all 5 metadata items?"),
        (None, "Operation cancelled"),
    ]


def test_metadata_clear_exports_protocol_mode_confirms_and_succeeds(tmp_path):
    output = ConfirmingProtocolOutputAdapter()
    client = _DummyClient(tmp_path)

    summary = slash_commands._execute_command(
        "/metadata clear exports",
        client,
        original_command="/metadata clear exports",
        output=output,
    )

    assert summary == "Cleared 1 export files"
    assert not (tmp_path / "exports" / "analysis_summary.csv").exists()
    assert output.messages[0][0] == "table"
    assert output.messages[0][1]["title"] == "About to Clear Export Files"
    assert output.messages[1] == (None, "Preview")
    assert output.messages[2] == (None, "- analysis_summary.csv (0.0 KB)")
    assert output.messages[3] == ("confirm", "Delete all 1 export files?")
    assert output.messages[4] == ("success", "Deleted 1 export files")


def test_metadata_clear_all_protocol_mode_confirms_then_cancels(tmp_path):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)

    summary = slash_commands._execute_command(
        "/metadata clear all",
        client,
        original_command="/metadata clear all",
        output=output,
    )

    assert summary is None
    assert output.messages == [
        ("warning", "This cannot be undone."),
        (
            "table",
            {
                "title": "About to Clear All Metadata",
                "columns": [{"name": "Field"}, {"name": "Value"}],
                "rows": [
                    ["Memory (metadata_store)", "1 entries"],
                    ["Memory (current_metadata)", "3 entries"],
                    ["Disk (workspace/metadata/)", "1 files"],
                    ["Disk (workspace/exports/)", "1 files"],
                    ["Total", "6 items"],
                ],
            },
        ),
        ("confirm", "Clear ALL 6 items? This cannot be undone!"),
        (None, "Operation cancelled"),
    ]


def test_metadata_unknown_subcommand_protocol_mode_uses_structured_output(tmp_path):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)

    summary = slash_commands._execute_command(
        "/metadata nope",
        client,
        original_command="/metadata nope",
        output=output,
    )

    assert summary is None
    assert output.messages == [
        ("warning", "Unknown metadata subcommand: nope"),
        (None, "Available: publications, samples, workspace, exports, list, clear"),
    ]


def test_workspace_remove_protocol_mode_renders_structured_output(tmp_path, monkeypatch):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)
    client.data_manager.list_modalities = lambda: ["rna", "atac"]

    class _FakeModalityManagementService:
        def __init__(self, data_manager):
            self.data_manager = data_manager

        def remove_modality(self, modality_name):
            assert modality_name == "rna"
            return (
                True,
                {
                    "removed_modality": "rna",
                    "shape": {"n_obs": 100, "n_vars": 200},
                },
                {"operation": "remove_modality"},
            )

    import lobster.services.data_management.modality_management_service as modality_management_service

    monkeypatch.setattr(
        modality_management_service,
        "ModalityManagementService",
        _FakeModalityManagementService,
    )

    summary = slash_commands._execute_command(
        "/workspace remove rna",
        client,
        original_command="/workspace remove rna",
        output=output,
    )

    assert summary == "Removed modality: rna"
    assert output.messages == [
        (None, "Removed: rna"),
        ("dim", "Shape: 100 obs × 200 vars"),
    ]


def test_config_provider_protocol_mode_renders_structured_output(tmp_path, monkeypatch):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)
    client.provider_override = "openai"

    class _Provider:
        def __init__(self, name, display_name):
            self.name = name
            self.display_name = display_name

    import lobster.config.llm_factory as llm_factory
    import lobster.config.providers as providers

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

    summary = slash_commands._execute_command(
        "/config provider",
        client,
        original_command="/config provider",
        output=output,
    )

    assert summary == "Listed providers (current: openai)"
    assert output.messages[0][0] == "table"
    assert output.messages[0][1]["title"] == "LLM Providers"
    assert output.messages[1] == (None, "Usage")
    assert output.messages[2][0] is None
    assert "/config provider <name>" in output.messages[2][1]
    assert output.messages[3] == ("dim", "Available providers: openai, ollama, anthropic")
    assert output.messages[4] == ("dim", "Current provider: openai")


def test_config_model_protocol_mode_renders_structured_output(tmp_path, monkeypatch):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)

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
        def __init__(self, name, display_name, description, is_default=False):
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

    summary = slash_commands._execute_command(
        "/config model",
        client,
        original_command="/config model",
        output=output,
    )

    assert summary == "Listed models for ollama provider"
    assert output.messages[0][0] == "table"
    assert output.messages[0][1]["title"] == "Active Model Selection"
    assert output.messages[1][0] == "table"
    assert output.messages[1][1]["title"] == "🦙 Available Ollama Models"
    assert output.messages[2][0] == "table"
    assert output.messages[2][1]["title"] == "Model Commands"


def test_config_show_protocol_mode_groups_agent_model_assignments(tmp_path, monkeypatch):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)

    class _FakeResolver:
        def __init__(self, workspace_path: Path | None = None):
            self.workspace_path = workspace_path

        def resolve_provider(self, runtime_override=None):
            _ = runtime_override
            return ("omics-os", "workspace config")

        def resolve_profile(self):
            return ("production", "workspace config")

        def resolve_model(self, agent_name=None, runtime_override=None, provider=None):
            _ = runtime_override, provider
            mapping = {
                "genomics_expert": (
                    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                    "provider default (omics-os)",
                ),
                "research_agent": (
                    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                    "provider default (omics-os)",
                ),
                "visualization_expert_agent": (
                    "us.anthropic.claude-opus-4-1-20260112-v1:0",
                    "workspace override",
                ),
            }
            return mapping[agent_name]

    class _FakeSettings:
        def get_agent_llm_params(self, _agent_name):
            return {}

    class _FakeProvider:
        def get_default_model(self):
            return "provider-default-model"

    fake_agent_registry = {
        "genomics_expert": SimpleNamespace(display_name="Genomics Expert"),
        "research_agent": SimpleNamespace(display_name="Research Agent"),
        "visualization_expert_agent": SimpleNamespace(
            display_name="Visualization Expert"
        ),
    }

    class _FakeWorkspaceProviderConfig:
        @staticmethod
        def exists(_workspace_path):
            return True

    class _FakeWorkspaceAgentConfig:
        @staticmethod
        def load(_workspace_path):
            return SimpleNamespace(
                preset=None,
                enabled_agents=[
                    "genomics_expert",
                    "research_agent",
                    "visualization_expert_agent",
                ],
            )

    import lobster.config.agent_registry as agent_registry
    import lobster.config.global_config as global_config
    import lobster.config.settings as settings_mod
    import lobster.config.subscription_tiers as subscription_tiers
    import lobster.config.workspace_agent_config as workspace_agent_config
    import lobster.config.workspace_config as workspace_config
    import lobster.config.providers as providers
    import lobster.core.component_registry as component_registry_mod
    import lobster.core.config_resolver as config_resolver
    import lobster.core.license_manager as license_manager

    monkeypatch.setattr(config_resolver, "ConfigResolver", _FakeResolver)
    monkeypatch.setattr(settings_mod, "get_settings", lambda: _FakeSettings())
    monkeypatch.setattr(providers, "get_provider", lambda _provider_name: _FakeProvider())
    monkeypatch.setattr(license_manager, "get_current_tier", lambda: "free")
    monkeypatch.setattr(
        subscription_tiers, "is_agent_available", lambda _agent_name, _tier: True
    )
    monkeypatch.setattr(agent_registry, "AGENT_REGISTRY", fake_agent_registry)
    monkeypatch.setattr(
        workspace_config, "WorkspaceProviderConfig", _FakeWorkspaceProviderConfig
    )
    monkeypatch.setattr(
        workspace_agent_config, "WorkspaceAgentConfig", _FakeWorkspaceAgentConfig
    )
    monkeypatch.setattr(global_config, "CONFIG_DIR", tmp_path / ".config")
    monkeypatch.setattr(
        component_registry_mod.component_registry,
        "list_agents",
        lambda: {
            agent_name: SimpleNamespace(
                display_name=cfg.display_name, tier_requirement="free"
            )
            for agent_name, cfg in fake_agent_registry.items()
        },
    )

    summary = slash_commands._execute_command(
        "/config",
        client,
        original_command="/config",
        output=output,
    )

    assert summary == "Displayed configuration (provider: omics-os, profile: production)"
    assert output.messages[0][0] == "table"
    assert output.messages[0][1]["title"] == "⚙️  Current Configuration"
    assert output.messages[1][0] == "table"
    assert output.messages[1][1]["title"] == "📁 Configuration Files"
    assert output.messages[2][0] == "table"
    assert output.messages[2][1]["title"] == "🤖 Agent Model Assignments"
    assert output.messages[2][1]["rows"] == [
        [
            "us.anthropic...20250929-v1:0",
            "provider de...t (omics-os)",
            "2",
            "Genomics Expert, Research Agent",
        ],
        [
            "us.anthropic...20260112-v1:0",
            "workspace override",
            "1",
            "Visualization Expert",
        ],
    ]
    assert output.messages[3][0] == "table"
    assert output.messages[3][1]["title"] == "Agent Configuration Summary"
    assert ["Config source", "Explicit agent list"] in output.messages[3][1]["rows"]
    assert ["Distinct model configs", "2"] in output.messages[3][1]["rows"]
    assert output.messages[4] == (
        "dim",
        "Use /status for the full agent roster and /config model for provider model catalogs.",
    )
    assert output.messages[5] == (
        "dim",
        "Use /config provider and /config model for focused follow-up commands.",
    )
    assert all(message[1] != "🔀 Agent Hierarchy" for message in output.messages)
    assert all(message[1] != "📦 Agent Composition" for message in output.messages)


def test_build_status_blocks_compact_omits_long_agent_lists(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("LOBSTER_LLM_PROVIDER=omics-os\n")

    import lobster.core.license_manager as license_manager
    import lobster.core.plugin_loader as plugin_loader
    import lobster.config.agent_registry as agent_registry
    import lobster.config.subscription_tiers as subscription_tiers

    monkeypatch.setattr(
        license_manager,
        "get_entitlement_status",
        lambda: {
            "tier": "free",
            "tier_display": "Free",
            "source": "default",
            "features": ["local_only", "community_support"],
        },
    )
    monkeypatch.setattr(
        plugin_loader,
        "get_installed_packages",
        lambda: {"lobster-ai": "1.0.0", "omics": "dev"},
    )
    monkeypatch.setattr(
        agent_registry,
        "get_worker_agents",
        lambda: {
            "research_agent": object(),
            "genomics_expert": object(),
            "visualization_expert_agent": object(),
        },
    )
    monkeypatch.setattr(
        subscription_tiers,
        "is_agent_available",
        lambda agent_name, _tier: agent_name != "visualization_expert_agent",
    )

    blocks = display_helpers.build_status_blocks(compact=True)
    titles = [block.data.get("title") for block in blocks if block.kind in {"list", "table", "kv", "section"}]

    assert "Runtime Summary" in titles
    assert "Optional Capabilities" in titles
    assert "Available Agents (2)" not in titles
    assert "Premium Agents (1)" not in titles
    assert "Enabled Features" not in titles


def test_status_panel_protocol_mode_renders_status_fallback(tmp_path):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)

    summary = slash_commands._execute_command(
        "/status-panel",
        client,
        original_command="/status-panel",
        output=output,
    )

    assert summary is None
    assert output.messages[0] == (
        "warning",
        "/status-panel renders Rich dashboard panels and is not available in Go TUI. Showing /status fallback.",
    )
    assert output.messages[1] == (None, "Lobster Status")
    assert any(
        message[0] == "table" and message[1]["title"] == "Initialization"
        for message in output.messages
    )


def test_workspace_info_protocol_mode_renders_workspace_fallback(tmp_path, monkeypatch):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)

    monkeypatch.setattr(
        slash_commands,
        "workspace_status",
        lambda _client, _output, compact=False: _output.render_blocks(
            [section_block(title="Workspace Status"), hint_block("workspace fallback")]
        ),
    )

    summary = slash_commands._execute_command(
        "/workspace-info",
        client,
        original_command="/workspace-info",
        output=output,
    )

    assert summary is None
    assert output.messages == [
        (
            "warning",
            "/workspace-info renders Rich dashboard panels and is not available in Go TUI. Showing /workspace fallback.",
        ),
        (None, "Workspace Status"),
        ("dim", "workspace fallback"),
    ]


def test_analysis_dash_protocol_mode_renders_metadata_and_plots_fallback(tmp_path, monkeypatch):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)

    monkeypatch.setattr(
        slash_commands,
        "metadata_overview",
        lambda _client, _output: _output.render_blocks(
            [section_block(title="Metadata Overview"), hint_block("metadata fallback")]
        ),
    )
    monkeypatch.setattr(
        slash_commands,
        "plots_list",
        lambda _client, _output: _output.render_blocks(
            [section_block(title="Plots"), hint_block("plots fallback")]
        ),
    )

    summary = slash_commands._execute_command(
        "/analysis-dash",
        client,
        original_command="/analysis-dash",
        output=output,
    )

    assert summary is None
    assert output.messages == [
        (
            "warning",
            "/analysis-dash renders Rich dashboard panels and is not available in Go TUI. Showing /metadata + /plots fallback.",
        ),
        (None, "Metadata Overview"),
        ("dim", "metadata fallback"),
        (None, "Plots"),
        ("dim", "plots fallback"),
    ]


def test_progress_protocol_mode_renders_compact_summary(tmp_path, monkeypatch):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)

    class _FakeProgressManager:
        def get_active_operations_count(self):
            return 3

    monkeypatch.setattr(
        slash_commands,
        "get_multi_progress_manager",
        lambda: _FakeProgressManager(),
    )

    summary = slash_commands._execute_command(
        "/progress",
        client,
        original_command="/progress",
        output=output,
    )

    assert summary is None
    assert output.messages == [
        (
            "warning",
            "/progress renders Rich live panels and is not available in Go TUI. Showing compact summary.",
        ),
        (
            "table",
            {
                "title": "Progress Summary",
                "columns": [{"name": "Field"}, {"name": "Value"}],
                "rows": [["Active Operations", "3"]],
            },
        ),
        ("dim", "Use transcript history and command-family output for detailed progress."),
    ]


def test_config_provider_accepts_direct_provider_name(tmp_path, monkeypatch):
    output = _DummyOutput()
    client = _DummyClient(tmp_path)

    def _fake_provider_switch(_client, _output, provider_name, save):
        return f"provider:{provider_name}:save={save}"

    monkeypatch.setattr(slash_commands, "config_provider_switch", _fake_provider_switch)

    summary = slash_commands._execute_command(
        "/config provider openai --save",
        client,
        original_command="/config provider openai --save",
        output=output,
    )

    assert summary == "provider:openai:save=True"


def test_config_model_accepts_direct_model_name(tmp_path, monkeypatch):
    output = _DummyOutput()
    client = _DummyClient(tmp_path)

    def _fake_model_switch(_client, _output, model_name, save):
        return f"model:{model_name}:save={save}"

    monkeypatch.setattr(slash_commands, "config_model_switch", _fake_model_switch)

    summary = slash_commands._execute_command(
        "/config model sonnet-4",
        client,
        original_command="/config model sonnet-4",
        output=output,
    )

    assert summary == "model:sonnet-4:save=False"


def test_status_protocol_mode_uses_compact_status_blocks(tmp_path):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)

    summary = slash_commands._execute_command(
        "/status",
        client,
        original_command="/status",
        output=output,
    )

    assert summary is None
    titles = [
        message[1]["title"]
        for message in output.messages
        if message[0] == "table"
    ]
    assert "Initialization" in titles
    assert "Subscription" in titles
    assert "Runtime Summary" in titles
    assert "Installed Packages" not in titles


def test_config_model_accepts_direct_model_name_with_save(tmp_path, monkeypatch):
    output = _DummyOutput()
    client = _DummyClient(tmp_path)

    def _fake_model_switch(_client, _output, model_name, save):
        return f"model:{model_name}:save={save}"

    monkeypatch.setattr(slash_commands, "config_model_switch", _fake_model_switch)

    summary = slash_commands._execute_command(
        "/config model sonnet-4 --save",
        client,
        original_command="/config model sonnet-4 --save",
        output=output,
    )

    assert summary == "model:sonnet-4:save=True"


def test_pipeline_list_protocol_mode_renders_structured_output(tmp_path):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)

    summary = slash_commands._execute_command(
        "/pipeline",
        client,
        original_command="/pipeline",
        output=output,
    )

    assert summary == "Found 1 notebooks"
    assert output.messages == [
        (
            "table",
            {
                "title": "Available Notebooks",
                "columns": [
                    {"name": "Name"},
                    {"name": "Steps"},
                    {"name": "Created By"},
                    {"name": "Created"},
                    {"name": "Size"},
                ],
                "rows": [
                    [
                        "RNA QC Workflow",
                        "7",
                        "research_agent",
                        "2026-03-01",
                        "43.2 KB",
                    ]
                ],
                "width": None,
            },
        )
    ]


def test_pipeline_export_protocol_mode_renders_structured_output(tmp_path):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)

    summary = slash_commands._execute_command(
        "/pipeline export analysis_workflow Demo notebook",
        client,
        original_command="/pipeline export analysis_workflow Demo notebook",
        output=output,
    )

    expected_path = tmp_path / "notebooks" / "analysis_workflow.ipynb"
    assert summary == f"Exported notebook: {expected_path}"
    assert output.messages[0] == (None, "Export Session as Jupyter Notebook")
    assert output.messages[1] == (None, "Exporting notebook...")
    assert output.messages[2] == (
        "success",
        f"Notebook exported: {expected_path}",
    )
    assert output.messages[3] == (None, "Next Steps")


def test_pipeline_run_protocol_mode_renders_structured_output(tmp_path):
    output = ConfirmingProtocolOutputAdapter()
    client = _DummyClient(tmp_path)

    summary = slash_commands._execute_command(
        "/pipeline run analysis_workflow.ipynb rna",
        client,
        original_command="/pipeline run analysis_workflow.ipynb rna",
        output=output,
    )

    assert summary == "Notebook executed successfully in 12.5s"
    assert output.messages == [
        (None, "Running validation..."),
        ("success", "Validation passed"),
        (
            "table",
            {
                "title": "Validation Summary",
                "columns": [{"name": "Field"}, {"name": "Value"}],
                "rows": [
                    ["Steps to execute", "4"],
                    ["Estimated time", "3 min"],
                ],
            },
        ),
        ("confirm", "Execute notebook?"),
        (None, "Executing notebook..."),
        ("success", "Execution complete"),
        (
            "table",
            {
                "title": "Execution Result",
                "columns": [{"name": "Field"}, {"name": "Value"}],
                "rows": [
                    [
                        "Output",
                        str(tmp_path / "notebooks" / "analysis_workflow_executed.ipynb"),
                    ],
                    ["Duration", "12.5s"],
                ],
            },
        ),
    ]


def test_help_protocol_mode_uses_narrow_table_metadata(tmp_path):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)

    summary = slash_commands._execute_command(
        "/help",
        client,
        original_command="/help",
        output=output,
    )

    assert summary is None

    help_tables = [
        payload
        for kind, payload in output.messages
        if kind == "table" and payload.get("title") == "Admin & UI Commands"
    ]
    assert len(help_tables) == 1

    admin_table = help_tables[0]
    assert admin_table["columns"] == [
        {"name": "Command", "width": 32, "no_wrap": True, "overflow": "ellipsis"},
        {"name": "Description", "max_width": 40, "overflow": "fold"},
    ]
    assert ["/status-panel", "Status dashboard; Go mode shows /status"] in admin_table["rows"]
    assert [
        "/analysis-dash",
        "Analysis dashboard; Go mode shows /metadata + /plots",
    ] in admin_table["rows"]


def test_pipeline_info_protocol_mode_renders_structured_output(tmp_path):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)

    summary = slash_commands._execute_command(
        "/pipeline info",
        client,
        original_command="/pipeline info",
        output=output,
    )

    assert summary == "Notebook info: RNA QC Workflow"
    assert output.messages[0] == (None, "Select Notebook")
    assert output.messages[1] == (None, "1. RNA QC Workflow")
    assert output.messages[2] == ("prompt", "Selection")
    assert output.messages[3] == (None, "RNA QC Workflow")
    assert output.messages[4][0] == "table"
    assert output.messages[4][1]["title"] == "Notebook Details"
    assert output.messages[5][0] == "table"
    assert output.messages[5][1]["title"] == "Dependencies"


def test_pipeline_unknown_subcommand_protocol_mode_uses_structured_output(tmp_path):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)

    summary = slash_commands._execute_command(
        "/pipeline nope",
        client,
        original_command="/pipeline nope",
        output=output,
    )

    assert summary is None
    assert output.messages == [
        ("warning", "Unknown pipeline subcommand: nope"),
        (None, "Available: export, list, run, info"),
    ]


def test_save_command_protocol_mode_avoids_direct_console_usage(tmp_path, monkeypatch):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)
    client.data_manager.modalities = {"rna": object(), "atac": object()}

    seen = {}

    def _fake_auto_save_state(force=False):
        seen["force"] = force
        return ["rna.h5ad", "Skipped atac.h5ad"]

    client.data_manager.auto_save_state = _fake_auto_save_state

    class _FailingConsole:
        def print(self, *_args, **_kwargs):
            raise AssertionError("direct console.print should not be used in protocol mode")

        def status(self, *_args, **_kwargs):
            raise AssertionError("direct console.status should not be used in protocol mode")

    monkeypatch.setattr(slash_commands, "console", _FailingConsole())

    summary = slash_commands._execute_command(
        "/save --force",
        client,
        original_command="/save --force",
        output=output,
    )

    assert seen["force"] is True
    assert summary == "Saved 1 items, skipped 1 unchanged"
    assert (None, "Saved: rna.h5ad") in output.messages
    assert (None, "Skipped 1 unchanged modalities") in output.messages


def test_restore_command_protocol_mode_renders_structured_output(tmp_path):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)

    summary = slash_commands._execute_command(
        "/restore",
        client,
        original_command="/restore",
        output=output,
    )

    assert summary == "Restored 1 datasets"
    assert output.messages == [
        (None, "Restored: geo_gse12345_rna"),
    ]


def test_clear_command_protocol_mode_avoids_direct_console_usage(tmp_path, monkeypatch):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)

    class _FailingConsole:
        def clear(self):
            raise AssertionError("direct console.clear should not be used in protocol mode")

    monkeypatch.setattr(slash_commands, "console", _FailingConsole())

    summary = slash_commands._execute_command(
        "/clear",
        client,
        original_command="/clear",
        output=output,
    )

    assert summary is None
    assert output.messages == []


def test_exit_command_protocol_mode_avoids_direct_console_and_confirm(tmp_path, monkeypatch):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)

    class _FailingConsole:
        def print(self, *_args, **_kwargs):
            raise AssertionError("direct console.print should not be used in protocol mode")

    class _FailingConfirm:
        @staticmethod
        def ask(*_args, **_kwargs):
            raise AssertionError("Confirm.ask should not be used in protocol mode")

    def _failing_display_goodbye(*_args, **_kwargs):
        raise AssertionError("display_goodbye should not be called in protocol mode")

    monkeypatch.setattr(slash_commands, "console", _FailingConsole())
    monkeypatch.setattr(slash_commands, "Confirm", _FailingConfirm)
    monkeypatch.setattr(slash_commands, "display_goodbye", _failing_display_goodbye)

    summary = slash_commands._execute_command(
        "/exit",
        client,
        original_command="/exit",
        output=output,
    )

    assert summary is None
    assert ("confirm", "exit?") in output.messages


def test_exit_command_protocol_mode_can_raise_without_rich_interactive_calls(tmp_path, monkeypatch):
    client = _DummyClient(tmp_path)
    output = ProtocolOutputAdapter()
    saved = {}

    def _confirm_true(question):
        output.messages.append(("confirm", question))
        return True

    def _fake_save_session_json():
        saved["called"] = True

    client._save_session_json = _fake_save_session_json
    monkeypatch.setattr(output, "confirm", _confirm_true)

    class _FailingConsole:
        def print(self, *_args, **_kwargs):
            raise AssertionError("direct console.print should not be used in protocol mode")

    class _FailingConfirm:
        @staticmethod
        def ask(*_args, **_kwargs):
            raise AssertionError("Confirm.ask should not be used in protocol mode")

    def _failing_display_goodbye(*_args, **_kwargs):
        raise AssertionError("display_goodbye should not be called in protocol mode")

    monkeypatch.setattr(slash_commands, "console", _FailingConsole())
    monkeypatch.setattr(slash_commands, "Confirm", _FailingConfirm)
    monkeypatch.setattr(slash_commands, "display_goodbye", _failing_display_goodbye)

    with pytest.raises(KeyboardInterrupt):
        slash_commands._execute_command(
            "/exit",
            client,
            original_command="/exit",
            output=output,
        )

    assert saved["called"] is True


def test_status_command_protocol_mode_resolves_provider_and_model(tmp_path, monkeypatch):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)
    captured = {}

    def _fake_status_blocks(compact=False):
        captured["compact"] = compact
        return [
            section_block(title="Lobster Status"),
            kv_block(
                [
                    ("Subscription Tier", "Free"),
                    ("Provider", "bedrock"),
                    ("Model", "anthropic.claude-sonnet-4"),
                ],
                title="Subscription",
            ),
            table_block(
                columns=[{"name": "Package"}, {"name": "Status"}],
                rows=[["lobster-ai", "Installed"]],
                title="Installed Packages",
            ),
            list_block(["research_agent"], title="Available Agents (1)"),
        ]

    monkeypatch.setattr(
        slash_commands,
        "build_status_blocks",
        _fake_status_blocks,
    )

    summary = slash_commands._execute_command(
        "/status",
        client,
        original_command="/status",
        output=output,
    )

    assert summary is None
    assert captured["compact"] is True
    assert output.messages == [
        (None, "Lobster Status"),
        (
            "table",
            {
                "title": "Subscription",
                "columns": [{"name": "Field"}, {"name": "Value"}],
                "rows": [
                    ["Subscription Tier", "Free"],
                    ["Provider", "bedrock"],
                    ["Model", "anthropic.claude-sonnet-4"],
                ],
            },
        ),
        (
            "table",
            {
                "title": "Installed Packages",
                "columns": [{"name": "Package"}, {"name": "Status"}],
                "rows": [["lobster-ai", "Installed"]],
                "width": None,
            },
        ),
        (None, "Available Agents (1)"),
        (None, "- research_agent"),
    ]
