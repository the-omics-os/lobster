from rich.console import Console

from lobster.cli_internal.commands.output_adapter import (
    ConsoleOutputAdapter,
    JsonOutputAdapter,
    ProtocolOutputAdapter,
    alert_block,
    code_block,
    hint_block,
    kv_block,
    list_block,
    section_block,
    table_block,
)


def test_protocol_output_adapter_print_table_strips_rich_markup():
    events = []
    adapter = ProtocolOutputAdapter(
        lambda msg_type, payload: events.append({"type": msg_type, "payload": payload})
    )

    adapter.print_table(
        {
            "title": "[bold cyan]Agent Status[/bold cyan]",
            "columns": [
                {"name": "[cyan]Agent[/cyan]"},
                {"name": "[green]Status[/green]"},
                {"name": "[yellow]Tier[/yellow]"},
            ],
            "rows": [
                [
                    "[cyan]Research Agent[/cyan]",
                    "[green]Installed[/green]",
                    "[yellow]Free[/yellow]",
                ],
                [
                    "[cyan]Data Expert[/cyan]",
                    "[green]Installed[/green]",
                    "[yellow]Free[/yellow]",
                ],
            ],
        }
    )

    assert events == [
        {"type": "text", "payload": {"content": "Agent Status\n"}},
        {
            "type": "table",
            "payload": {
                "headers": ["Agent", "Status", "Tier"],
                "columns": [
                    {"name": "Agent"},
                    {"name": "Status"},
                    {"name": "Tier"},
                ],
                "rows": [
                    ["Research Agent", "Installed", "Free"],
                    ["Data Expert", "Installed", "Free"],
                ],
            },
        },
    ]


def test_protocol_output_adapter_preserves_literal_bracket_syntax():
    events = []
    adapter = ProtocolOutputAdapter(
        lambda msg_type, payload: events.append({"type": msg_type, "payload": payload})
    )

    adapter.render_blocks(
        [
            table_block(
                columns=[{"name": "Command"}, {"name": "Description"}],
                rows=[
                    ["/queue clear [download|all]", "Clear queue(s)"],
                    ["/save [--force]", "Persist loaded modalities"],
                    ["/restore [pattern]", "Restore datasets from session cache"],
                ],
                title="Power Commands",
            )
        ]
    )

    assert events == [
        {"type": "text", "payload": {"content": "Power Commands\n"}},
        {
            "type": "table",
            "payload": {
                "headers": ["Command", "Description"],
                "columns": [
                    {"name": "Command"},
                    {"name": "Description"},
                ],
                "rows": [
                    ["/queue clear [download|all]", "Clear queue(s)"],
                    ["/save [--force]", "Persist loaded modalities"],
                    ["/restore [pattern]", "Restore datasets from session cache"],
                ],
            },
        },
    ]


def test_protocol_output_adapter_includes_table_layout_metadata():
    events = []
    adapter = ProtocolOutputAdapter(
        lambda msg_type, payload: events.append({"type": msg_type, "payload": payload})
    )

    adapter.render_blocks(
        [
            table_block(
                columns=[
                    {"name": "#", "width": 4, "justify": "right", "no_wrap": True},
                    {"name": "Name", "max_width": 18, "overflow": "ellipsis"},
                ],
                rows=[["1", "geo_gse247686_transcriptomics_single_cell_autosave"]],
                title="Datasets",
            )
        ]
    )

    assert events[1] == {
        "type": "table",
        "payload": {
            "headers": ["#", "Name"],
            "columns": [
                {"name": "#", "width": 4, "justify": "right", "no_wrap": True},
                {"name": "Name", "max_width": 18, "overflow": "ellipsis"},
            ],
            "rows": [["1", "geo_gse247686_transcriptomics_single_cell_autosave"]],
        },
    }


def test_protocol_output_adapter_print_table_coerces_non_string_cells():
    events = []
    adapter = ProtocolOutputAdapter(
        lambda msg_type, payload: events.append({"type": msg_type, "payload": payload})
    )

    adapter.print_table(
        {
            "columns": [{"name": "Key"}, {"name": "Value"}],
            "rows": [["count", 12], ["active", True], ["ratio", 0.42]],
        }
    )

    assert events == [
        {
            "type": "table",
            "payload": {
                "headers": ["Key", "Value"],
                "columns": [{"name": "Key"}, {"name": "Value"}],
                "rows": [["count", "12"], ["active", "True"], ["ratio", "0.42"]],
            },
        },
    ]


def test_protocol_output_adapter_render_blocks_maps_structured_blocks():
    events = []
    adapter = ProtocolOutputAdapter(
        lambda msg_type, payload: events.append({"type": msg_type, "payload": payload})
    )

    adapter.render_blocks(
        [
            section_block(
                title="[bold cyan]Summary[/bold cyan]", body="[dim]Ready[/dim]"
            ),
            kv_block(
                [("Tier", "free"), ("Calls", 2)],
                title="[green]Metrics[/green]",
            ),
            list_block(
                ["[cyan]research_agent[/cyan]", "data_expert"],
                title="Agents",
            ),
            code_block("print('hi')", language="python", title="Snippet"),
            alert_block("[red]Careful[/red]", level="warning"),
            hint_block("Use [cyan]/help[/cyan]", title="Tip"),
        ]
    )

    assert events == [
        {"type": "text", "payload": {"content": "Summary\n"}},
        {"type": "text", "payload": {"content": "Ready\n"}},
        {"type": "text", "payload": {"content": "Metrics\n"}},
        {
            "type": "table",
            "payload": {
                "headers": ["Field", "Value"],
                "columns": [{"name": "Field"}, {"name": "Value"}],
                "rows": [["Tier", "free"], ["Calls", "2"]],
            },
        },
        {"type": "text", "payload": {"content": "Agents\n"}},
        {
            "type": "text",
            "payload": {"content": "- research_agent\n- data_expert\n"},
        },
        {"type": "text", "payload": {"content": "Snippet\n"}},
        {
            "type": "code",
            "payload": {"content": "print('hi')", "language": "python"},
        },
        {
            "type": "alert",
            "payload": {"level": "warning", "message": "Careful"},
        },
        {"type": "text", "payload": {"content": "Tip\nUse /help\n"}},
    ]


def test_protocol_output_adapter_uses_confirm_callback_when_available():
    events = []
    questions = []
    adapter = ProtocolOutputAdapter(
        lambda msg_type, payload: events.append({"type": msg_type, "payload": payload}),
        confirm_fn=lambda question: questions.append(question) or True,
    )

    assert adapter.confirm("Delete all files?") is True
    assert questions == ["Delete all files?"]
    assert events == []


def test_protocol_output_adapter_uses_prompt_callback_when_available():
    adapter = ProtocolOutputAdapter(
        lambda *_args, **_kwargs: None,
        prompt_fn=lambda question, default: f"{question}::{default}",
    )

    assert adapter.prompt("Name", default="demo") == "Name::demo"


def test_json_output_adapter_to_dict_preserves_blocks_and_legacy_views():
    adapter = JsonOutputAdapter()

    adapter.render_blocks(
        [
            section_block(title="[bold]Overview[/bold]", body="[cyan]Ready[/cyan]"),
            table_block(
                columns=[{"name": "[green]Agent[/green]"}, {"name": "Status"}],
                rows=[["[cyan]research_agent[/cyan]", "[green]ok[/green]"]],
                title="[bold]Workers[/bold]",
            ),
            kv_block([("Tier", "free")], title="Subscription"),
            code_block("print('hi')", language="python"),
            alert_block("[red]Watch out[/red]", level="warning"),
            hint_block("Use [cyan]/help[/cyan]"),
        ]
    )

    assert adapter.to_dict() == {
        "blocks": [
            {"kind": "section", "title": "Overview", "body": "Ready"},
            {
                "kind": "table",
                "title": "Workers",
                "columns": [{"name": "Agent"}, {"name": "Status"}],
                "rows": [["research_agent", "ok"]],
                "width": None,
            },
            {
                "kind": "kv",
                "title": "Subscription",
                "rows": [["Tier", "free"]],
                "key_label": "Field",
                "value_label": "Value",
            },
            {
                "kind": "code",
                "code": "print('hi')",
                "language": "python",
                "title": None,
            },
            {
                "kind": "alert",
                "message": "Watch out",
                "level": "warning",
                "title": None,
            },
            {"kind": "hint", "message": "Use /help", "title": None},
        ],
        "messages": [
            {"text": "Ready"},
            {"text": "Watch out", "style": "warning"},
            {"text": "Use /help", "style": "dim"},
        ],
        "tables": [
            {
                "title": "Workers",
                "columns": ["Agent", "Status"],
                "rows": [["research_agent", "ok"]],
            },
            {
                "title": "Subscription",
                "columns": ["Field", "Value"],
                "rows": [["Tier", "free"]],
            },
        ],
        "code_blocks": [
            {"code": "print('hi')", "language": "python"},
        ],
    }


def test_console_output_adapter_preserves_literal_brackets_in_table_cells():
    console = Console(record=True, width=120)
    adapter = ConsoleOutputAdapter(console)

    adapter.render_blocks(
        [
            table_block(
                columns=[{"name": "Command"}, {"name": "Description"}],
                rows=[
                    ["/queue clear [download|all]", "Clear queue(s)"],
                    ["/metadata clear [exports|all]", "Clear metadata"],
                ],
                title="Help",
            )
        ]
    )

    out = console.export_text()

    assert "/queue clear [download|all]" in out
    assert "/metadata clear [exports|all]" in out
