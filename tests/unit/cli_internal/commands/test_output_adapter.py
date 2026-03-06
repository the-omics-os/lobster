from lobster.cli_internal.commands.output_adapter import (
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
                ["[cyan]Research Agent[/cyan]", "[green]Installed[/green]", "[yellow]Free[/yellow]"],
                ["[cyan]Data Expert[/cyan]", "[green]Installed[/green]", "[yellow]Free[/yellow]"],
            ],
        }
    )

    assert events == [
        {"type": "text", "payload": {"content": "Agent Status\n"}},
        {
            "type": "table",
            "payload": {
                "headers": ["Agent", "Status", "Tier"],
                "rows": [
                    ["Research Agent", "Installed", "Free"],
                    ["Data Expert", "Installed", "Free"],
                ],
            },
        },
    ]


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
            section_block(title="[bold cyan]Summary[/bold cyan]", body="[dim]Ready[/dim]"),
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
