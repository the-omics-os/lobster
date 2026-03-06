from lobster.cli_internal.commands.output_adapter import ProtocolOutputAdapter


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
