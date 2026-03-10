"""
Output adapters for command results.

Commands can emit a small structured block contract and let adapters render it
for Rich console, dashboard, JSON, or the Go TUI protocol.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import re
from typing import Any, Callable, Dict, Iterable, Optional, Sequence

from rich import box
from rich.console import Console
from rich.markup import escape
from rich.table import Table


_RICH_TAG_RE = re.compile(r"\[([^\[\]]+)\]")
_ALERT_STYLES = {"error", "warning", "success", "info"}


def _strip_markup(text: str) -> str:
    matches = list(_RICH_TAG_RE.finditer(text))
    if not matches:
        return text

    removable: set[int] = set()
    open_stack: list[int] = []

    for idx, match in enumerate(matches):
        raw_tag = match.group(1).strip()
        if not raw_tag:
            continue

        if raw_tag == "/":
            if open_stack:
                removable.add(open_stack.pop())
                removable.add(idx)
            continue

        if raw_tag.startswith("/"):
            closing_tag = raw_tag[1:].strip()
            if not closing_tag:
                if open_stack:
                    removable.add(open_stack.pop())
                    removable.add(idx)
                continue

            for stack_idx in range(len(open_stack) - 1, -1, -1):
                open_idx = open_stack[stack_idx]
                if matches[open_idx].group(1).strip() == closing_tag:
                    removable.add(open_idx)
                    removable.add(idx)
                    del open_stack[stack_idx]
                    break
            continue

        open_stack.append(idx)

    if not removable:
        return text

    parts: list[str] = []
    cursor = 0
    for idx, match in enumerate(matches):
        if idx not in removable:
            continue
        start, end = match.span()
        parts.append(text[cursor:start])
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


@dataclass(frozen=True)
class OutputBlock:
    kind: str
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, **self.data}


def section_block(
    *,
    title: Optional[str] = None,
    body: Optional[str] = None,
    style: Optional[str] = None,
) -> OutputBlock:
    data: Dict[str, Any] = {}
    if title is not None:
        data["title"] = title
    if body is not None:
        data["body"] = body
    if style is not None:
        data["style"] = style
    return OutputBlock(kind="section", data=data)


def kv_block(
    rows: Iterable[tuple[str, Any]],
    *,
    title: Optional[str] = None,
    key_label: str = "Field",
    value_label: str = "Value",
) -> OutputBlock:
    return OutputBlock(
        kind="kv",
        data={
            "title": title,
            "rows": [[str(key), str(value)] for key, value in rows],
            "key_label": key_label,
            "value_label": value_label,
        },
    )


def table_block(
    columns: Sequence[Dict[str, Any]],
    rows: Sequence[Sequence[Any]],
    *,
    title: Optional[str] = None,
    width: Optional[int] = None,
) -> OutputBlock:
    return OutputBlock(
        kind="table",
        data={
            "title": title,
            "columns": [dict(col) for col in columns],
            "rows": [[str(cell) for cell in row] for row in rows],
            "width": width,
        },
    )


def list_block(
    items: Sequence[str],
    *,
    title: Optional[str] = None,
    ordered: bool = False,
) -> OutputBlock:
    return OutputBlock(
        kind="list",
        data={
            "title": title,
            "items": list(items),
            "ordered": ordered,
        },
    )


def code_block(
    code: str,
    *,
    language: str = "python",
    title: Optional[str] = None,
) -> OutputBlock:
    return OutputBlock(
        kind="code",
        data={
            "title": title,
            "code": code,
            "language": language,
        },
    )


def alert_block(
    message: str,
    *,
    level: str = "info",
    title: Optional[str] = None,
) -> OutputBlock:
    return OutputBlock(
        kind="alert",
        data={
            "title": title,
            "message": message,
            "level": level,
        },
    )


def hint_block(message: str, *, title: Optional[str] = None) -> OutputBlock:
    return OutputBlock(
        kind="hint",
        data={
            "title": title,
            "message": message,
        },
    )


def _legacy_table_to_block(table_data: Dict[str, Any]) -> OutputBlock:
    return table_block(
        columns=table_data.get("columns", []),
        rows=table_data.get("rows", []),
        title=table_data.get("title"),
        width=table_data.get("width"),
    )


class OutputAdapter(ABC):
    """Abstract interface for command output."""

    @abstractmethod
    def render_blocks(self, blocks: Sequence[OutputBlock]) -> None:
        """Render structured output blocks."""

    @abstractmethod
    def confirm(self, question: str) -> bool:
        """Ask user for confirmation."""

    @abstractmethod
    def prompt(self, question: str, default: str = "") -> str:
        """Prompt user for text input."""

    def print(self, message: str, style: Optional[str] = None) -> None:
        if style in _ALERT_STYLES:
            self.render_blocks([alert_block(message, level=style)])
            return
        if style == "dim":
            self.render_blocks([hint_block(message)])
            return
        self.render_blocks([section_block(body=message, style=style)])

    def print_table(self, table_data: Dict[str, Any]) -> None:
        self.render_blocks([_legacy_table_to_block(table_data)])

    def print_code_block(self, code: str, language: str = "python") -> None:
        self.render_blocks([code_block(code, language=language)])


class ConsoleOutputAdapter(OutputAdapter):
    """OutputAdapter for Rich Console (CLI mode)."""

    _SEMANTIC_STYLES: Dict[str, str] = {
        "warning": "bold yellow",
        "info": "bold cyan",
        "success": "bold green",
        "error": "bold red",
        "dim": "dim",
    }

    def __init__(self, console: Console):
        self.console = console

    def render_blocks(self, blocks: Sequence[OutputBlock]) -> None:
        for block in blocks:
            self._render_block(block)

    def _render_block(self, block: OutputBlock) -> None:
        data = block.data
        if block.kind == "section":
            title = data.get("title")
            body = data.get("body")
            style = self._resolve_style(data.get("style"))
            if title:
                self.console.print(title, style=style)
            if body:
                self.console.print(body, style=style)
            return

        if block.kind == "kv":
            rows = data.get("rows", [])
            self._print_table_data(
                {
                    "title": data.get("title"),
                    "columns": [
                        {"name": data.get("key_label", "Field")},
                        {"name": data.get("value_label", "Value")},
                    ],
                    "rows": rows,
                }
            )
            return

        if block.kind == "table":
            self._print_table_data(
                {
                    "title": data.get("title"),
                    "columns": data.get("columns", []),
                    "rows": data.get("rows", []),
                    "width": data.get("width"),
                }
            )
            return

        if block.kind == "list":
            title = data.get("title")
            if title:
                self.console.print(title)
            ordered = bool(data.get("ordered"))
            for index, item in enumerate(data.get("items", []), start=1):
                prefix = f"{index}." if ordered else "-"
                self.console.print(f"{prefix} {item}")
            return

        if block.kind == "code":
            title = data.get("title")
            if title:
                self.console.print(title)
            self._print_code(data.get("code", ""), data.get("language", "python"))
            return

        if block.kind == "alert":
            title = data.get("title")
            message = data.get("message", "")
            text = f"{title}\n{message}" if title else message
            self.console.print(text, style=self._resolve_style(data.get("level")))
            return

        if block.kind == "hint":
            title = data.get("title")
            message = data.get("message", "")
            text = f"{title}\n{message}" if title else message
            self.console.print(text, style=self._resolve_style("dim"))
            return

        raise ValueError(f"Unsupported output block kind: {block.kind}")

    def _print_table_data(self, table_data: Dict[str, Any]) -> None:
        table = Table(
            box=box.ROUNDED,
            title=table_data.get("title"),
            width=table_data.get("width"),
        )

        for col in table_data.get("columns", []):
            table.add_column(
                str(col.get("name", "")),
                style=col.get("style", "white"),
                width=col.get("width"),
                max_width=col.get("max_width"),
                overflow=col.get("overflow", "fold"),
                justify=col.get("justify", "left"),
            )

        for row in table_data.get("rows", []):
            table.add_row(*[escape(str(cell)) for cell in row])

        self.console.print(table)

    def prompt(self, question: str, default: str = "") -> str:
        from rich.prompt import Prompt

        return Prompt.ask(question, default=default, console=self.console)

    def confirm(self, question: str) -> bool:
        from rich.prompt import Confirm

        return Confirm.ask(question)

    def _print_code(self, code: str, language: str = "python") -> None:
        from rich.syntax import Syntax

        syntax = Syntax(code, language, theme="monokai")
        self.console.print(syntax)

    def _resolve_style(self, style: Optional[str]) -> Optional[str]:
        if not style:
            return None
        return self._SEMANTIC_STYLES.get(style, style)


class JsonOutputAdapter(OutputAdapter):
    """OutputAdapter that collects structured data for JSON serialization."""

    def __init__(self):
        self.blocks: list[dict] = []

    def render_blocks(self, blocks: Sequence[OutputBlock]) -> None:
        for block in blocks:
            self.blocks.append(self._sanitize_block(block))

    def prompt(self, question: str, default: str = "") -> str:
        self.render_blocks(
            [
                alert_block(
                    f"Prompt skipped (non-interactive): {_strip_markup(question)} -> {default!r}",
                    level="warning",
                )
            ]
        )
        return default

    def confirm(self, question: str) -> bool:
        self.render_blocks(
            [
                alert_block(
                    f"Confirmation skipped (non-interactive): {_strip_markup(question)}",
                    level="warning",
                )
            ]
        )
        return False

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        if self.blocks:
            result["blocks"] = self.blocks

        messages = []
        tables = []
        code_blocks = []
        for block in self.blocks:
            kind = block["kind"]
            if kind in {"section", "alert", "hint"}:
                body = block.get("body")
                if kind == "alert":
                    body = block.get("message")
                if kind == "hint":
                    body = block.get("message")
                if body:
                    entry = {"text": body}
                    if kind == "alert":
                        entry["style"] = block.get("level")
                    elif kind == "hint":
                        entry["style"] = "dim"
                    elif block.get("style"):
                        entry["style"] = block.get("style")
                    messages.append(entry)
            elif kind == "table":
                tables.append(
                    {
                        "title": block.get("title"),
                        "columns": [
                            _strip_markup(str(col.get("name", "")))
                            for col in block.get("columns", [])
                        ],
                        "rows": [
                            [_strip_markup(str(cell)) for cell in row]
                            for row in block.get("rows", [])
                        ],
                    }
                )
            elif kind == "kv":
                tables.append(
                    {
                        "title": block.get("title"),
                        "columns": [
                            block.get("key_label", "Field"),
                            block.get("value_label", "Value"),
                        ],
                        "rows": [
                            [_strip_markup(str(cell)) for cell in row]
                            for row in block.get("rows", [])
                        ],
                    }
                )
            elif kind == "code":
                code_blocks.append(
                    {
                        "code": block.get("code", ""),
                        "language": block.get("language", "python"),
                    }
                )

        if messages:
            result["messages"] = messages
        if tables:
            result["tables"] = tables
        if code_blocks:
            result["code_blocks"] = code_blocks
        return result

    def _sanitize_block(self, block: OutputBlock) -> Dict[str, Any]:
        data = block.to_dict()
        sanitized: Dict[str, Any] = {"kind": data["kind"]}
        for key, value in data.items():
            if key == "kind":
                continue
            sanitized[key] = self._sanitize_value(value)
        return sanitized

    def _sanitize_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return _strip_markup(value)
        if isinstance(value, list):
            return [self._sanitize_value(item) for item in value]
        if isinstance(value, tuple):
            return [self._sanitize_value(item) for item in value]
        if isinstance(value, dict):
            return {str(key): self._sanitize_value(item) for key, item in value.items()}
        return value


class DashboardOutputAdapter(OutputAdapter):
    """OutputAdapter for Textual ResultsDisplay (Dashboard mode)."""

    def __init__(self, results_display):
        self.results_display = results_display

    def render_blocks(self, blocks: Sequence[OutputBlock]) -> None:
        for block in blocks:
            self._render_block(block)

    def _render_block(self, block: OutputBlock) -> None:
        data = block.data
        if block.kind == "section":
            parts = []
            if data.get("title"):
                parts.append(f"**{_strip_markup(data['title'])}**")
            if data.get("body"):
                parts.append(_strip_markup(data["body"]))
            if parts:
                self.results_display.append_system_message("\n\n".join(parts))
            return

        if block.kind == "kv":
            self._render_block(
                table_block(
                    columns=[
                        {"name": data.get("key_label", "Field")},
                        {"name": data.get("value_label", "Value")},
                    ],
                    rows=data.get("rows", []),
                    title=data.get("title"),
                )
            )
            return

        if block.kind == "table":
            columns = block.data.get("columns", [])
            rows = block.data.get("rows", [])
            if not columns or not rows:
                return

            header = "| " + " | ".join(
                _strip_markup(str(col.get("name", ""))) for col in columns
            ) + " |"
            separator = "|" + "|".join("---" for _ in columns) + "|"
            table_rows = [
                "| "
                + " | ".join(_strip_markup(str(cell)) for cell in row)
                + " |"
                for row in rows
            ]
            title = data.get("title")
            markdown = f"**{_strip_markup(title)}**\n\n" if title else ""
            markdown += header + "\n" + separator + "\n" + "\n".join(table_rows)
            self.results_display.append_system_message(markdown)
            return

        if block.kind == "list":
            items = data.get("items", [])
            if not items:
                return
            lines = []
            if data.get("title"):
                lines.append(f"**{_strip_markup(data['title'])}**")
                lines.append("")
            ordered = bool(data.get("ordered"))
            for index, item in enumerate(items, start=1):
                prefix = f"{index}." if ordered else "-"
                lines.append(f"{prefix} {_strip_markup(item)}")
            self.results_display.append_system_message("\n".join(lines))
            return

        if block.kind == "code":
            lines = []
            if data.get("title"):
                lines.append(f"**{_strip_markup(data['title'])}**")
                lines.append("")
            lines.append(
                f"```{data.get('language', 'python')}\n{data.get('code', '')}\n```"
            )
            self.results_display.append_system_message("\n".join(lines))
            return

        if block.kind == "alert":
            title = data.get("title")
            message = _strip_markup(data.get("message", ""))
            parts = []
            if title:
                parts.append(f"**{_strip_markup(title)}**")
            parts.append(message)
            self.results_display.append_system_message("\n".join(parts))
            return

        if block.kind == "hint":
            title = data.get("title")
            message = _strip_markup(data.get("message", ""))
            text = f"**{_strip_markup(title)}**\n{message}" if title else message
            self.results_display.append_system_message(text)
            return

        raise ValueError(f"Unsupported output block kind: {block.kind}")

    def prompt(self, question: str, default: str = "") -> str:
        clean_question = _strip_markup(question)
        self.results_display.append_system_message(
            f"ℹ️ Using default for '{clean_question}': {default!r}"
        )
        return default

    def confirm(self, question: str) -> bool:
        clean_question = _strip_markup(question)
        self.results_display.append_system_message(
            f"⚠️ Confirmation required: {clean_question}\n"
            "Interactive confirmations not supported in dashboard. "
            "Use CLI mode for destructive operations."
        )
        return False


class ProtocolOutputAdapter(OutputAdapter):
    """OutputAdapter that emits protocol messages to the Go TUI."""

    def __init__(
        self,
        send_fn,
        *,
        confirm_fn: Optional[Callable[[str], bool]] = None,
        prompt_fn: Optional[Callable[[str, str], str]] = None,
    ):
        self._send = send_fn
        self._confirm_fn = confirm_fn
        self._prompt_fn = prompt_fn

    def render_blocks(self, blocks: Sequence[OutputBlock]) -> None:
        for block in blocks:
            self._render_block(block)

    @staticmethod
    def _table_payload_columns(columns: Sequence[Dict[str, Any]]) -> list[Dict[str, Any]]:
        payload_columns: list[Dict[str, Any]] = []
        for col in columns:
            payload_col: Dict[str, Any] = {
                "name": _strip_markup(str(col.get("name", ""))),
            }
            for key in ("width", "max_width", "justify", "overflow"):
                value = col.get(key)
                if value is not None:
                    payload_col[key] = value
            if "no_wrap" in col:
                payload_col["no_wrap"] = bool(col.get("no_wrap"))
            payload_columns.append(payload_col)
        return payload_columns

    def _render_block(self, block: OutputBlock) -> None:
        data = block.data
        if block.kind == "section":
            title = data.get("title")
            body = data.get("body")
            if title:
                self._send("text", {"content": _strip_markup(title) + "\n"})
            if body:
                self._send("text", {"content": _strip_markup(body) + "\n"})
            return

        if block.kind == "kv":
            title = data.get("title")
            if title:
                self._send("text", {"content": _strip_markup(title) + "\n"})
            columns = [
                {"name": _strip_markup(data.get("key_label", "Field"))},
                {"name": _strip_markup(data.get("value_label", "Value"))},
            ]
            self._send(
                "table",
                {
                    "headers": [column["name"] for column in columns],
                    "columns": columns,
                    "rows": [
                        [_strip_markup(str(cell)) for cell in row]
                        for row in data.get("rows", [])
                    ],
                },
            )
            return

        if block.kind == "table":
            payload_columns = self._table_payload_columns(data.get("columns", []))
            headers = [col["name"] for col in payload_columns]
            rows = [
                [_strip_markup(str(cell)) for cell in row]
                for row in data.get("rows", [])
            ]
            title = data.get("title")
            if title:
                self._send("text", {"content": _strip_markup(title) + "\n"})
            payload = {"headers": headers, "rows": rows}
            if payload_columns:
                payload["columns"] = payload_columns
            self._send("table", payload)
            return

        if block.kind == "list":
            title = data.get("title")
            if title:
                self._send("text", {"content": _strip_markup(title) + "\n"})
            ordered = bool(data.get("ordered"))
            lines = []
            for index, item in enumerate(data.get("items", []), start=1):
                prefix = f"{index}." if ordered else "-"
                lines.append(f"{prefix} {_strip_markup(item)}")
            if lines:
                self._send("text", {"content": "\n".join(lines) + "\n"})
            return

        if block.kind == "code":
            title = data.get("title")
            if title:
                self._send("text", {"content": _strip_markup(title) + "\n"})
            self._send(
                "code",
                {
                    "content": data.get("code", ""),
                    "language": data.get("language", "python"),
                },
            )
            return

        if block.kind == "alert":
            self._send(
                "alert",
                {
                    "level": data.get("level", "info"),
                    "message": _strip_markup(data.get("message", "")),
                },
            )
            return

        if block.kind == "hint":
            title = data.get("title")
            message = data.get("message", "")
            text = f"{_strip_markup(title)}\n{_strip_markup(message)}" if title else _strip_markup(message)
            self._send("text", {"content": text + "\n"})
            return

        raise ValueError(f"Unsupported output block kind: {block.kind}")

    def confirm(self, question: str) -> bool:
        if self._confirm_fn is not None:
            return bool(self._confirm_fn(question))
        return False

    def prompt(self, question: str, default: str = "") -> str:
        if self._prompt_fn is not None:
            return str(self._prompt_fn(question, default))
        return default
