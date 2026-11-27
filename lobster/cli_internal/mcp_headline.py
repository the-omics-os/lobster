"""The ``headline`` of a tool reply: one line a host can show its user.

Pure functions, stdlib only. Hosts show the start of a reply (pi's MCP adapter
its first line, Codex a short preview), so the reply's first key carries a short
summary of the rest; the other keys stay as they were.
"""

from __future__ import annotations

import math
from typing import Any, Dict

HEADLINE_MAX_CHARS = 120


def elapsed(seconds: Any) -> str:
    """``42s``, ``4m12s`` or ``1h03m``; empty when *seconds* is not a number."""
    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or not math.isfinite(seconds)
    ):
        return ""
    s = max(0, int(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{s % 3600 // 60:02d}m"


def _count(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _status_parts(payload: Dict[str, Any]) -> list:
    status = payload["status"]
    took = elapsed(payload.get("elapsed_s"))
    if status == "running":
        phase = payload.get("phase")
        parts = [f"⏳ {phase if phase in ('starting', 'cancelling') else 'running'}"]
        if payload.get("step"):
            parts.append(str(payload["step"]))
        if isinstance(payload.get("steps_done"), int) and payload["steps_done"]:
            parts.append(_count(payload["steps_done"], "step"))
        download = payload.get("download")
        if isinstance(download, dict) and download.get("file"):
            parts.append(f"↓ {download['file']}")
        return parts + [took]
    if status == "needs_decision":
        decision = payload.get("decision") or {}
        return ["? needs decision", str(decision.get("question", ""))]
    if status == "completed":
        files = [f for f in payload.get("files") or [] if isinstance(f, dict)]
        plots = payload.get("plots_total")
        if not isinstance(plots, int):
            plots = sum(1 for f in files if f.get("category") == "plots")
        warnings = payload.get("warnings") or []
        parts = ["✓ completed", took, _count(len(files), "file")]
        if plots:
            parts.append(_count(plots, "plot"))
        if warnings:
            parts.append(_count(len(warnings), "warning"))
        return parts
    if status == "failed":
        error = payload.get("error") or {}
        return [
            "✗ failed",
            str(error.get("code") or payload.get("summary") or ""),
            took,
        ]
    if status == "cancelled":
        return ["■ cancelled", str(payload.get("cancel_reason") or ""), took]
    return [str(status), took]


def headline(tool: str, payload: Dict[str, Any]) -> str:
    """One line of at most ``HEADLINE_MAX_CHARS`` characters, without braces."""
    if "status" in payload:
        parts = _status_parts(payload)
    elif payload.get("outcome") == "answered":
        parts = [f"↳ answered ({payload.get('source')})"]
    elif "outcome" in payload:
        parts = [f"■ cancel · {payload['outcome']}"]
    else:
        parts = [f"ℹ {tool.removeprefix('lobster_')}"]
    text = " · ".join(p for p in parts if p)
    text = " ".join(text.replace("{", "(").replace("}", ")").split())
    if len(text) > HEADLINE_MAX_CHARS:
        text = text[: HEADLINE_MAX_CHARS - 1] + "…"
    return text
