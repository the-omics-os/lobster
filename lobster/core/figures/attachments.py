"""Content-addressed immutable data snapshots. No client filesystem paths accepted."""

from __future__ import annotations

import hashlib
import mimetypes
import re
from .errors import FigureError, IntegrityError

MAX_ATTACHMENT_BYTES = 10_000_000


def save_attachment(
    store, payload, *, label, role, description="", columns=None, origin=None
):
    if role not in ("plotted_data", "related_data", "source"):
        raise ValueError("Invalid attachment role")
    if not isinstance(payload, bytes) or len(payload) > MAX_ATTACHMENT_BYTES:
        raise ValueError("Attachment exceeds the 10 MB limit")
    if not isinstance(label, str) or not label or len(label) > 255:
        raise ValueError("Attachment needs a label of at most 255 characters")
    digest = hashlib.sha256(payload).hexdigest()
    with store.writer_lock:
        store._write(store.root / "_attachments" / (digest + ".blob"), payload)
    return {
        "id": digest,
        "sha256": digest,
        "bytes": len(payload),
        "label": label,
        "role": role,
        "description": str(description)[:2000],
        "columns": columns or [],
        "schema": {
            "columns": columns or [],
            "format": label.rsplit(".", 1)[-1].lower() if "." in label else "unknown",
        },
        "origin": origin,
        "mime": mimetypes.guess_type(label)[0] or "application/octet-stream",
    }


def read_attachment(store, descriptor):
    digest = descriptor["id"]
    if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
        raise ValueError("Invalid attachment identifier")
    payload = store._read(store.root / "_attachments" / (digest + ".blob"))
    if payload is None:
        raise ValueError("Attachment is unavailable")
    if hashlib.sha256(payload).hexdigest() != digest:
        raise IntegrityError("Attachment content changed")
    return payload


def manifest(store, record):
    source = record.get("source") or {}
    descriptors = list((source.get("declared") or {}).get("attachments") or [])
    result = []
    for item in descriptors:
        try:
            read_attachment(store, item)
            available = True
        except (OSError, ValueError, FigureError):
            available = False
        result.append({**item, "available": available})
    # Older figures still expose their retained source and input/provenance edges.
    if source:
        result.append(
            {
                "id": source["source_id"],
                "label": "Retained plot source",
                "role": "source",
                "available": True,
            }
        )
    for edge in record.get("inputs", []):
        result.append(
            {
                "label": edge.get("name") or "Input dataset",
                "role": "related_data",
                "instance_id": edge.get("instance_id"),
                "verified": edge.get("verified") is True,
                "available": False,
                "description": "Dataset provenance reference; no immutable snapshot attached.",
            }
        )
    execution = record.get("execution") or {}
    if execution.get("execution_id"):
        result.append(
            {
                "id": execution["execution_id"],
                "label": "Generating execution",
                "role": "source",
                "available": True,
            }
        )
    return result
