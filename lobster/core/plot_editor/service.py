"""Presentation-independent editor adapter over the production FigureStore."""

from __future__ import annotations

import hashlib
import base64
import secrets
import threading
import copy
import re
from typing import TYPE_CHECKING

from lobster.core.figures.ids import FigureRef
from lobster.core.figures.payload import (
    strict_json_loads,
    validate_config_bytes,
    validate_figure_bytes,
)
from lobster.core.figures.records import (
    Candidate,
    ExecutionInfo,
    SourcePayload,
    json_blob_bytes,
)
from lobster.core.figures.saver import TrustedSaver

if TYPE_CHECKING:
    from lobster.core.figures.store import FigureStore


class EditorService:
    """One figure and its revisions. Drafts are plain JSON, independent of the UI.

    Drafts name the source revision and observed history generation. Optional
    insertion and upload fields are interpreted here; clients cannot submit
    arbitrary timeline links, configuration or provenance records.
    """

    def __init__(self, store: FigureStore, ref: str):
        self.store = store
        self.initial_ref = str(FigureRef.parse(ref))
        self.figure_id = FigureRef.parse(ref).figure_id
        self._namespace = secrets.token_hex(16)
        self._lock = threading.Lock()
        self._requests = {}
        self.load(ref)  # refuse unavailable or corrupt figures before opening a server

    def reference(self, ref):
        parsed = FigureRef.parse(ref)
        if parsed.figure_id != self.figure_id:
            raise ValueError("This revision is outside the editor's figure")
        return str(parsed)

    def load(self, ref):
        ref = self.reference(ref)
        record, figure, config = self.store.read(ref)
        return {
            "ref": ref,
            "figure": validate_figure_bytes(figure),
            "config": validate_config_bytes(config),
            "record": strict_json_loads(record, what="revision"),
            "attachments": self.attachments(ref),
        }

    def history(self):
        return self.store.history(self.figure_id)

    def data_page(self, ref, *, offset=0, limit=100):
        """Read a bounded page from the revision's verified retained drawing source."""
        from lobster.core.figures.data_view import column_mapping

        if (
            type(offset) is not int
            or offset < 0
            or type(limit) is not int
            or not 1 <= limit <= 200
        ):
            raise ValueError("Expected nonnegative offset and page size from 1 to 200")
        record_bytes, figure_bytes, _ = self.store.read(self.reference(ref))
        record = strict_json_loads(record_bytes, what="revision")
        source = record.get("source")
        if not source:
            return {
                "status": "unavailable",
                "message": "No drawing-source table was retained for this revision.",
            }
        table = strict_json_loads(
            self.store.read_source(source["source_id"]), what="drawing source"
        )
        columns, rows = (
            (table.get("columns"), table.get("rows"))
            if isinstance(table, dict)
            else (None, None)
        )
        if (
            not isinstance(columns, list)
            or not isinstance(rows, list)
            or not all(isinstance(c, str) for c in columns)
        ):
            return {
                "status": "unavailable",
                "message": "The retained source is not a supported table.",
            }
        if any(not isinstance(row, list) or len(row) != len(columns) for row in rows):
            raise ValueError("Invalid retained table rows")
        figure = strict_json_loads(figure_bytes, what="saved figure")
        mapping = column_mapping(figure, columns)
        return {
            "status": "available",
            "ref": ref,
            "source_id": source["source_id"],
            "columns": [
                {
                    "name": name,
                    "roles": (
                        [item for item in mapping["roles"] if item["column"] == name]
                        if mapping
                        else []
                    ),
                }
                for name in columns
            ],
            "mapping_status": "available" if mapping else "unavailable",
            "transformations": mapping.get("transformations", []) if mapping else [],
            "selection": mapping.get("selection", {}) if mapping else {},
            "rows": rows[offset : offset + limit],
            "total_rows": len(rows),
            "offset": offset,
            "limit": limit,
        }

    def attachments(self, ref):
        from lobster.core.figures.attachments import manifest

        return manifest(self.store, self.store.read_record(self.reference(ref)))

    def attachment(self, ref, attachment_id):
        from lobster.core.figures.attachments import read_attachment

        record = self.store.read_record(self.reference(ref))
        source = record.get("source") or {}
        for descriptor in (source.get("declared") or {}).get("attachments", []):
            if descriptor["id"] == attachment_id:
                return {
                    "label": descriptor["label"],
                    "mime": descriptor["mime"],
                    "base64": base64.b64encode(
                        read_attachment(self.store, descriptor)
                    ).decode(),
                }
        if attachment_id == source.get("source_id"):
            return {
                "label": "source.json",
                "mime": "application/json",
                "base64": base64.b64encode(
                    self.store.read_source(attachment_id)
                ).decode(),
            }
        if attachment_id == (record.get("execution") or {}).get("execution_id"):
            payload = json_blob_bytes(self.store.read_execution(attachment_id))
            return {
                "label": "execution.json",
                "mime": "application/json",
                "base64": base64.b64encode(payload).decode(),
            }
        raise ValueError("Attachment does not belong to this revision")

    def delete(self, data):
        if not isinstance(data, dict) or set(data) != {"ref", "generation"}:
            raise ValueError("Expected ref and generation")
        with self._lock:
            result = self.store.delete_revision(
                self.reference(data["ref"]), data["generation"]
            )
            if self.initial_ref == result["deleted"]:
                self.initial_ref = result["load_ref"]
            return result

    def validate(self, draft):
        if (
            not isinstance(draft, dict)
            or not {"parent", "figure", "request_id"} <= set(draft)
            or not set(draft)
            <= {
                "parent",
                "figure",
                "request_id",
                "generation",
                "insert_after",
                "attachments",
                "explanation",
            }
        ):
            raise ValueError("Expected parent, figure, request_id")
        if draft.get("insert_after") is not None:
            self.reference(draft["insert_after"])
        if (
            not isinstance(draft.get("explanation", ""), str)
            or len(draft.get("explanation", "")) > 2000
        ):
            raise ValueError("Explanation must be at most 2000 characters")
        if (
            not isinstance(draft.get("attachments", []), list)
            or len(draft.get("attachments", [])) > 20
        ):
            raise ValueError("At most 20 attachments per save")
        parent = self.reference(draft["parent"])
        request_id = draft["request_id"]
        if (
            not isinstance(request_id, str)
            or not 16 <= len(request_id) <= 128
            or not request_id.isascii()
            or not request_id.isalnum()
        ):
            raise ValueError("Invalid request identifier")
        payload = json_blob_bytes(draft["figure"])
        validate_figure_bytes(payload)
        record, before, config = self.store.read(parent)
        validate_config_bytes(config)
        from lobster.core.figures.data_view import refresh_column_mapping

        proposed = strict_json_loads(payload, what="figure")
        if refresh_column_mapping(
            strict_json_loads(before, what="parent figure"), proposed
        ):
            payload = json_blob_bytes(proposed)
        return (
            parent,
            payload,
            config,
            strict_json_loads(record, what="parent revision"),
            request_id,
        )

    def edit_visuals(
        self,
        ref,
        *,
        generation,
        request_id,
        layout_updates=None,
        trace_updates=None,
        explanation="",
        require_latest=False,
    ):
        """Apply AI styling to a saved revision, then use the manual save seam.

        Trace coordinates, identities and retained source data cannot be replaced.
        Only Plotly presentation properties are accepted; Plotly validates values.
        """
        layout_updates = {} if layout_updates is None else layout_updates
        trace_updates = [] if trace_updates is None else trace_updates
        if not isinstance(layout_updates, dict) or not isinstance(trace_updates, list):
            raise ValueError("Expected layout object and trace update list")
        layout_fields = {
            "title",
            "font",
            "paper_bgcolor",
            "plot_bgcolor",
            "colorway",
            "showlegend",
            "legend",
            "margin",
            "width",
            "height",
            "autosize",
            "annotations",
            "shapes",
            "template",
            "barmode",
            "bargap",
            "bargroupgap",
            "hovermode",
            "hoverlabel",
            "uniformtext",
        }
        trace_fields = {
            "name",
            "visible",
            "opacity",
            "showlegend",
            "legendrank",
            "marker",
            "line",
            "mode",
            "text",
            "textposition",
            "textfont",
            "texttemplate",
            "hovertemplate",
            "hoverinfo",
            "hoverlabel",
            "fill",
            "fillcolor",
            "colorscale",
            "reversescale",
            "showscale",
            "colorbar",
            "contours",
            "selected",
            "unselected",
        }
        if any(
            k not in layout_fields
            and not re.fullmatch(
                r"(?:[xy]axis|coloraxis|scene|polar|ternary)(?:[2-9]|[1-9][0-9]+)?", k
            )
            for k in layout_updates
        ):
            raise ValueError("Only visual layout properties can be edited")

        def merge(target, updates):
            for key, value in updates.items():
                if key in {"__proto__", "constructor", "prototype"}:
                    raise ValueError("Invalid property name")
                if isinstance(value, dict):
                    if not isinstance(target.get(key), dict):
                        target[key] = {}
                    merge(target[key], value)
                else:
                    target[key] = copy.deepcopy(value)

        figure = self.load(ref)["figure"]
        merge(figure.setdefault("layout", {}), layout_updates)
        for update in trace_updates:
            if not isinstance(update, dict) or set(update) != {"index", "properties"}:
                raise ValueError("Each trace update needs index and properties")
            index, properties = update["index"], update["properties"]
            if type(index) is not int or not 0 <= index < len(figure["data"]):
                raise ValueError("Invalid trace index")
            if not isinstance(properties, dict) or not set(properties) <= trace_fields:
                raise ValueError("Only visual trace properties can be edited")
            merge(figure["data"][index], properties)
        return self.save(
            {
                "parent": ref,
                "figure": figure,
                "request_id": request_id,
                "generation": generation,
                "explanation": explanation,
            },
            actor="assistant",
            require_latest=require_latest,
        )

    def save(self, draft, *, actor="user", require_latest=False):
        import plotly
        from lobster.version import __version__

        with self._lock, self.store.writer_lock:
            capture = getattr(self.store, "attempt_fence", None)
            fence = capture() if callable(capture) else None
            parent, payload, config, record, key = self.validate(draft)
            identity = hashlib.sha256(
                json_blob_bytes([actor, require_latest, draft])
            ).hexdigest()
            previous = self._requests.get(key)
            if previous:
                if previous["identity"] != identity:
                    raise ValueError("Request identifier was used for another draft")
                if previous["result"]["status"] == "available":
                    # A cached ACK must not resurrect a deleted/corrupt revision.
                    self.store.read(previous["result"]["ref"])
                    return previous["result"]
                fence = previous["fence"]
            from lobster.core.figures.history import Timeline

            timeline = Timeline(self.store, self.figure_id)
            state, _ = timeline.read()
            if not previous:
                Timeline.check(state, draft.get("generation"))
                if (
                    require_latest
                    and FigureRef.parse(parent).revision_id != state["tail"]
                ):
                    raise ValueError(
                        "A newer saved revision exists. Inspect the latest figure before editing."
                    )
            anchor = draft.get("insert_after")
            if anchor and FigureRef.parse(anchor).revision_id not in state["nodes"]:
                raise ValueError("Insertion target is unavailable")
            source = record.get("source")
            source_payload = (
                SourcePayload(
                    self.store.read_source(source["source_id"]),
                    source.get("declared") or {},
                )
                if source
                else None
            )
            if draft.get("attachments"):
                from lobster.core.figures.attachments import save_attachment

                total_bytes = 0

                declared = dict(source_payload.declared) if source_payload else {}
                attached = list(declared.get("attachments") or [])
                for upload in draft["attachments"]:
                    if not isinstance(upload, dict) or set(upload) != {
                        "label",
                        "base64",
                        "description",
                    }:
                        raise ValueError("Invalid attachment upload")
                    if (
                        not isinstance(upload["base64"], str)
                        or len(upload["base64"]) > 13_333_336
                    ):
                        raise ValueError("Attachment exceeds the 10 MB limit")
                    try:
                        raw = base64.b64decode(upload["base64"], validate=True)
                    except (ValueError, TypeError):
                        raise ValueError("Invalid attachment encoding") from None
                    total_bytes += len(raw)
                    if total_bytes > 20_000_000:
                        raise ValueError("Attachments exceed the 20 MB total per save")
                    attached.append(
                        save_attachment(
                            self.store,
                            raw,
                            label=upload["label"],
                            role="related_data",
                            description=upload["description"],
                            origin="user upload",
                        )
                    )
                declared["attachments"] = attached
                source_payload = SourcePayload(
                    (
                        source_payload.payload_bytes
                        if source_payload
                        else json_blob_bytes({"columns": [], "rows": []})
                    ),
                    declared,
                )
            attempt = "editor_" + self._namespace + ":" + key
            candidate = Candidate(
                attempt_key=attempt + ":0",
                candidate_index=0,
                figure_bytes=payload,
                config_bytes=config,
                parent_ref=parent,
                source=source_payload,
                kind="edited",
                actor=actor,
                label=(
                    draft["figure"].get("layout", {}).get("title", {}).get("text")
                    if isinstance(draft["figure"].get("layout", {}).get("title"), dict)
                    else draft["figure"].get("layout", {}).get("title")
                )
                or record.get("label"),
                request=draft.get("explanation")
                or (
                    "AI visual edit"
                    if actor == "assistant"
                    else "Manual Plotly editor save"
                ),
            )
            info = ExecutionInfo(
                attempt_base=attempt,
                tool="plot_editor",
                agent="supervisor" if actor == "assistant" else None,
                parameters={
                    "parent_ref": parent,
                    "timeline_insert_after": draft.get("insert_after"),
                },
                persist=True,
                code_bytes=None,
                activity=None,
                engine_version=__version__,
                plotly_version=plotly.__version__,
            )
            result = (
                TrustedSaver(self.store)
                .commit_attempt(info, [candidate], succeeded=True, fence=fence)[0]
                .to_dict()
            )
            if result["status"] == "available":
                self.store.history(self.figure_id)
            self._requests[key] = {
                "identity": identity,
                "result": result,
                "fence": fence,
            }
            return result

    def wait_for_saves(self):
        """Let an active trusted save finish before the hosting surface shuts down."""
        with self._lock:
            pass
