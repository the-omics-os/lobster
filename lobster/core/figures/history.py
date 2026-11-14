"""Mutable timeline index over immutable revision payloads and ancestry.

Only the store calls this module, while holding its workspace writer lock.
A single atomic replacement publishes all forward/backward links together.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile

from .errors import StorageError
from .ids import FigureRef


def change_details(before, after):
    """Bounded, factual JSON diff; large arrays are summarized by size and hash."""
    import hashlib

    changes = []

    def value(v):
        raw = json.dumps(v, sort_keys=True, separators=(",", ":")).encode()
        if len(raw) <= 512:
            return v
        return {
            "summary": "Large value",
            "sha256": hashlib.sha256(raw).hexdigest(),
            "length": len(v) if isinstance(v, (list, dict, str)) else None,
        }

    def walk(a, b, path):
        if a == b:
            return
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(a.keys() | b.keys()):
                p = path + "/" + key.replace("~", "~0").replace("/", "~1")
                if key not in a or key not in b:
                    changes.append(
                        {
                            "path": p,
                            "operation": "added" if key in b else "removed",
                            "before": value(a.get(key)),
                            "after": value(b.get(key)),
                        }
                    )
                else:
                    walk(a[key], b[key], p)
        elif (
            isinstance(a, list)
            and isinstance(b, list)
            and path == "/data"
            and len(a) == len(b)
        ):
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{path}/{i}")
        else:
            changes.append(
                {
                    "path": path,
                    "operation": "changed",
                    "before": value(a),
                    "after": value(b),
                }
            )

    walk(before, after, "")
    data_keys = {
        "x",
        "y",
        "z",
        "values",
        "labels",
        "customdata",
        "ids",
        "dimensions",
        "type",
    }
    touches = any(
        c["path"] == "/data"
        or c["path"].startswith("/frames")
        or (
            c["path"].startswith("/data/")
            and (len(c["path"].split("/")) <= 3 or c["path"].split("/")[3] in data_keys)
        )
        for c in changes
    )
    paths = [c["path"] for c in changes]
    return {
        "summary": (
            (
                "Changed "
                + ", ".join(paths[:6])
                + (f" and {len(paths)-6} more properties" if len(paths) > 6 else "")
            )
            if paths
            else "No figure properties changed"
        ),
        "properties": changes[:500],
        "property_count": len(changes),
        "truncated": len(changes) > 500,
        "touches_data": touches,
    }


class Timeline:
    def __init__(self, store, figure_id):
        # Parsing validates both IDs and forbids paths supplied by a client.
        FigureRef.parse(figure_id + "@r_aaaaaaaaaa")
        self.store = store
        self.figure_id = figure_id
        self.directory = store.root / "_history"
        self.path = self.directory / (figure_id + ".json")

    def _publish(self, state):
        self.store._mkdirs(self.directory)
        self.store._read(self.path)  # reject symlinks/non-regular entries
        data = json.dumps(state, sort_keys=True, allow_nan=False).encode()
        fd, temporary = tempfile.mkstemp(prefix=".tmp-", dir=self.directory)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            from .store import _fsync_dir

            _fsync_dir(self.directory)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def read(self):
        records = list(self.store._iter_records(self.figure_id))
        records.sort(key=lambda r: (r.get("created_at", ""), r["revision_id"]))
        # Wall clocks can move backwards or give several revisions the same time.
        # Migrate ancestors/anchors before descendants, preserving every parent edge.
        remaining, records, emitted = records, [], set()
        known = {r["revision_id"] for r in remaining}
        while remaining:
            ready = []
            for record in remaining:
                dependencies = set(record.get("parents") or []) & known
                anchor = record.get("timeline_insert_after")
                if anchor:
                    target = FigureRef.parse(anchor)
                    if target.figure_id != self.figure_id:
                        raise StorageError("Invalid insertion anchor")
                    if target.revision_id in known:
                        dependencies.add(target.revision_id)
                if dependencies <= emitted:
                    ready.append(record)
            if not ready:
                raise StorageError("Cyclic revision ancestry")
            for record in ready:
                records.append(record)
                emitted.add(record["revision_id"])
                remaining.remove(record)
        raw = self.store._read(self.path)
        try:
            decoded = json.loads(raw) if raw else None
        except (ValueError, RecursionError) as exc:
            raise StorageError("Invalid figure history") from exc
        state = (
            decoded
            if raw
            else {
                "schema": "lobster.figure_history/1",
                "figure_id": self.figure_id,
                "head": None,
                "tail": None,
                "generation": 0,
                "nodes": {},
                "deleted": [],
            }
        )
        if (
            not isinstance(state, dict)
            or state.get("schema") != "lobster.figure_history/1"
            or state.get("figure_id") != self.figure_id
        ):
            raise StorageError("Invalid figure history")
        self._validate(state)
        original = copy.deepcopy(state)
        live = {r["revision_id"] for r in records}
        # Reconcile store deletion fences, including interrupted timeline deletes.
        for rid in list(state["nodes"]):
            if rid not in live:
                self._unlink(state, rid)
        for record in records:
            rid = record["revision_id"]
            if rid not in state["nodes"]:
                anchor = record.get("timeline_insert_after")
                after = FigureRef.parse(anchor).revision_id if anchor else state["tail"]
                if anchor and after not in state["nodes"]:
                    if self.store._ref_deleted(FigureRef.parse(anchor)):
                        after = state["tail"]
                    else:
                        raise StorageError("Insertion anchor is unavailable")
                self._insert(state, rid, after)
        if state != original:
            state["generation"] += 1
            self._publish(state)
        return state, {r["revision_id"]: r for r in records}

    def _validate(self, state):
        try:
            if (
                type(state["generation"]) is not int
                or state["generation"] < 0
                or not isinstance(state["deleted"], list)
            ):
                raise ValueError("Invalid history generation")
            nodes = state["nodes"]
            if not isinstance(nodes, dict):
                raise ValueError("Invalid nodes")
            seen, rid, previous = set(), state["head"], None
            while rid is not None:
                FigureRef(self.figure_id, rid)
                if rid in seen or nodes[rid]["previous"] != previous:
                    raise ValueError("Invalid links")
                seen.add(rid)
                previous, rid = rid, nodes[rid]["next"]
            if len(seen) != len(nodes) or previous != state["tail"]:
                raise ValueError("Disconnected history")
        except (KeyError, TypeError, ValueError) as exc:
            raise StorageError("Invalid figure history links") from exc

    @staticmethod
    def _insert(state, rid, after):
        next_id = state["nodes"][after]["next"] if after else None
        state["nodes"][rid] = {"previous": after, "next": next_id}
        if after:
            state["nodes"][after]["next"] = rid
        else:
            state["head"] = rid
        if next_id:
            state["nodes"][next_id]["previous"] = rid
        else:
            state["tail"] = rid

    @staticmethod
    def _unlink(state, rid):
        node = state["nodes"].pop(rid)
        if node["previous"]:
            state["nodes"][node["previous"]]["next"] = node["next"]
        else:
            state["head"] = node["next"]
        if node["next"]:
            state["nodes"][node["next"]]["previous"] = node["previous"]
        else:
            state["tail"] = node["previous"]
        if rid not in state["deleted"]:
            state["deleted"].append(rid)

    @staticmethod
    def check(state, generation):
        if type(generation) is not int or generation != state["generation"]:
            raise ValueError(
                "History changed in another window. Refresh history and retry."
            )

    def view(self):
        state, records = self.read()
        rows, rid, seen = [], state["head"], set()
        while rid:
            if rid in seen or rid not in state["nodes"]:
                raise StorageError("Invalid history links")
            seen.add(rid)
            record = records[rid]
            rows.append(
                {
                    **record,
                    **state["nodes"][rid],
                    "ref": f"{self.figure_id}@{rid}",
                    "availability": "available",
                    "derived_from": record.get("derived_from")
                    or (
                        f"{self.figure_id}@{record['parents'][0]}"
                        if record.get("parents")
                        else None
                    ),
                }
            )
            rid = state["nodes"][rid]["next"]
        if len(seen) != len(state["nodes"]):
            raise StorageError("Disconnected history")
        return {**state, "revisions": rows}

    def delete(self, ref, generation):
        state, _ = self.read()
        self.check(state, generation)
        parsed = FigureRef.parse(ref)
        if (
            parsed.figure_id != self.figure_id
            or parsed.revision_id not in state["nodes"]
        ):
            raise ValueError("Revision is unavailable")
        if parsed.revision_id == state["head"]:
            raise ValueError(
                "The original revision is pinned and cannot be deleted individually."
            )
        node = state["nodes"][parsed.revision_id]
        fallback = node["previous"] or node["next"]
        report = self.store.fence_deletion([ref])
        if not report.complete:
            raise StorageError("Deletion confirmation pending")
        self._unlink(state, parsed.revision_id)
        state["generation"] += 1
        self._publish(state)
        return {
            "deleted": ref,
            "load_ref": f"{self.figure_id}@{fallback}",
            "generation": state["generation"],
        }
