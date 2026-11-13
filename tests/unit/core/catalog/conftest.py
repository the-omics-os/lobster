"""Shared fakes for the catalog seam tests."""

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from lobster.core.catalog import factory
from lobster.core.catalog.port import ArtifactRef, CatalogStatus, TransformRef


class RecordingCatalog:
    """A synchronous engine that records every call (thread included)."""

    def __init__(self, workspace_path: Optional[Path] = None, instance=None):
        self.workspace_path = workspace_path
        self.instance = instance
        self.calls: List[Tuple[str, Any]] = []
        self.threads: List[int] = []
        self.next_keep_path: Any = None
        self.existing_ref: Optional[ArtifactRef] = None
        self.raise_on: Dict[str, BaseException] = {}
        self.block: Dict[str, threading.Event] = {}

    def _enter(self, op: str, payload: Any) -> None:
        self.threads.append(threading.get_ident())
        self.calls.append((op, payload))
        if op in self.block:
            self.block[op].wait(10)
        if op in self.raise_on:
            raise self.raise_on[op]

    def ops(self) -> List[str]:
        return [op for op, _ in self.calls]

    def begin_run(self, session):
        self._enter("begin_run", session)

    def keep_path(self, path):
        self._enter("keep_path", path)
        return self.next_keep_path

    def register_artifact(self, path, *, key, kind, otype, description, meta):
        self._enter(
            "register_artifact",
            {
                "path": Path(path),
                "key": key,
                "kind": kind,
                "otype": otype,
                "description": description,
                "meta": meta,
            },
        )
        if self.existing_ref is not None:
            return self.existing_ref
        return ArtifactRef(uid="u1", key=key, path=Path(path), version="1")

    def note_unrecorded(self, path):
        self._enter("note_unrecorded", Path(path))

    def link_input(self, path):
        self._enter("link_input", Path(path))
        return None

    def register_transform(self, *, key, kind, source, description):
        self._enter(
            "register_transform",
            {"key": key, "kind": kind, "source": source, "description": description},
        )
        return TransformRef(uid="t1", key=key)

    def end_run(self, status, report_path=None, environment_path=None):
        self._enter("end_run", (status, report_path))

    def status(self):
        self._enter("status", None)
        return CatalogStatus(engine="fake", mode="workspace", location="here")


@pytest.fixture
def recording_catalog_cls():
    return RecordingCatalog


@pytest.fixture(autouse=True)
def _fresh_process_catalog():
    factory._reset_for_tests()
    yield
    factory._reset_for_tests()
