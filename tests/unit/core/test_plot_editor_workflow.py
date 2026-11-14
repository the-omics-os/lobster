"""Real-store editor workflows: links, ancestry, snapshots and deterministic selection."""

import base64
import copy
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from lobster.core.figures.attachments import save_attachment
from lobster.core.figures.errors import FigureError
from lobster.core.figures.history import change_details
from lobster.core.figures.records import Candidate, ExecutionInfo, SourcePayload
from lobster.core.figures.saver import TrustedSaver
from lobster.core.figures.store import LocalFigureStore
from lobster.core.plot_editor.service import EditorService
from lobster.tools.plot_editor_tool import plot_catalog, resolve_plot


def create(store, name="Scatter", source=None, base="create:one"):
    candidate = Candidate(
        attempt_key=base + ":0",
        candidate_index=0,
        figure_bytes=json.dumps(
            {
                "data": [{"type": "scatter", "x": [1, 2], "y": [3, 4]}],
                "layout": {"title": {"text": name}},
            }
        ).encode(),
        config_bytes=b"{}",
        label=name,
        source=source,
    )
    info = ExecutionInfo(base, "test", None, {}, True, None, None, "test", "test")
    result = TrustedSaver(store).commit_attempt(info, [candidate], succeeded=True)[0]
    assert result.status == "available", result
    return result.ref


@pytest.fixture
def editor(tmp_path):
    store = LocalFigureStore(tmp_path)
    ref = create(store)
    return store, ref, EditorService(store, ref)


def draft(service, ref, title, **extra):
    figure = copy.deepcopy(service.load(ref)["figure"])
    figure["layout"]["title"]["text"] = title
    return {
        "parent": ref,
        "figure": figure,
        "request_id": title.replace(" ", "") + "x" * 20,
        "generation": service.history()["generation"],
        **extra,
    }


def links(history):
    rows = history["revisions"]
    assert history["head"] == rows[0]["revision_id"]
    assert history["tail"] == rows[-1]["revision_id"]
    for i, row in enumerate(rows):
        assert row["previous"] == (rows[i - 1]["revision_id"] if i else None)
        assert row["next"] == (
            rows[i + 1]["revision_id"] if i + 1 < len(rows) else None
        )
    return [r["ref"] for r in rows]


def test_save_historical_edit_appends_and_preserves_original(editor):
    store, root, service = editor
    original = store.read(root)
    second = service.save(draft(service, root, "Second"))["ref"]
    third = service.save(draft(service, root, "Third"))["ref"]
    assert links(service.history()) == [root, second, third]
    assert store.read_record(third)["derived_from"] == root
    assert store.read(root) == original
    record = store.read_record(third)
    assert "/layout/title/text" in record["change"]["summary"]
    assert record["change"]["details"]["touches_data"] is False
    assert record["change"]["details"]["properties"][0]["before"] == "Scatter"


def test_insert_and_delete_middle_and_tail(editor):
    store, root, service = editor
    second = service.save(draft(service, root, "Second"))["ref"]
    last = service.save(draft(service, second, "Last"))["ref"]
    inserted = service.save(draft(service, root, "Inserted", insert_after=root))["ref"]
    assert links(service.history()) == [root, inserted, second, last]
    assert store.read_record(second)["derived_from"] == root
    result = service.delete(
        {"ref": inserted, "generation": service.history()["generation"]}
    )
    assert result["load_ref"] == root
    assert links(service.history()) == [root, second, last]
    service.delete({"ref": last, "generation": service.history()["generation"]})
    assert links(service.history()) == [root, second]
    with pytest.raises(ValueError, match="pinned"):
        service.delete({"ref": root, "generation": service.history()["generation"]})
    with pytest.raises(Exception):
        store.read(inserted)
    # Deleted revision metadata and payload remain for ancestry/audit.
    assert (
        store.root / inserted.split("@")[0] / inserted.split("@")[1] / "revision.json"
    ).exists()


def test_stale_save_delete_and_idempotent_retry(editor):
    store, root, service = editor
    first = draft(service, root, "Second")
    stale = draft(service, root, "Stale")
    second = service.save(first)["ref"]
    assert service.save(first)["ref"] == second
    with pytest.raises(ValueError, match="History changed"):
        service.save(stale)
    with pytest.raises(ValueError, match="History changed"):
        service.delete({"ref": second, "generation": stale["generation"]})
    changed = {**first, "explanation": "different request"}
    with pytest.raises(ValueError, match="another draft"):
        service.save(changed)


def test_two_windows_do_not_overwrite_each_other(editor):
    store, root, service = editor
    other = EditorService(LocalFigureStore(store.workspace), root)
    drafts = [draft(service, root, "WindowOne"), draft(other, root, "WindowTwo")]

    def save(pair):
        try:
            return pair[0].save(pair[1])["status"]
        except ValueError:
            return "conflict"

    with ThreadPoolExecutor(2) as pool:
        outcomes = list(pool.map(save, zip([service, other], drafts)))
    assert sorted(outcomes) == ["available", "conflict"]
    assert len(links(service.history())) == 2


def test_legacy_history_migration_keeps_ancestry(editor):
    store, root, service = editor
    second = service.save(draft(service, root, "Second"))["ref"]
    index = store.root / "_history" / (service.figure_id + ".json")
    index.unlink()
    migrated = EditorService(store, root).history()
    assert links(migrated) == [root, second]
    assert store.read_record(second)["parents"] == [root.split("@")[1]]


def test_attachments_are_snapshots_inherited_and_downloadable(tmp_path):
    store = LocalFigureStore(tmp_path)
    path = tmp_path / "input.csv"
    path.write_bytes(b"x,y\n1,2\n")
    attachment = save_attachment(
        store,
        path.read_bytes(),
        label="input.csv",
        role="plotted_data",
        columns=["x", "y"],
    )
    source = SourcePayload(
        b'{"columns":["x","y"],"rows":[[1,2]]}', {"attachments": [attachment]}
    )
    root = create(store, source=source)
    path.write_bytes(b"changed")
    service = EditorService(store, root)
    request = draft(
        service,
        root,
        "Edited",
        attachments=[
            {
                "label": "stats.csv",
                "base64": base64.b64encode(b"p\n0.01\n").decode(),
                "description": "Related statistical results",
            }
        ],
    )
    edited = service.save(request)["ref"]
    assert (
        base64.b64decode(service.attachment(edited, attachment["id"])["base64"])
        == b"x,y\n1,2\n"
    )
    rows = service.load(edited)["attachments"]
    assert any(r["label"] == "stats.csv" and r["available"] for r in rows)
    assert len([r for r in rows if r.get("role") == "plotted_data"]) == 1
    assert not any(r["label"] == "stats.csv" for r in service.load(root)["attachments"])
    with pytest.raises(ValueError, match="does not belong"):
        service.attachment(
            root, next(r["id"] for r in rows if r["label"] == "stats.csv")
        )
    # Deleting an edit does not remove the shared source blob.
    service.delete({"ref": edited, "generation": service.history()["generation"]})
    assert service.attachment(root, attachment["id"])


def test_catalog_groups_revisions_and_resolves_latest(editor):
    store, root, service = editor
    latest = service.save(draft(service, root, "Edited"))["ref"]
    catalog = plot_catalog(store)
    assert len(catalog) == 1 and catalog[0]["revision_count"] == 2
    assert resolve_plot(store, "Scatter")[0] == latest
    assert resolve_plot(store, root)[0] == root
    create(store, name="Scatter", base="create:two")
    ref, choices = resolve_plot(store, "Scatter")
    assert ref is None and len(choices) == 2
    assert resolve_plot(store, "missing")[0] is None
    assert resolve_plot(store, "")[0] is None


def test_diff_marks_data_changes_and_bounds_large_arrays():
    before = {"data": [{"x": [1, 2], "y": [3, 4]}]}
    after = {"data": [{"x": list(range(10000)), "y": [3, 4]}]}
    details = change_details(before, after)
    assert details["touches_data"] is True
    assert details["properties"][0]["after"]["length"] == 10000
    assert len(json.dumps(details)) < 2000


def test_ai_visual_edit_uses_same_revision_saver_as_manual(editor):
    store, root, service = editor
    original = store.read(root)
    manual = service.save(draft(service, root, "Manual"))["ref"]
    edited = service.edit_visuals(
        manual,
        generation=service.history()["generation"],
        request_id="ai" * 16,
        layout_updates={"xaxis": {"title": {"text": "Effect"}}},
        trace_updates=[{"index": 0, "properties": {"marker": {"size": 12}}}],
        explanation="Make points larger and label the x axis",
    )
    assert edited["status"] == "available"
    ref = edited["ref"]
    record = store.read_record(ref)
    assert record["parents"] == [manual.split("@")[1]]
    assert record["change"]["actor"] == "assistant"
    assert record["change"]["details"]["touches_data"] is False
    assert record["change"]["request"] == "Make points larger and label the x axis"
    loaded = service.load(ref)["figure"]
    assert loaded["data"][0]["marker"]["size"] == 12
    assert loaded["data"][0]["x"] == [1, 2]
    assert store.read(root) == original
    assert links(service.history()) == [root, manual, ref]


@pytest.mark.parametrize(
    "update",
    [
        {"index": 0, "properties": {"x": [5, 6]}},
        {"index": 0, "properties": {"ids": ["other", "points"]}},
        {"index": -1, "properties": {"opacity": 0}},
        {"index": 0, "properties": {"marker": {"size": -10}}},
    ],
)
def test_ai_edit_rejects_data_changes_and_invalid_styles(editor, update):
    store, root, service = editor
    with pytest.raises((ValueError, FigureError)):
        service.edit_visuals(
            root,
            generation=service.history()["generation"],
            request_id="ai" * 16,
            trace_updates=[update],
        )
    assert links(service.history()) == [root]


def test_ai_tools_inspect_edit_retry_and_stale_history(editor):
    from types import SimpleNamespace
    from lobster.tools.plot_editor_tool import create_figure_edit_tools

    store, root, service = editor
    inspect, edit = create_figure_edit_tools(SimpleNamespace(figure_store=store))
    found = json.loads(inspect.invoke({"figure_ref": "Scatter"}))
    stale_inspection = json.loads(inspect.invoke({"figure_ref": "Scatter"}))
    request = {
        "figure_ref": found["ref"],
        "generation": found["generation"],
        "request_id": found["request_id"],
        "layout_updates": {"font": {"size": 18}},
        "trace_updates": [],
        "explanation": "Increase font size",
    }
    result = json.loads(edit.invoke(request))
    assert result["status"] == "available"
    assert json.loads(edit.invoke(request))["ref"] == result["ref"]
    assert links(service.history()) == [root, result["ref"]]
    stale = {**request, "request_id": stale_inspection["request_id"]}
    assert "History changed" in json.loads(edit.invoke(stale))["message"]


def test_ai_edit_from_old_chat_ref_uses_intervening_gui_save(editor):
    from types import SimpleNamespace
    from lobster.tools.plot_editor_tool import create_figure_edit_tools

    store, root, service = editor
    inspect, edit = create_figure_edit_tools(SimpleNamespace(figure_store=store))
    # Warm the same service cache before saving from a separate GUI adapter.
    json.loads(inspect.invoke({"figure_ref": root}))
    gui = EditorService(store, root)
    manual = draft(gui, root, "Manual styling")
    manual["figure"]["data"][0]["marker"] = {"size": 23, "color": "purple"}
    manual_ref = gui.save(manual)["ref"]
    found = json.loads(inspect.invoke({"figure_ref": root}))
    assert found["ref"] == manual_ref
    assert found["selection"] == "latest"
    assert found["figure"]["data"][0]["marker"]["size"] == 23
    result = json.loads(
        edit.invoke(
            {
                "figure_ref": found["ref"],
                "generation": found["generation"],
                "request_id": found["request_id"],
                "layout_updates": {"font": {"size": 18}},
                "trace_updates": [],
                "explanation": "Increase font size",
            }
        )
    )
    assert result["status"] == "available"
    assert store.read_record(result["ref"])["derived_from"] == manual_ref
    assert service.load(result["ref"])["figure"]["data"][0]["marker"] == {
        "size": 23,
        "color": "purple",
    }
    assert links(service.history()) == [root, manual_ref, result["ref"]]


def test_explicit_historical_ai_edit_remains_supported(editor):
    from types import SimpleNamespace
    from lobster.tools.plot_editor_tool import create_figure_edit_tools

    store, root, service = editor
    latest = service.save(draft(service, root, "Manual"))["ref"]
    inspect, edit = create_figure_edit_tools(SimpleNamespace(figure_store=store))
    found = json.loads(inspect.invoke({"figure_ref": root, "use_exact_revision": True}))
    assert found["ref"] == root and found["latest_ref"] == latest
    result = json.loads(
        edit.invoke(
            {
                "figure_ref": root,
                "generation": found["generation"],
                "request_id": found["request_id"],
                "layout_updates": {"title": {"text": "Historical edit"}},
                "trace_updates": [],
                "explanation": "Explicitly edit the original",
            }
        )
    )
    assert result["status"] == "available"
    assert store.read_record(result["ref"])["derived_from"] == root
    assert links(service.history()) == [root, latest, result["ref"]]


def test_ai_cannot_substitute_old_parent_or_invent_inspection(editor):
    from types import SimpleNamespace
    from lobster.tools.plot_editor_tool import create_figure_edit_tools

    store, root, service = editor
    service.save(draft(service, root, "Manual"))
    inspect, edit = create_figure_edit_tools(SimpleNamespace(figure_store=store))
    found = json.loads(inspect.invoke({"figure_ref": root}))
    args = {
        "figure_ref": root,
        "generation": found["generation"],
        "request_id": found["request_id"],
        "layout_updates": {"font": {"size": 20}},
        "trace_updates": [],
        "explanation": "Edit",
    }
    assert "inspected revision" in json.loads(edit.invoke(args))["message"]
    args["request_id"] = "invented" * 4
    assert "Inspect" in json.loads(edit.invoke(args))["message"]
    assert len(service.history()["revisions"]) == 2


def test_shared_saver_rejects_old_ai_parent_with_fresh_generation(editor):
    _, root, service = editor
    service.save(draft(service, root, "Manual"))
    with pytest.raises(ValueError, match="newer saved revision"):
        service.edit_visuals(
            root,
            generation=service.history()["generation"],
            request_id="latestguard" * 3,
            layout_updates={"font": {"size": 22}},
            require_latest=True,
        )


def test_delete_initial_revision_moves_server_default_and_rejects_bad_payload(editor):
    store, root, service = editor
    second = service.save(draft(service, root, "Second"))["ref"]
    opened = EditorService(store, second)
    opened.delete({"ref": second, "generation": opened.history()["generation"]})
    assert opened.initial_ref == root
    for invalid in [None, [], {}, {"ref": root, "generation": 1, "extra": True}]:
        with pytest.raises(ValueError, match="Expected ref"):
            opened.delete(invalid)


@pytest.mark.parametrize("answer", ["select", "cancel"])
def test_plot_picker_uses_real_interrupt_ids_and_selected_plot(
    editor, monkeypatch, answer
):
    from types import SimpleNamespace
    from typing import TypedDict
    from langgraph.graph import StateGraph, START, END
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command
    from lobster.tools.plot_editor_tool import create_plot_editor_tools
    from lobster.core import plot_editor

    store, root, service = editor
    other = create(store, name="Scatter", base="create:other")
    calls = []

    def launch(store, ref):
        calls.append(ref)
        return {"status": "editor_connected", "mode": "app"}

    monkeypatch.setattr(plot_editor, "launch_editor", launch)
    _, opener = create_plot_editor_tools(SimpleNamespace(figure_store=store))

    class State(TypedDict):
        outcome: str

    def open_node(state):
        return {
            "outcome": opener.invoke({"figure_ref": "Scatter", "user_approved": True})
        }

    builder = StateGraph(State)
    builder.add_node("open", open_node)
    builder.add_edge(START, "open")
    builder.add_edge("open", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "picker"}}
    paused = graph.invoke({"outcome": ""}, config)
    question = paused["__interrupt__"][0]
    assert not calls
    assert question.value["component"] == "select"
    options = question.value["data"]["options"]
    assert len(options) == 2
    # Choose by the server-issued label, not an AI-created reference.
    selected = next(label for label in options if other.split("@")[0] in label)
    response = {"selected": selected} if answer == "select" else {"action": "cancel"}
    result = graph.invoke(Command(resume={question.id: response}), config)
    outcome = json.loads(result["outcome"])
    if answer == "select":
        assert calls == [other]
        assert outcome["status"] == "editor_connected"
    else:
        assert not calls and outcome["status"] == "cancelled"


def test_explicit_plot_opens_latest_without_extra_question(editor, monkeypatch):
    from types import SimpleNamespace
    from lobster.tools.plot_editor_tool import create_plot_editor_tools
    from lobster.core import plot_editor

    store, root, service = editor
    latest = service.save(draft(service, root, "Latest"))["ref"]
    calls = []
    monkeypatch.setattr(
        plot_editor,
        "launch_editor",
        lambda store, ref: calls.append(ref)
        or {"status": "launch_requested", "mode": "app"},
    )
    listing, opener = create_plot_editor_tools(SimpleNamespace(figure_store=store))
    assert len(json.loads(listing.invoke({}))["plots"]) == 1
    assert (
        json.loads(opener.invoke({"figure_ref": "Scatter", "user_approved": True}))[
            "status"
        ]
        == "launch_requested"
    )
    assert calls == [latest]
    assert (
        json.loads(opener.invoke({"figure_ref": root, "user_approved": True}))["ref"]
        == root
    )
    assert (
        json.loads(opener.invoke({"figure_ref": "missing", "user_approved": True}))[
            "status"
        ]
        == "not_found"
    )
    assert (
        json.loads(opener.invoke({"figure_ref": root}))["status"] == "approval_required"
    )
    assert calls == [latest, root]


def test_interrupted_index_publication_is_recoverable(editor, monkeypatch):
    from lobster.core.figures.history import Timeline
    from lobster.core.figures.errors import StorageError

    store, root, service = editor
    request = draft(service, root, "Recovered")
    publish = Timeline._publish
    fail = [True]

    def once(self, state):
        if fail[0]:
            fail[0] = False
            raise StorageError("Simulated history publication failure")
        return publish(self, state)

    monkeypatch.setattr(Timeline, "_publish", once)
    pending = service.save(request)
    assert pending["status"] == "pending"
    recovered = service.save(request)
    assert recovered["status"] == "available"
    assert links(service.history()) == [root, recovered["ref"]]
    assert service.save(request)["ref"] == recovered["ref"]


def test_invalid_history_links_are_refused(editor):
    from lobster.core.figures.errors import StorageError

    store, root, service = editor
    history = service.history()
    history.pop("revisions")
    history["nodes"][history["head"]]["next"] = history["head"]
    index = store.root / "_history" / (service.figure_id + ".json")
    index.write_text(json.dumps(history))
    with pytest.raises(StorageError, match="Invalid figure history links"):
        service.history()


def test_migration_orders_ancestry_even_when_clock_moves_backwards(editor, monkeypatch):
    from lobster.core.figures import store as store_module

    store, root, service = editor
    monkeypatch.setattr(store_module, "_utcnow", lambda: "2000-01-01T00:00:00Z")
    second = service.save(draft(service, root, "Second"))["ref"]
    third = service.save(draft(service, second, "Third"))["ref"]
    (store.root / "_history" / (service.figure_id + ".json")).unlink()
    assert links(service.history()) == [root, second, third]
