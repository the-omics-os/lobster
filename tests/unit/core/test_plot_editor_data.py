"""Read-only table pages use retained production sources and declared column roles."""

import copy
import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pandas as pd
import pytest

from lobster.core.figures.payload import source_to_bytes
from lobster.core.figures.records import Candidate, ExecutionInfo, SourcePayload
from lobster.core.figures.saver import TrustedSaver
from lobster.core.figures.serialize import figure_to_bytes
from lobster.core.figures.store import LocalFigureStore
from lobster.core.plot_editor.server import EditorSession
from lobster.core.plot_editor.service import EditorService
from lobster.figure_templates import get_template

pytestmark = pytest.mark.no_auto_config


@pytest.fixture
def table_editor(tmp_path):
    data = pd.DataFrame(
        {
            "point_id": [f"P{i:03}" for i in range(205)],
            "effect": list(range(205)),
            "score": [i / 2 for i in range(205)],
            "group": ["A"] * 205,
            "note": ["<script>not executable</script>"] * 205,
        }
    )
    figure = get_template("generic.scatter")(
        data,
        x="effect",
        y="score",
        color="group",
        id_col="point_id",
    )
    store = LocalFigureStore(tmp_path)
    source = SourcePayload(source_to_bytes(data), {})
    candidate = Candidate(
        attempt_key="data:test:0",
        candidate_index=0,
        figure_bytes=figure_to_bytes(figure),
        config_bytes=b"{}",
        source=source,
    )
    info = ExecutionInfo(
        "data:test", "test", None, {}, True, None, None, "test", "test"
    )
    result = TrustedSaver(store).commit_attempt(info, [candidate], succeeded=True)[0]
    assert result.status == "available"
    return store, result.ref, EditorService(store, result.ref)


def test_pages_and_roles_use_exact_retained_table(table_editor):
    store, ref, service = table_editor
    before = store.read(ref)
    page = service.data_page(ref)
    assert page["total_rows"] == 205
    assert len(page["rows"]) == 100
    assert page["rows"][0] == ["P000", 0, 0.0, "A", "<script>not executable</script>"]
    roles = {
        col["name"]: [item["role"] for item in col["roles"]] for col in page["columns"]
    }
    assert roles == {
        "point_id": ["id"],
        "effect": ["x"],
        "score": ["y"],
        "group": ["color"],
        "note": [],
    }
    assert page["mapping_status"] == "available"
    last = service.data_page(ref, offset=200)
    assert len(last["rows"]) == 5 and last["rows"][0][0] == "P200"
    assert service.data_page(ref, offset=999)["rows"] == []
    assert store.read(ref) == before


@pytest.mark.parametrize(
    "offset,limit", [(-1, 100), (True, 100), (0, 0), (0, 201), (0, True)]
)
def test_invalid_page_bounds_are_rejected(table_editor, offset, limit):
    _, ref, service = table_editor
    with pytest.raises(ValueError):
        service.data_page(ref, offset=offset, limit=limit)


def test_visual_revision_inherits_table_and_roles(table_editor):
    store, ref, service = table_editor
    original = service.data_page(ref)
    edited = service.edit_visuals(
        ref,
        generation=service.history()["generation"],
        request_id="visual" * 5,
        layout_updates={"title": {"text": "Edited"}},
        trace_updates=[{"index": 0, "properties": {"marker": {"color": "red"}}}],
    )
    assert edited["status"] == "available"
    new = service.data_page(edited["ref"])
    assert new["source_id"] == original["source_id"]
    assert new["columns"] == original["columns"]
    assert new["rows"] == original["rows"]
    assert new["mapping_status"] == "available"


def test_stale_and_missing_mapping_never_infer_column_roles(table_editor):
    from lobster.core.figures.data_view import column_mapping

    _, ref, service = table_editor
    fig = service.load(ref)["figure"]
    columns = [col["name"] for col in service.data_page(ref)["columns"]]
    assert column_mapping(fig, columns)
    altered = copy.deepcopy(fig)
    altered["data"][0]["x"] = [1, 2]
    assert column_mapping(altered, columns) is None
    fig["layout"]["meta"].pop("lobster_data")
    assert column_mapping(fig, columns) is None


def test_older_figure_keeps_table_without_claiming_roles(table_editor):
    _, ref, service = table_editor
    figure = service.load(ref)["figure"]
    del figure["layout"]["meta"]["lobster_data"]
    result = service.save(
        {
            "parent": ref,
            "figure": figure,
            "request_id": "legacy" * 5,
            "generation": service.history()["generation"],
        }
    )
    assert result["status"] == "available"
    page = service.data_page(result["ref"])
    assert page["status"] == "available"
    assert page["mapping_status"] == "unavailable"
    assert all(not column["roles"] for column in page["columns"])
    assert page["rows"] == service.data_page(ref)["rows"]


def test_numeric_json_normalization_preserves_roles_without_hiding_data_changes():
    import plotly.graph_objects as go
    from lobster.core.figures.data_view import record_column_roles, column_mapping

    figure = go.Figure(go.Scatter(x=[1.0, 2.0, -0.0], y=[3.0, 4.0, 5.0]))
    record_column_roles(
        figure, [{"column": "x", "role": "x"}, {"column": "y", "role": "y"}]
    )
    browser = json.loads(figure_to_bytes(figure))
    browser["data"][0].update(x=[1, 2, 0], y=[3, 4, 5])
    assert column_mapping(browser, ["x", "y"]) is not None
    browser["data"][0]["x"][0] = 1.01
    assert column_mapping(browser, ["x", "y"]) is None


def test_legacy_numeric_hash_upgrades_on_save_without_rewriting_parent(tmp_path):
    import plotly.graph_objects as go
    from lobster.core.figures.data_view import record_column_roles, _binding

    figure = go.Figure(go.Scatter(x=[1.0, 2.0], y=[3.0, 4.0]))
    roles = [{"column": "x", "role": "x"}, {"column": "y", "role": "y"}]
    record_column_roles(figure, roles)
    parsed = json.loads(figure_to_bytes(figure))
    parsed["layout"]["meta"]["lobster_data"]["binding"] = _binding(
        parsed, roles, normalize=False
    )
    store = LocalFigureStore(tmp_path)
    source = SourcePayload(
        source_to_bytes([{"x": 1.0, "y": 3.0}, {"x": 2.0, "y": 4.0}]), {}
    )
    candidate = Candidate(
        attempt_key="legacy:numbers:0",
        candidate_index=0,
        figure_bytes=json.dumps(parsed).encode(),
        config_bytes=b"{}",
        source=source,
    )
    info = ExecutionInfo(
        "legacy:numbers", "test", None, {}, True, None, None, "test", "test"
    )
    root = TrustedSaver(store).commit_attempt(info, [candidate], succeeded=True)[0].ref
    service = EditorService(store, root)
    before = store.read(root)
    assert service.data_page(root)["mapping_status"] == "available"
    # Simulate a browser retaining its initial metadata across several saves.
    browser = copy.deepcopy(parsed)
    browser["data"][0].update(x=[1, 2], y=[3, 4])
    parent = root
    for n in range(2):
        browser["layout"]["title"] = {"text": f"Title {n}"}
        saved = service.save(
            {
                "parent": parent,
                "figure": browser,
                "request_id": f"legacy{n}" * 4,
                "generation": service.history()["generation"],
            }
        )
        assert saved["status"] == "available"
        parent = saved["ref"]
        assert service.data_page(parent)["mapping_status"] == "available"
        assert (
            service.data_page(parent)["source_id"]
            == service.data_page(root)["source_id"]
        )
    assert store.read(root) == before


def test_numeric_normalization_keeps_distinct_large_integers_distinct():
    import plotly.graph_objects as go
    from lobster.core.figures.data_view import record_column_roles, column_mapping

    figure = go.Figure(go.Scatter(x=[9007199254740993], y=[1]))
    record_column_roles(figure, [{"column": "x", "role": "x"}])
    parsed = json.loads(figure_to_bytes(figure))
    parsed["data"][0]["x"] = [9007199254740992]
    assert column_mapping(parsed, ["x"]) is None


def test_data_endpoint_is_authenticated_and_scoped(table_editor):
    store, ref, service = table_editor
    session = EditorSession(service)
    try:
        with pytest.raises(HTTPError) as caught:
            urlopen(session.origin + "/api/data")
        assert caught.value.code == 403
        headers = {"Authorization": "Bearer " + session.token}
        with urlopen(
            Request(session.origin + "/api/data?offset=200&limit=10", headers=headers)
        ) as response:
            page = json.load(response)
        assert len(page["rows"]) == 5
        assert page["ref"] == ref
        for query in [
            "ref=f_aaaaaaaaaa%40r_aaaaaaaaaa",
            "ref=../../secret",
            "limit=201",
            "offset=1&offset=2",
        ]:
            with pytest.raises(HTTPError) as caught:
                urlopen(Request(session.origin + "/api/data?" + query, headers=headers))
            assert caught.value.code in (400, 409)
        with pytest.raises(HTTPError) as caught:
            urlopen(
                Request(
                    session.origin + "/api/data",
                    headers={**headers, "Origin": "https://external.invalid"},
                )
            )
        assert caught.value.code == 403
    finally:
        session.close()


def test_general_plot_tool_records_roles_and_source_metadata(tmp_path):
    from types import SimpleNamespace
    from lobster.core.plot_manager import PlotManager
    from lobster.tools.plot_tools import create_plot_tools

    manager = PlotManager(tmp_path)
    store = LocalFigureStore(tmp_path)
    tools = create_plot_tools(
        SimpleNamespace(plot_manager=manager, figure_store=store), tmp_path
    )
    tool = next(t for t in tools if t.name == "create_interactive_plot")
    result = tool.invoke(
        {
            "type": "tool_call",
            "id": "data_roles",
            "name": tool.name,
            "args": {
                "plot_type": "scatter",
                "data_json": '[{"a":1,"b":2,"extra":3}]',
                "x": "a",
                "y": "b",
            },
        }
    )
    assert "saved:" in result.content
    row = store.list().items[0]
    ref = row["figure_id"] + "@" + row["revision_id"]
    page = EditorService(store, ref).data_page(ref)
    assert page["mapping_status"] == "available"
    assert page["selection"]["sampled"] is False
    assert (
        store.read_record(ref)["source"]["declared"]["column_mapping"]["schema"]
        == "lobster.plot_data/1"
    )


def test_missing_source_is_an_explicit_empty_state(tmp_path):
    import plotly.graph_objects as go

    store = LocalFigureStore(tmp_path)
    candidate = Candidate(
        attempt_key="empty:test:0",
        candidate_index=0,
        figure_bytes=figure_to_bytes(go.Figure()),
        config_bytes=b"{}",
    )
    info = ExecutionInfo(
        "empty:test", "test", None, {}, True, None, None, "test", "test"
    )
    result = TrustedSaver(store).commit_attempt(info, [candidate], succeeded=True)[0]
    assert (
        EditorService(store, result.ref).data_page(result.ref)["status"]
        == "unavailable"
    )


def test_volcano_template_records_transformed_y_and_filter():
    from lobster.core.figures.data_view import column_mapping
    from lobster.figure_templates import list_templates

    if "proteomics.volcano" not in {item["name"] for item in list_templates()}:
        pytest.skip("Optional proteomics templates are not installed")

    data = pd.DataFrame(
        {
            "protein": ["P1", "P2"],
            "comparison": ["treated", "treated"],
            "log2_fold_change": [-2.0, 1.5],
            "p_adjusted": [0.001, 0.1],
        }
    )
    fig = get_template("proteomics.volcano")(data, comparison="treated")
    mapping = column_mapping(json.loads(figure_to_bytes(fig)), list(data.columns))
    assert mapping is not None
    y = next(item for item in mapping["roles"] if item["role"] == "y")
    assert y["column"] == "p_adjusted" and "log10" in y["transform"]
    assert mapping["selection"]["equals"] == "treated"
    assert mapping["transformations"][0]["p_floor"] == 1e-300


def test_data_tab_is_read_only_paged_and_tracks_selected_revision(table_editor):
    playwright = pytest.importorskip("playwright.sync_api")
    store, ref, service = table_editor
    before = store.read(ref)
    session = EditorSession(service)
    try:
        with playwright.sync_playwright() as pw:
            try:
                browser = pw.chromium.launch(headless=True)
            except playwright.Error as exc:
                if "Executable doesn't exist" in str(exc):
                    pytest.skip("Install Playwright Chromium for browser checks")
                raise
            try:
                page = browser.new_page(viewport={"width": 1440, "height": 940})
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(session.url(ref))
                playwright.expect(page.locator("#status")).to_have_text(
                    "Revision loaded"
                )
                page.locator("#data-tab").click()
                playwright.expect(page.locator("#data-table tbody tr")).to_have_count(
                    100
                )
                playwright.expect(page.locator("#data-count")).to_have_text(
                    "205 rows · 5 columns"
                )
                playwright.expect(page.locator("#data-tab")).to_have_attribute(
                    "aria-selected", "true"
                )
                playwright.expect(page.locator("#figure-view")).to_be_hidden()
                assert page.locator("#data-table th.column-used").count() == 4
                assert page.locator("#data-table [data-role=unused]").count() == 1
                assert page.locator("#data-table script").count() == 0
                assert (
                    page.locator(
                        "#data-table input, #data-table [contenteditable]"
                    ).count()
                    == 0
                )
                playwright.expect(
                    page.locator("#data-table tbody tr").first
                ).to_contain_text("<script>not executable</script>")
                page.locator("#data-next").click()
                playwright.expect(page.locator("#data-range")).to_have_text(
                    "101–200 of 205 rows"
                )
                page.locator("#data-next").click()
                playwright.expect(page.locator("#data-table tbody tr")).to_have_count(5)
                playwright.expect(page.locator("#data-next")).to_be_disabled()
                page.locator("#data-previous").click()
                playwright.expect(page.locator("#data-range")).to_have_text(
                    "101–200 of 205 rows"
                )
                page.locator("#figure-tab").click()
                page.locator("#title").fill("Unsaved visual edit")
                page.locator("#data-tab").click()
                playwright.expect(page.locator("#data-notice")).to_contain_text(
                    "Unsaved figure edits"
                )
                playwright.expect(page.locator("#save")).to_be_disabled()
                playwright.expect(page.locator("#title")).to_have_attribute(
                    "readonly", ""
                )
                page.locator("#data-tab").press("ArrowLeft")
                playwright.expect(page.locator("#figure-tab")).to_have_attribute(
                    "aria-selected", "true"
                )
                playwright.expect(page.locator("#title")).to_have_value(
                    "Unsaved visual edit"
                )
                playwright.expect(page.locator("#save")).to_be_enabled()
                page.once("dialog", lambda dialog: dialog.accept())
                page.locator("#reset").click()
                playwright.expect(page.locator("#save")).to_be_disabled()
                # A new visual revision inherits the table. Polling follows it in Data view.
                page.locator("#data-tab").click()
                edited = service.edit_visuals(
                    ref,
                    generation=service.history()["generation"],
                    request_id="databrowser" * 3,
                    layout_updates={"title": {"text": "Saved edit"}},
                )
                playwright.expect(page.locator("#title")).to_have_value(
                    "Saved edit", timeout=12000
                )
                playwright.expect(page.locator("#data-table tbody tr")).to_have_count(
                    100
                )
                playwright.expect(page.locator("#data-tab")).to_have_attribute(
                    "aria-selected", "true"
                )
                page.locator("#history > button").first.click()
                playwright.expect(page.locator("#title")).to_have_value("Scatter")
                playwright.expect(page.locator("#data-table tbody tr")).to_have_count(
                    100
                )
                page.set_viewport_size({"width": 390, "height": 844})
                assert page.evaluate(
                    "document.documentElement.scrollWidth <= innerWidth"
                )
                page.screenshot(path="/tmp/lobster-data-tab-narrow.png")
                page.set_viewport_size({"width": 1440, "height": 940})
                page.screenshot(path="/tmp/lobster-data-tab.png")
                assert store.read(ref) == before
                assert (
                    service.data_page(edited["ref"])["columns"]
                    == service.data_page(ref)["columns"]
                )
                assert not errors
            finally:
                browser.close()
    finally:
        session.close()
