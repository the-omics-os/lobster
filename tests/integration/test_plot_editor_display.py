"""Optional browser regressions; Playwright/Chromium are test tools, not dependencies."""

import json

import pytest

from lobster.core.figures.records import Candidate, ExecutionInfo
from lobster.core.figures.saver import TrustedSaver
from lobster.core.figures.store import LocalFigureStore
from lobster.core.plot_editor.server import EditorSession
from lobster.core.plot_editor.service import EditorService

pytestmark = [pytest.mark.integration, pytest.mark.no_auto_config]


def test_fixed_size_revisions_fit_and_cards_show_only_title_and_date(tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    store = LocalFigureStore(tmp_path)
    figure = {
        "data": [
            {
                "type": "scatter",
                "mode": "markers",
                "marker": {"size": 14},
                "x": [1, 1.5, 2],
                "y": [3, 3.5, 4],
            }
        ],
        "layout": {
            "title": {"text": "Original"},
            "width": 2600,
            "height": 1800,
            "autosize": False,
            "margin": {"l": 600, "r": 600, "t": 600, "b": 600},
        },
    }
    candidate = Candidate(
        attempt_key="display:test:0",
        candidate_index=0,
        figure_bytes=json.dumps(figure).encode(),
        config_bytes=b"{}",
        label="Original",
    )
    info = ExecutionInfo(
        "display:test", "test", None, {}, True, None, None, "test", "test"
    )
    result = TrustedSaver(store).commit_attempt(info, [candidate], succeeded=True)[0]
    assert result.status == "available"
    service = EditorService(store, result.ref)
    session = EditorSession(service)

    def fits(page):
        page.wait_for_function("""() => {
          const plot=document.getElementById('plot'), surface=plot.parentElement;
          const svg=plot.querySelector('.svg-container')?.getBoundingClientRect();
          const bounds=surface.getBoundingClientRect();
          return plot._fullLayout && svg &&
                 svg.left>=bounds.left && svg.top>=bounds.top &&
                 svg.right<=bounds.right && svg.bottom<=bounds.bottom &&
                 document.documentElement.scrollWidth<=innerWidth;
        }""")
        assert page.evaluate("""() => {
          const surface=document.querySelector('.plot-surface').getBoundingClientRect();
          const svg=document.querySelector('#plot .svg-container').getBoundingClientRect();
          return svg.width<=surface.width && svg.height<=surface.height &&
                 document.documentElement.scrollWidth<=innerWidth;
        }""")

    try:
        with playwright.sync_playwright() as pw:
            try:
                browser = pw.chromium.launch(headless=True)
            except playwright.Error as exc:
                if "Executable doesn't exist" in str(exc):
                    pytest.skip(
                        "Install Playwright Chromium to run browser regressions"
                    )
                raise
            try:
                page = browser.new_page(viewport={"width": 1440, "height": 940})
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(session.url(result.ref))
                playwright.expect(page.locator("#status")).to_have_text(
                    "Revision loaded"
                )
                for width, height in [(1440, 940), (1024, 768), (390, 844)]:
                    page.set_viewport_size({"width": width, "height": height})
                    fits(page)
                page.set_viewport_size({"width": 1440, "height": 940})
                for _ in range(2):
                    page.locator("#properties-toggle").click()
                    fits(page)
                playwright.expect(page.locator("#save")).to_be_disabled()
                description = "Requested styling change " * 60
                edited = service.edit_visuals(
                    result.ref,
                    generation=service.history()["generation"],
                    request_id="displaytest" * 3,
                    layout_updates={
                        "title": {"text": "AI edit"},
                        "width": 1100,
                        "height": 760,
                    },
                    explanation=description,
                )
                assert edited["status"] == "available"
                playwright.expect(page.locator("#title")).to_have_value(
                    "AI edit", timeout=12000
                )
                fits(page)
                cards = page.locator("#history > button")
                playwright.expect(cards).to_have_count(2)
                assert "Requested styling" not in page.locator("#history").inner_text()
                assert cards.last.locator(":scope > *").count() == 2
                assert cards.last.locator("small").count() == 1
                assert cards.last.bounding_box()["width"] <= 165
                playwright.expect(page.locator("#revision-change-text")).to_have_text(
                    description.strip()
                )
                page.locator("#revision-change-detail summary").click()
                playwright.expect(
                    page.locator("#revision-change-properties")
                ).to_contain_text("/layout/title/text")
                fits(page)
                # Save a manual edit: display fitting must never leak into figure JSON.
                page.locator("#title").fill("Manual edit")
                page.locator("#save").click()
                playwright.expect(page.locator("#status")).to_have_text(
                    "Revision saved"
                )
                saved = service.load(service.history()["revisions"][-1]["ref"])[
                    "figure"
                ]
                assert saved["layout"]["width"] == 1100
                assert saved["layout"]["height"] == 760
                assert saved["layout"]["margin"]["l"] == 600
                fits(page)
                page.locator("#history > button").first.click()
                playwright.expect(page.locator("#title")).to_have_value("Original")
                playwright.expect(page.locator("#status")).to_have_text(
                    "Revision loaded"
                )
                fits(page)
                page.wait_for_function(
                    "document.getElementById('plot')._fullLayout.width === 2600"
                )
                # Plotly's drag overlay intentionally intercepts point events.
                # Move a real pointer to the rendered point instead of forcing
                # a locator action through that overlay during a redraw.
                bounds = (
                    page.locator("#plot .scatterlayer .point").nth(1).bounding_box()
                )
                assert bounds is not None
                page.mouse.move(
                    bounds["x"] + bounds["width"] / 2,
                    bounds["y"] + bounds["height"] / 2,
                )
                playwright.expect(
                    page.locator("#plot .hoverlayer .hovertext")
                ).not_to_have_count(0)
                playwright.expect(page.locator("#save")).to_be_disabled()
                assert not errors
            finally:
                browser.close()
    finally:
        session.close()


@pytest.mark.parametrize("edit_during_retry", [False, True])
def test_lost_save_response_retries_exact_draft_after_history_refresh(
    tmp_path, edit_during_retry
):
    playwright = pytest.importorskip("playwright.sync_api")
    store = LocalFigureStore(tmp_path)
    candidate = Candidate(
        attempt_key="retry:browser:0",
        candidate_index=0,
        figure_bytes=b'{"data":[{"type":"scatter","x":[1,2],"y":[3,4]}],"layout":{"title":{"text":"Original"}}}',
        config_bytes=b"{}",
    )
    info = ExecutionInfo(
        "retry:browser", "test", None, {}, True, None, None, "test", "test"
    )
    root = TrustedSaver(store).commit_attempt(info, [candidate], succeeded=True)[0].ref
    service = EditorService(store, root)
    before = store.read(root)
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
                page = browser.new_page()
                page.goto(session.url(root))
                playwright.expect(page.locator("#status")).to_have_text(
                    "Revision loaded"
                )
                bodies = []

                def lose_first_ack(route):
                    bodies.append(route.request.post_data_json)
                    if len(bodies) == 1:
                        # The real production saver commits; only its HTTP ACK is lost.
                        response = route.fetch()
                        assert response.ok
                        route.abort("failed")
                    else:
                        route.continue_()

                page.route("**/api/save", lose_first_ack)
                page.locator("#title").fill("Saved draft")
                page.locator("#save").click()
                playwright.expect(page.locator("#status")).to_contain_text(
                    "Retry Save revision"
                )
                playwright.expect(page.locator("#history > button")).to_have_count(
                    2, timeout=12000
                )
                # Even an explicit refresh must not alter a pending request's generation/ID.
                page.locator("#refresh-history").click()
                playwright.expect(page.locator("#status")).to_contain_text(
                    "History refreshed"
                )
                if edit_during_retry:
                    page.locator("#title").fill("Newer draft")
                page.locator("#save").click()
                if edit_during_retry:
                    playwright.expect(page.locator("#status")).to_contain_text(
                        "Newer edits are still unsaved"
                    )
                    playwright.expect(page.locator("#save")).to_be_enabled()
                    playwright.expect(page.locator("#title")).to_have_value(
                        "Newer draft"
                    )
                else:
                    playwright.expect(page.locator("#status")).to_have_text(
                        "Revision saved"
                    )
                    playwright.expect(page.locator("#save")).to_be_disabled()
                assert bodies[0] == bodies[1]
                rows = service.history()["revisions"]
                assert len(rows) == 2
                saved_ref = rows[-1]["ref"]
                assert (
                    service.load(saved_ref)["figure"]["layout"]["title"]["text"]
                    == "Saved draft"
                )
                if edit_during_retry:
                    page.locator("#save").click()
                    playwright.expect(page.locator("#status")).to_have_text(
                        "Revision saved"
                    )
                    rows = service.history()["revisions"]
                    assert len(rows) == 3
                    assert (
                        store.read_record(rows[-1]["ref"])["derived_from"] == saved_ref
                    )
                    assert (
                        service.load(rows[-1]["ref"])["figure"]["layout"]["title"][
                            "text"
                        ]
                        == "Newer draft"
                    )
                assert store.read(root) == before
            finally:
                browser.close()
    finally:
        session.close()
