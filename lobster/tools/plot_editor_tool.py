"""Deterministic post-save plot selection and editor opening."""

import json


def plot_catalog(store):
    """One row per plot; latest means timeline tail, never a guessed filename."""
    rows, cursor = [], None
    while True:
        page = store.list(cursor=cursor)
        rows.extend(page.items)
        cursor = page.next_cursor
        if cursor is None:
            break
    result = []
    for fid in sorted(
        {r["figure_id"] for r in rows if r.get("availability") == "available"}
    ):
        history = store.history(fid)
        available = history["revisions"]
        if not available:
            continue
        first, latest = available[0], available[-1]
        result.append(
            {
                "figure_id": fid,
                "ref": latest["ref"],
                "original_ref": first["ref"],
                "label": latest.get("label") or first.get("label") or "Untitled plot",
                "aliases": sorted({r["label"] for r in available if r.get("label")}),
                "created_at": first["created_at"],
                "updated_at": max(r["created_at"] for r in available),
                "revision_count": len(available),
            }
        )
    return result


def resolve_plot(store, selector):
    """Return an exact revision or candidate rows. No fuzzy auto-selection."""
    from lobster.core.figures.ids import FigureRef

    catalog = plot_catalog(store)
    if "@" in selector:
        try:
            ref = str(FigureRef.parse(selector))
            store.read(ref)
            return ref, []
        except ValueError:
            return None, []
    selector = selector.strip().casefold()
    exact = [
        r
        for r in catalog
        if selector
        in [
            r["figure_id"].casefold(),
            r["label"].casefold(),
            *[a.casefold() for a in r["aliases"]],
        ]
    ]
    matches = exact or [
        r
        for r in catalog
        if any(selector in a.casefold() for a in [r["label"], *r["aliases"]])
    ]
    return (
        (matches[0]["ref"], []) if len(matches) == 1 and selector else (None, matches)
    )


def create_plot_editor_tools(data_manager):
    from langchain_core.tools import tool

    @tool
    def list_editable_figures() -> str:
        """List existing saved plots once each, with title, dates and revision count.

        For an interactive selection, call open_plot_editor with an empty selector
        after the user requests editing. Never invent refs or use export filenames.
        """
        return json.dumps(
            {"status": "listed", "plots": plot_catalog(data_manager.figure_store)}
        )

    @tool
    def open_plot_editor(figure_ref: str = "", user_approved: bool = False) -> str:
        """Open a saved plot: exact ref, figure ID or unique title; blank shows a picker.

        User explicitly requesting the GUI or selecting a plot to edit grants approval;
        do not ask again. A newly-created plot MUST use the exact saved ref returned
        by the plotting tool. Existing plot title/ID opens its latest revision.
        Ambiguous/blank selectors present a real selection widget. No match is an
        error with available plots; never substitute a guessed plot. PNG/HTML export
        paths are not editable refs. Report saved, launch_requested and
        editor_connected separately; launch acknowledgement is not a visible window.
        """
        if not user_approved:
            return json.dumps(
                {
                    "status": "approval_required",
                    "message": "Offer to open the saved plot editor.",
                }
            )
        from lobster.core.plot_editor import launch_editor
        from lobster.core.figures.errors import FigureError, safe_reason
        from langgraph.types import interrupt

        store = data_manager.figure_store
        try:
            ref, choices = resolve_plot(store, figure_ref)
            if not ref:
                if not choices:
                    return json.dumps(
                        {"status": "not_found", "plots": plot_catalog(store)}
                    )
                labels = [
                    f"{r['label']} · {r['figure_id']} · {r['revision_count']} revisions"
                    for r in choices
                ]
                response = interrupt(
                    {
                        "component": "select",
                        "data": {
                            "question": "Choose a plot to edit",
                            "options": labels,
                        },
                        "fallback_prompt": "Choose a plot to edit",
                    }
                )
                if not isinstance(response, dict) or response.get("action") in (
                    "cancel",
                    "error",
                ):
                    return json.dumps({"status": "cancelled"})
                selected = response.get("selected", response.get("value"))
                if selected not in labels:
                    return json.dumps(
                        {
                            "status": "cancelled",
                            "message": "No matching plot selection received.",
                        }
                    )
                ref = choices[labels.index(selected)]["ref"]
            store.read(ref)
            outcome = launch_editor(store, ref)
            # The server returns a structured, observed connection outcome.
            return json.dumps(
                {"ref": ref, **outcome}
                if isinstance(outcome, dict)
                else {"ref": ref, "status": "launch_requested", "mode": outcome}
            )
        except FigureError as exc:
            return json.dumps(
                {"status": "failed", "message": safe_reason(exc), "plot_saved": True}
            )
        except (ValueError, RuntimeError) as exc:
            return json.dumps(
                {"status": "failed", "message": str(exc), "plot_saved": True}
            )

    for fn in (list_editable_figures, open_plot_editor):
        fn.metadata = {"categories": ["UTILITY"], "provenance": False}
        fn.tags = ["UTILITY"]
    from lobster.tools._async_compat import enable_sync_fallback

    return [list_editable_figures, enable_sync_fallback(open_plot_editor)]


def create_figure_edit_tools(data_manager):
    """Chat styling shares the editor draft validation and trusted revision saver."""
    import secrets
    from langchain_core.tools import tool
    from lobster.core.plot_editor.service import EditorService
    from lobster.core.figures.errors import FigureError, safe_reason

    services = {}
    inspections = {}

    def service_for(ref):
        from lobster.core.figures.ids import FigureRef

        fid = FigureRef.parse(ref).figure_id
        if fid not in services:
            services[fid] = EditorService(data_manager.figure_store, ref)
        return services[fid]

    @tool
    def inspect_saved_figure(figure_ref: str, use_exact_revision: bool = False) -> str:
        """Read the LATEST saved plot, including GUI saves, before a visual edit.

        Returns figure JSON, exact ref, history generation and request_id for
        edit_saved_figure. Ambiguous selectors return choices: ask the user.
        Use this for styling existing plots; do not recreate them from source.
        An old ref from chat identifies the figure, not the version to edit.
        Only set use_exact_revision=True when the USER explicitly requests an
        older revision; it requires a full figure@revision reference.
        """
        try:
            from lobster.core.figures.ids import FigureRef

            if use_exact_revision and "@" not in figure_ref:
                raise ValueError(
                    "An explicit historical edit needs a full revision reference"
                )
            selector = (
                FigureRef.parse(figure_ref).figure_id
                if "@" in figure_ref and not use_exact_revision
                else figure_ref
            )
            # Resolve, read and capture the history generation together. An intervening
            # GUI save after inspection is rejected by the shared revision saver.
            with data_manager.figure_store.writer_lock:
                ref, choices = resolve_plot(data_manager.figure_store, selector)
                if ref is None:
                    return json.dumps(
                        {"status": "selection_required", "plots": choices}
                    )
                service = service_for(ref)
                loaded = service.load(ref)
                history = service.history()
                generation = history["generation"]
                request_id = secrets.token_hex(16)
                inspections[request_id] = (ref, generation, use_exact_revision)
                return json.dumps(
                    {
                        "status": "available",
                        "ref": ref,
                        "figure": loaded["figure"],
                        "generation": generation,
                        "request_id": request_id,
                        "latest_ref": service.figure_id + "@" + history["tail"],
                        "selection": (
                            "exact_revision" if use_exact_revision else "latest"
                        ),
                    }
                )
        except (FigureError, ValueError) as exc:
            return json.dumps(
                {
                    "status": "failed",
                    "message": (
                        safe_reason(exc) if isinstance(exc, FigureError) else str(exc)
                    ),
                }
            )

    @tool
    def edit_saved_figure(
        figure_ref: str,
        generation: int,
        request_id: str,
        layout_updates: dict,
        trace_updates: list[dict],
        explanation: str,
    ) -> str:
        """Save requested visual changes as a new revision of an existing saved plot.

        First inspect_saved_figure; copy its exact ref, generation and request_id.
        Always use the reference returned by inspection, never the older chat ref.
        layout_updates is a nested Plotly layout object (e.g. {"xaxis":{"title":{"text":""}}}).
        trace_updates contains {"index":0,"properties":{"marker":{"size":12}}} objects.
        Nested objects merge, arrays replace. Use empty {} / [] for untouched parts.
        Coordinates, point IDs, trace types and source tables cannot be changed.
        explanation describes the user's requested edit. This is the same save path
        as the GUI Save button; only status available confirms a saved revision.
        Retry pending saves with identical arguments. On stale history inspect again.
        Do not open the GUI unless requested. Do not regenerate or export the plot.
        """
        try:
            inspected = inspections.get(request_id)
            if inspected is None:
                raise ValueError(
                    "Inspect the saved figure first to obtain a request identifier"
                )
            if inspected[:2] != (figure_ref, generation):
                raise ValueError(
                    "Edit must use the inspected revision and history generation"
                )
            result = service_for(figure_ref).edit_visuals(
                figure_ref,
                generation=generation,
                request_id=request_id,
                layout_updates=layout_updates,
                trace_updates=trace_updates,
                explanation=explanation,
                require_latest=not inspected[2],
            )
            return json.dumps(result)
        except (FigureError, ValueError, TypeError) as exc:
            return json.dumps(
                {
                    "status": "failed",
                    "message": (
                        safe_reason(exc) if isinstance(exc, FigureError) else str(exc)
                    ),
                }
            )

    for fn in (inspect_saved_figure, edit_saved_figure):
        fn.metadata = {"categories": ["UTILITY"], "provenance": False}
        fn.tags = ["UTILITY"]
    return [inspect_saved_figure, edit_saved_figure]
