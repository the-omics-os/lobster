# Local Plotly editor

The editor opens an **already-saved Plotly revision**. Figure generation and the
existing plotting save path stay with Lobster's plotting tools. In chat:

> Create the scatter plot from data/synthetic_plot_points.csv, save it, and open it in the graphical editor.

For an existing plot, ask to open its graphical editor by title, figure ID, or exact
revision reference. A title/ID opens the timeline tail; an explicit revision opens
that revision. Ask "What plots exist?" to see one entry per plot with dates and
revision count, then ask to choose a plot to edit to open a selection widget.
Blank or ambiguous edit requests use the same picker; missing matches return the
available plots instead of guessing a reference. Explicitly requesting the GUI grants
launch approval; otherwise the supervisor offers it first. Keep Lobster running.
The editor loads the saved Plotly figure, not a PNG, and never needs Kaleido.

## Architecture and guarantees

- `service.py` accepts JSON drafts (`parent`, `figure`, `request_id`, `generation`)
  with optional `insert_after`, `explanation`, and related-data uploads. A generation
  mismatch refuses a stale save/delete; Refresh history preserves the current draft.
  The same validation/save interfaces can support future non-browser clients. It uses the
  production `FigureStore` and `TrustedSaver`. Verified parent config, label,
  retained source and inherited inputs are preserved. Requests are idempotent
  within the editor session; reuse of a request ID with a different draft is refused.
  The browser retains the full pending request across lost responses and history
  refreshes. Retrying confirms that request without duplicate revisions; visual
  edits made after an uncertain save remain unsaved until a subsequent save.
- `server.py` uses standard-library `ThreadingHTTPServer` on `127.0.0.1` and an
  OS-allocated port. API calls require a random bearer token. Exact Host/Origin
  checks, same-origin JSON POSTs, request caps and a restrictive CSP protect it.
  There are no filesystem, code-execution or arbitrary-file routes. Tokens travel
  in the initial URL fragment and then tab session storage, not server request logs.
- One server is reused per store/figure; separate tabs have independent working
  drafts. Heartbeats keep it alive. Five minutes without an authenticated request
  closes an idle server. Normal Lobster exit also closes servers; browser processes
  are left alone because Chromium may share a user's running process.
- `browser.py` discovers Chrome, Chromium, Edge, Brave and Vivaldi via PATH,
  macOS application bundles and Windows registry/installation roots. Under WSL,
  it also checks Windows browser installations under `/mnt/c/Program Files`
  and `/mnt/c/Program Files (x86)`. It tries `--app=<loopback URL>`, then a graphical
  browser tab, then reports failure. On Linux it invokes known graphical browsers
  directly; generic openers such as `xdg-open` are not accepted because they can
  silently fall back to a text browser that cannot run the editor.
  Launch acknowledgement is best effort, not proof that an OS displayed a window.
- HTML/CSS/vanilla JS ship as package data, with no frontend build. Plotly.js is
  served from the installed Plotly package. No Python dependencies are added.
- Plot creation keeps the existing plotting pipeline. The supervisor can inspect
  and style saved figures through `EditorService`; GUI launch remains optional
  and is reported separately from saving.

## Editing

Controls cover titles, Cartesian axes/ranges/scales, fonts and backgrounds,
trace visibility/opacity, supported markers/lines/color scales, legends, margins,
bar layout and annotations. Four presets modify the working figure: Original,
Publication, Minimal and Presentation. Reset restores the loaded revision.

Save becomes available after visual edits. Saving appends a revision to the
timeline tail, even when editing an older revision. The Properties panel contains
visual controls only; file downloads, uploads and save-position controls are not
shown. Refresh and Delete revision sit beside the timeline.
Revision cards contain only a revision title and date. The selected revision's
optional request/summary and property diff appear in a separate bounded area
below the timeline; missing descriptions are allowed.
Immutable `derived_from` and `parents` fields identify the revision actually edited;
timeline order is independent of ancestry.

Each figure has an atomically replaced `plots/_history/<figure_id>.json` index:
`head` is the original, `tail` is the end of the current timeline, and each node
contains `previous`/`next` links. The original is pinned. Deleting another revision
fences it from reads, reconnects both neighbors, and loads the previous available
revision. Deleted payloads/provenance and tombstones are retained; shared data is
not removed. A whole-workspace plot purge also clears history and attachment blobs.

Existing revision trees are indexed deterministically on first access without
rewriting payloads or ancestry. New revisions retain their insertion target so an
interrupted index publication can be reconciled on retry. Links publish together
via atomic replace and fsync under the store's workspace writer lock. As with the
existing local store, this coordinates tabs/threads in one Lobster process, not
independent Lobster processes writing the same workspace simultaneously.

New edited revisions record a factual JSON diff, readable summary, data-change
flag, and attachment/configuration changes. Large values are summarized by length
and hash. An optional user explanation is retained separately as the edit request.
Unsaved changes prompt before leaving or loading history. A stale save/delete
requires selecting a revision from the refreshed history before retrying.
Refreshing preserves the draft and its original generation; it never silently
rebases unsaved edits onto newer history. Load the latest revision and reapply
the desired changes to retain another editor's intervening updates.

## AI edits and retained data

The Figure / Data tabs switch between editing and a read-only spreadsheet view
of the selected revision's retained drawing table. Data is fetched in pages of
100 rows (API maximum 200), with sticky column headings and row numbers. Headers
show recorded X/Y/Z, color/group, size, identifier, label and filter roles. The
Data page has no cell editing, uploads or downloads; unsaved visual drafts are
preserved when switching tabs and are not represented as saved source data.

General table plotting and supported generic/proteomics volcano templates record
`layout.meta.lobster_data` (`lobster.plot_data/1`) during generation. This declared
metadata lists column roles, explicit transformations and selection/sampling,
and binds them to a digest of the plotted data arrays. General plotting also
retains it in the source declaration. Visual revisions preserve this information;
changed data arrays invalidate the binding. Templates with non-tabular inputs do
not claim column mappings. Custom drawing code may call
`lobster.core.figures.data_view.record_column_roles` before `submit_figure`.
Numeric bindings treat JSON-equivalent integral floats/integers and signed zero
as the same values. Existing raw-number bindings remain readable and are upgraded
on a new visual revision when the data and declared roles are unchanged; parent
revisions and retained source tables are never rewritten for this upgrade.
New bindings also verify per-point marker colors. Replacing a numeric color
encoding with a solid color invalidates the mapping, while palette-only changes
preserve it. Legacy color mappings lack that verification and are shown as
unavailable rather than trusted; their figures and source tables remain readable.

Older/custom figures without valid mapping metadata show the retained table with
"mapping unavailable"; the editor never matches values or guesses from axis labels.
Missing retained sources get an explicit empty state. These tables are saved
drawing snapshots, not live views of the uploaded workbook. No existing revision
or original data file is rewritten to add mappings retroactively.

`inspect_saved_figure` resolves a saved plot and returns its exact revision,
figure JSON, history generation and request ID. `edit_saved_figure` applies nested
visual layout/trace updates and calls the same `EditorService.save` and
`TrustedSaver` as the browser Save button. It records the assistant as the actor,
the user's request, a factual diff and the actual parent revision. Coordinates,
trace types, identities and retained data cannot be changed by this styling tool.
Ambiguous plots require clarification. Scientific recalculation continues through
the existing analysis agents and plot-generation tools.

AI inspection defaults to the current timeline tail even when chat supplies an
older revision reference. Only an explicitly requested historical edit uses
`use_exact_revision=True`. The edit request is bound to the inspected reference
and generation, and ordinary AI saves enforce a latest-parent check under the
store writer lock. Manual saves between inspection and editing require a fresh
inspection; retries of an already saved request remain idempotent. Unsaved GUI
drafts remain local to the browser and are not exposed to chat.

An open editor checks history every three seconds. When viewing the latest saved
revision without unsaved changes, it follows new revisions. Unsaved manual drafts
and deliberately loaded historical versions stay in place while history refreshes.
Stale writes are rejected using the existing history generation check.

Saved source data and content-addressed attachments remain in the production
store and are inherited from the edited revision. Removing file controls from
the GUI does not remove scientific data or alter the existing save infrastructure.

Launch results distinguish `launch_requested` from `editor_connected`. The latter
requires an authenticated figure fetch from the editor. Neither claims proof that
the operating system visibly displayed a window.

Entity controls appear only for 2D scatter/scattergl with unique string IDs in
`ids`, or a declared `meta.lobster_ids.customdata_index`, and matching x/y arrays.
Numeric Plotly typed arrays are supported. Arbitrary customdata/text are not
assumed to be identifiers. Search shows up to 60 matches. Highlighting is transient;
labels and hidden-marker opacity can be saved. Hiding markers does not remove data
or line segments. Restore uses opacity from the loaded revision. Ordinary table
plots without saved identity metadata do not show entity controls.

Future scatter plots made by `create_interactive_plot` or `generic.scatter`
automatically retain one valid conventional identity column (for example
`point_id`, `entity_id`, `gene_id`, or `sample_id`). Values must be present,
nonblank and unique across the input table before sampling. Several valid
candidates require an explicit `id_col`; `inspect_plot_data` lists those candidates.
An explicit custom-named identity column is also supported. Grouping and sampling
carry each point's ID through Plotly's row selection, without joining by x/y or
changing numeric measurement columns. IDs are saved as strings, included in data
role metadata, and inherited by later visual revisions. No row-number identities
are fabricated. Specialized templates/custom drawing code must continue declaring
their point identities; arbitrary scatter arrays cannot safely be matched back to
a source table. Existing saved figures are not rewritten by this change.

The editor has no image/JSON download section or Plotly camera-download button.
Frames and properties without a control remain in the saved figure JSON.
The preview scales proportionally to fit the canvas, including when loading AI
edits with fixed width/height or resizing the window, Properties panel or history
details. Fixed-size figures retain their composition, margins and label spacing;
figures without fixed dimensions use the available space with a minimum logical
canvas of 640×480. Rendering adjustments apply only to the display copy (including
animation frames), so viewport changes neither dirty the draft nor rewrite saved
document dimensions.

## Platform limits

This is a local desktop GUI. A remote server's browser/localhost is not the user's
computer; nontechnical hosted access needs a separate authenticated HTTPS hosting
integration. A usable modern graphical browser is required for local operation.
Windows/macOS/Linux browser detection is included, but the existing production
`LocalFigureStore` only supports POSIX filesystems. Windows needs a FigureStore
implementation that supports its platform; the editor does not bypass this guard.

Native Windows support should qualify atomic no-replace file publication, durable
file/directory updates and protection against symlinks/reparse points before
relaxing the store guard. Timeline publication must use the same storage seam,
and the figure-store protocol must expose history/deletion operations explicitly.
Then run storage recovery, deletion, HTTP-security and browser tests on a Windows
runner with a local NTFS workspace. Windows can store Plotly JSON; the current
limitation is this backend's qualified guarantees, not the figure format or web UI.

External map tiles, external images, CDN fonts/MathJax and other network-dependent
assets are blocked. Every Plotly trace can be retained, but only supported common
properties have controls in this initial editor. Unsaved drafts live in the tab;
a suspended computer or idle tab may need the editor reopened from Lobster.

## Verification

Follow-up checklist for the local editor:

- [x] Complete editor/backend type hints and pass focused mypy checks for
  `attachments.py`, `data_view.py`, `history.py`, `store.py`, `plot_editor/` and
  `plot_editor_tool.py` (`--follow-imports=silent --ignore-missing-imports`).
- [x] Verify all four editor tools with shared AQUADIF contract checks. They are
  `UTILITY` tools with `provenance=False`; analysis provenance IR is not required.
- [x] Verify persisted manual/assistant edits retain figure revision provenance:
  parent ancestry, actor, request, factual diff, retained source and execution record.
- [x] Run Black, isort (using the repository configuration), Ruff and focused
  editor, tabular-routing, save-reporting and local-store regression tests.
- [ ] Run local Playwright/Chromium regressions: four browser cases skipped in
  this environment because Playwright is not installed.
- [ ] Complete the native macOS visible-window smoke test described below.
- [x] Commit and push the follow-up changes.

The Figure lifecycle GitHub Actions workflow runs the storage/custom-code tests
and editor backend tests on `ubuntu-latest` and `macos-latest`. It installs
Playwright/Chromium as CI-only tools and runs the headless editor checks separately;
browser checks may not silently skip. Browser discovery tests simulate both macOS
application folders, paths with spaces, failed application launches and tab fallback.
These simulated tests also run locally on Linux.

Native macOS qualification still needs a visible-window smoke test: launch an
existing figure from chat, confirm a Chromium app window appears without tabs or
an address bar, edit/save a property, and reopen the resulting revision. Headless
CI validates the webpage and storage, not the operating system's visible app window.
Adding the CI matrix does not itself mean the macOS job has already passed.

The integration is exercised through a fresh Lobster chat using the production
workspace and tools. Browser verification checks loading, immediate edits,
presets, AI/manual saving, revision deletion, source integrity and HTTP security against the
actual launched session. A temporary Playwright/Chromium installation can drive
these checks; neither is a project/runtime requirement. No mock store is used by
the editor. Browser launch fallbacks and unavailable-revision validation can also
be exercised without changing the plotting pipeline.

## Interface design references

The editor uses [IBM Carbon's spacing system](https://www.carbondesignsystem.com/building-blocks/foundations/spacing/overview)
for consistent 4/8/12/16/24px spacing and clear separation between the workspace,
toolbar and inspector. [Adobe Spectrum's field-label guidance](https://spectrum.adobe.com/web/rsp/components/field-label)
and [accordion patterns](https://spectrum.adobe.com/web/swc/components/accordion)
inform the labeled controls and collapsible property groups. These are references,
not added component-library dependencies. Lobster's coral accent is used for the
primary action and loaded revision; the surrounding interface uses neutral surfaces.

Long text fields use top labels; compact numeric and selection controls use side
labels. Color fields pair a swatch with a hex value. Panel expansion is retained
while editing, the Properties panel can be hidden on smaller windows, and save
state is visible in the toolbar. Interface styling does not apply a plot preset
or rewrite saved figure styling automatically.
