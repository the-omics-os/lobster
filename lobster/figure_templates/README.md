# Figure templates

Templates are ordinary, versioned Python functions returning editable Plotly
figures. They are not model-facing tools and do not save files, modify datasets,
fit models, or rerun statistical tests.

## Discover and draw

```python
from lobster.figure_templates import get_template, list_templates, show_template_source

list_templates()                       # names, versions, purposes, signatures
show_template_source("proteomics.volcano")  # readable provider code and helpers

fig = get_template("proteomics.volcano@1")(
    retained_de_rows,
    comparison="treated_vs_control",
    fdr=0.05,
    log2fc_min=1.0,                     # LOG2 units, not a linear fold change
    boundary="strict",                 # p < FDR and abs(log2FC) > cutoff
)
fig.update_layout(title="Treated vs control")
fig.add_annotation(text="Custom annotation", x=0, y=1)
```

The volcano filters one explicit comparison and refuses duplicate protein IDs
within it. It changes only the classification shown in the figure, not the
stored DE results or downstream enrichment. `boundary="inclusive"` uses `<=`
for p and `>=` for absolute log2FC. Zero p-values use the explicitly recorded
`p_floor`. Source rows with nonfinite effects or invalid p-values are refused,
not silently removed.

Inside `execute_custom_code`, hand the result to the shared `submit_figure`
helper. Creating a Python variable is not submission. A source must be supplied
explicitly to retain regeneration data; submitting without one does not imply
that the figure can be regenerated. Saving and revision identity belong to the
trusted wrapper, not to a template.

## Inventory

- Core: `generic.scatter`, `line`, `bar`, `histogram`, `box`, `heatmap`,
  `correlation`.
- Proteomics: `proteomics.volcano`, `proteomics.pathway_bubble`.
- Drug discovery: `chem.similarity_heatmap` (a **precomputed** matrix).
- Transcriptomics: `sc.umap`, `violin`, `feature`, `dot`, `heatmap`, `elbow`,
  `cluster_composition`, `qc`.

Single-cell functions use an in-memory AnnData, its stored embedding, and
explicitly selected genes/groups/layers. The QC template requires metrics from
`assess_data_quality`; it never calculates missing metrics or writes `obs`.
Dot plots record the descriptive group means, percent expressing and optional
per-gene z-score settings. Aggregated marks do not claim single-cell selection.

Fingerprint analysis returns a workspace metadata result reference and key.
Read that retained result's `data.analysis.similarity_matrix` and
`data.analysis.compound_ids` for the chemistry template; do not rerun chemistry
just to draw it. The metadata file is mutable workspace content, **not** an
immutable scientific identity. Freeze the full payload explicitly on submission
when a regeneration source is needed.

## Style, identity and recipe

`initialize_house_style()` makes `omics_os` the Plotly default for custom code
while the default is still Plotly's stock `plotly`;
`apply_house_style(fig)` applies the same embedded style to an existing figure.
Every template applies it by default. Users may subsequently change any Plotly
property.

The style comes from the Omics-OS design tokens. `lobster.figure_templates._style`
exposes the values:

- `HOUSE_COLORS`: 14 category colours in order, graphite first; a second series
  is red. Above 14 categories they cycle. Never sample a value scale for
  categories.
- `SEQUENTIAL` (white to dark red, the heatmap, contour and 2D histogram
  default), `SEQUENTIAL_EXPRESSION` (starts at grey `#E0E0E0`; use it for
  marker colours, whose lowest values would vanish on white) and `DIVERGING`
  (green, white at 0.5, magenta; use it with zmid 0 and a symmetric range).
  `value_scale(values, unsigned=...)` picks by sign: DIVERGING and its
  half-range when any finite value is negative, else the unsigned scale.
  The template's own sequential default, which Plotly Express and plain
  markers use, is `SEQUENTIAL_EXPRESSION`; heatmap, contour and 2D histogram
  traces default to `SEQUENTIAL`.
- Roles: `INK`, `TICK`, `GRID`, `AXIS`, `NO_CALL`, `UP`, `DOWN` and `KEY`.

The template uses Manrope (falling back to Arial) at 14 in `INK`, a transparent
legend, colorbars without an outline and dashed `INK` shapes for thresholds.
Scatter and scattergl traces cycle marker symbols and line dashes in step with
the colours. In scatter charts, up to 14 series differ by marker symbol. Line
dashes are the 6 named ones (WebGL accepts no others): the first 6 series
differ by dash, and green lines have a dash no other colour uses.

Lobster's visualization tools, plot tools and templates apply these rules:
the symbol and dash cycles (except the volcano template, which uses circles;
the side of 0 separates the series), signed values diverging around 0, and
marker colours starting at grey. Hand-written
figures get the template defaults (font, colours, colorway, scales and, for
`go.Scatter` traces, the symbol and dash cycles). Plotly Express pins every
trace to circle markers and solid lines unless you pass `symbol=` (scatter) or
`line_dash=` (line) on the same column as `color=`; for signed values pass
`DIVERGING` with `zmid=0` and a symmetric range.

Templates record their name, version and resolved data-free arguments in
`layout.meta.lobster_template`. This is **declared** recipe metadata, not proof
that the figure was produced by that template. Settings must be finite JSON and
fit within 16 KiB; oversized settings are refused rather than truncated. Data
belongs in the source, not template arguments. A template version pins its
arguments and data semantics, not its style.

Trace `uid`s are stable by semantic class/gene/panel. Individual cell/protein
labels are strings in `customdata[:, 0]`; their entity kind/comparison are in
`trace.meta.lobster_ids`. Their source namespace comes from the saved revision,
not a guessed dataset name or a row position. Without a frozen source they are
labels only, not bound scientific selections. Matrix and aggregate marks have
no single-entity binding.

## Add a provider

A package may ship a uniquely named module beneath the PEP 420 namespace
`lobster.figure_template_providers`. Do not add `__init__.py` there. Expose a
cheap `TEMPLATE_MANIFEST` of descriptors with `name`, `version`, `purpose` and
`callable` (function name), and decorate each callable with `figure_template`.
Keep heavy imports inside drawing functions; discovery should not load a
scientific stack. The decorator's default data argument is `data`; additional
input arrays must be declared with `data_args=(...)` so they are not recorded as
settings.

Each provider needs fixture tests for Plotly validity, input immutability,
style, stable identities and explicit transform/threshold settings. Source code
and a modified returned figure remain available for designs beyond the template.
