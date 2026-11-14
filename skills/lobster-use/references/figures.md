# Figures: Save, Then Edit the Saved Revision

These figure rules describe the **local 1.2 engine**, not Cloud feature availability.

- Run `assess_data_quality` before `create_qc_plots`, then draw the full QC-assessed
  modality, not a DE/small-gene subset. Drawing does not compute or repair QC metrics.
- Templates draw from retained results; they do not rerun scientific analyses or
  save a figure. Discover with `list_templates()`, `get_template(name)` and
  `show_template_source(name)` inside `execute_custom_code`.
- A Python `fig` variable is **not saved automatically**. Call
  `submit_figure(fig, source=rows, intent="what it shows", label="...")` for each
  figure to retain. `publish_plot(fig, title)` is a compatibility alias without
  an explicit regeneration source.
- Figures are submitted only for saving after successful code execution. Check
  the tool's save outcome: code success alone is not proof that a figure saved.
  `Save confirmation pending.` means confirmation is uncertain, not a definite
  absence; do not report it as a completed save.
- `save_plot=False` on plotting tools is preview-only, not a saved revision.
- For an edit request, load the literal saved ref with
  `load_figure("<figure_id>@<revision_id>")`, change it, then submit with that
  same ref as `parent`. This retains the original and creates a new revision.
  Restyling or filtering an existing figure is an edit, not a new figure.
- An edit inherits the parent's saved Plotly config unless `config` is supplied;
  `config={}` explicitly clears it. To retain regeneration data, supply `source`
  explicitly; neither a filename nor a variable name establishes source identity.
- Interactive `/plots` lists saved figures; `/plot <identifier>` opens one.
  Programmatic listing is `lobster command plots --json` with workspace/session.

See [CLI reference](cli-reference.md) and
[orchestration patterns](agent-patterns.md) for examples.
