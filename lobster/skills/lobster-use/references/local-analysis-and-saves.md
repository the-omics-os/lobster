# Local Analysis Example — After Approved Setup

The examples below assume `lobster` resolves to Python. In the `.venv` above,
replace it with `.venv/bin/python -m lobster` (or use the activated environment).
Downloads and paid inference require their own permission, not just setup consent.

```bash
# Teaching dataset; download and inference also require permission
WORKSPACE="./pbmc3k"
mkdir -p "$WORKSPACE/data"
curl -fL -o "$WORKSPACE/data/pbmc3k.tar.gz" \
  https://cf.10xgenomics.com/samples/cell/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz
tar -xzf "$WORKSPACE/data/pbmc3k.tar.gz" -C "$WORKSPACE/data"
WORKSPACE="$(cd "$WORKSPACE" && pwd)"

# Load and assess in ONE query; do not assume a later query retains QC state
lobster query -w "$WORKSPACE" --session-id pbmc3k --json \
  "Load the 10x matrix in $WORKSPACE/data/filtered_gene_bc_matrices/hg19
   (matrix.mtx, genes.tsv, barcodes.tsv; human PBMCs, hg19) as pbmc3k.
   Use gene symbols as var names. Report the loaded shape, then assess quality:
   per-cell genes, total counts and percent mitochondrial (MT- genes), with
   medians and 95th percentiles from tools. Do not filter or run DE."
lobster command "workspace list" -w "$WORKSPACE" --session-id pbmc3k --json
```

PBMC 3k's local 1.2.1 load was checked at **2,700 cells × 32,738 genes**. This
one-query load/QC request is an illustrative agent-drivable combination, not an
additional executed acceptance run. The checked 1.2.1 workflow used separate
requests in one human chat, explicit `/save` and a restart check;
that evidence is not query-to-query memory persistence or population-level DE.

`-w` names the workspace folder; given a project folder, Lobster uses the
project's `.lobster_workspace`. Do not append another `.lobster_workspace` to a
workspace path when inspecting outputs. Without an override, the local
workspace defaults to
`./.lobster_workspace` (or `LOBSTER_WORKSPACE`).

### Local Session History Is Not a Data Save

Each `lobster query` invocation starts a new client. Reusing the workspace and
`--session-id` can restore conversation history, but session JSON does not carry
the previous process's in-memory modalities into the next command. A completed
query, a session JSON file or an answer written with `--output` is not proof that
analysis data was saved.

**Saved inputs:** local startup scans `<workspace>/data/` for supported files;
size/read guards can prevent loading. A later query can load saved `*_raw.h5ad`
or DE-result H5ADs under their **file-stem names**, not unsaved QC-assessed state.
Inspect current files and modalities. When an input is absent, ask the Data Expert
to verify its file path/adapter and call `load_modality` **within the query that
uses it**. A separate inspection/load command does not keep a live client for
the next query.

**Query-mode saves:** some tools write files during a query. GEO imports can write
`*_raw.h5ad` and call autosave at import time; that is not a universal save of
changes made later in the query. `assess_data_quality` creates a QC-assessed
modality in memory without saving it. `run_de_with_formula(..., save_results=True)`
can write its result table and H5AD modality. `run_bulk_de_direct(..., save_result=True)`
also has a native save option, but its default marker-style out-group filter can
remove most or all bulk genes; prefer the formula tool for an approved bulk design.
`export_de_results` separately writes a table. These saves do not promise every
intermediate or every analysis tool's output was written.

**Local H5AD save outcomes:** a failure before the new file replaces an existing
file keeps the previous file. If replacement completed but save bookkeeping fails,
the error says the data was saved and bookkeeping failed; inspect the reported
file before retrying. This does not save other memory-only outputs.

**Coding-agent path:** use one non-interactive query for dependent steps with
explicit supported tool-save options, or load saved inputs within each new query.
See the [agent-drivable bulk example](agent-patterns.md#count-safe-bulk-rna-workflow).
Inspect save errors, returned paths, files and loaded data before another step;
report any memory-only outputs. A partial save is not complete durability.

**Human/interactive option:** keep dependent steps in one `lobster chat` session
and run `/save` before ending it. Inspect the save results and output files; after
a restart, check startup-loaded modalities or explicitly load the saved file
before continuing. Do not use a TUI as a required coding-agent interface.

For other datasets, inspect identity, files and supported representation before
approving a download or QC. Some text-only GEO series, including GSE109564,
are not selected by GEO download and their compressed text can be refused by the
single-cell loader. Do not claim they load or silently switch studies. See the
[1.2.1 known limitations](https://github.com/the-omics-os/lobster/releases/tag/v1.2.1#known-limitations).
