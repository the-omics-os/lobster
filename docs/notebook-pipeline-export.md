# Jupyter Notebook Export & Pipeline Replay

**Version:** 2.0
**Status:** ✅ Production Ready
**Added:** October 2025

## Overview

Lobster's **Jupyter Notebook Export** feature transforms your interactive analysis sessions into reproducible, executable Jupyter notebooks. This enables:

- 📓 **Reproducibility**: Every analysis can be replayed with new data
- 🔄 **Shareability**: Distribute workflows via Git as standard .ipynb files
- 🧪 **Validation**: Test pipelines on different datasets before production
- 📚 **Documentation**: Self-documenting analysis with embedded provenance
- 🚀 **Zero Infrastructure Cost**: No custom pipeline engines needed

## Key Features

### ✨ Core Capabilities

- **Provenance-to-Code**: Converts W3C-PROV activity records to executable code
- **Tool Mapping**: Maps 10+ Lobster tools to standard library code (scanpy, pyDESeq2)
- **Papermill Integration**: Parameter injection for batch execution
- **Schema Validation**: Pre-execution checks for data compatibility
- **Dry Run**: Simulate execution without running
- **Partial Preservation**: Save results even if execution fails
- **Version Tracking**: Captures dependency versions for reproducibility

### 🎯 Supported Operations

Currently supports exporting these analysis types:
- Single-cell RNA-seq QC & clustering
- Bulk RNA-seq differential expression
- Data filtering & normalization
- Dimensionality reduction (PCA, UMAP)
- Marker gene identification
- Pseudobulk aggregation

## Quick Start

### 1. Perform Analysis

```bash
lobster chat
> Load dataset GSE12345 from GEO
> Perform quality control
> Filter low-quality cells
> Normalize and identify highly variable genes
> Cluster cells and generate UMAP
```

### 2. Export as Notebook

```bash
> /pipeline export
Notebook name: qc_clustering_workflow
Description: Standard QC and clustering for 10X data

✓ Notebook exported: ~/.lobster/notebooks/qc_clustering_workflow.ipynb

Next steps:
  1. Review:  jupyter notebook qc_clustering_workflow.ipynb
  2. Commit:  git add ~/.lobster/notebooks/qc_clustering_workflow.ipynb
  3. Run:     /pipeline run qc_clustering_workflow.ipynb <modality>
```

### 3. Review & Share

```bash
# Open in Jupyter
jupyter notebook ~/.lobster/notebooks/qc_clustering_workflow.ipynb

# Commit to Git
git add ~/.lobster/notebooks/qc_clustering_workflow.ipynb
git commit -m "Add QC clustering workflow"
git push
```

### 4. Execute on New Data

```bash
lobster chat
> /read new_dataset.h5ad
> /pipeline run qc_clustering_workflow.ipynb new_dataset

Running validation...
✓ Validation passed
  Steps to execute: 10
  Estimated time: 20 min

Execute notebook? [y/n]: y

Executing notebook...
✓ Execution complete!
  Output: qc_clustering_workflow_output.ipynb
  Duration: 18.3s
```

## CLI Commands

### `/pipeline export`

Export current session as Jupyter notebook.

**Usage:**
```bash
/pipeline export
```

**Interactive Prompts:**
- Notebook name (no extension)
- Description (optional)

**Options:**
- Filter strategy: `successful` (default), `all`, or `manual`

**Output:**
- Notebook saved to `~/.lobster/notebooks/<name>.ipynb`
- Includes Papermill-compatible parameters
- Embeds complete provenance metadata

---

### `/pipeline list`

List all available notebooks.

**Usage:**
```bash
/pipeline list
```

**Output:**
```
┌─────────────────────┬───────┬────────────┬────────────┬─────────┐
│ Name                │ Steps │ Created By │ Created    │ Size    │
├─────────────────────┼───────┼────────────┼────────────┼─────────┤
│ qc_clustering       │   10  │ researcher │ 2025-10-27 │ 15.3 KB │
│ bulk_de_analysis    │    8  │ researcher │ 2025-10-26 │ 12.1 KB │
└─────────────────────┴───────┴────────────┴────────────┴─────────┘
```

---

### `/pipeline run <notebook> <modality>`

Execute notebook with validation.

**Usage:**
```bash
# Interactive (prompts for notebook & modality)
/pipeline run

# Direct execution
/pipeline run qc_clustering.ipynb my_dataset
```

**Workflow:**
1. **Validation**: Checks data shape, required columns, data type
2. **Dry Run**: Shows steps, estimated time, warnings
3. **Confirmation**: User approves execution
4. **Execution**: Runs with Papermill, preserves partial results
5. **Report**: Shows output notebook path, duration, status

**Parameters:**
- Custom parameters can be specified in the notebook's `parameters` cell
- Override with Papermill: `papermill notebook.ipynb out.ipynb -p param value`

---

### `/pipeline info`

Show detailed notebook metadata.

**Usage:**
```bash
/pipeline info
```

**Shows:**
- Creator & creation date
- Lobster version
- Dependency versions
- Number of steps
- File size
- Provenance summary

---

## Notebook Structure

Generated notebooks follow this structure:

```python
# ============================================
# Header (Markdown)
# ============================================
# Workflow name, description, metadata

# ============================================
# Parameters Cell (Tagged for Papermill)
# ============================================
input_data = "dataset.h5ad"
output_prefix = "results"
random_seed = 42
min_genes = 200
min_cells = 3
max_mito_percent = 20.0

# ============================================
# Step 1: Quality Control
# ============================================
import scanpy as sc
adata = sc.read_h5ad(input_data)

sc.pp.calculate_qc_metrics(
    adata,
    qc_vars=['mt', 'ribo'],
    percent_top=None,
    log1p=False,
    inplace=True
)

print(f"Cells: {adata.n_obs}, Genes: {adata.n_vars}")

# ============================================
# Step 2: Filter Cells
# ============================================
print(f"Before filtering: {adata.n_obs} cells")

sc.pp.filter_cells(adata, min_genes=min_genes)
sc.pp.filter_genes(adata, min_cells=min_cells)
adata = adata[adata.obs['pct_counts_mt'] < max_mito_percent].copy()

print(f"After filtering: {adata.n_obs} cells")

# ... more steps ...

# ============================================
# Footer (Markdown)
# ============================================
# Results export, usage instructions, Git workflow
```

## Tool Mapping

Lobster tools are mapped to standard library code:

| Lobster Tool | Maps To | Library |
|--------------|---------|---------|
| `quality_control` | `scanpy.pp.calculate_qc_metrics` | scanpy |
| `filter_cells` | `scanpy.pp.filter_cells/genes` | scanpy |
| `normalize` | `scanpy.pp.normalize_total` + `log1p` | scanpy |
| `highly_variable_genes` | `scanpy.pp.highly_variable_genes` | scanpy |
| `pca` | `scanpy.tl.pca` | scanpy |
| `neighbors` | `scanpy.pp.neighbors` | scanpy |
| `cluster` | `scanpy.tl.leiden` | scanpy |
| `umap` | `scanpy.tl.umap` | scanpy |
| `find_markers` | `scanpy.tl.rank_genes_groups` | scanpy |
| `differential_expression` | pyDESeq2 workflow | pyDESeq2 |

**Unmapped Tools:** Tools without mappings get placeholders with `# TODO: Manual review` comments.

## Advanced Usage

### Custom Parameters

Modify the `parameters` cell in your notebook:

```python
# Parameters (tagged for Papermill)
input_data = "dataset.h5ad"
output_prefix = "results"
random_seed = 42

# Custom parameters
n_top_genes = 3000  # Increased from default 2000
leiden_resolution = 0.8  # Higher resolution for finer clusters
```

### Batch Execution

Use Papermill for batch processing:

```bash
# Run on multiple datasets
for dataset in data/*.h5ad; do
    papermill workflow.ipynb "output_$(basename $dataset .h5ad).ipynb" \
        -p input_data "$dataset" \
        -p output_prefix "batch_$(date +%Y%m%d)"
done
```

### Filter Strategies

Control which activities are exported:

```python
# Python API
dm.export_notebook(
    name="workflow",
    filter_strategy="successful"  # "successful" | "all" | "manual"
)
```

- **`successful`** (default): Exports only successful operations
- **`all`**: Exports all activities including failed ones
- **`manual`**: Reserved for future manual selection UI

### Validation & Dry Run

Validate before execution:

```python
# Python API
result = dm.run_notebook(
    "workflow.ipynb",
    "my_dataset",
    dry_run=True
)

print(f"Steps: {result['steps_to_execute']}")
print(f"Estimated time: {result['estimated_duration_minutes']} min")
print(f"Validation: {result['validation']}")

# Check warnings
if result['validation'].has_warnings:
    for warning in result['validation'].warnings:
        print(f"⚠️  {warning}")
```

### Error Handling

Notebooks preserve partial results on failure:

```python
try:
    result = dm.run_notebook("workflow.ipynb", "dataset")
    if result['status'] == 'success':
        print(f"✓ Complete: {result['output_notebook']}")
    else:
        print(f"✗ Failed: {result['error']}")
        print(f"Partial results: {result.get('output_notebook')}")
except Exception as e:
    print(f"Execution error: {e}")
```

## Python API

### DataManagerV2 Methods

```python
from lobster.core.data_manager_v2 import DataManagerV2

dm = DataManagerV2(enable_provenance=True)

# Export notebook
path = dm.export_notebook(
    name="my_workflow",
    description="Custom analysis pipeline",
    filter_strategy="successful"
)

# List notebooks
notebooks = dm.list_notebooks()
for nb in notebooks:
    print(f"{nb['name']}: {nb['n_steps']} steps, {nb['size_kb']:.1f} KB")

# Run notebook
result = dm.run_notebook(
    notebook_path="my_workflow.ipynb",
    input_modality="dataset_name",
    parameters={"random_seed": 123},
    dry_run=False
)
```

### NotebookExporter API

```python
from lobster.core.notebook_exporter import NotebookExporter
from lobster.core.provenance import ProvenanceTracker

provenance = ProvenanceTracker()
exporter = NotebookExporter(provenance, dm)

# Export
path = exporter.export(
    name="workflow",
    description="Analysis pipeline",
    filter_strategy="successful"
)

# Access metadata
metadata = exporter._create_metadata()
print(f"Dependencies: {metadata['dependencies']}")
```

### NotebookExecutor API

```python
from lobster.core.notebook_executor import NotebookExecutor, ValidationResult

executor = NotebookExecutor(dm)

# Validate
validation = executor.validate_input(notebook_path, input_data_path)
if validation.has_errors:
    for error in validation.errors:
        print(f"✗ {error}")

# Dry run
result = executor.dry_run(notebook_path, input_data_path)
print(f"Steps: {result['steps_to_execute']}")
print(f"Time: {result['estimated_duration_minutes']} min")

# Execute
result = executor.execute(
    notebook_path=notebook_path,
    input_data=input_data_path,
    parameters={"random_seed": 42},
    output_path=custom_output_path
)
```

## Best Practices

### 📝 Naming Conventions

- Use descriptive, lowercase names with underscores
- Include analysis type: `qc_clustering_workflow`, `bulk_de_analysis`
- Version if iterating: `qc_workflow_v2`, `de_analysis_robust`

### 🔖 Git Workflow

```bash
# Store notebooks in dedicated directory
mkdir -p notebooks/
git add notebooks/*.ipynb

# Use meaningful commit messages
git commit -m "Add QC clustering workflow for 10X data"

# Tag stable versions
git tag -a v1.0-qc-workflow -m "Production QC workflow"
git push --tags
```

### ✅ Testing Notebooks

1. **Validate First**: Always run dry_run before execution
2. **Test on Subset**: Run on small datasets first
3. **Check Outputs**: Verify output notebook has expected results
4. **Document Parameters**: Comment parameter choices in notebook
5. **Version Control**: Track notebook changes in Git

### 🔒 Security

- **Review Code**: Inspect generated notebooks before execution
- **Validate Inputs**: Check data shape/schema before running
- **Sandbox Execution**: Run notebooks in isolated environments
- **Secret Management**: Never commit notebooks with API keys

### 🎯 Performance

- **Batch Processing**: Use Papermill for multiple datasets
- **Resource Limits**: Set memory/CPU limits for execution
- **Checkpoints**: Add save points for long workflows
- **Parallelization**: Use `n_jobs` parameter in scanpy operations

## Troubleshooting

### Common Issues

**Issue:** `Provenance tracking disabled - cannot export notebook`

**Solution:** Enable provenance when creating DataManagerV2:
```python
dm = DataManagerV2(enable_provenance=True)
```

---

**Issue:** `No activities recorded - nothing to export`

**Solution:** Perform analysis before exporting:
```bash
> Load data and perform at least one operation before /pipeline export
```

---

**Issue:** `Validation failed: Missing required columns`

**Solution:** Check data schema matches notebook expectations:
```python
result = dm.run_notebook("notebook.ipynb", "dataset", dry_run=True)
print(result['validation'].errors)
```

---

**Issue:** `Papermill not installed`

**Solution:** Install dependencies:
```bash
pip install nbformat papermill nbconvert jupytext
# OR
make dev-install
```

---

**Issue:** `FileNotFoundError: Notebook not found`

**Solution:** Check notebook location:
```bash
ls ~/.lobster/notebooks/
/pipeline list
```

---

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now run export/execution
dm.export_notebook("debug_workflow")
```

## Limitations & Future Work

### Current Limitations

- **Tool Coverage**: Only 10 core tools mapped (expanding to 20+)
- **Proteomics**: Limited proteomics tool mappings
- **Complex Formulas**: R-style formulas may need manual adjustment
- **Custom Code**: User-defined functions not captured
- **Interactive Plots**: Plotly plots exported as static images

### Roadmap

#### Near-Term (Month 2)
- [ ] Expand tool mapping to 20+ tools
- [ ] Add proteomics-specific mappings
- [ ] Notebook templates library
- [ ] Parameter tuning suggestions
- [ ] HTML export via nbconvert

#### Medium-Term (Quarter 2)
- [ ] GUI notebook editor integration
- [ ] Advanced validation (statistical checks)
- [ ] Notebook composition (chaining)
- [ ] Cloud execution runners (AWS Batch)

#### Long-Term (Quarter 3+)
- [ ] Marketplace for shared notebooks
- [ ] AI-powered notebook generation
- [ ] Automated parameter optimization
- [ ] Integration with workflow engines (Nextflow, Snakemake)

## FAQ

**Q: Do I need Jupyter installed to export notebooks?**
A: No! Export works with just `nbformat`. You need Jupyter only to view/edit notebooks interactively.

**Q: Can I edit exported notebooks?**
A: Yes! They're standard .ipynb files. Edit in Jupyter, VS Code, or any notebook editor.

**Q: Does execution require Lobster?**
A: No! Notebooks use standard libraries (scanpy, pandas). Execute anywhere Python is installed.

**Q: How do I version notebooks?**
A: Use Git! Notebooks are text files that work great with version control.

**Q: Can I share notebooks publicly?**
A: Yes! Notebooks use standard libraries and don't contain proprietary code.

**Q: What about large datasets?**
A: Notebooks reference data by path. Store data separately and track locations.

**Q: Does this replace Lobster agents?**
A: No! Agents provide interactive analysis. Notebooks enable reproducibility and batch execution.

**Q: How do I add custom code?**
A: Edit the notebook and add cells. Modifications won't break Papermill execution.

## References

- [Jupyter Notebook Format](https://nbformat.readthedocs.io/)
- [Papermill Documentation](https://papermill.readthedocs.io/)
- [Scanpy Documentation](https://scanpy.readthedocs.io/)
- [Lobster Provenance Tracking](./provenance-tracking.md)
- [DataManagerV2 API](./data-manager-v2.md)

## Support

For issues, questions, or feature requests:
- GitHub Issues: https://github.com/the-omics-os/lobster/issues
- Documentation: https://github.com/the-omics-os/lobster.wiki
- Email: support@omics-os.com

---

**Last Updated:** October 27, 2025
**Authors:** Omics-OS Team
**License:** MIT
