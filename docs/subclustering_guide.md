# Sub-clustering Guide

## Overview

Sub-clustering is a powerful technique in single-cell RNA-seq analysis for refining initial clustering results. It enables researchers to re-cluster specific cell populations at higher resolution to discover finer-grained subpopulations that may be biologically meaningful.

## Purpose

The `subcluster_cells()` method in `ClusteringService` allows you to:

1. **Refine heterogeneous clusters**: Break down broad cell type categories into more specific subtypes
2. **Targeted analysis**: Focus computational resources on clusters of interest without re-processing the entire dataset
3. **Hierarchical cell type identification**: Build a hierarchical taxonomy of cell types (e.g., T cells → CD4+ T cells → Th1/Th2/Treg)
4. **Maintain traceability**: Sub-cluster labels include parent cluster prefixes for easy tracking

## When to Use Sub-clustering

### Good Use Cases

- **Initial clustering groups distinct populations**: When your first-pass clustering produces clusters that you know contain multiple cell types
- **Rare cell type discovery**: When you want to identify rare subtypes within a specific cluster
- **Biological heterogeneity**: When you expect biological heterogeneity within a cluster (e.g., activation states, cell cycle phases)
- **Focused marker gene analysis**: When you want to find markers specific to subpopulations within a cluster

### When NOT to Use Sub-clustering

- **Over-clustering concerns**: Avoid excessive sub-clustering that may split technical rather than biological variation
- **Low cell counts**: Sub-clustering very small clusters (<50 cells) may produce unstable results
- **Already optimal resolution**: If your initial clustering already captures all relevant biological structure

## Method Signature

```python
def subcluster_cells(
    self,
    adata: anndata.AnnData,
    cluster_key: str = "leiden",
    clusters_to_refine: Optional[List[str]] = None,
    resolution: float = 0.5,
    resolutions: Optional[List[float]] = None,
    n_pcs: int = 20,
    n_neighbors: int = 15,
    batch_key: Optional[str] = None,
    demo_mode: bool = False,
) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
```

## Parameters

### Required Parameters

- **`adata`** (AnnData): AnnData object with existing clustering results and PCA coordinates in `adata.obsm['X_pca']`

### Optional Parameters

- **`cluster_key`** (str, default: "leiden"): Column name in `adata.obs` containing cluster assignments to refine
- **`clusters_to_refine`** (List[str], default: None): List of cluster IDs to re-cluster
  - `None`: Re-clusters ALL cells (full re-clustering)
  - `["0", "3"]`: Only re-clusters clusters 0 and 3
- **`resolution`** (float, default: 0.5): Single resolution parameter for Leiden clustering
  - Lower values (0.1-0.5): Fewer, broader sub-clusters
  - Higher values (1.0-2.0): More, finer-grained sub-clusters
- **`resolutions`** (List[float], default: None): List of resolutions for multi-resolution testing
  - Example: `[0.25, 0.5, 1.0]` creates 3 columns with different granularities
- **`n_pcs`** (int, default: 20): Number of principal components to use for sub-clustering
  - Default is lower than full clustering (30) since we're working on subsets
- **`n_neighbors`** (int, default: 15): Number of neighbors for KNN graph construction
- **`batch_key`** (str, default: None): **Not yet implemented** - Future support for batch-aware sub-clustering
- **`demo_mode`** (bool, default: False): Use faster parameters for testing (caps n_pcs and n_neighbors at 10)

## Return Values

Returns a 3-tuple following the standard Lobster service pattern:

1. **`adata`** (AnnData): Modified AnnData with new sub-cluster columns in `.obs`
2. **`stats`** (Dict[str, Any]): Statistics dictionary containing:
   - `analysis_type`: "sub-clustering"
   - `parent_clusters`: List of clusters that were refined
   - `n_cells_subclustered`: Number of cells re-clustered
   - `resolutions_tested`: List of resolutions used
   - `subclustering_results`: Per-resolution sub-cluster counts
   - `cluster_sizes`: Cell counts per sub-cluster
   - `execution_time_seconds`: Processing time
3. **`ir`** (AnalysisStep): Intermediate representation for reproducible notebook export

## Output Column Naming

### Single Resolution

When using a single `resolution` parameter:

- **Output column**: `leiden_subcluster`
- **Label format**: `{parent}.{sub}` (e.g., `0.0`, `0.1`, `3.0`, `3.1`)
- **Non-refined clusters**: Retain original cluster IDs (e.g., `1`, `2`, `4`)

### Multi-Resolution

When using multiple `resolutions`:

- **Output columns**: `leiden_sub_res0_25`, `leiden_sub_res0_5`, `leiden_sub_res1_0`
- **Label format**: Same prefix system (`{parent}.{sub}`) in each column
- **Use case**: Explore different granularities to choose optimal resolution

## Examples

### Example 1: Basic Sub-clustering of a Single Cluster

```python
from lobster.tools.clustering_service import ClusteringService

# Initialize service
service = ClusteringService()

# Sub-cluster cluster "0" at resolution 0.5
adata_result, stats, ir = service.subcluster_cells(
    adata,
    cluster_key="leiden",
    clusters_to_refine=["0"],
    resolution=0.5,
    n_pcs=20,
    n_neighbors=15,
)

# Inspect results
print(f"Refined {stats['n_cells_subclustered']} cells")
print(f"Sub-clusters: {adata_result.obs['leiden_subcluster'].unique()}")
```

**Expected output**:
```
Refined 1234 cells
Sub-clusters: ['0.0', '0.1', '0.2', '1', '2', '3']
```

### Example 2: Sub-clustering Multiple Clusters

```python
# Refine clusters 0, 3, and 5
adata_result, stats, ir = service.subcluster_cells(
    adata,
    cluster_key="leiden",
    clusters_to_refine=["0", "3", "5"],
    resolution=0.8,  # Higher resolution for finer granularity
)

# Check sub-clusters per parent
for parent, n_sub in stats['subclustering_results'][0.8]['n_subclusters_per_parent'].items():
    print(f"Cluster {parent} split into {n_sub} sub-clusters")
```

**Expected output**:
```
Cluster 0 split into 3 sub-clusters
Cluster 3 split into 2 sub-clusters
Cluster 5 split into 4 sub-clusters
```

### Example 3: Multi-Resolution Sub-clustering

```python
# Test multiple resolutions to find optimal granularity
adata_result, stats, ir = service.subcluster_cells(
    adata,
    cluster_key="leiden",
    clusters_to_refine=["0"],
    resolutions=[0.25, 0.5, 1.0, 2.0],
)

# Compare sub-cluster counts at different resolutions
for res in stats['resolutions_tested']:
    key = f"leiden_sub_res{res}".replace(".", "_")
    n_subclusters = adata_result.obs[key].nunique()
    print(f"Resolution {res}: {n_subclusters} sub-clusters")
```

**Expected output**:
```
Resolution 0.25: 5 sub-clusters
Resolution 0.5: 8 sub-clusters
Resolution 1.0: 12 sub-clusters
Resolution 2.0: 18 sub-clusters
```

### Example 4: Full Re-clustering (All Cells)

```python
# Re-cluster all cells at higher resolution
adata_result, stats, ir = service.subcluster_cells(
    adata,
    cluster_key="leiden",
    clusters_to_refine=None,  # None = all clusters
    resolution=1.0,
)

print(f"Re-clustered {stats['n_total_cells']} cells")
print(f"Original clusters: {len(stats['parent_clusters'])}")
print(f"Total sub-clusters: {adata_result.obs['leiden_subcluster'].nunique()}")
```

### Example 5: Demo Mode for Fast Testing

```python
# Use demo mode for quick validation
adata_result, stats, ir = service.subcluster_cells(
    adata,
    cluster_key="leiden",
    clusters_to_refine=["0", "1"],
    resolution=0.5,
    demo_mode=True,  # Caps n_pcs=10, n_neighbors=10
)

print(f"Processing time: {stats['execution_time_seconds']} seconds")
```

## Best Practices

### Choosing Resolution

**Low resolution (0.1 - 0.5)**: Use when you want broad subpopulations
- Suitable for initial exploration
- Reduces risk of over-clustering

**Medium resolution (0.5 - 1.0)**: Balanced approach (default)
- Good starting point for most datasets
- Captures biologically meaningful variation

**High resolution (1.0 - 2.0)**: Fine-grained subpopulations
- Use when you know there's biological heterogeneity
- Requires careful interpretation to avoid technical artifacts

### Choosing n_pcs and n_neighbors

**Default values (n_pcs=20, n_neighbors=15)**: Good for most cases
- Lower than full clustering (30 PCs) since we're on subsets

**Smaller datasets (<100 cells per cluster)**:
```python
n_pcs=10, n_neighbors=10
```

**Larger datasets (>1000 cells per cluster)**:
```python
n_pcs=30, n_neighbors=20
```

### Validation Workflow

1. **Run multi-resolution first**: Test multiple resolutions to explore the structure
2. **Visualize results**: Use UMAP to visualize sub-clusters
3. **Check marker genes**: Validate that sub-clusters have distinct gene signatures
4. **Biological validation**: Ensure sub-clusters correspond to known biology

### Common Pitfalls to Avoid

1. **Over-clustering technical variation**: Don't mistake cell cycle or stress signatures for distinct cell types
2. **Sub-clustering tiny clusters**: Avoid sub-clustering clusters with <50 cells
3. **Ignoring batch effects**: If batches are present, consider the (future) `batch_key` parameter
4. **Too many levels**: Don't recursively sub-cluster more than 2-3 levels deep

## Integration with Lobster Workflow

### Typical Analysis Pipeline

```python
# 1. Initial clustering
adata, stats, ir = service.cluster_and_visualize(
    adata,
    resolution=0.5,
)

# 2. Identify clusters of interest (e.g., based on marker genes)
# Suppose cluster "0" is T cells and shows heterogeneity

# 3. Sub-cluster the T cell cluster
adata, stats, ir = service.subcluster_cells(
    adata,
    cluster_key="leiden",
    clusters_to_refine=["0"],
    resolution=0.8,
)

# 4. Find markers for sub-clusters
import scanpy as sc
sc.tl.rank_genes_groups(adata, "leiden_subcluster", method="wilcoxon")

# 5. Annotate sub-cluster types
# Map "0.0" -> "CD4+ T cells", "0.1" -> "CD8+ T cells", etc.
```

### Provenance Tracking

Sub-clustering generates full W3C-PROV compatible provenance:

```python
# Access provenance from IR
print(ir.operation)  # "scanpy.tl.leiden"
print(ir.tool_name)  # "subcluster_cells"
print(ir.parameters)  # Full parameter set

# Export to Jupyter notebook
from lobster.core.notebook_exporter import NotebookExporter
exporter = NotebookExporter(data_manager)
notebook_path = exporter.export_notebook("analysis_pipeline.ipynb")
```

## Technical Details

### Algorithm

1. **Validation**: Check cluster_key exists, validate cluster IDs, ensure PCA available
2. **Subset selection**: Create boolean mask for cells in `clusters_to_refine`
3. **Neighbor graph**: Compute KNN graph on subset using existing PCA coordinates
4. **Leiden clustering**: Run Leiden algorithm at specified resolution(s)
5. **Prefix labels**: Add parent cluster ID as prefix (e.g., "0" → "0.0", "0.1", "0.2")
6. **Merge results**: Transfer sub-cluster labels back to original AnnData

### Performance

- **Small clusters (<100 cells)**: <1 second
- **Medium clusters (100-1000 cells)**: 1-5 seconds
- **Large clusters (>1000 cells)**: 5-15 seconds

Use `demo_mode=True` for 2-3x speedup during testing.

## Troubleshooting

### Error: "Cluster key 'leiden' not found in adata.obs"

**Cause**: AnnData doesn't have the specified clustering column
**Solution**: Run initial clustering first:
```python
adata, stats, ir = service.cluster_and_visualize(adata)
```

### Error: "Invalid cluster IDs: ['999']"

**Cause**: Specified cluster ID doesn't exist
**Solution**: Check available clusters:
```python
print(adata.obs['leiden'].unique())
```

### Error: "PCA results not found in adata.obsm['X_pca']"

**Cause**: PCA hasn't been computed
**Solution**: Run full clustering pipeline first, which includes PCA

### Warning: "Only 1 sub-cluster detected"

**Possible causes**:
- Resolution too low: Try increasing resolution to 0.8 or 1.0
- Cluster too homogeneous: May not need sub-clustering
- Too few cells: Need at least 50 cells for reliable sub-clustering

## References

1. **Leiden algorithm**: Traag, V.A., Waltman, L. & van Eck, N.J. "From Louvain to Leiden: guaranteeing well-connected communities." Sci Rep 9, 5233 (2019). https://doi.org/10.1038/s41598-019-41695-z

2. **Sub-clustering best practices**: Luecken, M.D., Theis, F.J. "Current best practices in single-cell RNA-seq analysis: a tutorial." Mol Syst Biol 15, e8746 (2019). https://doi.org/10.15252/msb.20188746

3. **Scanpy implementation**: Wolf, F.A., Angerer, P. & Theis, F.J. "SCANPY: large-scale single-cell gene expression data analysis." Genome Biol 19, 15 (2018). https://doi.org/10.1186/s13059-017-1382-0

## Future Enhancements

Planned features for future versions:

- **Batch-aware sub-clustering**: Use `batch_key` parameter for batch correction during sub-clustering
- **Automatic resolution selection**: Heuristic-based resolution selection
- **Sub-cluster stability**: Bootstrap-based stability metrics
- **Recursive sub-clustering**: Helper method for multi-level hierarchical refinement
