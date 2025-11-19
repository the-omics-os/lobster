# Clustering Quality Metrics Guide

## Overview

Single-cell RNA-seq clustering is inherently subjective. The resolution parameter controls cluster granularity, but there's no universal "correct" resolution. Clustering quality metrics provide **quantitative guidance** to help you:

- Evaluate whether your clustering is good
- Compare multiple resolutions objectively
- Detect over-clustering or under-clustering
- Make informed decisions about optimal resolution

This guide explains the three clustering quality metrics implemented in Lobster and how to use them effectively.

---

## The Three Metrics

### 1. Silhouette Score

**Range**: -1 to 1 (higher is better)

**What it measures**: How similar cells are to their own cluster compared to neighboring clusters.

**Interpretation**:
- **>0.7**: EXCELLENT separation - clusters are very well-defined
- **>0.5**: GOOD separation - clusters are reasonably distinct
- **>0.25**: MODERATE separation - some overlap between clusters
- **<0.25**: POOR separation - high overlap, likely over-clustering

**Best for**: Detecting over-clustering or under-clustering

**Scientific basis**:
Rousseeuw, P. J. (1987). "Silhouettes: A graphical aid to the interpretation and validation of cluster analysis". *Journal of Computational and Applied Mathematics*, 20, 53-65.

### 2. Davies-Bouldin Index

**Range**: 0 to ∞ (lower is better)

**What it measures**: Ratio of within-cluster distances to between-cluster distances.

**Interpretation**:
- **<1.0**: GOOD cluster compactness - clusters are tight and well-separated
- **1.0-2.0**: MODERATE compactness - reasonable clustering
- **>2.0**: POOR compactness - clusters are diffuse or overlapping

**Best for**: Comparing multiple resolutions to find the optimal balance

**Scientific basis**:
Davies, D. L. & Bouldin, D. W. (1979). "A Cluster Separation Measure". *IEEE Transactions on Pattern Analysis and Machine Intelligence*, PAMI-1(2), 224-227.

### 3. Calinski-Harabasz Score

**Range**: 0 to ∞ (higher is better)

**What it measures**: Ratio of between-cluster variance to within-cluster variance.

**Interpretation**:
- **>1000**: HIGH variance ratio - very distinct clusters
- **100-1000**: MODERATE variance ratio - reasonable separation
- **<100**: LOW variance ratio - weak separation

**Best for**: Detecting the optimal number of clusters

**Scientific basis**:
Caliński, T. & Harabasz, J. (1974). "A dendrite method for cluster analysis". *Communications in Statistics*, 3(1), 1-27.

---

## Usage Examples

### Example 1: Evaluate Single Clustering Result

```python
from lobster.tools.clustering_service import ClusteringService

service = ClusteringService()

# After clustering, evaluate quality
result, stats, ir = service.compute_clustering_quality(
    adata,
    cluster_key="leiden"
)

# View interpretation
print(stats["interpretation"])
# Output:
# Silhouette score 0.612: GOOD cluster separation
# Davies-Bouldin index 0.847: GOOD cluster compactness
# Calinski-Harabasz score 1234.5: HIGH variance ratio

# View recommendations
print("\nRecommendations:")
for rec in stats["recommendations"]:
    print(f"  {rec}")
# Output:
#   ✓ Excellent separation. This resolution works well.
```

### Example 2: Compare Multiple Resolutions

```python
# Test 5 different resolutions
resolutions = [0.1, 0.25, 0.5, 1.0, 2.0]
results = []

for res in resolutions:
    # Cluster at this resolution
    cluster_key = f"leiden_res{str(res).replace('.', '_')}"

    # Compute quality metrics
    result, stats, ir = service.compute_clustering_quality(
        adata,
        cluster_key=cluster_key
    )

    results.append({
        "resolution": res,
        "n_clusters": stats["n_clusters"],
        "silhouette": stats["silhouette_score"],
        "davies_bouldin": stats["davies_bouldin_index"],
        "calinski_harabasz": stats["calinski_harabasz_score"]
    })

# Find optimal resolution
import pandas as pd
df = pd.DataFrame(results)
print(df)

# Output:
#    resolution  n_clusters  silhouette  davies_bouldin  calinski_harabasz
# 0        0.10           3       0.721           0.654              1456.2
# 1        0.25           7       0.612           0.847              1234.5
# 2        0.50          15       0.498           1.123               987.3
# 3        1.00          28       0.312           1.654               743.1
# 4        2.00          52       0.189           2.341               521.4

# Best resolution: 0.10 (highest silhouette, lowest DB index)
```

### Example 3: Use Specific Metrics Only

```python
# Only compute silhouette score (faster)
result, stats, ir = service.compute_clustering_quality(
    adata,
    cluster_key="leiden",
    metrics=["silhouette"]
)

# Only compute Davies-Bouldin and Calinski-Harabasz
result, stats, ir = service.compute_clustering_quality(
    adata,
    cluster_key="leiden",
    metrics=["davies_bouldin", "calinski_harabasz"]
)
```

### Example 4: Use Different Representations

```python
# Use UMAP coordinates instead of PCA
result, stats, ir = service.compute_clustering_quality(
    adata,
    cluster_key="leiden",
    use_rep="X_umap"
)

# Use custom embedding (e.g., scVI)
result, stats, ir = service.compute_clustering_quality(
    adata,
    cluster_key="leiden",
    use_rep="X_scvi"
)
```

### Example 5: Control Dimensionality

```python
# Use only first 10 PCs (faster, less sensitive to noise)
result, stats, ir = service.compute_clustering_quality(
    adata,
    cluster_key="leiden",
    n_pcs=10
)

# Use all PCs (default)
result, stats, ir = service.compute_clustering_quality(
    adata,
    cluster_key="leiden",
    n_pcs=None
)
```

---

## Multi-Resolution Selection Workflow

### Step 1: Run Clustering at Multiple Resolutions

```python
from lobster.tools.clustering_service import ClusteringService
import scanpy as sc

service = ClusteringService()

# Run clustering at multiple resolutions
result, stats, ir = service.cluster_and_visualize(
    adata,
    resolutions=[0.1, 0.25, 0.5, 1.0, 2.0]
)

# This creates: leiden_res0_1, leiden_res0_25, leiden_res0_5, etc.
```

### Step 2: Evaluate Quality for Each Resolution

```python
resolutions = [0.1, 0.25, 0.5, 1.0, 2.0]
quality_results = []

for res in resolutions:
    cluster_key = f"leiden_res{str(res).replace('.', '_')}"

    result, stats, ir = service.compute_clustering_quality(
        adata,
        cluster_key=cluster_key
    )

    quality_results.append({
        "resolution": res,
        "n_clusters": stats["n_clusters"],
        "silhouette": stats["silhouette_score"],
        "davies_bouldin": stats["davies_bouldin_index"],
        "calinski_harabasz": stats["calinski_harabasz_score"],
        "recommendations": stats["recommendations"]
    })
```

### Step 3: Visualize Results

```python
import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame(quality_results)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Silhouette score (higher better)
axes[0].plot(df["resolution"], df["silhouette"], marker='o')
axes[0].axhline(y=0.5, color='r', linestyle='--', label='Good threshold')
axes[0].set_xlabel("Resolution")
axes[0].set_ylabel("Silhouette Score")
axes[0].set_title("Silhouette Score (higher better)")
axes[0].legend()
axes[0].grid(True)

# Davies-Bouldin index (lower better)
axes[1].plot(df["resolution"], df["davies_bouldin"], marker='o', color='orange')
axes[1].axhline(y=1.0, color='r', linestyle='--', label='Good threshold')
axes[1].set_xlabel("Resolution")
axes[1].set_ylabel("Davies-Bouldin Index")
axes[1].set_title("Davies-Bouldin Index (lower better)")
axes[1].legend()
axes[1].grid(True)

# Number of clusters
axes[2].plot(df["resolution"], df["n_clusters"], marker='o', color='green')
axes[2].set_xlabel("Resolution")
axes[2].set_ylabel("Number of Clusters")
axes[2].set_title("Cluster Count")
axes[2].grid(True)

plt.tight_layout()
plt.show()
```

### Step 4: Select Optimal Resolution

```python
# Approach 1: Highest silhouette score
best_silhouette = df.loc[df["silhouette"].idxmax()]
print(f"Best by silhouette: resolution={best_silhouette['resolution']}")

# Approach 2: Lowest Davies-Bouldin index
best_db = df.loc[df["davies_bouldin"].idxmin()]
print(f"Best by Davies-Bouldin: resolution={best_db['resolution']}")

# Approach 3: Highest Calinski-Harabasz score
best_ch = df.loc[df["calinski_harabasz"].idxmax()]
print(f"Best by Calinski-Harabasz: resolution={best_ch['resolution']}")

# Approach 4: Combined scoring (normalize and average)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df["silhouette_norm"] = scaler.fit_transform(df[["silhouette"]])
df["davies_bouldin_norm"] = 1 - scaler.fit_transform(df[["davies_bouldin"]])  # Invert (lower is better)
df["calinski_harabasz_norm"] = scaler.fit_transform(df[["calinski_harabasz"]])

# Combined score (equal weights)
df["combined_score"] = (
    df["silhouette_norm"] +
    df["davies_bouldin_norm"] +
    df["calinski_harabasz_norm"]
) / 3

best_combined = df.loc[df["combined_score"].idxmax()]
print(f"Best by combined score: resolution={best_combined['resolution']}")
```

---

## Best Practices

### When to Use Which Metric

| Scenario | Recommended Metric | Reasoning |
|----------|-------------------|-----------|
| **Comparing 2-5 resolutions** | Silhouette Score | Intuitive, single number, well-established |
| **Comparing 5+ resolutions** | Davies-Bouldin Index | More stable across wide resolution ranges |
| **Finding optimal cluster count** | Calinski-Harabasz Score | Directly measures variance ratio |
| **General-purpose evaluation** | All three metrics | Provides comprehensive view |

### Typical Value Ranges by Dataset Type

| Dataset Type | Typical Silhouette | Typical Davies-Bouldin |
|--------------|-------------------|------------------------|
| **Well-defined cell types** (e.g., PBMC) | 0.5-0.8 | 0.6-1.2 |
| **Continuous differentiation** (e.g., development) | 0.3-0.6 | 1.0-2.0 |
| **Heterogeneous tissue** (e.g., tumor) | 0.2-0.5 | 1.5-3.0 |

### Resolution Selection Guidelines

1. **Start broad**: Test resolutions spanning 0.1 to 2.0
2. **Narrow down**: Focus on resolutions with silhouette >0.5
3. **Consider biology**: High metrics don't always mean biologically meaningful
4. **Validate with markers**: Use known marker genes to verify clusters make sense
5. **Check batch effects**: Quality metrics can be misleading if batches aren't corrected

### Common Pitfalls

**❌ Over-reliance on metrics alone**
- Metrics guide you, but don't replace biological knowledge
- Always validate with marker genes and literature

**❌ Ignoring batch effects**
- Quality metrics can be artificially high if batches cluster separately
- Always check batch effects before interpreting metrics

**❌ Using UMAP for quality assessment**
- UMAP distorts global distances (not suitable for distance-based metrics)
- Always use PCA or other linear embeddings for quality metrics

**❌ Not considering rare cell types**
- Small rare populations may lower overall metrics
- Use per-cluster silhouette scores to identify which clusters are problematic

**❌ Comparing metrics across different datasets**
- Metrics are dataset-specific
- Don't compare absolute values between different experiments

---

## Limitations

### When Metrics Might Be Misleading

1. **Rare cell types**: Small populations may have low silhouette scores even if biologically real
2. **Continuous differentiation**: Developmental trajectories don't have discrete boundaries
3. **Batch effects**: Uncorrected batch effects can inflate metrics artificially
4. **High-dimensional noise**: PCA captures noise, affecting metric reliability
5. **Imbalanced cluster sizes**: Very different cluster sizes can skew metrics

### Complementary Approaches

Quality metrics should be used alongside:

- **Marker gene validation**: Check if clusters express expected markers
- **Differential expression**: Verify clusters have distinct gene expression profiles
- **Batch effect visualization**: Ensure batches are well-mixed within clusters
- **Trajectory analysis**: For continuous processes, use pseudotime instead
- **Expert knowledge**: Biologists understand cell types better than metrics

---

## Per-Cluster Analysis

### Identify Problematic Clusters

```python
result, stats, ir = service.compute_clustering_quality(
    adata,
    cluster_key="leiden"
)

# Get per-cluster silhouette scores
per_cluster = stats["per_cluster_silhouette"]

# Find clusters with low silhouette
for cluster_id, score in per_cluster.items():
    if score < 0.3:
        print(f"⚠️ Cluster {cluster_id}: silhouette={score:.3f} (low quality)")

# Output:
# ⚠️ Cluster 5: silhouette=0.187 (low quality)
# ⚠️ Cluster 12: silhouette=0.254 (low quality)
```

### Investigate Low-Quality Clusters

```python
# Visualize problematic cluster
import scanpy as sc

# Subset to low-quality cluster
adata_subset = adata[adata.obs["leiden"] == "5"].copy()

# Find marker genes
sc.tl.rank_genes_groups(adata_subset, groupby="leiden", method="wilcoxon")

# Check if it's a doublet or transition state
sc.pl.umap(adata, color=["leiden", "n_genes", "pct_counts_mt"])
```

---

## Advanced Usage

### Custom Metric Combinations

```python
def custom_quality_score(stats):
    """
    Custom scoring function emphasizing silhouette and cluster count.
    """
    silhouette = stats["silhouette_score"]
    n_clusters = stats["n_clusters"]

    # Penalize very high or very low cluster counts
    if n_clusters < 5 or n_clusters > 30:
        penalty = 0.8
    else:
        penalty = 1.0

    return silhouette * penalty

# Compare resolutions
scores = []
for res in [0.1, 0.25, 0.5, 1.0, 2.0]:
    cluster_key = f"leiden_res{str(res).replace('.', '_')}"
    result, stats, ir = service.compute_clustering_quality(adata, cluster_key=cluster_key)

    scores.append({
        "resolution": res,
        "custom_score": custom_quality_score(stats)
    })

best_res = max(scores, key=lambda x: x["custom_score"])
print(f"Best resolution by custom score: {best_res['resolution']}")
```

### Integration with Sub-Clustering

```python
# Initial clustering
result, stats, ir = service.cluster_and_visualize(adata, resolution=0.5)

# Evaluate quality
result, stats, ir = service.compute_clustering_quality(result, cluster_key="leiden")

# Identify low-quality clusters for sub-clustering
per_cluster = stats["per_cluster_silhouette"]
low_quality_clusters = [
    cluster_id for cluster_id, score in per_cluster.items()
    if score < 0.3
]

print(f"Sub-clustering recommended for: {low_quality_clusters}")

# Sub-cluster low-quality populations
if low_quality_clusters:
    result, stats, ir = service.subcluster_cells(
        result,
        cluster_key="leiden",
        clusters_to_refine=low_quality_clusters,
        resolution=0.8
    )
```

---

## Reproducibility

### Storing Quality Metrics

Quality metrics are automatically stored in `adata.uns` for later retrieval:

```python
# Compute metrics
result, stats, ir = service.compute_clustering_quality(
    adata,
    cluster_key="leiden"
)

# Access stored metrics later
quality_data = result.uns["leiden_quality"]

print(f"Silhouette: {quality_data['silhouette_score']}")
print(f"Davies-Bouldin: {quality_data['davies_bouldin_index']}")
print(f"Calinski-Harabasz: {quality_data['calinski_harabasz_score']}")
print(f"Recommendations: {quality_data['recommendations']}")
```

### Exporting to Notebooks

The IR (Intermediate Representation) enables automatic Jupyter notebook generation:

```python
from lobster.core.notebook_exporter import NotebookExporter

# Compute quality metrics
result, stats, ir = service.compute_clustering_quality(
    adata,
    cluster_key="leiden"
)

# Export to notebook
exporter = NotebookExporter()
exporter.add_analysis_step(ir)
exporter.export("clustering_quality_analysis.ipynb")
```

---

## References

### Scientific Literature

1. **Silhouette Score**:
   - Rousseeuw, P. J. (1987). "Silhouettes: A graphical aid to the interpretation and validation of cluster analysis". *Journal of Computational and Applied Mathematics*, 20, 53-65.

2. **Davies-Bouldin Index**:
   - Davies, D. L. & Bouldin, D. W. (1979). "A Cluster Separation Measure". *IEEE Transactions on Pattern Analysis and Machine Intelligence*, PAMI-1(2), 224-227.

3. **Calinski-Harabasz Score**:
   - Caliński, T. & Harabasz, J. (1974). "A dendrite method for cluster analysis". *Communications in Statistics*, 3(1), 1-27.

4. **Single-Cell Clustering Best Practices**:
   - Kiselev, V. Y., et al. (2019). "Challenges in unsupervised clustering of single-cell RNA-seq data". *Nature Reviews Genetics*, 20(5), 273-282.
   - Luecken, M. D. & Theis, F. J. (2019). "Current best practices in single‐cell RNA‐seq analysis: a tutorial". *Molecular Systems Biology*, 15(6), e8746.

### External Resources

- [Scikit-learn Clustering Metrics](https://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation)
- [Scanpy Clustering Tutorial](https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html)
- [Single-Cell Best Practices Book](https://www.sc-best-practices.org/)

---

## Support

For issues or questions:
- GitHub Issues: [https://github.com/the-omics-os/lobster/issues](https://github.com/the-omics-os/lobster/issues)
- Documentation: [https://github.com/the-omics-os/lobster/wiki](https://github.com/the-omics-os/lobster/wiki)
