# Error Hardening Audit — Tool Error Actionability

**Date**: 2026-03-12
**Purpose**: Catalog every tool error return in Lobster AI, grade it for LLM-agent recoverability, and prioritize fixes.

**Grading rubric:**
- **A**: Agent gets enough context to recover in 1 retry
- **B**: Agent knows what failed but must guess the fix
- **C**: Agent gets almost nothing useful (raw `str(e)`)

**Totals: 117 A, 105 B, 170 C** across ~392 error returns.

---

## Priority Tiers

### P0 — Demo failures / confirmed death spirals (fix immediately)

| File | Line | Tool | Grade | Issue |
|------|------|------|-------|-------|
| `packages/lobster-proteomics/.../de_analysis_expert.py` | 417-426 | `find_differential_proteins` | B | **Emory demo failure.** Has traceback but loses `group_column` values, detected platform, method. Agent can't know whether to fix group_column, switch method, or increase sample size. |
| `packages/lobster-research/.../data_expert.py` | 653 | `load_modality` | C | No `file_path`, no adapter name. Highest-traffic tool — causes 8+ retry loops. |
| `packages/lobster-research/.../data_expert.py` | 496 | `get_modality_details` | C | No modality name echoed. Blind retry. |
| `packages/lobster-research/.../data_expert.py` | 866 | `concatenate_samples` | C | No modality names, no hint whether auto-detection failed. |

### P1 — High-traffic tools with C-grade errors (fix this week)

| File | Line | Tool | Grade | Fix pattern |
|------|------|------|-------|-------------|
| `packages/lobster-transcriptomics/.../shared_tools.py` | 941 | `compute_neighbors_and_embedding` | C | Include `list(adata.obsm.keys())` — PCA absence is #1 cause |
| `packages/lobster-transcriptomics/.../shared_tools.py` | 851 | `run_pca` | C | Include `adata.shape`, `n_comps` |
| `packages/lobster-transcriptomics/.../shared_tools.py` | 480 | `filter_and_normalize` | C | Include `data_type`, `adata.shape`, resolved filter params |
| `packages/lobster-transcriptomics/.../transcriptomics_expert.py` | 1095 | `integrate_batches` | C | Include `batch_key`, `list(adata.obs.columns)` |
| `packages/lobster-transcriptomics/.../transcriptomics_expert.py` | 885 | `find_marker_genes` | B | Include `list(adata.obs.columns)` — cluster key mis-name is #1 mistake |
| `packages/lobster-transcriptomics/.../annotation_expert.py` | 412 | `apply_manual_annotations` | C | Include `cluster_key`, `list(adata.obs[cluster_key].unique())` |
| `packages/lobster-transcriptomics/.../de_analysis_expert.py` | 1461 | `run_de_with_formula` | C | Include `formula`, `contrast`, `pseudobulk_modality` |
| `packages/lobster-proteomics/.../shared_tools.py` | 1637 | `correct_batch_effects` | C | Include `modality_name`, `method`, batch values |
| `packages/lobster-proteomics/.../shared_tools.py` | 1455 | `import_proteomics_data` | C | Include `file_path`, `software` (or detected) |
| `packages/lobster-proteomics/.../de_analysis_expert.py` | 976 | `run_pathway_enrichment` | C | Include `modality_name`, `databases`, `n_proteins` |
| `packages/lobster-proteomics/.../de_analysis_expert.py` | 1447 | `run_string_network_analysis` | C | Include `modality_name`, `species`, `score_threshold` |
| `packages/lobster-metabolomics/.../shared_tools.py` | 557 | `run_metabolomics_statistics` | C | Include `group_column`, `list(adata.obs.columns)` |
| `packages/lobster-metabolomics/.../shared_tools.py` | 465 | `correct_batch_effects` | C | Include `batch_key`, `list(adata.obs.columns)` |
| `packages/lobster-genomics/.../genomics_expert.py` | 270 | `load_vcf_file` | C | Include `file_path`, `region` |
| `packages/lobster-research/.../research_agent.py` | 334 | `search_literature` | C | Include `query`, `sources` |

### P2 — Visualization + ML (all C-grade, low actionability)

| Package | C-grade count | Pattern |
|---------|--------------|---------|
| `lobster-visualization` | 10/11 | Every plot tool: no `modality_name`, no `color_by`/`groupby`/`genes` |
| `lobster-ml` | 8/9 | Every tool: no `modality_name`, no `target_column`, no `method` |
| `lobster-structural-viz` | 5/10 | `"Unexpected error:"` catch-alls drop `pdb_id` |

### P3 — Providers + drug discovery catch-alls

| Package | C-grade count | Pattern |
|---------|--------------|---------|
| All providers (pubmed, geo, pride, massive, metabolights, biorxiv) | 16 | `f"Error searching X: {str(e)}"` — no query echoed |
| `lobster-drug-discovery` (cheminformatics) | 10/18 | Catch-alls drop SMILES strings |
| `lobster-drug-discovery` (clinical) | 6/15 | Catch-alls drop `target_id`/`chembl_id` |
| `lobster-drug-discovery` (pharmacogenomics) | 7/18 | Catch-alls drop sequence/mutation context |

### P4 — Workspace/filesystem (mostly fine, 7 C-grades)

| File | Line | Tool | Fix |
|------|------|------|-----|
| `workspace_tool.py` | 1180, 1851, 2257 | catch-all outer wrappers | Include `identifier`, `workspace`, `level` |
| `filesystem_tools.py` | 478 | `shell_execute` catch-all | Include `command` and `workspace_path` |
| `filesystem_tools.py` | 335 | `glob_files` catch-all | Include `pattern` and base path |

---

## Systemic Fix Patterns

### Pattern A: Include in-scope parameters in catch-all

```python
# BEFORE (C-grade):
except Exception as e:
    return f"Error in batch correction: {str(e)}"

# AFTER (B+ grade):
except Exception as e:
    return f"Error in batch correction for '{modality_name}' (method={method}, batch_key={batch_key}): {str(e)}"
```

### Pattern B: Include available columns/keys for "not found" errors

```python
# BEFORE (B-grade):
except Exception as e:
    return f"Error clustering: {str(e)}"

# AFTER (A-grade):
except Exception as e:
    cols = list(adata.obs.columns) if adata is not None else []
    obsm = list(adata.obsm.keys()) if adata is not None else []
    return (
        f"Error clustering '{modality_name}': {str(e)}\n"
        f"obs.columns: {cols[:15]}\n"
        f"obsm keys: {obsm}"
    )
```

### Pattern C: Fix "Unexpected error:" prefix in genomics

```python
# BEFORE (misleading):
except Exception as e:
    return f"Unexpected error: {str(e)}"

# AFTER (honest):
except Exception as e:
    return f"Error in filter_samples for '{modality_name}': {str(e)}. obs.columns={list(adata.obs.columns)}"
```

### Pattern D: Echo query in provider catch-alls

```python
# BEFORE:
except Exception as e:
    return f"Error searching PRIDE: {str(e)}"

# AFTER:
except Exception as e:
    return f"Error searching PRIDE for query='{query}': {str(e)}"
```

---

## Execution Plan

- **P0**: Fix in this session (4 tools, confirmed death spirals)
- **P1**: Fix this week (15 high-traffic tools)
- **P2**: Fix next week (ML + viz blanket fix)
- **P3**: Fix next sprint (providers + drug discovery)
- **P4**: Fix when touched (workspace/filesystem, already mostly fine)
