"""The IR code_templates must reproduce the services they came from.

Each template is a second implementation of a service. Focused tests pin the behavior that
must remain aligned while allowing harmless changes in generated text.
"""

import pytest

from lobster.services.analysis.clustering_service import ClusteringService
from lobster.services.analysis.enhanced_singlecell_service import (
    EnhancedSingleCellService,
)
from lobster.services.quality.preprocessing_service import PreprocessingService
from lobster.services.quality.quality_service import QualityService

# --- NotebookExporter selection-aware wording ------------------------------


@pytest.mark.parametrize("selection", ["exportable", "replayable"])
@pytest.mark.parametrize("excluded", [[], ["loading", "loading"]])
def test_exporter_coverage_wording_matches_selection(selection, excluded):
    from types import SimpleNamespace

    from lobster.core.notebooks.exporter import NotebookExporter

    exporter = NotebookExporter.__new__(NotebookExporter)
    exporter.provenance = SimpleNamespace(namespace="synthetic-session")
    footer = exporter._create_footer_cell(excluded, select_on=selection).source
    summary = exporter._create_provenance_summary_cell(
        3, 1, excluded, select_on=selection
    ).source
    other_selection = "replayable" if selection == "exportable" else "exportable"
    assert f"{selection} IR" in footer
    assert f"{selection} IR" in summary
    assert f"{other_selection} IR" not in footer + summary
    assert "**1 activities** have" in summary
    assert "**2 activities** are excluded" in summary
    assert (
        "IR coverage alone does not show that running it gives the same results."
        in footer
    )
    if excluded:
        assert "All selected activities" not in footer
        assert footer.count("`loading`") == 1
    else:
        assert f"All selected activities have {selection} IR." in footer


def test_exporter_default_wording_remains_exportable():
    from types import SimpleNamespace

    from lobster.core.notebooks.exporter import NotebookExporter

    exporter = NotebookExporter.__new__(NotebookExporter)
    exporter.provenance = SimpleNamespace(namespace="synthetic-session")
    assert "exportable IR" in exporter._create_footer_cell([]).source
    assert "exportable IR" in exporter._create_provenance_summary_cell(1, 1, []).source


# --- QualityService.assess_quality -----------------------------------------


@pytest.fixture
def quality_code():
    ir = QualityService()._create_quality_ir(
        min_genes=500,
        max_genes=5000,
        max_mt_pct=20.0,
        max_ribo_pct=50.0,
        min_housekeeping_score=1.0,
    )
    return ir.render()


def test_quality_template_writes_the_services_column_names(quality_code):
    """The service writes mt_pct / ribo_pct / housekeeping_score, not scanpy's."""
    for column in ("mt_pct", "ribo_pct", "housekeeping_score"):
        assert f"adata.obs['{column}']" in quality_code, f"missing {column}"


def test_quality_template_qc_pass_includes_the_housekeeping_floor(quality_code):
    """The scanpy-based version had only four conditions and dropped this one.

    That let a replay keep cells the session had filtered out, which changed
    which genes cleared min_cells downstream.
    """
    # The qc_pass expression is everything up to its closing paren.
    qc_block = quality_code.split("adata.obs['qc_pass'] = (")[1].split("\n)")[0]

    # Five criteria means four '&' operators.
    assert qc_block.count("&") == 4, (
        f"qc_pass must combine exactly five criteria, got "
        f"{qc_block.count('&') + 1}:\n{qc_block}"
    )
    # And specifically the housekeeping floor, the one that was dropped.
    assert "housekeeping_score" in qc_block
    assert ">= 1.0" in qc_block, "housekeeping threshold not interpolated"
    # Both genes-per-cell bounds and both percentage ceilings.
    assert "_n_genes >= 500" in qc_block
    assert "_n_genes <= 5000" in qc_block
    assert "mt_pct" in qc_block and "ribo_pct" in qc_block


def test_quality_template_does_not_use_scanpy_qc_metric_names(quality_code):
    """Building qc_pass from pct_counts_* would use mismatched metric names."""
    assert "sc.pp.calculate_qc_metrics" not in quality_code
    qc_block = quality_code.split("adata.obs['qc_pass']")[1]
    assert "pct_counts_mt" not in qc_block
    assert "n_genes_by_counts" not in qc_block


# --- PreprocessingService.filter_and_normalize_cells -----------------------


@pytest.fixture
def filter_code():
    ir = PreprocessingService()._create_filter_normalize_ir(
        min_genes_per_cell=200,
        max_genes_per_cell=5000,
        min_cells_per_gene=3,
        max_mito_percent=20.0,
        max_ribo_percent=50.0,
        target_sum=10000,
        normalization_method="log1p",
    )
    return ir.render()


def test_filter_template_filters_genes_before_the_mt_ribo_cell_filter(filter_code):
    """The order is load-bearing: it decides which genes clear min_cells.

    The service calls filter_cells(min_genes) then filter_genes(min_cells) and
    only then applies the mt/ribo/max-genes mask. Reordering these filters can
    change the retained genes and downstream analysis.
    """
    assert "sc.pp.filter_genes" in filter_code
    assert "pct_counts_mt" in filter_code
    gene_filter_at = filter_code.index("sc.pp.filter_genes")
    mt_filter_at = filter_code.index("pct_counts_mt")
    assert (
        gene_filter_at < mt_filter_at
    ), "sc.pp.filter_genes must run BEFORE the mt/ribo cell filter"


def test_filter_template_uses_filter_cells_so_n_genes_is_written(filter_code):
    """sc.pp.filter_cells writes obs['n_genes']; a boolean mask does not."""
    assert "sc.pp.filter_cells(adata, min_genes=" in filter_code


def test_filter_template_stores_raw_before_normalizing(filter_code):
    assert "adata.raw = adata.copy()" in filter_code
    assert filter_code.index("adata.raw = adata.copy()") < filter_code.index(
        "sc.pp.normalize_total"
    )


# --- ClusteringService.cluster_and_visualize -------------------------------


@pytest.mark.parametrize("method", ["deviance", "hvg"])
def test_clustering_template_does_not_leave_adata_as_the_subset(
    method, tmp_path, monkeypatch
):
    """The structural drift: the template rebound `adata` to the gene subset.

    Every later cell — annotation, DE — must use the full expression object,
    not the feature subset used to fit the clustering representation.
    """
    from types import SimpleNamespace

    import anndata
    import nbformat
    import numpy as np
    import pandas as pd

    from lobster.core.notebooks.exporter import NotebookExporter
    from lobster.core.provenance.provenance import ProvenanceTracker

    ir = ClusteringService()._create_clustering_ir(
        resolution=1.0,
        feature_selection_method=method,
        algorithm="leiden",
        n_features=12,
        n_pcs=5,
        n_neighbors=5,
    )
    code = ir.render()
    assert "adata_selected =" in code, "subset must bind to a separate name"
    assert "\nadata = adata[:," not in code, "template must not rebind adata"

    # S3 transfers every requested resolution under its actual key; the property
    # is preserved even though the former literal `cluster_key` assignment moved.
    source = anndata.AnnData(
        np.random.default_rng(42)
        .negative_binomial(2, 0.5, (48, 600))
        .astype(np.float32),
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(48)]),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(600)]),
    )
    source.write_h5ad(tmp_path / "dataset.h5ad")
    tracker = ProvenanceTracker()
    tracker.create_activity(activity_type="cluster_cells", agent="test", ir=ir)
    context = SimpleNamespace(
        workspace_path=tmp_path, workspace_dir=tmp_path, modalities={"fixture": source}
    )
    path = NotebookExporter(tracker, context).export(f"clustering_{method}")
    notebook = nbformat.read(path, as_version=4)
    monkeypatch.chdir(tmp_path)
    namespace = {}
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            exec(
                compile(cell.source, f"clustering_notebook_cell_{index}", "exec"),
                namespace,
            )  # nosec B102
    replay = namespace["adata"]
    selected = namespace["adata_selected"]
    assert (
        selected.n_vars < source.n_vars
    ), "fixture must exercise a real feature subset"
    assert replay.shape == source.shape
    assert list(replay.obs_names) == list(source.obs_names)
    assert list(replay.var_names) == list(source.var_names)
    for key in ("leiden", "leiden_res1_0"):
        pd.testing.assert_series_equal(replay.obs[key], selected.obs[key])
        assert replay.obs[key].notna().all()
    np.testing.assert_array_equal(replay.obsm["X_umap"], selected.obsm["X_umap"])


@pytest.mark.parametrize("method", ["deviance", "hvg"])
def test_clustering_template_declares_n_comps(method):
    """Without n_comps scanpy defaults to 50 while the service uses n_pcs."""
    code = (
        ClusteringService()
        ._create_clustering_ir(
            resolution=1.0,
            n_pcs=30,
            feature_selection_method=method,
            algorithm="leiden",
        )
        .render()
    )
    assert "n_comps=30" in code
    assert "sc.tl.pca(adata_selected, svd_solver='arpack', n_comps=30)" in code


@pytest.mark.parametrize("method", ["deviance", "hvg"])
def test_clustering_template_runs_marker_genes(method):
    """The service ranks marker genes in the same call, so uns must match."""
    code = (
        ClusteringService()
        ._create_clustering_ir(
            resolution=1.0, feature_selection_method=method, algorithm="leiden"
        )
        .render()
    )
    assert "sc.tl.rank_genes_groups(adata, 'leiden'" in code


@pytest.mark.parametrize("method", ["deviance", "hvg"])
def test_clustering_template_sets_the_uns_keys_the_service_sets(method):
    code = (
        ClusteringService()
        ._create_clustering_ir(
            resolution=1.0, feature_selection_method=method, algorithm="leiden"
        )
        .render()
    )
    for key in ("resolutions_tested", "clustering_results", "umap_distance_warning"):
        assert f"adata.uns['{key}']" in code, f"missing uns['{key}']"


# --- EnhancedSingleCellService.annotate_cell_types -------------------------


def test_annotation_template_is_not_a_stub():
    """It documented four confidence columns and computed none of them.

    The template carried a literal '# ... (confidence calculation logic) ...'
    placeholder, so a replay produced only 'cell_type' and could not reproduce
    the confidence figures the session reported.
    """
    code = (
        EnhancedSingleCellService()
        ._create_annotation_ir(
            reference_markers={"T cells": ["CD3D"]}, cluster_key="leiden_res1_0"
        )
        .render()
    )

    assert "confidence calculation logic" not in code, "placeholder still present"
    # It delegates to the service rather than restating the vectorised scorer.
    assert "EnhancedSingleCellService" in code
    assert "annotate_cell_types(" in code
    # And it must pass the resolved cluster key through.
    assert "leiden_res1_0" in code


def _replay(ir, source):
    """Execute only generated analysis code on an isolated synthetic object."""
    import numpy as np
    import scanpy as sc

    namespace = {"adata": source.copy(), "np": np, "sc": sc}
    for helper in ir.helper_code or []:
        exec(helper, namespace)  # nosec B102: trusted IR on synthetic fixtures
    exec(ir.render(), namespace)  # nosec B102: trusted IR on synthetic fixtures
    return namespace["adata"]


@pytest.mark.parametrize("annotated", ["none", "matching", "different"])
def test_quality_replay_leaves_existing_gene_annotations_alone(annotated):
    """The service never reads or rewrites var['mt'] / var['ribo'].

    A data adapter adds both columns on load. The template used to delete
    them (and, when present, read its masks from them), so a replay changed
    the column order, or the QC metrics when the columns disagreed with the
    gene names.
    """
    import anndata as ad
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(3)
    names = [f"MT-{i}" for i in range(4)] + [f"RPL{i}" for i in range(4)]
    names += [f"G{i}" for i in range(40)]
    var = pd.DataFrame(index=names)
    if annotated != "none":
        is_mt = np.array([n.startswith("MT-") for n in names])
        is_ribo = np.array([n.startswith("RPL") for n in names])
        if annotated == "different":
            is_mt, is_ribo = is_ribo, np.zeros(len(names), dtype=bool)
        var["mt"] = is_mt
        var["ribo"] = is_ribo
        var["other"] = np.arange(len(names))
    source = ad.AnnData(
        X=rng.poisson(3, size=(60, len(names))).astype(np.float32),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(60)]),
        var=var,
    )
    live, _, ir = QualityService().assess_quality(
        source.copy(), min_genes=5, max_genes=60, max_mt_pct=50.0
    )
    replayed = _replay(ir, source)
    assert list(replayed.var.columns) == list(live.var.columns)
    pd.testing.assert_frame_equal(replayed.var, live.var)
    pd.testing.assert_frame_equal(replayed.obs, live.obs)


@pytest.mark.parametrize("method", ["log1p", "sctransform_like"])
@pytest.mark.parametrize("sparse_input", [False, True])
def test_filter_normalization_replay_matches_live(method, sparse_input):
    import anndata as ad
    import numpy as np
    import pandas as pd
    from scipy import sparse

    counts = np.array(
        [
            [1, 5, 3, 2],
            [3, 1, 4, 9],
            [2, 2, 8, 1],
            [9, 3, 1, 5],
            [4, 7, 2, 3],
            [1, 4, 6, 7],
        ],
        dtype=float,
    )
    source = ad.AnnData(
        X=sparse.csr_matrix(counts) if sparse_input else counts,
        var=pd.DataFrame(index=["MT-ND1", "RPS1", "CD3D", "MS4A1"]),
    )
    live, _, ir = PreprocessingService().filter_and_normalize_cells(
        source,
        min_genes_per_cell=1,
        max_genes_per_cell=10,
        min_cells_per_gene=1,
        max_mito_percent=100,
        max_ribo_percent=100,
        normalization_method=method,
        target_sum=100,
    )
    replay = _replay(ir, source)
    dense = lambda x: x.toarray() if sparse.issparse(x) else np.asarray(x)
    np.testing.assert_allclose(dense(replay.X), dense(live.X))
    np.testing.assert_allclose(dense(replay.raw.X), dense(live.raw.X))
    pd.testing.assert_frame_equal(replay.obs, live.obs)
    pd.testing.assert_index_equal(replay.var_names, live.var_names)


@pytest.mark.parametrize("sparse_input", [False, True])
@pytest.mark.parametrize(
    "branch", ["cpm", "deseq2_layer", "deseq2_fallback", "vst_layer", "vst_fallback"]
)
def test_bulk_normalization_replay_matches_live(branch, sparse_input, monkeypatch):
    """Force both supported pyDESeq2 layer layouts, not just the default one."""
    import anndata as ad
    import numpy as np
    import pandas as pd
    import pydeseq2.dds
    from scipy import sparse

    from lobster.services.analysis.bulk_preprocessing_service import (
        BulkPreprocessingService,
    )

    class Dataset:
        def __init__(self, counts, metadata, design):
            assert design == "~1"
            self.obs = pd.DataFrame(
                {"size_factors": [2.0, 4.0, 1.0]}, index=counts.index
            )
            self.layers = {}
            if branch.endswith("_layer"):
                # Distinct from counts / size_factors: ensure the layer wins.
                self.layers["normed_counts"] = counts.to_numpy() + 10.0
                self.layers["vst_counts"] = counts.to_numpy() + 20.0

        def fit_size_factors(self):
            pass

        def fit_genewise_dispersions(self):
            pass

        def fit_dispersion_trend(self):
            pass

        def fit_MAP_dispersions(self):
            pass

        def vst(self):
            pass

    monkeypatch.setattr(pydeseq2.dds, "DeseqDataSet", Dataset)
    counts = np.array([[2.5, 4, 8], [8, 4.5, 12], [0, 0, 0]])
    source = ad.AnnData(X=sparse.csr_matrix(counts) if sparse_input else counts)
    method = branch.split("_")[0]
    live, _, ir = BulkPreprocessingService().normalize_counts(
        source, method=method, target_sum=100
    )
    replay = _replay(ir, source)
    np.testing.assert_allclose(replay.X, live.X)
    np.testing.assert_array_equal(replay.layers["counts"], live.layers["counts"])
    pd.testing.assert_frame_equal(replay.obs, live.obs)
    pd.testing.assert_frame_equal(replay.var, live.var)


@pytest.mark.parametrize("method", ["deseq2", "vst"])
def test_bulk_normalization_real_pydeseq2_replay(method, monkeypatch):
    import anndata as ad
    import numpy as np
    import pydeseq2.dds

    from lobster.services.analysis.bulk_preprocessing_service import (
        BulkPreprocessingService,
    )

    dataset = pydeseq2.dds.DeseqDataSet
    monkeypatch.setattr(
        pydeseq2.dds, "DeseqDataSet", lambda **kwargs: dataset(**kwargs, n_cpus=1)
    )
    counts = np.random.default_rng(41).negative_binomial(5, 0.25, size=(8, 30)) + 1
    source = ad.AnnData(X=counts.astype(float))
    live, _, ir = BulkPreprocessingService().normalize_counts(source, method=method)
    replay = _replay(ir, source)
    np.testing.assert_allclose(replay.X, live.X)
    np.testing.assert_array_equal(replay.layers["counts"], live.layers["counts"])
