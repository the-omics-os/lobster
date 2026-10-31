"""Regression tests for cluster-key mismatches between clustering and downstream readers.

The clustering service names its columns by resolution (``leiden_res0_5``,
``louvain_res1_0``, ``leiden_subcluster``) while downstream readers may receive a
literal ``"leiden"`` default. Where that column is absent, annotation can fail
immediately after clustering succeeds.

Resolution order under test, in priority order:

1. explicit ``cluster_key``
2. exact ``leiden`` / ``louvain``
3. the *unique* ``leiden_res*`` column
4. raise ONLY when genuinely ambiguous

The implementation accounts for paths that write an unsuffixed alias and paths
that do not. Readers must resolve a real column rather than assume one exists.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.core.utils.cluster_keys import (
    ClusterKeyError,
    list_cluster_key_candidates,
    resolve_cluster_key,
)


def _adata_with_cluster_columns(*names: str, n_obs: int = 30) -> "ad.AnnData":
    rng = np.random.default_rng(0)
    adata = ad.AnnData(X=rng.random((n_obs, 5)))
    for name in names:
        adata.obs[name] = pd.Categorical([str(i % 3) for i in range(n_obs)])
    return adata


# --- 1. explicit cluster_key wins -----------------------------------------


def test_explicit_key_wins_even_when_others_exist():
    adata = _adata_with_cluster_columns("leiden", "leiden_res0_5", "leiden_res1_0")
    assert resolve_cluster_key(adata, "leiden_res1_0") == "leiden_res1_0"


def test_explicit_non_cluster_column_is_accepted():
    """Seurat/other conventions must work: the caller knows best."""
    adata = _adata_with_cluster_columns("leiden")
    adata.obs["RNA_snn_res.1"] = pd.Categorical(["a"] * adata.n_obs)
    assert resolve_cluster_key(adata, "RNA_snn_res.1") == "RNA_snn_res.1"


def test_explicit_missing_key_raises_and_lists_candidates():
    """A wrong explicit key must fail loudly, never be silently replaced."""
    adata = _adata_with_cluster_columns("leiden_res0_5")
    with pytest.raises(ClusterKeyError) as excinfo:
        resolve_cluster_key(adata, "leiden")
    message = str(excinfo.value)
    assert "leiden" in message
    assert "leiden_res0_5" in message, "the error must name the available column"


# --- 2. exact leiden/louvain ----------------------------------------------


def test_exact_leiden_preferred_over_suffixed():
    adata = _adata_with_cluster_columns("leiden", "leiden_res0_5", "leiden_res1_0")
    assert resolve_cluster_key(adata) == "leiden"


def test_exact_louvain_resolves_without_leiden_present():
    """The louvain path must resolve an actual clustering column."""
    adata = _adata_with_cluster_columns("louvain", "louvain_res0_5")
    assert resolve_cluster_key(adata) == "louvain"


def test_both_exact_columns_is_ambiguous_unless_algorithm_given():
    adata = _adata_with_cluster_columns("leiden", "louvain")
    with pytest.raises(ClusterKeyError):
        resolve_cluster_key(adata)
    assert resolve_cluster_key(adata, algorithm="louvain") == "louvain"
    assert resolve_cluster_key(adata, algorithm="leiden") == "leiden"


# --- 3. the unique leiden_res* column -------------------------------------


def test_unique_suffixed_column_resolves():
    """The writer and reader must agree on the cluster column."""
    adata = _adata_with_cluster_columns("leiden_res0_5")
    assert resolve_cluster_key(adata) == "leiden_res0_5"


def test_unique_subcluster_column_resolves():
    adata = _adata_with_cluster_columns("leiden_subcluster")
    assert resolve_cluster_key(adata) == "leiden_subcluster"


def test_algorithm_disambiguates_across_algorithms():
    adata = _adata_with_cluster_columns("leiden_res0_5", "louvain_res0_5")
    assert resolve_cluster_key(adata, algorithm="leiden") == "leiden_res0_5"
    assert resolve_cluster_key(adata, algorithm="louvain") == "louvain_res0_5"


# --- 4. raise only when genuinely ambiguous -------------------------------


def test_multiple_suffixed_columns_is_genuinely_ambiguous():
    adata = _adata_with_cluster_columns("leiden_res0_5", "leiden_res1_0")
    with pytest.raises(ClusterKeyError) as excinfo:
        resolve_cluster_key(adata)
    message = str(excinfo.value)
    assert "leiden_res0_5" in message and "leiden_res1_0" in message


def test_no_cluster_columns_raises_with_actionable_message():
    rng = np.random.default_rng(0)
    adata = ad.AnnData(X=rng.random((10, 3)))
    with pytest.raises(ClusterKeyError) as excinfo:
        resolve_cluster_key(adata)
    assert "clustering" in str(excinfo.value).lower()


def test_cluster_key_error_is_a_value_error():
    """Existing `except ValueError` handlers around annotation must still work."""
    assert issubclass(ClusterKeyError, ValueError)


def test_list_cluster_key_candidates_is_sorted_and_filtered():
    adata = _adata_with_cluster_columns("leiden_res1_0", "leiden", "louvain_res0_5")
    adata.obs["total_counts"] = np.arange(adata.n_obs)
    assert list_cluster_key_candidates(adata) == [
        "leiden",
        "leiden_res1_0",
        "louvain_res0_5",
    ]


# --- end-to-end: annotation after clustering ------------------------------


def test_annotation_resolves_suffixed_key_written_by_clustering():
    """Only a suffixed column exists in this end-to-end scenario.

    Pre-fix this raised "No clustering results found" with a 'leiden' default
    against an AnnData whose only cluster column was 'leiden_res0_5'.
    """
    pytest.importorskip("lobster.services.analysis.enhanced_singlecell_service")
    from lobster.services.analysis.enhanced_singlecell_service import (
        EnhancedSingleCellService,
    )

    rng = np.random.default_rng(0)
    n_obs = 60
    genes = ["CD3D", "CD3E", "MS4A1", "CD79A", "LYZ", "NKG7", "GNLY", "FCGR3A"]
    X = rng.random((n_obs, len(genes))) * 5
    adata = ad.AnnData(X=X)
    adata.var_names = genes
    adata.obs_names = [f"cell{i}" for i in range(n_obs)]
    # The writer's actual output: a resolution-suffixed column and no alias.
    adata.obs["leiden_res0_5"] = pd.Categorical([str(i % 3) for i in range(n_obs)])

    service = EnhancedSingleCellService()
    annotated, stats, ir = service.annotate_cell_types(adata)

    assert stats["cluster_key"] == "leiden_res0_5"
    assert "cell_type" in annotated.obs.columns
    # The IR must record the column actually read, so the exported notebook
    # replays against a column that exists.
    assert "leiden_res0_5" in str(ir.parameters) or "leiden_res0_5" in str(ir.code)


def test_clustering_ir_template_names_the_same_column_the_service_writes(
    tmp_path, monkeypatch
):
    """Execute complete notebooks: runtime-generated keys must match the service."""
    from types import SimpleNamespace

    import nbformat

    from lobster.core.notebooks.exporter import NotebookExporter
    from lobster.core.provenance.provenance import ProvenanceTracker
    from lobster.services.analysis.clustering_service import ClusteringService

    for method in ("deviance", "hvg"):
        source = ad.AnnData(
            np.random.default_rng(42)
            .negative_binomial(2, 0.5, (48, 600))
            .astype(np.float32),
            obs=pd.DataFrame(index=[f"cell_{i}" for i in range(48)]),
            var=pd.DataFrame(index=[f"gene_{i}" for i in range(600)]),
        )
        result, _, ir = ClusteringService().cluster_and_visualize(
            source,
            resolution=0.5,
            feature_selection_method=method,
            algorithm="leiden",
            n_neighbors=5,
            n_pcs=5,
        )
        workspace = tmp_path / method
        workspace.mkdir()
        source.write_h5ad(workspace / "dataset.h5ad")
        tracker = ProvenanceTracker()
        tracker.create_activity(activity_type="cluster_cells", agent="test", ir=ir)
        context = SimpleNamespace(
            workspace_path=workspace,
            workspace_dir=workspace,
            modalities={"fixture": source},
        )
        path = NotebookExporter(tracker, context).export(f"cluster_keys_{method}")
        notebook = nbformat.read(path, as_version=4)
        monkeypatch.chdir(workspace)
        namespace = {}
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                exec(
                    compile(cell.source, f"cluster_keys_cell_{index}", "exec"),
                    namespace,
                )  # nosec B102
        replay = namespace["adata"]
        assert replay.shape == result.shape == source.shape
        assert (
            list(replay.obs_names) == list(result.obs_names) == list(source.obs_names)
        )
        assert (
            list(replay.var_names) == list(result.var_names) == list(source.var_names)
        )
        service_keys = {key for key in result.obs if key.startswith("leiden")}
        replay_keys = {key for key in replay.obs if key.startswith("leiden")}
        assert {"leiden", "leiden_res0_5"} <= service_keys
        assert replay_keys == service_keys
        for key in service_keys:
            assert replay.obs[key].notna().all()
            pd.testing.assert_series_equal(replay.obs[key], result.obs[key])
        np.testing.assert_allclose(replay.obsm["X_umap"], result.obsm["X_umap"])


def test_clustering_ir_template_respects_louvain():
    """Check key construction structurally, without executing optional Louvain."""
    import ast
    import inspect
    import textwrap

    from lobster.services.analysis.clustering_service import ClusteringService

    service = ClusteringService()
    service_tree = ast.parse(
        textwrap.dedent(inspect.getsource(service._perform_clustering))
    )
    service_keys = [
        node.value
        for node in ast.walk(service_tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "key_name"
            for target in node.targets
        )
    ]
    assert service_keys, "service key construction must be checked, not guessed"
    for resolution in (1.0, 0.5):
        ir = service._create_clustering_ir(
            resolution=resolution,
            feature_selection_method="deviance",
            algorithm="louvain",
        )
        code = ir.render()
        assert "sc.tl.louvain(" in code
        template_tree = ast.parse(code)
        template_keys = [
            node.value
            for node in ast.walk(template_tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id in {"cluster_key", "key", "primary_key"}
                for target in node.targets
            )
        ]
        assert len(template_keys) == 3, "check clustering, transfer and primary keys"
        namespace = {
            "algorithm": "louvain",
            "res": resolution,
            "resolutions_to_test": [resolution],
        }
        expected = f"louvain_res{resolution}".replace(".", "_")
        for expression in service_keys + template_keys:
            actual = eval(
                compile(ast.Expression(expression), "cluster_key_expression", "eval"),
                {"__builtins__": {}},
                namespace,
            )  # nosec B307: only owned key expressions, not the clustering algorithm
            assert actual == expected


def test_annotation_still_raises_when_no_clustering_was_run():
    """The fix must not paper over a genuinely missing clustering step."""
    pytest.importorskip("lobster.services.analysis.enhanced_singlecell_service")
    from lobster.services.analysis.enhanced_singlecell_service import (
        EnhancedSingleCellService,
        SingleCellError,
    )

    rng = np.random.default_rng(0)
    adata = ad.AnnData(X=rng.random((20, 4)))
    adata.var_names = ["CD3D", "MS4A1", "LYZ", "NKG7"]

    with pytest.raises(SingleCellError):
        EnhancedSingleCellService().annotate_cell_types(adata)
