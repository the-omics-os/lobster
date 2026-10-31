"""Benign S5 requested/applied clustering contract controls.

Real Scanpy numerical operations and actual tool/service/DM/IR are exercised.
Only model construction/graph capture are fake; no model call or freshness claim.
"""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from scipy import sparse

from lobster.agents.transcriptomics.prompts import create_transcriptomics_expert_prompt
from lobster.core.provenance.analysis_ir import AnalysisStep
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.core.utils.h5ad_utils import sanitize_value
from lobster.services.analysis import clustering_service as service_module
from lobster.services.analysis.clustering_service import (
    ClusteringError,
    ClusteringService,
)
from lobster.testing.provenance_replay import replay_ir

parent_module = importlib.import_module(
    "lobster.agents.transcriptomics.transcriptomics_expert"
)


def _normalized(csr=False):
    rng = np.random.default_rng(23)
    rates = np.linspace(1.0, 15.0, 96)
    x = rng.poisson(rates, size=(96, 96)).astype(np.float64)
    obs = pd.DataFrame({"source": "synthetic"}, index=[f"cell_{i}" for i in range(96)])
    var = pd.DataFrame(
        {"highly_variable": np.arange(96) < 64}, index=[f"GENE_{i}" for i in range(96)]
    )
    result = ad.AnnData(sparse.csr_matrix(x) if csr else x, obs=obs, var=var)
    sc.pp.normalize_total(result, target_sum=10000)
    sc.pp.log1p(result)
    return result


@pytest.fixture(params=[False, True], ids=["dense", "csr"])
def prepared(request):
    source = _normalized(request.param)
    result, stats, _ = ClusteringService().run_pca(source, n_comps=50)
    assert stats["n_comps_computed"] == 50
    assert stats["n_features_used"] == 64
    assert result.obsm["X_pca"].shape == (96, 50)
    assert np.isfinite(result.obsm["X_pca"]).all()
    return result


def _dense(x):
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def _same_source(source, snapshot):
    np.testing.assert_array_equal(_dense(source.X), _dense(snapshot.X))
    pd.testing.assert_frame_equal(source.obs, snapshot.obs)
    pd.testing.assert_frame_equal(source.var, snapshot.var)
    np.testing.assert_array_equal(source.obsm["X_pca"], snapshot.obsm["X_pca"])
    np.testing.assert_array_equal(_dense(source.raw.X), _dense(snapshot.raw.X))


def _tools(tmp_path, monkeypatch, source):
    dm = DataManagerV2(workspace_path=tmp_path / "workspace")
    dm.store_modality("prepared", source.copy(), step="reduced")
    settings = SimpleNamespace(get_agent_llm_params=lambda name: {})
    monkeypatch.setattr(parent_module, "get_settings", lambda: settings)
    monkeypatch.setattr(parent_module, "create_llm", lambda *args, **kwargs: object())
    monkeypatch.setattr(parent_module, "create_react_agent", lambda **kwargs: kwargs)
    switch = importlib.import_module("lobster.tools.code_tool_switch")
    monkeypatch.setattr(
        switch, "attach_code_tool", lambda tools, *args, **kwargs: tools
    )
    graph = parent_module.transcriptomics_expert(dm)
    return dm, {tool.name: tool for tool in graph["tools"]}


def _json_blocks(response):
    return [
        json.loads(part.split("\n```", 1)[0])
        for part in response.split("```json\n")[1:]
    ]


@pytest.fixture(scope="session", autouse=True)
def _require_owned_origins():
    # Once installed at the registered public-safe test path, stop at SETUP
    # rather than let editable-source drift masquerade as feature RED/GREEN.
    root = Path(__file__).resolve().parents[4]
    assert Path(service_module.__file__).resolve().is_relative_to(root)
    assert Path(parent_module.__file__).resolve().is_relative_to(root)


def test_clustering_origins_are_this_checkout():
    root = Path(__file__).resolve().parents[4]
    assert Path(service_module.__file__).resolve().is_relative_to(root)
    assert Path(parent_module.__file__).resolve().is_relative_to(root)


def test_actual_supported_selected_pca50_positive(prepared):
    assert prepared.shape == (96, 96)
    assert prepared.var["highly_variable"].sum() == 64
    assert prepared.obsm["X_pca"].shape == (96, 50)
    assert "pca" in prepared.uns and prepared.raw is not None


@pytest.mark.parametrize("demo_mode", [False, True], ids=["ordinary", "demo"])
def test_explicit_prepared_pca_first40_and_neighbors10_no_upstream_repeat(
    prepared, demo_mode
):
    source = prepared
    snapshot = source.copy()
    reference = source.copy()
    sc.pp.neighbors(reference, n_neighbors=10, n_pcs=40, use_rep="X_pca")
    with (
        patch.object(
            sc.pp, "normalize_total", wraps=sc.pp.normalize_total
        ) as normalize,
        patch.object(sc.pp, "log1p", wraps=sc.pp.log1p) as log,
        patch.object(
            sc.pp, "highly_variable_genes", wraps=sc.pp.highly_variable_genes
        ) as hvg,
        patch.object(sc.pp, "scale", wraps=sc.pp.scale) as scale,
        patch.object(sc.tl, "pca", wraps=sc.tl.pca) as pca,
        patch.object(sc.pp, "neighbors", wraps=sc.pp.neighbors) as neighbors,
    ):
        result, stats, ir = ClusteringService().cluster_and_visualize(
            source,
            resolution=0.7,
            use_rep="X_pca",
            n_neighbors=10,
            n_pcs=40,
            batch_correction=False,
            demo_mode=demo_mode,
            skip_steps=["marker_genes"],
        )
        assert (
            normalize.call_count
            == log.call_count
            == hvg.call_count
            == scale.call_count
            == pca.call_count
            == 0
        )
        assert neighbors.call_count == 1
        assert neighbors.call_args.kwargs["n_neighbors"] == 10
        assert neighbors.call_args.kwargs["n_pcs"] == 40
        assert neighbors.call_args.kwargs["use_rep"] == "X_pca"
    _same_source(source, snapshot)
    assert result.shape == source.shape
    assert "leiden" in result.obs and "X_umap" in result.obsm
    np.testing.assert_array_equal(result.obsm["X_pca"], source.obsm["X_pca"])
    np.testing.assert_allclose(
        result.obsp["distances"].toarray(), reference.obsp["distances"].toarray()
    )
    np.testing.assert_allclose(
        result.obsp["connectivities"].toarray(),
        reference.obsp["connectivities"].toarray(),
    )
    assert stats["applied_parameters"]["n_neighbors"] == 10
    assert stats["applied_parameters"]["n_pcs"] == 40
    assert stats["applied_parameters"]["use_rep"] == "X_pca"
    assert ir.parameters["n_neighbors"] == 10 and ir.parameters["n_pcs"] == 40
    namespace = {"adata": source.copy()}
    for statement in ir.imports:
        exec(compile(statement, "<clustering-ir-import>", "exec"), namespace)
    for helper in getattr(ir, "helper_code", []):
        exec(compile(helper, "<clustering-ir-helper>", "exec"), namespace)
    replayed = replay_ir(ir, **namespace)["adata"]
    np.testing.assert_allclose(
        replayed.obsp["distances"].toarray(), result.obsp["distances"].toarray()
    )
    np.testing.assert_allclose(
        replayed.obsp["connectivities"].toarray(),
        result.obsp["connectivities"].toarray(),
    )
    np.testing.assert_array_equal(replayed.obs["leiden"], result.obs["leiden"])
    assert "rank_genes_groups" not in replayed.uns


def test_actual_parent_tool_exposes_explicit_dimensions(
    tmp_path, monkeypatch, prepared
):
    _, tools = _tools(tmp_path, monkeypatch, prepared)
    assert {"n_neighbors", "n_pcs", "use_rep"} <= set(tools["cluster_cells"].args)


@pytest.mark.parametrize(
    "bad_rep", ["missing", "too_narrow", "nonfinite", "nonnumeric", "zero_width"]
)
def test_prepared_representation_refuses_without_fallback_or_analysis(
    prepared, bad_rep
):
    source = prepared.copy()
    key = "X_pca"
    if bad_rep == "missing":
        key = "missing_representation"
    elif bad_rep == "too_narrow":
        source.obsm[key] = source.obsm[key][:, :39].copy()
    elif bad_rep == "nonfinite":
        source.obsm[key][0, 0] = np.nan
    elif bad_rep == "nonnumeric":
        source.obsm[key] = np.full((96, 50), "benign text")
    elif bad_rep == "zero_width":
        source.obsm[key] = np.empty((96, 0))
    snapshot = source.copy()
    with (
        patch.object(
            sc.pp,
            "normalize_total",
            side_effect=AssertionError("unexpected normalization"),
        ) as normalize,
        patch.object(sc.tl, "pca", side_effect=AssertionError("unexpected PCA")) as pca,
        patch.object(
            sc.pp, "neighbors", side_effect=AssertionError("unexpected neighbors")
        ) as neighbors,
    ):
        with pytest.raises((ClusteringError, ValueError)):
            ClusteringService().cluster_and_visualize(
                source, use_rep=key, n_neighbors=10, n_pcs=40, batch_correction=False
            )
        assert normalize.call_count == pca.call_count == neighbors.call_count == 0
    _same_source(source, snapshot)


@pytest.mark.parametrize("requested", [0, 1, 97, 98])
def test_explicit_unhonorable_neighbors_refuse(prepared, requested):
    with patch.object(
        sc.pp, "neighbors", side_effect=AssertionError("unexpected neighbors")
    ) as neighbors:
        with pytest.raises((ClusteringError, ValueError)):
            ClusteringService().cluster_and_visualize(
                prepared,
                use_rep="X_pca",
                n_neighbors=requested,
                n_pcs=40,
                batch_correction=False,
            )
        assert neighbors.call_count == 0


def test_exact_full_hvg_feature_request_refuses_before_upstream(prepared):
    with patch.object(
        sc.pp, "normalize_total", side_effect=AssertionError("unexpected normalization")
    ) as normalize:
        with pytest.raises(
            (ClusteringError, ValueError),
            match="(?i)(n_features|exact|unsupported|not supported)",
        ):
            ClusteringService().cluster_and_visualize(
                prepared,
                feature_selection_method="hvg",
                n_features=17,
                batch_correction=False,
            )
        assert normalize.call_count == 0


def test_explicit_unhonorable_standalone_pca_refuses_before_scaling(prepared):
    snapshot = prepared.copy()
    with (
        patch.object(sc.pp, "scale", wraps=sc.pp.scale) as scale,
        patch.object(sc.tl, "pca", wraps=sc.tl.pca) as pca,
    ):
        with pytest.raises((ClusteringError, ValueError)):
            ClusteringService().run_pca(prepared, n_comps=64)
        assert scale.call_count == pca.call_count == 0
    _same_source(prepared, snapshot)


def test_unset_prepared_dimensions_keep_ordinary_defaults_and_disclose_adjustment(
    prepared,
):
    source = prepared.copy()
    source.obsm["X_pca"] = source.obsm["X_pca"][:, :12].copy()
    with patch.object(sc.pp, "neighbors", wraps=sc.pp.neighbors) as neighbors:
        _, stats, ir = ClusteringService().cluster_and_visualize(
            source, use_rep="X_pca", demo_mode=True, skip_steps=["marker_genes"]
        )
    assert neighbors.call_args.kwargs["n_neighbors"] == 15
    assert neighbors.call_args.kwargs["n_pcs"] == 12
    assert stats["requested_parameters"]["n_neighbors"] is None
    assert stats["requested_parameters"]["n_pcs"] is None
    assert stats["applied_parameters"]["n_neighbors"] == 15
    assert stats["applied_parameters"]["n_pcs"] == 12
    assert stats["parameter_adjustments"]
    assert ir.parameters["n_neighbors"] == 15 and ir.parameters["n_pcs"] == 12


def test_named_prepared_rep_and_explicit_batch_correction_refuse(prepared):
    with (
        patch.object(
            sc.pp, "normalize_total", wraps=sc.pp.normalize_total
        ) as normalize,
        patch.object(sc.pp, "neighbors", wraps=sc.pp.neighbors) as neighbors,
    ):
        with pytest.raises(
            (ClusteringError, ValueError), match="(?i)(batch|prepared|representation)"
        ):
            ClusteringService().cluster_and_visualize(
                prepared, use_rep="X_pca", batch_correction=True
            )
        assert normalize.call_count == neighbors.call_count == 0


def test_parent_unset_batch_skips_for_named_prepared_rep_and_reports_reason(
    tmp_path, monkeypatch, prepared
):
    source = prepared.copy()
    source.obs["batch"] = pd.Categorical(["a"] * 48 + ["b"] * 48)
    dm, tools = _tools(tmp_path, monkeypatch, source)
    assert {"n_neighbors", "n_pcs"} <= set(tools["cluster_cells"].args)
    with patch.object(
        sc.pp, "normalize_total", wraps=sc.pp.normalize_total
    ) as normalize:
        response = tools["cluster_cells"].invoke(
            {
                "modality_name": "prepared",
                "use_rep": "X_pca",
                "n_neighbors": 10,
                "n_pcs": 40,
                "demo_mode": True,
                "save_result": False,
            }
        )
        assert normalize.call_count == 0
    assert "using your prepared X_pca; batch correction not applied" in response
    blocks = _json_blocks(response)
    assert blocks and blocks[-1]["applied_parameters"]["batch_correction"] is False
    assert "prepared_clustered" in dm.list_modalities()


def _activity(dm, name):
    return next(a for a in reversed(dm.provenance.activities) if a["type"] == name)


def _replay_recorded(record, source):
    ir = AnalysisStep.from_dict(record)
    namespace = {"adata": source.copy()}
    for statement in ir.imports:
        exec(compile(statement, "<recorded-clustering-import>", "exec"), namespace)
    for helper in getattr(ir, "helper_code", []):
        exec(compile(helper, "<recorded-clustering-helper>", "exec"), namespace)
    return replay_ir(ir, **namespace)["adata"]


def test_shared_pca_actual_json_features_identity_and_recorded_ir(
    tmp_path, monkeypatch, prepared
):
    source = _normalized(sparse.issparse(prepared.X))
    dm, tools = _tools(tmp_path, monkeypatch, source)
    response = tools["run_pca"].invoke({"modality_name": "prepared", "n_comps": 50})
    result = dm.get_modality("prepared_pca")
    assert result.obsm["X_pca"].shape == (
        96,
        50,
    )  # Actual native positive before reporting assertions.
    blocks = _json_blocks(response)
    assert blocks and blocks[-1]["applied_parameters"]["n_comps"] == 50
    assert blocks[-1]["applied_parameters"]["n_features_used"] == 64
    assert blocks[-1]["result_modality_name"] == "prepared_pca"
    activity = _activity(dm, "run_pca")
    assert activity["parameters"]["n_comps"] == "50"
    assert activity["parameters"]["n_features_used"] == "64"
    assert activity["ir"]["parameters"]["n_comps"] == 50
    replayed = _replay_recorded(activity["ir"], source)
    np.testing.assert_allclose(replayed.obsm["X_pca"], result.obsm["X_pca"])
    np.testing.assert_array_equal(_dense(source.X), _dense(result.X))


def test_shared_hvg_selection_reports_observed_count_not_fabricated_count(
    tmp_path, monkeypatch, prepared
):
    dm, tools = _tools(tmp_path, monkeypatch, _normalized(sparse.issparse(prepared.X)))
    response = tools["select_variable_features"].invoke(
        {
            "modality_name": "prepared",
            "method": "hvg",
            "n_top_genes": 40,
            "flavor": "seurat",
        }
    )
    result = dm.get_modality("prepared_hvg_selected")
    observed = int(result.var["highly_variable"].sum())
    assert (
        0 < observed <= result.n_vars
    )  # Native selected mask is the oracle, not request40.
    blocks = _json_blocks(response)
    assert (
        blocks and blocks[-1]["applied_parameters"]["n_features_selected"] == observed
    )
    assert blocks[-1]["applied_parameters"]["flavor"] == "seurat"
    activity = _activity(dm, "select_variable_features")
    assert activity["parameters"]["n_features_selected"] == str(observed)
    assert activity["ir"]["parameters"]["flavor"] == "seurat"


def test_shared_named_representation_actual_graph_and_top_ir_parity(
    tmp_path, monkeypatch, prepared
):
    dm, tools = _tools(tmp_path, monkeypatch, prepared)
    assert "use_rep" in tools["compute_neighbors_and_embed"].args
    response = tools["compute_neighbors_and_embed"].invoke(
        {
            "modality_name": "prepared",
            "n_neighbors": 10,
            "n_pcs": 40,
            "use_rep": "X_pca",
        }
    )
    result = dm.get_modality("prepared_embedded")
    payload = _json_blocks(response)[-1]
    assert payload["applied_parameters"]["use_rep"] == "X_pca"
    assert payload["applied_parameters"]["n_pcs"] == 40
    activity = _activity(dm, "compute_neighbors_and_embed")
    for key in ("use_rep", "n_neighbors", "n_pcs"):
        assert activity["parameters"][key] == sanitize_value(
            activity["ir"]["parameters"][key]
        )
    replayed = _replay_recorded(activity["ir"], prepared)
    np.testing.assert_allclose(
        replayed.obsp["distances"].toarray(), result.obsp["distances"].toarray()
    )
    np.testing.assert_array_equal(result.obsm["X_pca"], prepared.obsm["X_pca"])


def test_parent_explicit_hvg_flavor_has_refusal_api(tmp_path, monkeypatch, prepared):
    dm, tools = _tools(tmp_path, monkeypatch, prepared)
    assert "flavor" in tools["cluster_cells"].args
    with patch.object(
        sc.pp, "normalize_total", wraps=sc.pp.normalize_total
    ) as normalize:
        response = tools["cluster_cells"].invoke(
            {
                "modality_name": "prepared",
                "feature_selection_method": "hvg",
                "flavor": "cell_ranger",
                "save_result": False,
                "batch_correction": False,
            }
        )
        assert normalize.call_count == 0
    assert response.startswith("Error") and "flavor" in response
    assert "prepared_clustered" not in dm.list_modalities()
    assert not any(a["type"] == "cluster_cells" for a in dm.provenance.activities)


def _raw_full(csr):
    rng = np.random.default_rng(29)
    rates = np.full((96, 96), 3.0)
    rates[:48, :32] = 22.0
    rates[48:, 32:64] = 22.0
    x = rng.poisson(rates).astype(np.float64)
    result = ad.AnnData(sparse.csr_matrix(x) if csr else x)
    result.obs_names = [f"cell_{i}" for i in range(96)]
    result.var_names = [f"GENE_{i}" for i in range(96)]
    return result


def test_full_low_gene_pipeline_honors_pcs_not_scanpy_auto_x(prepared):
    source = _raw_full(sparse.issparse(prepared.X))
    with patch.object(sc.pp, "neighbors", wraps=sc.pp.neighbors) as neighbors:
        result, stats, ir = ClusteringService().cluster_and_visualize(
            source, n_features=48, n_pcs=30, n_neighbors=10, skip_steps=["marker_genes"]
        )
    assert result.shape == source.shape and result.obsm["X_pca"].shape == (96, 30)
    assert neighbors.call_args.kwargs.get("use_rep") == "X_pca"
    assert stats["applied_parameters"]["n_pcs"] == 30
    assert ir.parameters["n_pcs"] == 30
    namespace = {"adata": source.copy()}
    for statement in ir.imports:
        exec(compile(statement, "<full-cluster-import>", "exec"), namespace)
    replayed = replay_ir(ir, **namespace)["adata"]
    np.testing.assert_allclose(
        replayed.obsp["distances"].toarray(), result.obsp["distances"].toarray()
    )
    assert replayed.shape == result.shape


def test_full_pipeline_recorded_skip_markers_matches_actual_branch(prepared):
    source = _raw_full(sparse.issparse(prepared.X))
    result, stats, ir = ClusteringService().cluster_and_visualize(
        source, n_features=64, skip_steps=["marker_genes"]
    )
    assert "rank_genes_groups" not in result.uns
    assert ir.parameters.get("find_markers") is False
    namespace = {"adata": source.copy()}
    for statement in ir.imports:
        exec(compile(statement, "<skip-cluster-import>", "exec"), namespace)
    replayed = replay_ir(ir, **namespace)["adata"]
    assert "rank_genes_groups" not in replayed.uns
    assert stats["applied_parameters"]["n_pcs"] == ir.parameters["n_pcs"]


def test_native_equal_size_neighbor_request_not_artificially_capped(prepared):
    result, stats, ir = ClusteringService().cluster_and_visualize(
        prepared, use_rep="X_pca", n_neighbors=96, n_pcs=40, skip_steps=["marker_genes"]
    )
    assert result.uns["neighbors"]["params"]["n_neighbors"] == 96
    assert stats["applied_parameters"]["n_neighbors"] == 96
    assert ir.parameters["n_neighbors"] == 96


def test_prepared_parent_prompt_requires_named_rep_and_hvg_only_pca():
    prompt = create_transcriptomics_expert_prompt()
    assert "use_rep='X_pca'" in prompt
    assert "HVG-only" in prompt and "run_pca" in prompt
    assert "suffix" in prompt and "prepared" in prompt


@pytest.mark.parametrize("helper_fails", [False, True], ids=["applied", "unchanged"])
def test_batch_attempt_status_and_same_helper_replayed(prepared, helper_fails):
    source = _raw_full(sparse.issparse(prepared.X))
    source.obs["batch"] = pd.Categorical(["a", "b"] * 48)
    if helper_fails:
        # Benign ordinary failure branch: existing helper returns its input.
        context = patch.object(
            ClusteringService,
            "_perform_batch_correction",
            lambda self, adata, key: adata,
        )
    else:
        context = patch.object(
            ClusteringService,
            "_perform_batch_correction",
            autospec=True,
            side_effect=ClusteringService._perform_batch_correction,
        )
    with context:
        result, stats, ir = ClusteringService().cluster_and_visualize(
            source,
            batch_correction=True,
            batch_key="batch",
            n_features=64,
            skip_steps=["marker_genes"],
        )
        assert result.shape == source.shape and result.obsm["X_pca"].shape[1] == 30
        status = stats["batch_correction_status"]
        assert status["attempted"] is True
        assert status["applied"] is (not helper_fails)
        assert status["failed"] is helper_fails
        assert status["effective_key"] == "batch"
        assert stats["batch_correction"] is (not helper_fails)
        assert ir.parameters["batch_correction_attempted"] is True
        assert ir.parameters["batch_key"] == "batch"
        replayed = _replay_recorded(ir.to_dict(), source)
    assert replayed.obs_names.tolist() == result.obs_names.tolist()
    np.testing.assert_allclose(_dense(replayed.X), _dense(result.X))
    np.testing.assert_allclose(
        replayed.obsp["distances"].toarray(), result.obsp["distances"].toarray()
    )


@pytest.mark.parametrize("reuse", [False, True], ids=["full", "prepared"])
def test_subsample_same_seed_and_cohort_in_both_ir_branches(prepared, reuse):
    source = prepared if reuse else _raw_full(sparse.issparse(prepared.X))
    reference = source.copy()
    sc.pp.subsample(reference, n_obs=48, random_state=42)
    result, stats, ir = ClusteringService().cluster_and_visualize(
        source,
        use_rep="X_pca" if reuse else None,
        subsample_size=48,
        n_features=None if reuse else 64,
        skip_steps=["marker_genes"],
    )
    assert result.n_obs == 48
    assert result.obs_names.tolist() == reference.obs_names.tolist()
    status = stats["subsample_status"]
    assert (
        status["performed"] is True and status["target"] == 48 and status["seed"] == 42
    )
    assert status["input_n_obs"] == 96 and status["result_n_obs"] == 48
    assert (
        ir.parameters["subsample_size"] == 48 and ir.parameters["subsample_seed"] == 42
    )
    replayed = _replay_recorded(ir.to_dict(), source)
    assert replayed.shape == result.shape
    assert replayed.obs_names.tolist() == result.obs_names.tolist()
    np.testing.assert_allclose(
        replayed.obsp["distances"].toarray(), result.obsp["distances"].toarray()
    )


def test_demo_reports_marked_versus_pca_used_and_replays_exact_branches(prepared):
    rng = np.random.default_rng(31)
    rates = np.full((96, 1200), 3.0)
    rates[:48, :400] = 22.0
    rates[48:, 400:800] = 22.0
    x = rng.poisson(rates).astype(np.float64)
    source = ad.AnnData(sparse.csr_matrix(x) if sparse.issparse(prepared.X) else x)
    source.obs_names = [f"cell_{i}" for i in range(96)]
    source.var_names = [f"GENE_{i}" for i in range(1200)]
    with patch.object(sc.tl, "umap", wraps=sc.tl.umap) as umap:
        result, stats, ir = ClusteringService().cluster_and_visualize(
            source, n_features=1100, demo_mode=True
        )
    assert int(result.var["highly_deviant"].sum()) == 1100
    assert (
        umap.call_args.kwargs["min_dist"] == 0.5
        and umap.call_args.kwargs["spread"] == 1.5
    )
    assert "rank_genes_groups" not in result.uns
    assert stats["features_marked"] == 1100 and stats["features_used"] == 1000
    assert "marker_genes" in stats["skipped_steps"]
    assert ir.parameters["demo_mode"] is True and ir.parameters["find_markers"] is False
    replayed = _replay_recorded(ir.to_dict(), source)
    assert (
        replayed.shape == result.shape
        and int(replayed.var["highly_deviant"].sum()) == 1100
    )
    np.testing.assert_allclose(
        replayed.obsp["distances"].toarray(), result.obsp["distances"].toarray()
    )
    np.testing.assert_allclose(replayed.obsm["X_umap"], result.obsm["X_umap"])
    assert "rank_genes_groups" not in replayed.uns


def test_standard_multi_resolution_actual_first_columns_and_replay_list(prepared):
    source = _raw_full(sparse.issparse(prepared.X))
    resolutions = [0.4, 0.8]
    result, stats, ir = ClusteringService().cluster_and_visualize(
        source, resolutions=resolutions, n_features=64, skip_steps=["marker_genes"]
    )
    assert {"leiden_res0_4", "leiden_res0_8", "leiden"} <= set(result.obs)
    assert stats["resolution"] == 0.4
    assert stats["applied_parameters"]["resolution"] == 0.4
    assert ir.parameters["resolutions"] == resolutions
    assert ir.parameters["resolution"] == 0.4
    replayed = _replay_recorded(ir.to_dict(), source)
    for key in ("leiden_res0_4", "leiden_res0_8", "leiden"):
        np.testing.assert_array_equal(replayed.obs[key], result.obs[key])
    assert replayed.uns["resolutions_tested"] == resolutions
    np.testing.assert_allclose(
        replayed.obsp["distances"].toarray(), result.obsp["distances"].toarray()
    )


def test_prepared_representation_multi_resolution_refuses_before_analysis(prepared):
    with patch.object(sc.pp, "neighbors", wraps=sc.pp.neighbors) as neighbors:
        with pytest.raises(
            (ClusteringError, ValueError), match="(?i)(resolutions|multi.resolution)"
        ):
            ClusteringService().cluster_and_visualize(
                prepared,
                use_rep="X_pca",
                resolutions=[0.4, 0.8],
                skip_steps=["marker_genes"],
            )
        assert neighbors.call_count == 0


@pytest.mark.parametrize(
    "helper_fails", [False, True], ids=["applied", "ordinary_failure"]
)
def test_final_parent_default_batch_json_top_ir_identity_and_real_helper_failure(
    tmp_path, monkeypatch, prepared, helper_fails
):
    source = _raw_full(sparse.issparse(prepared.X))
    source.obs["batch"] = pd.Categorical(["a", "b"] * 48)
    snapshot = source.copy()
    dm, tools = _tools(tmp_path, monkeypatch, source)
    context = (
        patch.object(
            ad, "concat", side_effect=ValueError("ordinary concat failure control")
        )
        if helper_fails
        else patch.object(ad, "concat", wraps=ad.concat)
    )
    with context:
        response = tools["cluster_cells"].invoke(
            {
                "modality_name": "prepared",
                "n_features": 64,
                "demo_mode": True,
                "save_result": False,
            }
        )
        assert "prepared_clustered" in dm.list_modalities()
        result = dm.get_modality("prepared_clustered")
        assert result.shape == source.shape and result.obsm["X_pca"].shape[1] == 30
        payload = _json_blocks(response)[-1]
        assert payload["requested_parameters"]["batch_correction"] is None
        assert payload["batch_correction_status"]["attempted"] is True
        assert payload["batch_correction_status"]["applied"] is (not helper_fails)
        assert payload["batch_correction_status"]["failed"] is helper_fails
        assert payload["batch_correction_status"]["effective_key"] == "batch"
        assert payload["applied_parameters"]["batch_correction"] is (not helper_fails)
        assert payload["result_modality_name"] == "prepared_clustered"
        assert payload["input_shape"] == payload["result_shape"] == [96, 96]
        activity = _activity(dm, "cluster_cells")
        assert (
            activity["parameters"]["result_modality_name"]
            == payload["result_modality_name"]
        )
        for key in (
            "features_marked",
            "features_used",
            "batch_correction_status",
            "subsample_status",
            "skipped_steps",
        ):
            assert activity["parameters"][key] == sanitize_value(payload[key])
            assert activity["ir"]["parameters"][key] == payload[key]
        assert activity["parameters"]["batch_correction"] == sanitize_value(
            not helper_fails
        )
        assert activity["ir"]["parameters"]["batch_correction_attempted"] is True
        replayed = _replay_recorded(activity["ir"], source)
    assert replayed.obs_names.tolist() == result.obs_names.tolist()
    np.testing.assert_allclose(_dense(replayed.X), _dense(result.X))
    np.testing.assert_allclose(
        replayed.obsp["distances"].toarray(), result.obsp["distances"].toarray()
    )
    np.testing.assert_array_equal(_dense(source.X), _dense(snapshot.X))
    pd.testing.assert_frame_equal(source.obs, snapshot.obs)
    pd.testing.assert_frame_equal(source.var, snapshot.var)


def test_final_hvg_only_shared_pca_then_parent_named_first40_no_repeat(
    tmp_path, monkeypatch, prepared
):
    source = _normalized(sparse.issparse(prepared.X))
    assert "X_pca" not in source.obsm and int(source.var.highly_variable.sum()) == 64
    dm, tools = _tools(tmp_path, monkeypatch, source)
    pca_response = tools["run_pca"].invoke({"modality_name": "prepared", "n_comps": 50})
    assert _json_blocks(pca_response)[-1]["result_modality_name"] == "prepared_pca"
    reduced = dm.get_modality("prepared_pca")
    assert reduced.obsm["X_pca"].shape == (96, 50)
    snapshot = reduced.copy()
    with (
        patch.object(
            sc.pp, "normalize_total", wraps=sc.pp.normalize_total
        ) as normalize,
        patch.object(
            sc.pp, "highly_variable_genes", wraps=sc.pp.highly_variable_genes
        ) as hvg,
        patch.object(sc.pp, "scale", wraps=sc.pp.scale) as scale,
        patch.object(sc.tl, "pca", wraps=sc.tl.pca) as pca,
    ):
        response = tools["cluster_cells"].invoke(
            {
                "modality_name": "prepared_pca",
                "use_rep": "X_pca",
                "n_neighbors": 10,
                "n_pcs": 40,
                "demo_mode": True,
                "save_result": False,
            }
        )
        assert (
            normalize.call_count
            == hvg.call_count
            == scale.call_count
            == pca.call_count
            == 0
        )
    result = dm.get_modality("prepared_pca_clustered")
    payload = _json_blocks(response)[-1]
    assert payload["input_modality_name"] == "prepared_pca"
    assert payload["result_modality_name"] == "prepared_pca_clustered"
    activity = _activity(dm, "cluster_cells")
    for key in ("use_rep", "n_neighbors", "n_pcs"):
        assert activity["parameters"][key] == sanitize_value(
            payload["applied_parameters"][key]
        )
        assert activity["ir"]["parameters"][key] == payload["applied_parameters"][key]
    assert payload["applied_parameters"]["n_pcs"] == 40
    assert payload["applied_parameters"]["batch_correction"] is False
    replayed = _replay_recorded(activity["ir"], reduced)
    np.testing.assert_allclose(
        replayed.obsp["distances"].toarray(), result.obsp["distances"].toarray()
    )
    np.testing.assert_array_equal(replayed.obs["leiden"], result.obs["leiden"])
    _same_source(reduced, snapshot)


def test_final_implicit_standalone_pca_adjusts_actual_capacity_discloses_and_replays(
    tmp_path, monkeypatch, prepared
):
    source = _normalized(sparse.issparse(prepared.X))
    source.var["highly_variable"] = np.arange(96) < 8
    snapshot = source.copy()
    dm, tools = _tools(tmp_path, monkeypatch, source)
    response = tools["run_pca"].invoke({"modality_name": "prepared"})
    result = dm.get_modality("prepared_pca")
    assert result.obsm["X_pca"].shape == (96, 7)
    payload = _json_blocks(response)[-1]
    assert payload["requested_parameters"]["n_comps"] is None
    assert payload["applied_parameters"]["n_comps"] == 7
    assert payload["applied_parameters"]["n_features_used"] == 8
    adjustment = payload["parameter_adjustments"][0]
    assert adjustment["parameter"] == "n_comps" and adjustment["default"] == 30
    assert adjustment["effective"] == 7 and adjustment["reason"]
    activity = _activity(dm, "run_pca")
    assert activity["parameters"]["n_comps"] == "7"
    assert activity["ir"]["parameters"]["n_comps"] == 7
    replayed = _replay_recorded(activity["ir"], source)
    np.testing.assert_allclose(replayed.obsm["X_pca"], result.obsm["X_pca"])
    np.testing.assert_array_equal(_dense(source.X), _dense(snapshot.X))
    pd.testing.assert_frame_equal(source.obs, snapshot.obs)
    pd.testing.assert_frame_equal(source.var, snapshot.var)
