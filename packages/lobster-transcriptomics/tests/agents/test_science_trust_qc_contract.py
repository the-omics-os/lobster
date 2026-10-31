"""Deterministic QC reporting, explicit filters and applied-parameter contracts."""

import inspect
import json
from pathlib import Path
from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from jinja2 import Environment, meta
from scipy import sparse

from lobster.agents.transcriptomics.prompts import create_transcriptomics_expert_prompt
from lobster.agents.transcriptomics.shared_tools import create_shared_tools
from lobster.core.provenance.analysis_ir import AnalysisStep
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.core.utils.h5ad_utils import sanitize_value
from lobster.services.quality.preprocessing_service import (
    PreprocessingError,
    PreprocessingService,
)
from lobster.services.quality.quality_service import QualityService
from lobster.testing.provenance_replay import replay_ir


@pytest.fixture(params=[False, True], ids=["dense", "csr"])
def counts(request):
    """Synthetic cohort, not a reconstruction of a particular biological dataset."""
    x = np.zeros((128, 2601), dtype=np.float64)
    x[:, :240] = 1
    x[:5, :2501] = 1  # Five above the requested gene ceiling.
    x[5:62, 0] = 20  # 57 above 5% mito but below the default 20%.
    x[63, 0] = 20
    x[63, 2] += 141  # Exactly 5% before denominator epsilon.
    x[64, :2500] = 1  # Exactly the gene ceiling.
    x[65, 1] = 1000  # High ribo only: must survive a disabled ribo filter.
    x[66, 150:] = 0  # Independent min-genes failure, not an exceedance.
    obs = pd.DataFrame(
        {"sample": pd.Categorical(["a"] * 64 + ["b"] * 64)},
        index=[f"cell_{i}" for i in range(128)],
    )
    var = pd.DataFrame(
        {"source_label": ["synthetic"] * 2601},
        index=["MT-ND1", "RPS3", "ACTB"] + [f"GENE{i}" for i in range(2598)],
    )
    return ad.AnnData(sparse.csr_matrix(x) if request.param else x, obs=obs, var=var)


def dense(x):
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def oracle(adata):
    x = dense(adata.X)
    total = x.sum(axis=1)
    return {
        "n_genes": (x > 0).sum(axis=1),
        "total_counts": total,
        "mt_pct": x[:, 0] / (total + 1e-8) * 100,
    }


def tool_context(tmp_path, counts):
    dm = DataManagerV2(workspace_path=tmp_path / "workspace")
    dm.store_modality("input", counts.copy())
    quality = QualityService()
    preprocessing = PreprocessingService()
    tools = {t.name: t for t in create_shared_tools(dm, quality, preprocessing)}
    return dm, tools


def filter_args(**overrides):
    args = {
        "modality_name": "input",
        "min_genes_per_cell": 200,
        "max_genes_per_cell": 2500,
        "min_cells_per_gene": 3,
        "max_mito_percent": 5.0,
        "normalization_method": "log1p",
        "target_sum": 10000,
        "save_result": False,
    }
    return {**args, **overrides}


def json_blocks(response):
    return [
        json.loads(part.split("\n```", 1)[0])
        for part in response.split("```json\n")[1:]
    ]


def test_sources_are_from_this_checkout():
    import importlib

    root = Path(__file__).resolve().parents[4]
    for name in (
        "lobster.core.runtime.data_manager",
        "lobster.agents.transcriptomics.shared_tools",
        "lobster.agents.transcriptomics.prompts",
        "lobster.services.quality.quality_service",
        "lobster.services.quality.preprocessing_service",
        "lobster.core.provenance.analysis_ir",
        "lobster.testing.provenance_replay",
    ):
        assert (
            Path(importlib.import_module(name).__file__).resolve().is_relative_to(root)
        )


def test_synthetic_oracle_threshold_boundaries(counts):
    metrics = oracle(counts)
    genes = metrics["n_genes"] > 2500
    mito = metrics["mt_pct"] > 5
    assert int(genes.sum()) == 5
    assert int(mito.sum()) == 57
    assert int((genes | mito).sum()) == 62
    assert metrics["n_genes"][64] == 2500
    assert not mito[63]


def test_qc_returns_real_distributions_and_requested_exceedances(counts):
    before = dense(counts.X).copy()
    result, stats, _ = QualityService().assess_quality(
        counts,
        min_genes=200,
        max_genes=2500,
        max_mt_pct=5,
        max_ribo_pct=50,
        min_housekeeping_score=0,
    )
    expected = oracle(counts)
    assert stats["qc_cohort"] == {
        "n_observations": 128,
        "stage": "pre_filter",
        "quantile_method": "linear",
    }
    for key, unit in (
        ("n_genes", "genes/observation"),
        ("total_counts", "counts/observation"),
        ("mt_pct", "percent"),
    ):
        summary = stats["qc_distributions"][key]
        assert summary["unit"] == unit
        assert summary["median"] == pytest.approx(np.median(expected[key]))
        assert summary["p95"] == pytest.approx(
            np.quantile(expected[key], 0.95, method="linear")
        )
    exceedances = stats["threshold_exceedances"]
    assert exceedances["n_genes"] == {"threshold": 2500, "operator": ">", "count": 5}
    assert exceedances["mt_pct"] == {"threshold": 5, "operator": ">", "count": 57}
    assert exceedances["either_genes_or_mt"] == 62
    assert stats["cells_removed"] != exceedances["either_genes_or_mt"]
    assert result.n_obs == counts.n_obs  # Assessment annotates, it does not filter.
    np.testing.assert_array_equal(dense(counts.X), before)


def test_real_qc_tool_exposes_computed_numbers_and_params(tmp_path, counts):
    dm, tools = tool_context(tmp_path, counts)
    response = tools["assess_data_quality"].invoke(
        {
            "modality_name": "input",
            "min_genes": 200,
            "max_genes": 2500,
            "max_mt_pct": 5,
            "max_ribo_pct": 50,
        }
    )
    blocks = json_blocks(response)
    computed = next(b for b in blocks if "qc_distributions" in b)
    assert computed["threshold_exceedances"]["n_genes"]["count"] == 5
    assert computed["threshold_exceedances"]["mt_pct"]["count"] == 57
    assert computed["qc_cohort"]["n_observations"] == 128
    activity = next(
        a for a in dm.provenance.activities if a["type"] == "assess_data_quality"
    )
    assert activity["parameters"]["max_genes"] == "2500"
    assert activity["parameters"]["max_mt_pct"] in {"5", "5.0"}
    assert computed["qc_distributions"]["n_genes"]["median"] == np.median(
        oracle(counts)["n_genes"]
    )


def test_tool_default_ribo_is_disabled_and_disclosed(tmp_path, counts):
    dm, tools = tool_context(tmp_path, counts)
    tool = tools["filter_and_normalize"]
    assert "max_ribo_percent" in tool.args
    assert tool.args["max_ribo_percent"]["default"] is None
    response = tool.invoke(filter_args())
    output = dm.get_modality("input_filtered_normalized")
    assert "cell_65" in output.obs_names
    assert "Max ribosomal %: disabled" in response
    assert "Raw counts location: adata.raw" in response
    assert "layers['counts']" not in response
    assert output.raw is not None
    params = next(b for b in json_blocks(response) if "max_ribo_percent" in b)
    assert params["max_ribo_percent"] is None


def test_sdk_default_stays_50_and_explicit_ribo_changes_only_expected_rows(counts):
    service = PreprocessingService()
    assert (
        inspect.signature(service.filter_and_normalize_cells)
        .parameters["max_ribo_percent"]
        .default
        == 50
    )
    kwargs = dict(
        min_genes_per_cell=200,
        max_genes_per_cell=2500,
        min_cells_per_gene=3,
        max_mito_percent=5,
    )
    disabled, stats, _ = service.filter_and_normalize_cells(
        counts, max_ribo_percent=None, **kwargs
    )
    legacy, _, _ = service.filter_and_normalize_cells(counts, **kwargs)
    assert set(disabled.obs_names) - set(legacy.obs_names) == {"cell_65"}
    independent = stats["filter_failures_independent"]
    assert independent["min_genes_per_cell"] == 1
    assert independent["max_genes_per_cell"] == 5
    assert independent["max_mito_percent"] == 57
    assert independent["max_ribo_percent"] is None
    assert stats["filter_failure_cohort"] == {
        "n_observations": 128,
        "n_genes": 2601,
        "stage": "pre_filter",
    }
    assert stats["cells_removed"] == 63


@pytest.mark.parametrize("ribo", [None, 50])
def test_real_activity_template_vars_recorded_and_ir_replays(tmp_path, counts, ribo):
    dm, tools = tool_context(tmp_path, counts)
    response = tools["filter_and_normalize"].invoke(filter_args(max_ribo_percent=ribo))
    assert not response.startswith("Error")
    activity = next(
        a for a in dm.provenance.activities if a["type"] == "filter_and_normalize"
    )
    ir = activity["ir"]
    refs = meta.find_undeclared_variables(Environment().parse(ir["code_template"]))
    assert refs <= set(ir["parameters"])
    serialized = sanitize_value(ir["parameters"])
    for key in refs:
        assert key in activity["parameters"]
        assert activity["parameters"][key] == serialized[key]
    expected = dm.get_modality("input_filtered_normalized")
    actual_ir = AnalysisStep.from_dict(ir)
    scope = {"adata": counts.copy()}
    for statement in actual_ir.imports:
        exec(compile(statement, "<recorded-ir-import>", "exec"), scope)
    for helper in actual_ir.helper_code or []:
        exec(compile(helper, "<recorded-ir-helper>", "exec"), scope)
    replayed = replay_ir(actual_ir, **scope)["adata"]
    assert list(replayed.obs_names) == list(expected.obs_names)
    assert list(replayed.var_names) == list(expected.var_names)
    np.testing.assert_allclose(dense(replayed.X), dense(expected.X))
    np.testing.assert_array_equal(dense(replayed.raw.X), dense(expected.raw.X))
    pd.testing.assert_series_equal(replayed.obs["sample"], expected.obs["sample"])


def test_quality_ir_scientific_variables_also_recorded_top_level(tmp_path, counts):
    dm, tools = tool_context(tmp_path, counts)
    tools["assess_data_quality"].invoke(
        {"modality_name": "input", "max_genes": 2500, "max_mt_pct": 5}
    )
    activity = next(
        a for a in dm.provenance.activities if a["type"] == "assess_data_quality"
    )
    ir = activity["ir"]
    refs = meta.find_undeclared_variables(Environment().parse(ir["code_template"]))
    assert refs <= set(ir["parameters"])
    for key in refs:
        assert key in activity["parameters"]
        assert activity["parameters"][key] == sanitize_value(ir["parameters"])[key]


@pytest.mark.parametrize("method", ["cpm", "tpm", "unknown"])
def test_unsupported_normalization_refuses_before_service_mutation(counts, method):
    service = PreprocessingService()
    before = dense(counts.X).copy()
    with (
        patch.object(
            service, "_calculate_qc_metrics", wraps=service._calculate_qc_metrics
        ) as qc,
        patch.object(
            service, "_apply_quality_filters", wraps=service._apply_quality_filters
        ) as filters,
    ):
        with pytest.raises(PreprocessingError):
            service.filter_and_normalize_cells(counts, normalization_method=method)
        qc.assert_not_called()
        filters.assert_not_called()
    np.testing.assert_array_equal(dense(counts.X), before)


def test_parent_prompt_grounding_and_applied_parameter_contract():
    prompt = create_transcriptomics_expert_prompt()
    assert "NOT COMPUTED" in prompt
    assert "numbers only from tool output" in prompt.lower()
    assert "applied parameters" in prompt.lower()
    assert "requested thresholds" in prompt.lower()
    assert "independent" in prompt.lower()
    assert "do not sum" in prompt.lower()


def test_filter_tool_doc_lists_only_supported_methods(tmp_path, counts):
    _, tools = tool_context(tmp_path, counts)
    doc = tools["filter_and_normalize"].description
    assert "sctransform_like" in doc
    assert "'cpm'" not in doc and "'tpm'" not in doc


def test_overlap_exceedances_and_filter_failures_are_not_additive(counts):
    overlapping = counts.copy()
    if sparse.issparse(overlapping.X):
        matrix = overlapping.X.tolil()
        matrix[0, 0] = 200
        overlapping.X = matrix.tocsr()
    else:
        overlapping.X[0, 0] = 200
    values = oracle(overlapping)
    genes = values["n_genes"] > 2500
    mito = values["mt_pct"] > 5
    assert int((genes & mito).sum()) == 1
    assert int(genes.sum() + mito.sum()) == 63
    assert int((genes | mito).sum()) == 62
    _, qc, _ = QualityService().assess_quality(
        overlapping,
        min_genes=200,
        max_genes=2500,
        max_mt_pct=5,
        max_ribo_pct=50,
        min_housekeeping_score=0,
    )
    assert qc["threshold_exceedances"]["n_genes"]["count"] == 5
    assert qc["threshold_exceedances"]["mt_pct"]["count"] == 58
    assert qc["threshold_exceedances"]["either_genes_or_mt"] == 62
    filtered, stats, _ = PreprocessingService().filter_and_normalize_cells(
        overlapping,
        min_genes_per_cell=200,
        max_genes_per_cell=2500,
        min_cells_per_gene=3,
        max_mito_percent=5,
        max_ribo_percent=None,
    )
    independent = stats["filter_failures_independent"]
    assert independent["max_genes_per_cell"] == 5
    assert independent["max_mito_percent"] == 58
    assert independent["min_genes_per_cell"] == 1
    assert stats["cells_removed"] == overlapping.n_obs - filtered.n_obs == 63
    assert (
        sum(
            independent[k]
            for k in ("min_genes_per_cell", "max_genes_per_cell", "max_mito_percent")
        )
        == 64
    )


@pytest.mark.parametrize("as_csr", [False, True], ids=["empty_dense", "empty_csr"])
def test_empty_qc_distributions_are_null_not_fabricated(as_csr):
    x = np.empty((0, 3))
    empty = ad.AnnData(sparse.csr_matrix(x) if as_csr else x)
    empty.var_names = ["MT-ND1", "RPS3", "ACTB"]
    result, stats, _ = QualityService().assess_quality(empty, min_housekeeping_score=0)
    assert result.n_obs == 0
    assert stats["qc_cohort"]["n_observations"] == 0
    for summary in stats["qc_distributions"].values():
        assert summary["median"] is None and summary["p95"] is None
    assert stats["threshold_exceedances"]["either_genes_or_mt"] == 0


@pytest.mark.parametrize("method", ["cpm", "tpm", "unknown"])
def test_actual_tool_unsupported_normalization_does_not_store_or_log(
    tmp_path, counts, method
):
    dm, tools = tool_context(tmp_path, counts)
    resident = dm.get_modality("input")
    snapshot = resident.copy()
    modalities = set(dm.modalities)
    activity_count = len(dm.provenance.activities)
    dirty_before = set(dm._modality_dirty)
    with (
        patch.object(dm, "store_modality", wraps=dm.store_modality) as store,
        patch.object(dm, "save_modality", wraps=dm.save_modality) as save,
        patch.object(dm, "log_tool_usage", wraps=dm.log_tool_usage) as log,
    ):
        response = tools["filter_and_normalize"].invoke(
            filter_args(normalization_method=method)
        )
    assert "unsupported normalization" in response.lower()
    store.assert_not_called()
    save.assert_not_called()
    log.assert_not_called()
    assert set(dm.modalities) == modalities
    assert len(dm.provenance.activities) == activity_count
    assert dm._modality_dirty == dirty_before
    assert resident.raw is None
    np.testing.assert_array_equal(dense(resident.X), dense(snapshot.X))
    pd.testing.assert_frame_equal(resident.obs, snapshot.obs)
    pd.testing.assert_frame_equal(resident.var, snapshot.var)


@pytest.mark.parametrize("method", ["log1p", "sctransform_like"])
def test_supported_normalization_preserves_existing_scanpy_algorithm(counts, method):
    import scanpy as sc

    source = counts[:16, :12].copy()
    expected = source.copy()
    sc.pp.normalize_total(expected, target_sum=10000)
    sc.pp.log1p(expected)
    if method == "sctransform_like":
        sc.pp.scale(expected, max_value=10)
    actual = source.copy()
    stats = PreprocessingService()._normalize_expression_data(actual, method, 10000)
    np.testing.assert_allclose(dense(actual.X), dense(expected.X))
    np.testing.assert_array_equal(dense(actual.raw.X), dense(source.X))
    assert stats["normalization_method"] == method
    assert stats["target_sum"] == 10000


@pytest.mark.parametrize("method", ["cpm", "tpm"])
def test_direct_normalizer_unsupported_method_does_not_create_raw(counts, method):
    service = PreprocessingService()
    source = counts[:16, :12].copy()
    snapshot = source.copy()
    with patch.object(source, "copy", wraps=source.copy) as copy:
        with pytest.raises(PreprocessingError):
            service._normalize_expression_data(source, method, 10000)
        copy.assert_not_called()
    assert source.raw is None
    np.testing.assert_array_equal(dense(source.X), dense(snapshot.X))


@pytest.mark.parametrize("method", ["cpm", "tpm", "unknown"])
def test_unsupported_tool_method_refuses_before_real_backed_retrieval(
    tmp_path, counts, method
):
    dm, tools = tool_context(tmp_path, counts)
    source_path = tmp_path / "native_backed_input.h5ad"
    counts.write_h5ad(source_path)
    backed = ad.read_h5ad(source_path, backed="r")
    assert backed.isbacked and backed.file.is_open
    assert float(backed.X[0, 0]) == float(dense(counts.X)[0, 0])
    source_bytes = source_path.read_bytes()
    dm.modalities["input"] = backed
    dm._modality_dirty.clear()
    modalities_before = set(dm.modalities)
    spilled_before = dict(dm._modality_spilled)
    dirty_before = set(dm._modality_dirty)
    activity_count = len(dm.provenance.activities)
    try:
        with (
            patch.object(dm, "get_modality", wraps=dm.get_modality) as fetch,
            patch.object(
                dm, "materialize_modality", wraps=dm.materialize_modality
            ) as materialize,
            patch.object(dm, "_spill_modality", wraps=dm._spill_modality) as spill,
            patch.object(dm, "store_modality", wraps=dm.store_modality) as store,
            patch.object(dm, "save_modality", wraps=dm.save_modality) as save,
            patch.object(dm, "log_tool_usage", wraps=dm.log_tool_usage) as log,
        ):
            response = tools["filter_and_normalize"].invoke(
                filter_args(normalization_method=method)
            )
        assert "unsupported normalization" in response.lower()
        fetch.assert_not_called()
        materialize.assert_not_called()
        spill.assert_not_called()
        store.assert_not_called()
        save.assert_not_called()
        log.assert_not_called()
        assert dm.modalities["input"] is backed
        assert backed.isbacked and backed.file.is_open
        assert set(dm.modalities) == modalities_before
        assert dm._modality_spilled == spilled_before
        assert dm._modality_dirty == dirty_before
        assert len(dm.provenance.activities) == activity_count
        assert source_path.read_bytes() == source_bytes
    finally:
        backed.file.close()
