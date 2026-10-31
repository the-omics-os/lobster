"""S7 reporting/alignment controls; numerical tests and thresholds unchanged.

Real Scanpy, service, tool, DM and IR; only model construction is fake.
Recorded-result fixtures exercise extraction, not a mocked DE estimator.
"""

import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from scipy import sparse

from lobster.agents.transcriptomics.prompts import (
    create_de_analysis_expert_prompt,
    create_transcriptomics_expert_prompt,
)
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.core.utils.h5ad_utils import repair_marker_names
from lobster.services.analysis import enhanced_singlecell_service as service_module
from lobster.services.analysis.enhanced_singlecell_service import (
    EnhancedSingleCellService,
)
from lobster.testing.provenance_replay import replay_ir

parent_module = importlib.import_module(
    "lobster.agents.transcriptomics.transcriptomics_expert"
)
KEYS = {
    "original": "rank_genes_groups",
    "filtered": "rank_genes_groups_filtered",
    "displayed": "rank_genes_groups_filtered",
}


@pytest.fixture(scope="session", autouse=True)
def require_owned_origins():
    root = Path(__file__).resolve().parents[4]
    assert Path(service_module.__file__).resolve().is_relative_to(root)
    assert Path(parent_module.__file__).resolve().is_relative_to(root)
    assert (
        Path(importlib.import_module("lobster.core.runtime.data_manager").__file__)
        .resolve()
        .is_relative_to(root)
    )
    assert (
        Path(importlib.import_module("lobster.agents.transcriptomics.prompts").__file__)
        .resolve()
        .is_relative_to(root)
    )


def normalized(csr=False):
    x = np.zeros((36, 6), dtype=np.float64)
    for group in range(3):
        rows = slice(group * 12, (group + 1) * 12)
        x[rows, group] = np.arange(12) % 3 + 9
    x[:, 3] = 3
    x[:12, 5] = np.arange(12) % 2
    x = np.log1p(x)
    return ad.AnnData(
        sparse.csr_matrix(x) if csr else x,
        obs=pd.DataFrame(
            {"cluster": pd.Categorical(np.repeat(["A", "B", "C"], 12))},
            index=[f"cell_{i}" for i in range(36)],
        ),
        var=pd.DataFrame(index=["GENE_A", "GENE_B", "GENE_C", "SHARED", "ZERO", "nan"]),
    )


def usable_names(adata, key, group):
    values = adata.uns[key]["names"][group]
    return [value for value in values if not pd.isna(value) and value != ""]


def native_reference(source, groups=None):
    result = source.copy()
    kwargs = dict(
        groupby="cluster",
        method="wilcoxon",
        n_genes=6,
        use_raw=False,
        key_added=KEYS["original"],
    )
    if groups is not None:
        kwargs["groups"] = groups
    sc.tl.rank_genes_groups(result, **kwargs)
    sc.tl.filter_rank_genes_groups(
        result,
        key=KEYS["original"],
        key_added=KEYS["filtered"],
        min_fold_change=1.5,
        min_in_group_fraction=0.25,
        max_out_group_fraction=0.5,
    )
    return result


@pytest.mark.parametrize("csr", [False, True], ids=["dense", "csr"])
def test_actual_native_original_and_filtered_positive(csr):
    source = normalized(csr)
    reference = native_reference(source)
    assert reference.uns[KEYS["original"]]["names"].dtype.names == ("A", "B", "C")
    assert len(usable_names(reference, KEYS["filtered"], "A")) < 6
    assert "GENE_A" in usable_names(reference, KEYS["filtered"], "A")
    assert KEYS["original"] not in source.uns
    assert KEYS["filtered"] not in source.uns


@pytest.mark.parametrize("csr", [False, True], ids=["dense", "csr"])
def test_actual_service_filtered_counts_rows_and_numerics(csr):
    source = normalized(csr)
    before = source.copy()
    expected = native_reference(source)
    result, stats, ir = EnhancedSingleCellService().find_marker_genes(
        source, groupby="cluster", n_genes=6
    )
    # Native positives precede new reporting assertions.
    assert KEYS["original"] in result.uns and KEYS["filtered"] in result.uns
    for group in expected.uns[KEYS["original"]]["names"].dtype.names:
        np.testing.assert_allclose(
            result.uns[KEYS["original"]]["logfoldchanges"][group],
            expected.uns[KEYS["original"]]["logfoldchanges"][group],
            equal_nan=True,
        )
    np.testing.assert_array_equal(
        source.X.toarray() if csr else source.X, before.X.toarray() if csr else before.X
    )
    assert KEYS["original"] not in source.uns
    expected_post = {
        g: len(usable_names(expected, KEYS["filtered"], g)) for g in ("A", "B", "C")
    }
    assert stats["post_filter_counts"] == expected_post
    assert stats["result_keys"] == KEYS == ir.parameters["result_keys"]
    assert stats["total_genes_filtered"] == 18 - sum(expected_post.values())
    for group, rows in stats["top_markers_per_group"].items():
        assert [row["gene"] for row in rows] == usable_names(
            expected, KEYS["filtered"], group
        )[:10]
    warnings = stats["statistic_warnings"]
    assert warnings["extreme_log2fc_threshold"] == 10
    assert warnings["threshold_status"] == "provisional_threshold"
    for group in ("A", "B", "C"):
        effects = expected.uns[KEYS["original"]]["logfoldchanges"][group]
        extreme = np.isfinite(effects) & (np.abs(effects) >= 10)
        selected = np.array(
            [
                not pd.isna(name) and name != ""
                for name in expected.uns[KEYS["filtered"]]["names"][group]
            ]
        )
        assert warnings["original"][group]["extreme_log2fc"] == int(extreme.sum())
        assert warnings["filtered"][group]["extreme_log2fc"] == int(
            (extreme & selected).sum()
        )


def test_actual_group_subset_report_and_ir_follow_executed_fields():
    source = normalized()
    result, stats, ir = EnhancedSingleCellService().find_marker_genes(
        source, groupby="cluster", groups=["B"], n_genes=6
    )
    assert result.uns[KEYS["original"]]["names"].dtype.names == ("B",)
    assert stats["groups_analyzed"] == ["B"]
    assert stats["n_groups"] == 1
    assert stats["groups_requested"] == ["B"] == ir.parameters["groups"]
    assert list(stats["post_filter_counts"]) == ["B"]


def recorded_results(absent=False, nonfinite=False, masked=False):
    result = normalized()
    genes = ["DROP", "nan", "KEEP", "BELOW"]
    names = np.rec.fromarrays([np.array(genes, dtype=object)], names=["A"])
    fields = {
        "names": names,
        "scores": np.rec.fromarrays([np.array([1.0, 2.0, 3.0, 4.0])], names=["A"]),
        "logfoldchanges": np.rec.fromarrays(
            [np.array([11.0, 10.0, -10.0, 9.999])], names=["A"]
        ),
    }
    if not absent:
        fields["pvals"] = np.rec.fromarrays(
            [np.array([0.1, 0.2, 0.3, 0.4])], names=["A"]
        )
        fields["pvals_adj"] = np.rec.fromarrays(
            [np.array([0.73, 0.8, 0.9, 1.0])], names=["A"]
        )
    if nonfinite:
        fields["scores"]["A"][1] = np.nan
        fields["logfoldchanges"]["A"][2] = np.inf
    result.uns[KEYS["original"]] = fields
    result.uns[KEYS["filtered"]] = dict(fields)
    if masked:
        result.uns[KEYS["filtered"]]["names"] = np.rec.fromarrays(
            [np.array([np.nan, "nan", "KEEP", ""], dtype=object)], names=["A"]
        )
    return result


def test_extraction_uses_filtered_mask_keeps_literal_nan_and_numeric_alignment():
    source = recorded_results(masked=True)
    original_names = source.uns[KEYS["original"]]["names"].copy()
    rows = EnhancedSingleCellService()._extract_marker_genes(source, "cluster")
    assert rows["gene"].tolist() == ["nan", "KEEP"]
    assert rows["score"].tolist() == [2.0, 3.0]
    assert rows["padj"].tolist() == [0.8, 0.9]
    assert rows["log2fc"].tolist() == [10.0, -10.0]
    np.testing.assert_array_equal(source.uns[KEYS["original"]]["names"], original_names)


def test_empty_filtered_result_does_not_display_original_candidates():
    source = recorded_results(masked=True)
    source.uns[KEYS["filtered"]]["names"]["A"][:] = np.nan
    rows = EnhancedSingleCellService()._extract_marker_genes(source, "cluster")
    assert rows.empty
    assert usable_names(source, KEYS["original"], "A") == [
        "DROP",
        "nan",
        "KEEP",
        "BELOW",
    ]


@pytest.mark.parametrize("kind", ["finite", "nonfinite", "absent"])
def test_supported_missing_nonfinite_extreme_flags_do_not_rewrite_results(kind):
    source = recorded_results(absent=kind == "absent", nonfinite=kind == "nonfinite")
    effect_before = source.uns[KEYS["original"]]["logfoldchanges"].copy()
    rows = EnhancedSingleCellService()._extract_marker_genes(source, "cluster")
    assert rows["gene"].tolist() == ["DROP", "nan", "KEEP", "BELOW"]
    assert "flags" in rows.columns
    assert "extreme_log2fc" in rows.iloc[0]["flags"]
    assert "extreme_log2fc" in rows.iloc[1]["flags"]
    assert "extreme_log2fc" not in rows.iloc[3]["flags"]
    if kind == "nonfinite":
        assert "score_non_finite" in rows.iloc[1]["flags"]
        assert "log2fc_non_finite" in rows.iloc[2]["flags"]
        assert "extreme_log2fc" not in rows.iloc[2]["flags"]
    if kind == "absent":
        assert "pval_absent" in rows.iloc[0]["flags"]
        assert "padj_absent" in rows.iloc[0]["flags"]
        assert rows.iloc[0]["pval"] is None and rows.iloc[0]["padj"] is None
    np.testing.assert_allclose(
        source.uns[KEYS["original"]]["logfoldchanges"]["A"],
        effect_before["A"],
        equal_nan=True,
    )


def actual_tools(tmp_path, monkeypatch, source):
    dm = DataManagerV2(workspace_path=tmp_path / "workspace", auto_scan=False)
    dm.store_modality("prepared", source.copy(), step="loaded")
    settings = SimpleNamespace(get_agent_llm_params=lambda name: {})
    monkeypatch.setattr(parent_module, "get_settings", lambda: settings)
    monkeypatch.setattr(parent_module, "create_llm", lambda *args, **kwargs: object())
    monkeypatch.setattr(parent_module, "create_react_agent", lambda **kwargs: kwargs)
    switch = importlib.import_module("lobster.tools.code_tool_switch")
    monkeypatch.setattr(
        switch, "attach_code_tool", lambda tools, *args, **kwargs: tools
    )
    graph = parent_module.transcriptomics_expert(dm)
    dm.log_tool_usage = Mock(wraps=dm.log_tool_usage)
    return dm, {tool.name: tool for tool in graph["tools"]}


@pytest.mark.parametrize("csr", [False, True], ids=["dense", "csr"])
def test_actual_parent_tool_reports_keys_units_candidates_and_matching_ir(
    tmp_path, monkeypatch, csr
):
    dm, tools = actual_tools(tmp_path, monkeypatch, normalized(csr))
    response = tools["find_marker_genes"].invoke(
        {
            "modality_name": "prepared",
            "groupby": "cluster",
            "groups": ["B"],
            "n_genes": 6,
            "save_result": False,
        }
    )
    assert "Unexpected error" not in response and "Error finding" not in response
    assert "prepared_markers" in dm.list_modalities()
    assert "rank_genes_groups_filtered" in response
    assert "specificity-filtered" in response.lower()
    assert "FDR" in response and "log2FC > 1.5" in response
    assert "single-donor" in response.lower()
    calls = [
        c
        for c in dm.log_tool_usage.call_args_list
        if c.kwargs.get("tool_name") == "find_marker_genes"
    ]
    assert len(calls) == 1
    params = calls[0].kwargs["parameters"]
    ir = calls[0].kwargs["ir"]
    assert params["result_keys"] == KEYS == ir.parameters["result_keys"]
    assert params["groups_analyzed"] == ["B"]
    assert (
        params["extreme_log2fc_threshold"]
        == 10
        == ir.parameters["extreme_log2fc_threshold"]
    )


def test_prepared_input_ir_keeps_numerics_keys_and_reporting_identity():
    source = normalized()
    result, stats, ir = EnhancedSingleCellService().find_marker_genes(
        source, groupby="cluster", groups=["B"], n_genes=6
    )
    replayed = replay_ir(
        ir, adata=source.copy(), repair_marker_names=repair_marker_names
    )["adata"]
    assert usable_names(replayed, KEYS["filtered"], "B") == usable_names(
        result, KEYS["filtered"], "B"
    )
    np.testing.assert_allclose(
        replayed.uns[KEYS["original"]]["logfoldchanges"]["B"],
        result.uns[KEYS["original"]]["logfoldchanges"]["B"],
        equal_nan=True,
    )
    assert ir.parameters["result_keys"] == stats["result_keys"] == KEYS
    assert (
        "key_added" in ir.code_template
        and "rank_genes_groups_filtered" in ir.code_template
    )


@pytest.mark.parametrize("csr", [False, True], ids=["dense", "csr"])
def test_actual_logreg_reports_unavailable_statistics_not_empty_discoveries(csr):
    source = normalized(csr)
    result, stats, _ = EnhancedSingleCellService().find_marker_genes(
        source, groupby="cluster", method="logreg", n_genes=6
    )
    assert "scores" in result.uns[KEYS["original"]]
    assert "pvals" not in result.uns[KEYS["original"]]
    assert "pvals_adj" not in result.uns[KEYS["original"]]
    assert "logfoldchanges" not in result.uns[KEYS["original"]]
    assert sum(stats["post_filter_counts"].values()) > 0
    for group in stats["groups_analyzed"]:
        assert stats["statistic_warnings"]["filtered"][group]["absent_statistics"] == [
            "pval",
            "padj",
            "log2fc",
        ]
        for row in stats["top_markers_per_group"][group]:
            assert row["pval"] is None and row["padj"] is None and row["log2fc"] is None
            assert (
                "padj_absent" in row["flags"] and "extreme_log2fc" not in row["flags"]
            )


def test_requested_order_and_rank_limit_report_actual_native_fields_and_slots():
    result, stats, ir = EnhancedSingleCellService().find_marker_genes(
        normalized(), groupby="cluster", groups=["B", "A"], n_genes=25
    )
    actual = list(result.uns[KEYS["original"]]["names"].dtype.names)
    assert stats["groups_requested"] == ["B", "A"] == ir.parameters["groups"]
    assert stats["groups_analyzed"] == actual == ir.parameters["groups_analyzed"]
    assert list(stats["pre_filter_counts"]) == actual
    assert stats["pre_filter_counts"] == {group: 6 for group in actual}
    assert stats["n_genes"] == 25
    assert stats["filter_semantics"] == {
        "log2fc": {"operator": ">", "threshold": 1.5},
        "in_group_fraction": {"operator": ">", "threshold": 0.25},
        "out_group_fraction": {"operator": "<", "threshold": 0.5},
        "compare_abs": False,
        "fdr_filter_applied": False,
    }


def test_filtered_mask_precedes_top10_cap_with_all_statistic_positions_retained():
    source = normalized()
    count = 24
    names = np.rec.fromarrays(
        [np.array([f"gene_{i}" for i in range(count)], dtype=object)], names=["A"]
    )
    fields = {"names": names}
    for key in ("scores", "pvals", "pvals_adj", "logfoldchanges"):
        fields[key] = np.rec.fromarrays([np.arange(count, dtype=float)], names=["A"])
    source.uns[KEYS["original"]] = fields
    source.uns[KEYS["filtered"]] = dict(fields)
    selected = names.copy()
    selected["A"][:15] = np.nan
    selected["A"][16] = "nan"
    source.uns[KEYS["filtered"]]["names"] = selected
    rows = EnhancedSingleCellService()._extract_marker_genes(source, "cluster")
    assert rows["gene"].tolist() == ["gene_15", "nan"] + [
        f"gene_{i}" for i in range(17, 24)
    ]
    assert rows["score"].tolist() == list(range(15, 24))
    assert rows["padj"].tolist() == list(range(15, 24))
    assert source.uns[KEYS["original"]]["names"]["A"][16] == "gene_16"


def test_already_logged_raw_selection_and_original_source_remain_unchanged():
    source = normalized()
    source.raw = source.copy()
    source.X = np.full(source.shape, np.log1p(2.0))
    before = source.copy()
    reference = native_reference(source.raw.to_adata())
    result, stats, _ = EnhancedSingleCellService().find_marker_genes(
        source, groupby="cluster", n_genes=6
    )
    for group in stats["groups_analyzed"]:
        np.testing.assert_allclose(
            result.uns[KEYS["original"]]["logfoldchanges"][group],
            reference.uns[KEYS["original"]]["logfoldchanges"][group],
            equal_nan=True,
        )
    np.testing.assert_array_equal(source.X, before.X)
    np.testing.assert_array_equal(source.raw.X, before.raw.X)
    assert KEYS["original"] not in source.uns


def test_actual_saved_parent_results_preserve_both_keys_and_numeric_statistics(
    tmp_path, monkeypatch
):
    dm, tools = actual_tools(tmp_path, monkeypatch, normalized())
    response = tools["find_marker_genes"].invoke(
        {
            "modality_name": "prepared",
            "groupby": "cluster",
            "groups": ["A"],
            "n_genes": 6,
            "save_result": True,
        }
    )
    assert "Saved to" in response and "Unexpected error" not in response
    saved = list(dm.workspace_path.rglob("prepared_markers.h5ad"))
    assert len(saved) == 1
    restored = ad.read_h5ad(saved[0])
    live = dm.get_modality("prepared_markers")
    for key in (KEYS["original"], KEYS["filtered"]):
        np.testing.assert_array_equal(
            restored.uns[key]["names"]["A"], live.uns[key]["names"]["A"]
        )
        np.testing.assert_allclose(
            restored.uns[key]["logfoldchanges"]["A"],
            live.uns[key]["logfoldchanges"]["A"],
            equal_nan=True,
        )
    assert "nan" in restored.uns[KEYS["original"]]["names"]["A"]


def test_provisional_extreme_threshold_is_one_named_constant_not_a_filter():
    assert getattr(service_module, "EXTREME_LOG2FC_THRESHOLD", None) == 10


def test_parent_and_de_prompts_distinguish_candidates_and_independent_units():
    parent = create_transcriptomics_expert_prompt().lower()
    de = create_de_analysis_expert_prompt().lower()
    assert "specificity-filtered" in parent and "fdr-significant" in parent
    assert "single-donor" in parent and "single-donor" in de
    assert "extreme" in parent and "log2" in parent
