"""Provenance IR replay for the annotation and formula-DE tools.

Each tool that changes data must log an AnalysisStep whose template, executed on
a copy of the tool's input, reproduces the tool's own output. Read-only checks
log IR that is exported but not replayable.
"""

import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.agents.transcriptomics.annotation_expert import annotation_expert
from lobster.agents.transcriptomics.de_analysis_expert import de_analysis_expert
from lobster.core.provenance.analysis_ir import AnalysisStep
from lobster.testing.provenance_replay import (
    RecordingDataManager,
    factory_tools,
    replay_ir,
)

ANNOTATION_MODULE = "lobster.agents.transcriptomics.annotation_expert"
DE_MODULE = "lobster.agents.transcriptomics.de_analysis_expert"


def _clustered_cells(n_cells: int = 60) -> ad.AnnData:
    rng = np.random.default_rng(0)
    clusters = np.array([str(i % 4) for i in range(n_cells)])
    obs = pd.DataFrame(
        {
            "seurat_clusters": pd.Categorical(clusters),
            "n_genes_by_counts": np.where(clusters == "3", 50, 800),
            "pct_counts_mt": np.where(clusters == "2", 70.0, 5.0),
            "total_counts": np.full(n_cells, 3000),
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    return ad.AnnData(
        X=rng.poisson(2, size=(n_cells, 20)).astype(float),
        obs=obs,
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(20)]),
    )


def _assert_real_ir(ir, replayable=True):
    assert isinstance(ir, AnalysisStep), f"expected AnalysisStep, got {ir!r}"
    assert ir.replayable is replayable
    ir.validate_rendered_code()


@pytest.fixture
def annotation_setup():
    adata = _clustered_cells()
    dm = RecordingDataManager({"cells": adata})
    tools = factory_tools(annotation_expert, ANNOTATION_MODULE, dm)
    return adata, dm, tools


class TestAnnotationToolIR:
    def test_manual_annotation_replays(self, annotation_setup):
        adata, dm, tools = annotation_setup
        tools["manually_annotate_clusters"].invoke(
            {
                "modality_name": "cells",
                "annotations": {"0": "T cells", "1": "B cells"},
                "cluster_key": "seurat_clusters",
                "save_result": False,
            }
        )
        ir = dm.last_ir("manually_annotate_clusters")
        _assert_real_ir(ir)
        produced = dm.modalities["cells_manually_annotated"]
        replayed = replay_ir(ir, adata=adata.copy())["adata"]
        pd.testing.assert_series_equal(
            replayed.obs["cell_type_manual"], produced.obs["cell_type_manual"]
        )
        assert set(produced.obs["cell_type_manual"]) == {
            "T cells",
            "B cells",
            "Unassigned",
        }

    def test_collapse_replays(self, annotation_setup):
        adata, dm, tools = annotation_setup
        tools["collapse_clusters_to_celltype"].invoke(
            {
                "modality_name": "cells",
                "cluster_list": ["0", "2"],
                "cell_type_name": "Myeloid",
                "cluster_key": "seurat_clusters",
                "save_result": False,
            }
        )
        ir = dm.last_ir("collapse_clusters_to_celltype")
        _assert_real_ir(ir)
        produced = dm.modalities["cells_collapsed"]
        replayed = replay_ir(ir, adata=adata.copy())["adata"]
        pd.testing.assert_series_equal(
            replayed.obs["cell_type_manual"], produced.obs["cell_type_manual"]
        )
        assert (produced.obs["cell_type_manual"] == "Myeloid").sum() == 30

    @pytest.mark.parametrize("remove_debris", [False, True])
    def test_mark_debris_replays(self, annotation_setup, remove_debris):
        adata, dm, tools = annotation_setup
        tools["mark_clusters_as_debris"].invoke(
            {
                "modality_name": "cells",
                "debris_clusters": ["3"],
                "remove_debris": remove_debris,
                "cluster_key": "seurat_clusters",
                "save_result": False,
            }
        )
        ir = dm.last_ir("mark_clusters_as_debris")
        _assert_real_ir(ir)
        suffix = "debris_removed" if remove_debris else "debris_marked"
        produced = dm.modalities[f"cells_{suffix}"]
        replayed = replay_ir(ir, adata=adata.copy())["adata"]
        assert list(replayed.obs_names) == list(produced.obs_names)
        pd.testing.assert_frame_equal(
            replayed.obs[["cell_type_manual", "is_debris"]],
            produced.obs[["cell_type_manual", "is_debris"]],
        )
        assert produced.n_obs == (45 if remove_debris else 60)

    def test_debris_suggestion_uses_cluster_key_and_is_not_replayable(
        self, annotation_setup
    ):
        adata, dm, tools = annotation_setup
        result = tools["suggest_debris_clusters"].invoke(
            {"modality_name": "cells", "cluster_key": "seurat_clusters"}
        )
        # Cluster 3 has 50 genes/cell, cluster 2 has 70% mito: both flagged.
        assert "Cluster 2" in result and "Cluster 3" in result
        ir = dm.last_ir("suggest_debris_clusters")
        _assert_real_ir(ir, replayable=False)
        assert ir.exportable
        suggested = replay_ir(ir, adata=adata.copy())["suggested_debris"]
        assert sorted(suggested) == ["2", "3"]
        assert sorted(ir.execution_context["suggested_debris_clusters"]) == ["2", "3"]

    def test_debris_suggestion_is_stateless_across_datasets(self, annotation_setup):
        """A second dataset must not reuse the first dataset's cluster summary."""
        adata, dm, tools = annotation_setup
        tools["suggest_debris_clusters"].invoke(
            {"modality_name": "cells", "cluster_key": "seurat_clusters"}
        )
        healthy = _clustered_cells()
        healthy.obs["n_genes_by_counts"] = 800
        healthy.obs["pct_counts_mt"] = 5.0
        dm.modalities["healthy"] = healthy
        result = tools["suggest_debris_clusters"].invoke(
            {"modality_name": "healthy", "cluster_key": "seurat_clusters"}
        )
        assert result.startswith("No debris clusters suggested")

    def test_template_records_resolved_mapping(self, annotation_setup, monkeypatch):
        from lobster.services.templates.annotation_templates import (
            AnnotationTemplateService,
        )

        mapping = {"0": "T cells", "1": "B cells", "2": "NK cells"}
        monkeypatch.setattr(
            AnnotationTemplateService,
            "apply_template_to_clusters",
            lambda self, **kwargs: dict(mapping),
        )
        adata, dm, tools = annotation_setup
        tools["apply_annotation_template"].invoke(
            {
                "modality_name": "cells",
                "tissue_type": "pbmc",
                "cluster_key": "seurat_clusters",
                "save_result": False,
            }
        )
        ir = dm.last_ir("apply_annotation_template")
        _assert_real_ir(ir)
        produced = dm.modalities["cells_template_pbmc"]
        replayed = replay_ir(ir, adata=adata.copy())["adata"]
        pd.testing.assert_series_equal(
            replayed.obs["cell_type_template"], produced.obs["cell_type_template"]
        )
        assert (produced.obs["cell_type_template"] == "Unknown").sum() == 15

    def test_import_mapping_replays(self, annotation_setup, tmp_path):
        adata, dm, tools = annotation_setup
        mapping_file = tmp_path / "mapping.json"
        mapping_file.write_text(
            json.dumps(
                {
                    "cluster_column": "seurat_clusters",
                    "cluster_to_celltype": {"0": "T cells", "3": "Debris"},
                }
            )
        )
        tools["import_annotation_mapping"].invoke(
            {
                "modality_name": "cells",
                "mapping_file": str(mapping_file),
                "save_result": False,
            }
        )
        ir = dm.last_ir("import_annotation_mapping")
        _assert_real_ir(ir)
        produced = dm.modalities["cells_imported_annotations"]
        replayed = replay_ir(ir, adata=adata.copy())["adata"]
        pd.testing.assert_series_equal(
            replayed.obs["cell_type_imported"], produced.obs["cell_type_imported"]
        )

    def test_missing_label_scalars_stay_unassigned(self, annotation_setup):
        from lobster.services.metadata.manual_annotation_service import (
            ManualAnnotationService,
        )

        adata, _, _ = annotation_setup
        mapping = {"0": None, "1": np.float32("nan"), "2": pd.NA, "3": "B cells"}
        out, _, _ = ManualAnnotationService().apply_cluster_mapping(
            adata, cluster_key="seurat_clusters", mapping=mapping
        )
        assert set(out.obs["cell_type_manual"]) == {"Unassigned", "B cells"}

    def test_imported_null_labels_stay_unassigned(self, annotation_setup, tmp_path):
        """JSON null must not become a cell type literally named 'None'."""
        adata, dm, tools = annotation_setup
        mapping_file = tmp_path / "mapping.json"
        mapping_file.write_text(
            json.dumps(
                {
                    "cluster_column": "seurat_clusters",
                    "cluster_to_celltype": {"0": None, "1": "B cells"},
                }
            )
        )
        tools["import_annotation_mapping"].invoke(
            {
                "modality_name": "cells",
                "mapping_file": str(mapping_file),
                "save_result": False,
            }
        )
        labels = dm.modalities["cells_imported_annotations"].obs["cell_type_imported"]
        assert "None" not in set(labels)
        assert set(labels) == {"B cells", "Unassigned"}
        replayed = replay_ir(
            dm.last_ir("import_annotation_mapping"), adata=adata.copy()
        )
        pd.testing.assert_series_equal(
            replayed["adata"].obs["cell_type_imported"], labels
        )


# ---------------------------------------------------------------------------
# Formula-based differential expression
# ---------------------------------------------------------------------------


def _pseudobulk(seed: int = 0) -> ad.AnnData:
    """8 pseudobulk samples x 60 genes; the first 10 genes are up in 'treated'."""
    rng = np.random.default_rng(seed)
    condition = ["control"] * 4 + ["treated"] * 4
    counts = rng.negative_binomial(20, 0.3, size=(8, 60)).astype(float)
    counts[4:, :10] *= 4
    obs = pd.DataFrame(
        {"condition": pd.Categorical(condition), "batch": ["a", "b"] * 4},
        index=[f"sample_{i}" for i in range(8)],
    )
    return ad.AnnData(
        X=counts, obs=obs, var=pd.DataFrame(index=[f"gene_{i}" for i in range(60)])
    )


@pytest.fixture
def de_setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the DE IR's BulkRNASeqService writes ./bulk_results
    adata = _pseudobulk()
    original = adata.copy()
    dm = RecordingDataManager({"pb": adata})
    tools = factory_tools(de_analysis_expert, DE_MODULE, dm)
    return original, dm, tools


CONTRAST = ["condition", "treated", "control"]
RESULTS_KEY = "de_results_formula_condition_treated_vs_control"


class TestFormulaDEToolIR:
    def test_prepare_design_replays(self, de_setup):
        original, dm, tools = de_setup
        tools["prepare_de_design"].invoke(
            {"modality_name": "pb", "formula": "~condition", "contrast": CONTRAST}
        )
        ir = dm.last_ir("prepare_de_design")
        _assert_real_ir(ir)
        replayed = replay_ir(ir, adata=original.copy())["adata"]
        stored = dm.modalities["pb"].uns["formula_design"]
        assert replayed.uns["formula_design"]["formula"] == stored["formula"]
        assert replayed.uns["formula_design"]["contrast"] == stored["contrast"]
        assert (
            replayed.uns["formula_design"]["validation_results"]["valid"]
            == stored["validation_results"]["valid"]
        )

    def test_design_validation_is_exported_not_replayed(self, de_setup):
        original, dm, tools = de_setup
        tools["validate_experimental_design"].invoke(
            {"modality_name": "pb", "formula": "~condition"}
        )
        ir = dm.last_ir("validate_experimental_design")
        _assert_real_ir(ir, replayable=False)
        check = replay_ir(ir, adata=original.copy())["design_check"]
        assert check["valid"] is True

    def test_formula_de_replays_identical_results(self, de_setup):
        original, dm, tools = de_setup
        result = tools["run_de_with_formula"].invoke(
            {
                "pseudobulk_modality": "pb",
                "formula": "~condition",
                "contrast": CONTRAST,
                "save_results": False,
            }
        )
        assert "Differential Expression Analysis Complete" in result
        ir = dm.last_ir("run_de_with_formula")
        _assert_real_ir(ir)
        produced = dm.modalities["pb"].uns[RESULTS_KEY]["results_df"]
        replayed = replay_ir(ir, adata=original.copy())["adata"]
        pd.testing.assert_frame_equal(replayed.uns[RESULTS_KEY]["results_df"], produced)
        assert replayed.uns["de_formula_design"]["formula"] == "~condition"

    def test_iteration_and_comparison_replay(self, de_setup):
        original, dm, tools = de_setup
        for name, formula in (("base", "~condition"), ("batch", "~batch + condition")):
            result = tools["iterate_de_analysis"].invoke(
                {
                    "pseudobulk_modality": "pb",
                    "new_formula": formula,
                    "new_contrast": CONTRAST,
                    "iteration_name": name,
                }
            )
            # Regression: the tool used to call run_de_with_formula directly,
            # which fails on a LangChain tool object, so every iteration errored.
            assert "Iteration" in result and "Error" not in result, result

        tracker = _ledger(dm.modalities["pb"])
        assert [i["name"] for i in tracker["iterations"]] == ["base", "batch"]

        # Replaying every logged step in order rebuilds the iteration state.
        replayed = original.copy()
        for call in dm.calls:
            if call.tool_name in ("run_de_with_formula", "iterate_de_analysis"):
                _assert_real_ir(call.ir)
                replayed = replay_ir(call.ir, adata=replayed)["adata"]
        assert [i["name"] for i in _ledger(replayed)["iterations"]] == [
            "base",
            "batch",
        ]
        assert _ledger(replayed)["current_iteration"] == 2

        tools["compare_de_iterations"].invoke({"pseudobulk_modality": "pb"})
        ir = dm.last_ir("compare_de_iterations")
        _assert_real_ir(ir)
        replayed = replay_ir(ir, adata=replayed)["adata"]
        key = "iteration_comparison_batch_vs_base"
        produced = dm.modalities["pb"].uns["iteration_comparisons"][key]
        replayed_cmp = replayed.uns["iteration_comparisons"][key]
        assert replayed_cmp["summary_stats"] == produced["summary_stats"]
        assert sorted(replayed_cmp["overlap_genes"]) == sorted(
            produced["overlap_genes"]
        )

    def test_same_contrast_iterations_keep_their_own_results(self, de_setup):
        """Two formulas for one contrast must not be compared as one result.

        The formula result key depends only on the contrast, so the second run
        overwrote the first and the comparison reported 100% overlap.
        """
        original, dm, tools = de_setup
        for name, formula in (("base", "~condition"), ("batch", "~batch + condition")):
            tools["iterate_de_analysis"].invoke(
                {
                    "pseudobulk_modality": "pb",
                    "new_formula": formula,
                    "new_contrast": CONTRAST,
                    "iteration_name": name,
                }
            )
        uns = dm.modalities["pb"].uns
        base = uns["de_iteration_results_1"]["results_df"]
        batch = uns["de_iteration_results_2"]["results_df"]
        lfc_delta = (base["log2FoldChange"] - batch["log2FoldChange"]).abs().max()
        assert lfc_delta > 1e-3, "the two formulas must give different results"

        tools["compare_de_iterations"].invoke({"pseudobulk_modality": "pb"})
        cmp = uns["iteration_comparisons"]["iteration_comparison_batch_vs_base"]
        expected_corr = batch.loc[cmp["overlap_genes"], "log2FoldChange"].corr(
            base.loc[cmp["overlap_genes"], "log2FoldChange"]
        )
        assert cmp["correlation"] == pytest.approx(expected_corr)
        assert cmp["correlation"] < 1.0 - 1e-9

        # Replaying every step reproduces the same, non-trivial comparison.
        replayed = original.copy()
        for call in dm.calls:
            replayed = replay_ir(call.ir, adata=replayed)["adata"]
        replayed_cmp = replayed.uns["iteration_comparisons"][
            "iteration_comparison_batch_vs_base"
        ]
        assert replayed_cmp["summary_stats"] == cmp["summary_stats"]
        assert replayed_cmp["correlation"] == pytest.approx(cmp["correlation"])

    def test_duplicate_iteration_names_replay_the_same_comparison(self, de_setup):
        """Iterations are matched by results_key, so repeated names cannot collide."""
        original, dm, tools = de_setup
        for formula in ("~condition", "~batch + condition"):
            tools["iterate_de_analysis"].invoke(
                {
                    "pseudobulk_modality": "pb",
                    "new_formula": formula,
                    "new_contrast": CONTRAST,
                    "iteration_name": "same",
                }
            )
        tools["compare_de_iterations"].invoke({"pseudobulk_modality": "pb"})
        cmp = dm.modalities["pb"].uns["iteration_comparisons"][
            "iteration_comparison_same_vs_same"
        ]
        assert cmp["correlation"] < 1.0 - 1e-9

        replayed = original.copy()
        for call in dm.calls:
            replayed = replay_ir(call.ir, adata=replayed)["adata"]
        replayed_cmp = replayed.uns["iteration_comparisons"][
            "iteration_comparison_same_vs_same"
        ]
        assert replayed_cmp["summary_stats"] == cmp["summary_stats"]
        assert replayed_cmp["correlation"] == pytest.approx(cmp["correlation"])

    def test_rejected_design_records_no_iteration(self, de_setup):
        """A design run_de_with_formula rejects must not reuse an older result."""
        _, dm, tools = de_setup
        tools["iterate_de_analysis"].invoke(
            {
                "pseudobulk_modality": "pb",
                "new_formula": "~condition",
                "new_contrast": CONTRAST,
                "iteration_name": "good",
            }
        )
        n_calls = len(dm.calls)
        result = tools["iterate_de_analysis"].invoke(
            {
                "pseudobulk_modality": "pb",
                "new_formula": "~missing_covariate",
                "new_contrast": CONTRAST,
                "iteration_name": "bad",
            }
        )
        assert result.startswith("Error in iteration 'bad'"), result
        tracker = _ledger(dm.modalities["pb"])
        assert [i["name"] for i in tracker["iterations"]] == ["good"]
        assert "de_iteration_results_2" not in dm.modalities["pb"].uns
        assert [c.tool_name for c in dm.calls[n_calls:]] == []

    def test_comparison_without_saving_is_not_replayable(self, de_setup):
        _, dm, tools = de_setup
        for name in ("one", "two"):
            tools["iterate_de_analysis"].invoke(
                {
                    "pseudobulk_modality": "pb",
                    "new_formula": "~condition",
                    "new_contrast": CONTRAST,
                    "iteration_name": name,
                }
            )
        tools["compare_de_iterations"].invoke(
            {"pseudobulk_modality": "pb", "save_comparison": False}
        )
        _assert_real_ir(dm.last_ir("compare_de_iterations"), replayable=False)


def _ledger(adata):
    """Decode persisted history, accepting legacy input for compatibility tests."""
    if "de_iterations_json" in adata.uns:
        return json.loads(adata.uns["de_iterations_json"])
    return adata.uns["de_iterations"]


def _iterate(tools, formula="~condition", name=None):
    response = tools["iterate_de_analysis"].invoke(
        {
            "pseudobulk_modality": "pb",
            "new_formula": formula,
            "new_contrast": CONTRAST,
            "iteration_name": name,
        }
    )
    assert "Complete" in response and "Error" not in response, response


def _save_with_exporter(adata, path):
    """Execute the real exporter's final save cell (including its fallback)."""
    from types import SimpleNamespace

    from lobster.core.notebooks.exporter import NotebookExporter

    exporter = NotebookExporter(SimpleNamespace(activities={}), None)
    scope = {"adata": adata, "output_prefix": str(path)}
    # Execute the real saving cell generated from controlled local test fixtures.
    exec(exporter._create_data_saving_cell().source, scope)  # nosec B102
    return ad.read_h5ad(f"{path}_processed.h5ad")


class TestDEExplicitResultKeys:
    @pytest.mark.parametrize("source", ["custom_contrast", "generated_filtered"])
    @pytest.mark.parametrize("tool_name", ["filter_de_results", "export_de_results"])
    def test_explicit_non_discovery_keys_filter_and_export(
        self, de_setup, tmp_path, source, tool_name
    ):
        _, dm, tools = de_setup
        adata = dm.modalities["pb"]
        frame = pd.DataFrame(
            {
                "padj": [0.02, 0.01, 0.03, 0.8],
                "log2FoldChange": [2.5, 0.5, -3.0, 4.0],
                "baseMean": [50.0] * 4,
            },
            index=["gene_up", "gene_weak", "gene_down", "gene_not_significant"],
        )
        if source == "generated_filtered":
            # Use the actual tool's output, not a hand-built filtered dictionary.
            adata.uns["de_results_source"] = {"results_df": frame}
            response = tools["filter_de_results"].invoke(
                {
                    "modality_name": "pb",
                    "de_results_key": "de_results_source",
                    "lfc_threshold": 0.0,
                }
            )
            assert "DE Results Filtered" in response, response
            key = "filtered_de_results_source"
            expected = frame[frame["padj"] < 0.05].copy()
            pd.testing.assert_frame_equal(adata.uns[key]["results_df"], expected)
        else:
            key = "custom_contrast"
            adata.uns[key] = {"results_df": frame}
            expected = frame.copy()
        # Explicit identity overrides both a current marker and insertion order.
        adata.uns["de_results_decoy"] = {
            "results_df": frame.rename(index=lambda gene: f"decoy_{gene}")
        }
        adata.uns["de_current_results_key"] = "de_results_decoy"
        before = adata.copy()
        dm.calls.clear()
        args = {"modality_name": "pb", "de_results_key": key}
        output_path = dm.workspace_path / "explicit.csv"  # inside the workspace
        if tool_name == "export_de_results":
            args["output_path"] = str(output_path)
        response = tools[tool_name].invoke(args)
        assert "Error" not in response and "error" not in response, response
        assert len(dm.calls) == 1
        assert dm.calls[0].parameters["de_results_key"] == key
        ir = dm.last_ir(tool_name)
        _assert_real_ir(ir)
        assert ir.parameters["de_results_key"] == key
        if tool_name == "filter_de_results":
            expected = expected[
                (expected["padj"] < 0.05) & (abs(expected["log2FoldChange"]) > 1.0)
            ]
            assert expected.index.tolist() == ["gene_up", "gene_down"]
            assert "DE Results Filtered" in response, response
            pd.testing.assert_frame_equal(
                adata.uns[f"filtered_{key}"]["results_df"], expected
            )
            replayed = replay_ir(ir, adata=before)["adata"]
            pd.testing.assert_frame_equal(
                replayed.uns[f"filtered_{key}"]["results_df"], expected
            )
        else:
            assert "DE Results Exported" in response, response
            expected = expected.sort_values("padj")
            pd.testing.assert_frame_equal(
                pd.read_csv(output_path, index_col=0), expected, check_names=False
            )
            # Execute the recorded export as well; its selected table must match.
            output_path.unlink()
            replay_ir(ir, adata=before)
            pd.testing.assert_frame_equal(
                pd.read_csv(output_path, index_col=0), expected, check_names=False
            )
        assert adata.uns["de_current_results_key"] == "de_results_decoy"

    @pytest.mark.parametrize("tool_name", ["filter_de_results", "export_de_results"])
    @pytest.mark.parametrize("key", ["custom_contrast", "de_results_malformed"])
    @pytest.mark.parametrize(
        "malformation", ["missing", "not_dict", "missing_frame", "not_frame", "null"]
    )
    def test_explicit_missing_or_malformed_results_reject(
        self, de_setup, tmp_path, tool_name, key, malformation
    ):
        _, dm, tools = de_setup
        adata = dm.modalities["pb"]
        if malformation != "missing":
            adata.uns[key] = {
                "not_dict": pd.DataFrame({"padj": [0.01]}),
                "missing_frame": {},
                "not_frame": {"results_df": {"padj": [0.01]}},
                "null": {"results_df": None},
            }[malformation]
        keys_before = set(adata.uns)
        output_path = tmp_path / "rejected.csv"
        args = {"modality_name": "pb", "de_results_key": key}
        if tool_name == "export_de_results":
            args["output_path"] = str(output_path)
        response = tools[tool_name].invoke(args)
        expected_error = (
            "unavailable" if malformation == "missing" else "Invalid DE results format"
        )
        assert expected_error in response, response
        assert key in response, response
        assert not dm.calls
        assert set(adata.uns) == keys_before
        assert not output_path.exists()

    @pytest.mark.parametrize("key", ["custom_contrast", "filtered_de_results_source"])
    def test_non_discovery_keys_are_not_automatic_candidates(self, de_setup, key):
        _, dm, tools = de_setup
        dm.modalities["pb"].uns[key] = {
            "results_df": pd.DataFrame(
                {"padj": [0.01], "log2FoldChange": [2.0], "baseMean": [50.0]},
                index=["gene_up"],
            )
        }
        response = tools["filter_de_results"].invoke({"modality_name": "pb"})
        assert "No DE results found" in response, response
        assert not dm.calls


class TestDECurrentResultsAndPersistence:
    def test_same_contrast_new_formula_filters_newest(self, de_setup):
        original, dm, tools = de_setup
        _iterate(tools)
        old_key = _ledger(dm.modalities["pb"])["iterations"][0]["results_key"]
        old = dm.modalities["pb"].uns[old_key]["results_df"].copy()
        result = tools["run_de_with_formula"].invoke(
            {
                "pseudobulk_modality": "pb",
                "formula": "~batch + condition",
                "contrast": CONTRAST,
                "save_results": False,
            }
        )
        assert "Complete" in result, result
        newest = dm.modalities["pb"].uns[RESULTS_KEY]["results_df"]
        assert (old["log2FoldChange"] - newest["log2FoldChange"]).abs().max() > 1e-3
        expected = newest[newest["padj"] <= 0.05]
        assert len(expected) == 12
        assert len(old[old["padj"] <= 0.05]) == 13
        tools["filter_de_results"].invoke(
            {"modality_name": "pb", "lfc_threshold": 0.0, "min_base_mean": 0.0}
        )
        ir = dm.last_ir("filter_de_results")
        assert ir.parameters["de_results_key"] == RESULTS_KEY
        pd.testing.assert_frame_equal(
            dm.modalities["pb"].uns[f"filtered_{RESULTS_KEY}"]["results_df"],
            expected,
        )
        assert not old_key.startswith("de_results")
        assert dm.modalities["pb"].uns["de_current_results_key"] == RESULTS_KEY
        replayed = original.copy()
        for call in dm.calls:
            replayed = replay_ir(call.ir, adata=replayed)["adata"]
        pd.testing.assert_frame_equal(
            replayed.uns[f"filtered_{RESULTS_KEY}"]["results_df"],
            dm.modalities["pb"].uns[f"filtered_{RESULTS_KEY}"]["results_df"],
        )
        assert replayed.uns["de_current_results_key"] == RESULTS_KEY

    def test_twelve_snapshots_roundtrip_continue_compare(self, de_setup, tmp_path):
        original, dm, tools = de_setup
        for i in range(1, 13):
            _iterate(tools, "~batch + condition" if i == 12 else "~condition")
        produced = dm.modalities["pb"]
        tracker = _ledger(produced)
        latest_key = tracker["iterations"][-1]["results_key"]
        snapshots = {
            i["results_key"]: produced.uns[i["results_key"]]["results_df"].copy()
            for i in tracker["iterations"]
        }
        # The real fallback may drop inherited design metadata, never the ledger.
        loaded = _save_with_exporter(produced.copy(), tmp_path / "twelve")
        assert "de_iterations_json" in loaded.uns
        assert _ledger(loaded) == tracker
        assert _ledger(loaded)["version"] == 1
        assert list(loaded.uns) == sorted(loaded.uns)  # HDF5 reorders keys
        dm.modalities["pb"] = loaded
        tools["filter_de_results"].invoke(
            {"modality_name": "pb", "lfc_threshold": 0.0, "min_base_mean": 0.0}
        )
        assert (
            dm.last_ir("filter_de_results").parameters["de_results_key"] == latest_key
        )
        assert len(loaded.uns[f"filtered_{latest_key}"]["results_df"]) == 12
        _iterate(tools, name="continued")
        continued_tracker = _ledger(loaded)
        assert continued_tracker["current_iteration"] == 13
        assert continued_tracker["iterations"][-1]["results_key"] not in snapshots
        for key, frame in snapshots.items():
            pd.testing.assert_frame_equal(loaded.uns[key]["results_df"], frame)
        result = tools["compare_de_iterations"].invoke({"pseudobulk_modality": "pb"})
        assert "DE Iteration Comparison" in result, result
        comparison = loaded.uns["iteration_comparisons"][
            "iteration_comparison_continued_vs_iteration_12"
        ]
        assert comparison["summary_stats"]["n_sig_1"] == 13
        assert comparison["summary_stats"]["n_sig_2"] == 12
        assert comparison["correlation"] < 1.0 - 1e-9
        replayed = original.copy()
        for call in dm.calls:
            replayed = replay_ir(call.ir, adata=replayed)["adata"]
        assert _ledger(replayed) == continued_tracker
        replay_comparison = replayed.uns["iteration_comparisons"][
            "iteration_comparison_continued_vs_iteration_12"
        ]
        assert replay_comparison["summary_stats"] == comparison["summary_stats"]
        assert replay_comparison["correlation"] == pytest.approx(
            comparison["correlation"]
        )
        final = _save_with_exporter(replayed, tmp_path / "continued")
        assert _ledger(final) == continued_tracker
        assert (
            final.uns["de_current_results_key"] == loaded.uns["de_current_results_key"]
        )

    @pytest.mark.parametrize("legacy_prefix", [False, True])
    def test_missing_ledger_never_overwrites_existing_snapshot(
        self, de_setup, legacy_prefix
    ):
        _, dm, tools = de_setup
        _iterate(tools)
        adata = dm.modalities["pb"]
        key = _ledger(adata)["iterations"][0]["results_key"]
        if legacy_prefix:
            adata.uns["de_results_iteration_10"] = adata.uns.pop(key)
            key = "de_results_iteration_10"
        frame = adata.uns[key]["results_df"].copy()
        adata.uns.pop("de_iterations", None)
        adata.uns.pop("de_iterations_json", None)
        _iterate(tools, "~batch + condition")
        pd.testing.assert_frame_equal(adata.uns[key]["results_df"], frame)
        assert _ledger(adata)["current_iteration"] == (11 if legacy_prefix else 2)
        assert _ledger(adata)["iterations"][-1]["results_key"] != key

    @pytest.mark.parametrize(
        "tool_name",
        [
            "filter_de_results",
            "export_de_results",
            "run_pathway_enrichment",
            "run_gsea_analysis",
            "extract_and_export_de_results",
        ],
    )
    def test_all_consumers_require_choice_for_ambiguous_legacy_results(
        self, de_setup, tool_name
    ):
        _, dm, tools = de_setup
        # Routing-only check: two independently named historical tables, neither
        # marked current. No network/scientific enrichment is needed or claimed.
        frame = pd.DataFrame(
            {"padj": [0.01], "log2FoldChange": [1.0], "baseMean": [20.0]},
            index=["gene_0"],
        )
        dm.modalities["pb"].uns["de_results_first"] = {"results_df": frame}
        dm.modalities["pb"].uns["de_results_second"] = {"results_df": frame.copy()}
        args = {"modality_name": "pb"}
        if tool_name == "run_pathway_enrichment":
            args["gene_list"] = []
        response = tools[tool_name].invoke(args)
        assert "explicit" in response.lower() and "de_results_key" in response, response
        assert not dm.calls

    def test_legacy_ledger_migrates_and_compares_after_reload(self, de_setup, tmp_path):
        _, dm, tools = de_setup
        _iterate(tools, name="legacy")
        adata = dm.modalities["pb"]
        tracker = _ledger(adata)
        tracker.pop("version", None)
        adata.uns.pop("de_iterations_json", None)
        # Model a pre-serialization session, with the old snapshot namespace.
        record = tracker["iterations"][0]
        adata.uns["de_results_iteration_1"] = adata.uns.pop(record["results_key"])
        record["results_key"] = "de_results_iteration_1"
        adata.uns["de_iterations"] = tracker
        _iterate(tools, "~batch + condition", name="new")
        assert "de_iterations" not in adata.uns
        assert _ledger(adata)["iterations"][0] == record
        loaded = _save_with_exporter(adata.copy(), tmp_path / "migrated")
        dm.modalities["pb"] = loaded
        result = tools["compare_de_iterations"].invoke({"pseudobulk_modality": "pb"})
        assert "DE Iteration Comparison" in result, result
        comparison = loaded.uns["iteration_comparisons"][
            "iteration_comparison_new_vs_legacy"
        ]
        assert comparison["summary_stats"]["n_sig_1"] == 12
        assert comparison["summary_stats"]["n_sig_2"] == 13
        assert comparison["correlation"] < 1.0 - 1e-9
        assert _ledger(loaded) == _ledger(adata)

    def test_snapshot_replay_refuses_overwrite(self, de_setup):
        _, dm, tools = de_setup
        _iterate(tools)
        adata = dm.modalities["pb"]
        info = _ledger(adata)["iterations"][0]
        original_snapshot = adata.uns[info["results_key"]]["results_df"].copy()
        record_ir = dm.last_ir("iterate_de_analysis")
        _iterate(tools, "~batch + condition")
        ledger_before = _ledger(adata)
        with pytest.raises(ValueError, match="Refusing to overwrite"):
            replay_ir(record_ir, adata=adata)
        assert _ledger(adata) == ledger_before
        pd.testing.assert_frame_equal(
            adata.uns[info["results_key"]]["results_df"], original_snapshot
        )

    @pytest.mark.parametrize(
        "invalid_ledger",
        [
            "not json",
            '{"version": 99, "iterations":[], "current_iteration":0}',
            '{"version":1, "iterations":[], "current_iteration":-1}',
        ],
    )
    def test_invalid_ledger_fails_without_running_de(self, de_setup, invalid_ledger):
        _, dm, tools = de_setup
        dm.modalities["pb"].uns["de_iterations_json"] = invalid_ledger
        response = tools["iterate_de_analysis"].invoke(
            {
                "pseudobulk_modality": "pb",
                "new_formula": "~condition",
                "new_contrast": CONTRAST,
            }
        )
        assert "Error" in response, response
        assert dm.modalities["pb"].uns["de_iterations_json"] == invalid_ledger
        assert not dm.calls

    def test_unique_legacy_fallback_and_explicit_snapshot_choice(self, de_setup):
        _, dm, tools = de_setup
        _iterate(tools)
        adata = dm.modalities["pb"]
        snapshot_key = _ledger(adata)["iterations"][0]["results_key"]
        adata.uns.pop("de_current_results_key")
        adata.uns.pop(RESULTS_KEY)
        response = tools["filter_de_results"].invoke({"modality_name": "pb"})
        assert "DE Results Filtered" in response, response
        assert (
            dm.last_ir("filter_de_results").parameters["de_results_key"] == snapshot_key
        )
        _iterate(tools, "~batch + condition")
        response = tools["filter_de_results"].invoke(
            {
                "modality_name": "pb",
                "de_results_key": snapshot_key,
                "lfc_threshold": 0.0,
                "min_base_mean": 0.0,
            }
        )
        assert "DE Results Filtered" in response, response
        assert len(adata.uns[f"filtered_{snapshot_key}"]["results_df"]) == 13
        adata.uns["de_current_results_key"] = "deleted_result"
        response = tools["filter_de_results"].invoke({"modality_name": "pb"})
        assert "explicit de_results_key" in response, response

    @pytest.mark.parametrize("path", ["pseudobulk", "direct", "bulk_direct"])
    def test_other_successful_de_paths_record_current_identity(self, de_setup, path):
        _, dm, tools = de_setup
        adata = dm.modalities["pb"]
        if path == "pseudobulk":
            adata.uns["is_pseudobulk"] = True
        else:
            # A success fixture must survive the marker-style out-group cap.
            adata.X[adata.obs["condition"] == "treated", :10] = 1
        if path == "bulk_direct":
            name = "run_bulk_de_direct"
            args = {
                "modality_name": "pb",
                "group_key": "condition",
                "group1": "control",
                "group2": "treated",
                "save_result": False,
            }
        else:
            name = "run_differential_expression"
            args = {
                "modality_name": "pb",
                "groupby": "condition",
                "group1": "control",
                "group2": "treated",
                "save_result": False,
            }
        response = tools[name].invoke(args)
        assert "Complete" in response and "Error" not in response, response
        output = dm.modalities[
            "pb" if path == "pseudobulk" else "pb_de_control_vs_treated"
        ]
        assert output.n_vars > 0
        expected_key = (
            "de_results_condition_treated_vs_control"
            if path == "pseudobulk"
            else "de_results_control_vs_treated"
        )
        assert output.uns["de_current_results_key"] == expected_key
        assert expected_key in output.uns
        assert (
            len(
                pd.DataFrame(
                    output.uns[expected_key]["results_df"]
                    if path == "pseudobulk"
                    else output.uns[expected_key]
                )
            )
            > 0
        )
        _assert_real_ir(dm.last_ir(name))
        # Legacy non-formula templates have independent replay/format debt; this
        # asserts runtime identity, not parity of those inherited algorithms.

    @pytest.mark.parametrize(
        "tool_name",
        [
            "filter_de_results",
            "export_de_results",
            "run_pathway_enrichment",
            "run_gsea_analysis",
            "extract_and_export_de_results",
        ],
    )
    def test_all_consumers_use_identity_not_dictionary_order(
        self, de_setup, tmp_path, monkeypatch, tool_name
    ):
        _, dm, tools = de_setup
        _iterate(tools)
        _iterate(tools, "~batch + condition")
        adata = dm.modalities["pb"]
        tracker = _ledger(adata)
        current_key = tracker["iterations"][-1]["results_key"]
        old_key = tracker["iterations"][0]["results_key"]
        expected = adata.uns[current_key]["results_df"]
        # Order the real, independent pyDESeq2 tables adversarially. Current
        # identity remains authoritative even if old history was inserted last.
        old_snapshot = adata.uns.pop(old_key)
        adata.uns[old_key] = old_snapshot
        args = {"modality_name": "pb"}
        captured = {}
        if tool_name == "run_pathway_enrichment":
            from lobster.services.analysis.bulk_rnaseq_service import BulkRNASeqService

            def capture_pathway(self, gene_list, **kwargs):
                captured["genes"] = gene_list
                raise RuntimeError("offline routing boundary")

            monkeypatch.setattr(
                BulkRNASeqService, "run_pathway_enrichment", capture_pathway
            )
            args["gene_list"] = []
        elif tool_name == "run_gsea_analysis":
            from lobster.services.analysis.pathway_enrichment_service import (
                PathwayEnrichmentService,
            )

            def capture_gsea(self, ranked_genes, **kwargs):
                captured["ranked"] = ranked_genes.copy()
                raise RuntimeError("offline routing boundary")

            monkeypatch.setattr(
                PathwayEnrichmentService, "gene_set_enrichment_analysis", capture_gsea
            )
        elif tool_name in ("export_de_results", "extract_and_export_de_results"):
            args["output_path"] = str(dm.workspace_path / "selected.csv")
            if tool_name == "extract_and_export_de_results":
                args["shrink_lfc"] = False
        else:
            args.update(lfc_threshold=0.0, min_base_mean=0.0)
        response = tools[tool_name].invoke(args)
        if tool_name == "run_pathway_enrichment":
            assert "offline routing boundary" in response, response
            assert captured["genes"] == expected[expected["padj"] < 0.05].index.tolist()
            assert len(captured["genes"]) == 12
        elif tool_name == "run_gsea_analysis":
            assert "offline routing boundary" in response, response
            ranked = (
                pd.DataFrame(
                    {
                        "gene": expected.index,
                        "score": expected["log2FoldChange"].fillna(0).values,
                    }
                )
                .sort_values("score", ascending=False)
                .reset_index(drop=True)
            )
            pd.testing.assert_frame_equal(captured["ranked"], ranked)
        elif tool_name == "filter_de_results":
            assert "DE Results Filtered" in response, response
            pd.testing.assert_frame_equal(
                adata.uns[f"filtered_{current_key}"]["results_df"],
                expected[expected["padj"] < 0.05],
            )
        else:
            assert "Error" not in response, response
            exported = pd.read_csv(args["output_path"], index_col=0)
            selected = (
                expected
                if tool_name == "export_de_results"
                else expected[expected["padj"] <= 0.05]
            )
            pd.testing.assert_frame_equal(
                exported.sort_index(), selected.sort_index(), check_names=False
            )
        # External enrichment was intercepted only at the transport boundary.
        # The real DE tables/selected genes/scores are the routing oracle; this
        # does not claim to validate GO/GSEA scientific results offline.

    @pytest.mark.parametrize("strategy", ["successful", "replay"])
    def test_real_manager_export_load_save_resume(
        self, tmp_path, monkeypatch, strategy
    ):
        """Execute every exported code cell offline; no kernel/network transport."""
        import nbformat

        from lobster.core.notebooks.exporter import NotebookExporter
        from lobster.core.runtime.data_manager import DataManagerV2

        monkeypatch.chdir(tmp_path)
        # Only infrastructure metadata and LLM construction are intercepted.
        # DE, provenance, export, load, save and comparison are the real code.
        monkeypatch.setattr(
            NotebookExporter, "_get_system_info", lambda self: {"mode": "offline"}
        )
        original = _pseudobulk()
        input_path = tmp_path / "input.h5ad"
        original.write_h5ad(input_path)
        dm = DataManagerV2(workspace_path=tmp_path / "runtime", auto_scan=False)
        dm.modalities["pb"] = original.copy()
        tools = factory_tools(de_analysis_expert, DE_MODULE, dm)
        _iterate(tools, name="base")
        _iterate(tools, "~batch + condition", name="batch")
        tools["filter_de_results"].invoke(
            {"modality_name": "pb", "lfc_threshold": 0.0, "min_base_mean": 0.0}
        )
        response = tools["compare_de_iterations"].invoke({"pseudobulk_modality": "pb"})
        assert "DE Iteration Comparison" in response, response
        notebook_path = dm.export_notebook(
            name=f"de_{strategy}", filter_strategy=strategy
        )
        notebook = nbformat.read(notebook_path, as_version=4)
        scope = {}
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            # Execute fixture-generated notebook cells, including real load/save.
            exec(compile(cell.source, str(notebook_path), "exec"), scope)  # nosec B102
            if "parameters" in cell.metadata.get("tags", []):
                scope["input_data"] = str(input_path)
                scope["output_prefix"] = str(tmp_path / f"de_{strategy}")
        loaded = ad.read_h5ad(scope["output_path"])
        produced = dm.modalities["pb"]
        assert _ledger(loaded) == _ledger(produced)
        assert (
            loaded.uns["de_current_results_key"]
            == produced.uns["de_current_results_key"]
        )
        snapshot = _ledger(produced)["iterations"][-1]["results_key"]
        for key in (RESULTS_KEY, snapshot, f"filtered_{snapshot}"):
            pd.testing.assert_frame_equal(
                loaded.uns[key]["results_df"], produced.uns[key]["results_df"]
            )
        cmp_key = "iteration_comparison_batch_vs_base"
        assert (
            loaded.uns["iteration_comparisons"][cmp_key]["summary_stats"]
            == produced.uns["iteration_comparisons"][cmp_key]["summary_stats"]
        )
        resumed = DataManagerV2(workspace_path=tmp_path / "resumed", auto_scan=False)
        resumed.modalities["pb"] = loaded
        resumed_tools = factory_tools(de_analysis_expert, DE_MODULE, resumed)
        _iterate(resumed_tools, name="continued")
        assert _ledger(loaded)["current_iteration"] == 3
        response = resumed_tools["compare_de_iterations"].invoke(
            {"pseudobulk_modality": "pb"}
        )
        assert "DE Iteration Comparison" in response, response
        cmp_key = "iteration_comparison_continued_vs_batch"
        assert (
            loaded.uns["iteration_comparisons"][cmp_key]["summary_stats"]["n_sig_1"]
            == 13
        )
        assert (
            loaded.uns["iteration_comparisons"][cmp_key]["summary_stats"]["n_sig_2"]
            == 12
        )
