"""The filter result states which cells the earlier QC flagged but the filter kept."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from lobster.agents.transcriptomics.shared_tools import create_shared_tools
from lobster.core.provenance.analysis_ir import AnalysisStep
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.services.quality.preprocessing_service import PreprocessingService
from lobster.services.quality.quality_service import QualityService
from lobster.testing.provenance_replay import replay_ir

THRESHOLD_ORDER = (
    "min_genes",
    "max_genes",
    "max_mt_pct",
    "max_ribo_pct",
    "min_housekeeping_score",
)


@pytest.fixture
def counts():
    """128 cells x 2601 genes: cell 65 is high-ribosomal only; the rest pass QC."""
    x = np.zeros((128, 2601), dtype=np.float64)
    x[:, :240] = 1
    x[65, 1] = 1000  # RPS3: ribosomal share far above 50 %
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(128)])
    var = pd.DataFrame(
        {"source_label": ["synthetic"] * 2601},
        index=["MT-ND1", "RPS3", "ACTB"] + [f"GENE{i}" for i in range(2598)],
    )
    return ad.AnnData(sparse.csr_matrix(x), obs=obs, var=var)


def context(tmp_path, counts):
    dm = DataManagerV2(workspace_path=tmp_path / "workspace")
    dm.store_modality("input", counts.copy())
    tools = {
        t.name: t
        for t in create_shared_tools(dm, QualityService(), PreprocessingService())
    }
    return dm, tools


def assess_then_filter(dm, tools, **filter_args):
    tools["assess_data_quality"].invoke({"modality_name": "input"})
    return tools["filter_and_normalize"].invoke(
        {
            "modality_name": "input_quality_assessed",
            "save_result": False,
            **filter_args,
        }
    )


def flag_line(response):
    lines = [ln for ln in response.splitlines() if "flagged by" in ln]
    assert len(lines) == 1, response
    return lines[0]


def test_flagged_cells_the_filter_kept_are_named_with_reason_and_unapplied_filter(
    tmp_path, counts
):
    dm, tools = context(tmp_path, counts)
    response = assess_then_filter(dm, tools)
    line = flag_line(response)
    assert "1 cell flagged by QC" in line and "kept" in line
    assert "ribosomal" in line and "> 50" in line
    assert "not applied" in line
    kept = dm.get_modality("input_quality_assessed_filtered_normalized")
    assert int((~kept.obs["qc_pass"].astype(bool)).sum()) == 1  # the line is true


def test_an_explicit_ribo_filter_leaves_nothing_flagged_and_says_so(tmp_path, counts):
    dm, tools = context(tmp_path, counts)
    response = assess_then_filter(dm, tools, max_ribo_percent=50)
    line = flag_line(response)
    assert "0 cells flagged by QC" in line and "kept" in line


def test_no_earlier_assessment_means_no_flag_line(tmp_path, counts):
    dm, tools = context(tmp_path, counts)
    response = tools["filter_and_normalize"].invoke(
        {"modality_name": "input", "save_result": False}
    )
    assert "flagged by" not in response
    assert "no earlier QC assessment" in response


def test_thresholds_absent_gives_counts_only_and_says_so(tmp_path, counts):
    dm, tools = context(tmp_path, counts)
    tools["assess_data_quality"].invoke({"modality_name": "input"})
    qc = dm.get_modality("input_quality_assessed")
    del qc.uns["qc_thresholds"]  # an older modality has neither key
    del qc.uns["qc_threshold_names"]
    response = tools["filter_and_normalize"].invoke(
        {"modality_name": "input_quality_assessed", "save_result": False}
    )
    line = flag_line(response)
    assert "1 cell flagged by QC" in line
    assert "reasons unknown" in line and "thresholds" in line


def test_the_reasons_survive_the_save_and_reload_of_the_assessed_modality(
    tmp_path, counts
):
    """The save path turns numbers in a dict into strings; the recorded arrays survive."""
    dm, tools = context(tmp_path, counts)
    tools["assess_data_quality"].invoke({"modality_name": "input"})
    dm.save_modality("input_quality_assessed", "qc.h5ad")
    dm.modalities["reloaded"] = ad.read_h5ad(dm.data_dir / "qc.h5ad")
    response = tools["filter_and_normalize"].invoke(
        {"modality_name": "reloaded", "save_result": False}
    )
    line = flag_line(response)
    assert "ribosomal" in line and "> 50" in line and "unknown" not in line


def test_the_filter_states_its_thresholds_are_fixed_defaults(tmp_path, counts):
    dm, tools = context(tmp_path, counts)
    response = assess_then_filter(dm, tools)
    assert "fixed defaults" in response and "not fitted" in response


def recorded(uns):
    """The recorded thresholds as {name: value}, read by name."""
    return dict(zip(uns["qc_threshold_names"], uns["qc_thresholds"]))


def test_the_service_records_the_thresholds_it_used_by_name(counts):
    out, _, _ = QualityService().assess_quality(
        counts.copy(), min_genes=150, max_genes=4000, max_mt_pct=15, max_ribo_pct=40
    )
    assert out.uns["qc_thresholds"].dtype == np.float64
    assert list(out.uns["qc_threshold_names"]) == list(THRESHOLD_ORDER)
    assert recorded(out.uns) == {
        "min_genes": 150.0,
        "max_genes": 4000.0,
        "max_mt_pct": 15.0,
        "max_ribo_pct": 40.0,
        "min_housekeeping_score": 1.0,  # the service default
    }


def render_threshold_block(**args):
    """Run only the template's thresholds block (the QC expressions above it need all values)."""
    from lobster.core.provenance.analysis_ir import _JINJA_ENV

    ir = QualityService()._create_quality_ir(**args)
    rendered = _JINJA_ENV.from_string(ir.code_template).render(**ir.parameters)
    start = rendered.index("_qc_given = {")
    end = rendered.index("# Display QC summary")
    scope = {"adata": ad.AnnData(np.zeros((1, 1))), "np": np}
    exec(compile(rendered[start:end], "<ir-block>", "exec"), scope)
    return scope["adata"].uns


ALL_SET = dict(
    min_genes=200,
    max_genes=5000,
    max_mt_pct=20.0,
    max_ribo_pct=50.0,
    min_housekeeping_score=1.0,
)
PARTLY_SET = dict(ALL_SET, max_ribo_pct=None, min_housekeeping_score=0.0, max_genes=5e3)
NONE_SET = dict.fromkeys(ALL_SET)


@pytest.mark.parametrize(
    "args", [ALL_SET, PARTLY_SET, NONE_SET], ids=["all", "partly", "none"]
)
def test_service_and_template_write_the_same_arrays(args, tmp_path):
    from lobster.services.quality.quality_service import set_qc_thresholds

    service_uns = {}
    set_qc_thresholds(service_uns, **args)
    template_uns = render_threshold_block(**args)
    assert set(service_uns) == set(template_uns)
    for key in service_uns:
        assert service_uns[key].dtype == template_uns[key].dtype
        np.testing.assert_array_equal(service_uns[key], template_uns[key])
    if args is NONE_SET:
        assert service_uns == {}  # neither key
    else:
        assert not np.isnan(service_uns["qc_thresholds"]).any()  # no NaN stand-ins
        unset = [k for k, v in args.items() if v is None]
        assert not set(unset) & set(service_uns["qc_threshold_names"])

    # and the same after the data manager's save and reload
    dm = DataManagerV2(workspace_path=tmp_path / "workspace")
    back = {}
    for label, uns in (("service", service_uns), ("template", template_uns)):
        adata = ad.AnnData(np.zeros((2, 2), dtype=np.float32))
        adata.uns.update({k: v for k, v in uns.items() if k.startswith("qc_")})
        dm.modalities[label] = adata
        dm.save_modality(label, f"{label}.h5ad")
        back[label] = ad.read_h5ad(dm.data_dir / f"{label}.h5ad").uns
    assert set(back["service"]) == set(back["template"]) == set(service_uns)
    for key in service_uns:
        np.testing.assert_array_equal(back["service"][key], back["template"][key])
        assert (
            back["service"][key].dtype == service_uns[key].dtype
        )  # object / float64 kept


def test_the_exported_ir_matches_the_tool_end_to_end(tmp_path, counts):
    dm, tools = context(tmp_path, counts)
    tools["assess_data_quality"].invoke({"modality_name": "input"})
    activity = next(
        a for a in dm.provenance.activities if a["type"] == "assess_data_quality"
    )
    ir = AnalysisStep.from_dict(activity["ir"])
    scope = {"adata": counts.copy()}
    for statement in ir.imports:
        exec(compile(statement, "<ir-import>", "exec"), scope)
    for helper in ir.helper_code or []:
        exec(compile(helper, "<ir-helper>", "exec"), scope)
    replayed = replay_ir(ir, **scope)["adata"]
    tool_made = dm.get_modality("input_quality_assessed")
    for key in ("qc_thresholds", "qc_threshold_names"):
        np.testing.assert_array_equal(replayed.uns[key], tool_made.uns[key])
    assert list(replayed.obs["qc_pass"]) == list(tool_made.obs["qc_pass"])


def test_the_thresholds_survive_the_save_and_reload(tmp_path, counts):
    dm = DataManagerV2(workspace_path=tmp_path / "workspace")
    out, _, _ = QualityService().assess_quality(counts.copy(), max_ribo_pct=40)
    dm.modalities["qc"] = out
    dm.save_modality("qc", "qc.h5ad")
    back = ad.read_h5ad(dm.data_dir / "qc.h5ad")
    assert back.uns["qc_thresholds"].dtype == np.float64
    assert list(back.uns["qc_threshold_names"]) == list(THRESHOLD_ORDER)
    assert recorded(back.uns) == recorded(out.uns)


def test_an_older_file_without_the_keys_reads_as_absent(counts):
    from lobster.services.quality.quality_service import _read_qc_thresholds

    assert _read_qc_thresholds({}) is None
    assert _read_qc_thresholds({"other": 1}) is None


@pytest.mark.parametrize(
    "uns",
    [
        {"qc_thresholds": np.array([1.0])},  # one key without the other
        {"qc_threshold_names": np.array(["min_genes"], dtype=object)},
        {
            "qc_thresholds": np.array([1.0, 2.0]),
            "qc_threshold_names": np.array(["min_genes"], dtype=object),
        },  # length mismatch
        {
            "qc_thresholds": np.array([1.0, 2.0]),
            "qc_threshold_names": np.array(["min_genes", "min_genes"], dtype=object),
        },  # repeated name
        {
            "qc_thresholds": np.array([1.0]),
            "qc_threshold_names": np.array([3], dtype=object),
        },  # not a string
        {
            "qc_thresholds": {"min_genes": "1"},
            "qc_threshold_names": np.array(["min_genes"], dtype=object),
        },  # a dict where values should be
    ],
    ids=["values-only", "names-only", "length", "repeat", "non-string", "dict"],
)
def test_unreadable_thresholds_give_counts_only_and_no_crash(counts, uns):
    from lobster.agents.transcriptomics.shared_tools import _qc_flags_line

    out, _, _ = QualityService().assess_quality(counts.copy())
    out.uns.pop("qc_thresholds"), out.uns.pop("qc_threshold_names")
    out.uns.update(uns)
    line = _qc_flags_line(out, {}, "cells")
    assert "flagged by QC" in line
    assert "unreadable" in line and "Reasons overlap" not in line


def test_the_reader_pairs_by_name_not_position(counts):
    from lobster.agents.transcriptomics.shared_tools import _qc_flags_line

    out, _, _ = QualityService().assess_quality(counts.copy())
    # the same pairs in another order must read the same
    names = list(out.uns["qc_threshold_names"])[::-1]
    values = list(out.uns["qc_thresholds"])[::-1]
    out.uns["qc_threshold_names"] = np.array(names, dtype=object)
    out.uns["qc_thresholds"] = np.array(values, dtype=np.float64)
    line = _qc_flags_line(out, {}, "cells")
    assert "ribosomal" in line and "> 50" in line


def test_a_flag_no_recorded_threshold_explains_is_said_plainly(counts):
    from lobster.agents.transcriptomics.shared_tools import _qc_flags_line
    from lobster.services.quality.quality_service import set_qc_thresholds

    out, _, _ = QualityService().assess_quality(counts.copy())
    out.obs["ribo_pct"] = (
        np.nan
    )  # a flagged cell whose metrics are gone: no reason counts
    set_qc_thresholds(out.uns, 200, 5000, 20.0, 50.0, 0.0)
    line = _qc_flags_line(out, {}, "cells")
    assert "no recorded threshold explains the flag" in line
    assert "Reasons overlap" not in line and "; ." not in line


def test_rerunning_the_assessment_replaces_stale_thresholds():
    from lobster.services.quality.quality_service import set_qc_thresholds

    uns = {}
    set_qc_thresholds(uns, 200, 5000, 20.0, 50.0, 1.0)
    set_qc_thresholds(uns, None, None, None, None, None)
    assert "qc_thresholds" not in uns and "qc_threshold_names" not in uns


def test_the_order_is_one_constant_used_by_both_paths(monkeypatch):
    """Reordering QC_THRESHOLD_KEYS reorders the service and the template together."""
    from lobster.services.quality import quality_service as qs

    monkeypatch.setattr(qs, "QC_THRESHOLD_KEYS", tuple(reversed(qs.QC_THRESHOLD_KEYS)))
    service_uns = {}
    qs.set_qc_thresholds(service_uns, **ALL_SET)
    template_uns = render_threshold_block(**ALL_SET)
    assert list(service_uns["qc_threshold_names"]) == list(qs.QC_THRESHOLD_KEYS)
    assert list(template_uns["qc_threshold_names"]) == list(qs.QC_THRESHOLD_KEYS)
    np.testing.assert_array_equal(
        service_uns["qc_thresholds"], template_uns["qc_thresholds"]
    )
