"""The survival biomarker screen counts the proteins it tested.

A protein with the same value in every sample puts every sample on one side of
the median: the log-rank test has nothing to compare and gives no p-value. The
screen still counted it as tested and passed its missing p-value into the FDR
correction, which raised the FDR of every other protein. It is now left out,
and the reply says how many proteins were not tested and why.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from lifelines.statistics import logrank_test

from lobster.agents.proteomics.biomarker_discovery_expert import (
    biomarker_discovery_expert,
)
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.services.analysis.proteomics_survival_service import (
    ProteomicsSurvivalService,
)
from lobster.testing.provenance_replay import factory_tools

pytestmark = pytest.mark.unit

BIOMARKER = "lobster.agents.proteomics.biomarker_discovery_expert"


def _patients(constant=True):
    """30 patients, 4 proteins: P0 and P3 testable, P2 has 5 values (fewer
    than 20), and P1 is the same in every patient unless ``constant`` is off."""
    rng = np.random.default_rng(0)
    x = rng.normal(10, 1, (30, 4))
    if constant:
        x[:, 1] = 5.0
    x[:25, 2] = np.nan
    x[:20, 3] = 1.0
    obs = pd.DataFrame(
        {"T": rng.uniform(10, 100, 30), "E": rng.integers(0, 2, 30).astype(float)},
        index=[f"s{i}" for i in range(30)],
    )
    return ad.AnnData(x, obs=obs, var=pd.DataFrame(index=[f"P{i}" for i in range(4)]))


def _logrank_p(adata, j):
    """The log-rank p of a median split, from lifelines directly."""
    v = adata.X[:, j]
    high = v > np.median(v)
    t, e = adata.obs["T"].values, adata.obs["E"].values
    return logrank_test(t[~high], t[high], e[~high], e[high]).p_value


def test_a_constant_protein_is_not_tested_nor_corrected():
    adata = _patients()
    out, stats, _ = ProteomicsSurvivalService().batch_kaplan_meier(
        adata, duration_col="T", event_col="E"
    )

    assert stats["n_proteins_tested"] == 2
    assert stats["proteins_processed"] == 4
    results = {r["protein"]: r for r in out.uns["batch_kaplan_meier"]["results"]}
    assert sorted(results) == ["P0", "P3"]
    # Benjamini-Hochberg over the two p-values, by hand
    p0, p3 = _logrank_p(adata, 0), _logrank_p(adata, 3)
    lo, hi = sorted([p0, p3])
    q_hi = hi
    q_lo = min(lo * 2, q_hi)
    want = {p0: q_lo if p0 == lo else q_hi, p3: q_lo if p3 == lo else q_hi}
    assert results["P0"]["fdr"] == pytest.approx(want[p0])
    assert results["P3"]["fdr"] == pytest.approx(want[p3])


def test_a_protein_with_no_events_is_tested_with_p_one():
    adata = _patients()
    adata.obs["E"] = 0.0
    out, stats, _ = ProteomicsSurvivalService().batch_kaplan_meier(
        adata, duration_col="T", event_col="E"
    )

    results = {r["protein"]: r for r in out.uns["batch_kaplan_meier"]["results"]}
    assert sorted(results) == ["P0", "P3"]
    assert stats["n_proteins_tested"] == 2
    # lifelines gives p = 1.0 with no events; Benjamini-Hochberg of [1, 1] is [1, 1]
    for prot, j in (("P0", 0), ("P3", 3)):
        assert _logrank_p(adata, j) == 1.0
        assert results[prot]["p_value"] == 1.0
        assert results[prot]["fdr"] == 1.0


def test_unknown_protein_names_are_not_counted_as_untested():
    out, stats, _ = ProteomicsSurvivalService().batch_kaplan_meier(
        _patients(),
        duration_col="T",
        event_col="E",
        proteins=["P0", "NOPE", "P1", "P3", "ALSO_NOPE"],
    )

    assert stats["n_proteins_tested"] == 2
    assert stats["proteins_processed"] == 3
    assert stats["proteins_not_found"] == ["NOPE", "ALSO_NOPE"]


def _reply(tmp_path, adata, **kw):
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    dm.store_modality("prot", adata)
    return factory_tools(biomarker_discovery_expert, BIOMARKER, dm)[
        "find_survival_biomarkers"
    ].invoke({"modality_name": "prot", "time_column": "T", "event_column": "E", **kw})


def test_the_reply_counts_the_proteins_tested_and_those_left_untested(tmp_path):
    reply = _reply(tmp_path, _patients())

    assert (
        "- Proteins tested: 2 of 4; not tested: 2 (fewer than 20 samples with a "
        "value, or no p-value: at least half the samples share the highest "
        "value)\n" in reply
    ), reply
    assert "no events" not in reply
    assert "not found" not in reply


def test_the_reply_names_unknown_proteins(tmp_path):
    reply = _reply(
        tmp_path, _patients(), proteins=["P0", "NOPE", "P1", "P3", "ALSO_NOPE"]
    )

    assert "- Proteins tested: 2 of 3; not tested: 1 (" in reply, reply
    assert "- Not found in the data: NOPE, ALSO_NOPE\n" in reply, reply


def test_the_reply_refuses_when_no_protein_is_found(tmp_path):
    reply = _reply(tmp_path, _patients(), proteins=["NOPE"])

    assert "None of the specified proteins were found" in reply, reply
    assert "Proteins tested" not in reply


def test_the_reply_counts_every_protein_when_all_were_tested(tmp_path):
    adata = _patients(constant=False)[:, ["P0", "P1", "P3"]].copy()

    assert "- Proteins tested: 3\n" in _reply(tmp_path, adata)
