"""The differential-expression reply counts the proteins it tested.

A protein with fewer than 2 values in a compared group is not tested, but the
reply's "Proteins tested" gave every protein of the modality: 3,000 where 2,995
were tested. It now gives the proteins tested, and how many were not and why.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.agents.proteomics.de_analysis_expert import de_analysis_expert
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.services.analysis.proteomics_differential_service import (
    ProteomicsDifferentialService,
)
from lobster.testing.provenance_replay import factory_tools

pytestmark = pytest.mark.unit

DE = "lobster.agents.proteomics.de_analysis_expert"


def _proteins(n_sparse):
    """12 samples (6 a, 6 b), 50 proteins; the last ``n_sparse`` have one value
    in group a."""
    rng = np.random.default_rng(0)
    x = rng.normal(10, 1, (12, 50))
    x[6:, :5] += 4
    if n_sparse:
        x[1:6, 50 - n_sparse :] = np.nan
    obs = pd.DataFrame(
        {"group": ["a"] * 6 + ["b"] * 6}, index=[f"s{i}" for i in range(12)]
    )
    return ad.AnnData(x, obs=obs, var=pd.DataFrame(index=[f"P{i}" for i in range(50)]))


def _reply(tmp_path, adata):
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    dm.store_modality("prot", adata)
    return factory_tools(de_analysis_expert, DE, dm)[
        "find_differential_proteins"
    ].invoke(
        {
            "modality_name": "prot",
            "group_column": "group",
            "platform_type": "mass_spec",
            "method": "t_test",
        }
    )


def test_the_stats_count_the_proteins_tested():
    _, stats, _ = ProteomicsDifferentialService().perform_differential_expression(
        _proteins(n_sparse=5), group_column="group", test_method="t_test"
    )

    assert stats["n_proteins_tested"] == 45
    assert stats["proteins_processed"] == 50


def test_the_reply_counts_the_proteins_tested_and_those_left_untested(tmp_path):
    reply = _reply(tmp_path, _proteins(n_sparse=5))

    assert (
        "- Proteins tested: 45 of 50 (5 with fewer than 2 values in a compared "
        "group were not tested)\n" in reply
    ), reply


def test_the_reply_counts_every_protein_when_all_were_tested(tmp_path):
    reply = _reply(tmp_path, _proteins(n_sparse=0))

    assert "- Proteins tested: 50\n" in reply, reply


def test_a_protein_tested_in_three_comparisons_counts_once():
    adata = _proteins(n_sparse=0)
    adata.obs["group"] = ["a"] * 4 + ["b"] * 4 + ["c"] * 4

    _, stats, _ = ProteomicsDifferentialService().perform_differential_expression(
        adata, group_column="group", test_method="t_test"
    )

    assert stats["n_proteins_tested"] == 50


def test_the_ptm_reply_counts_the_sites_and_proteins_tested(tmp_path):
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    dm.store_modality("sites", _proteins(n_sparse=5))
    dm.store_modality("prot", _proteins(n_sparse=2))

    reply = factory_tools(de_analysis_expert, DE, dm)[
        "run_differential_ptm_analysis"
    ].invoke(
        {
            "modality_name": "sites",
            "protein_modality_name": "prot",
            "group_column": "group",
        }
    )

    assert (
        "- PTM sites tested: 45 of 50 (5 with fewer than 2 values in a compared "
        "group were not tested)\n" in reply
    ), reply
    assert (
        "- Proteins tested: 48 of 50 (2 with fewer than 2 values in a compared "
        "group were not tested)\n" in reply
    ), reply
