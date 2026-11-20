"""When no test has a p-value, the DE, time course and correlation tools refuse.

A protein with the same value in every sample has no p-value from any of them,
and neither has any protein against a target with one value. When that is
every test, the tool reported "Complete" with 0 significant and stored a result.
It now refuses, says why (naming a target with one value) and stores nothing.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.agents.proteomics.de_analysis_expert import de_analysis_expert
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.testing.provenance_replay import factory_tools

DE = "lobster.agents.proteomics.de_analysis_expert"


def _proteins(flat):
    """12 samples (6 a, 6 b; times 0-3; a dose), 20 proteins; with ``flat``
    every protein has one value in every sample (its own)."""
    rng = np.random.default_rng(0)
    x = rng.normal(10, 1, (12, 20))
    if flat:
        x = np.tile(np.arange(1.0, 21.0), (12, 1))
    obs = pd.DataFrame(
        {
            "group": ["a"] * 6 + ["b"] * 6,
            "time": [0.0, 1.0, 2.0, 3.0] * 3,
            "dose": rng.uniform(0, 5, 12),
            "batch_dose": [2.5] * 12,
        },
        index=[f"s{i}" for i in range(12)],
    )
    return ad.AnnData(x, obs=obs, var=pd.DataFrame(index=[f"P{i}" for i in range(20)]))


def _call(tmp_path, adata, tool, **args):
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    dm.store_modality("prot", adata)
    reply = factory_tools(de_analysis_expert, DE, dm)[tool].invoke(
        {"modality_name": "prot", **args}
    )
    return reply, [m for m in dm.list_modalities() if m != "prot"]


def _logged(tmp_path, monkeypatch, adata, tool, **args):
    logged = []
    monkeypatch.setattr(
        DataManagerV2,
        "log_tool_usage",
        lambda self, **kw: logged.append(kw["tool_name"]),
    )
    _call(tmp_path, adata, tool, **args)
    return logged


@pytest.mark.parametrize("flat", [True, False])
def test_de_with_no_p_value_refuses(tmp_path, flat):
    reply, stored = _call(
        tmp_path,
        _proteins(flat),
        "find_differential_proteins",
        group_column="group",
        platform_type="affinity",
        method="t_test",
    )

    refusal = (
        "**DE refused: no protein has a p-value.**\n\n"
        "Of 20 tests, none has a p-value: every protein has the same value in "
        "every sample compared."
    )
    assert reply.startswith(refusal) is flat, reply
    assert ("No result modality was stored." in reply) is flat
    assert (stored == []) is flat


@pytest.mark.parametrize("flat", [True, False])
def test_time_course_with_no_p_value_refuses(tmp_path, flat):
    reply, stored = _call(
        tmp_path, _proteins(flat), "run_time_course_analysis", time_column="time"
    )

    refusal = (
        "**Time course refused: no protein has a p-value.**\n\n"
        "Of 20 tests, none has a p-value: every protein has the same value at "
        "every time point."
    )
    assert reply.startswith(refusal) is flat, reply
    assert (stored == []) is flat


@pytest.mark.parametrize(
    "flat, target, cause",
    [
        (
            True,
            "dose",
            "every protein has the same value in every sample.",
        ),
        (
            False,
            "batch_dose",
            "every value of 'batch_dose' is the same (2.5), so no protein can be "
            "correlated with it.",
        ),
        (False, "dose", None),
    ],
    ids=["flat_proteins", "flat_target", "neither"],
)
def test_correlation_with_no_p_value_refuses(tmp_path, flat, target, cause):
    reply, stored = _call(
        tmp_path, _proteins(flat), "run_correlation_analysis", target_column=target
    )

    head = "**Correlation refused: no protein has a p-value.**\n\n"
    if cause is None:
        assert not reply.startswith(head), reply
        assert stored == ["prot_correlation"]
    else:
        assert reply.startswith(f"{head}Of 20 tests, none has a p-value: {cause}")
        assert stored == []


FLAT_ADVICE = (
    "Check the values loaded for this modality. No result modality was stored."
)


@pytest.mark.parametrize(
    "flat, target, refusal",
    [
        (
            True,
            "dose",
            "**Correlation refused: no protein has a p-value.**\n\n"
            "Of 20 tests, none has a p-value: every protein has the same value in "
            f"every sample. {FLAT_ADVICE}",
        ),
        (
            False,
            "batch_dose",
            "**Correlation refused: no protein has a p-value.**\n\n"
            "Of 20 tests, none has a p-value: every value of 'batch_dose' is the "
            "same (2.5), so no protein can be correlated with it. Choose a target "
            "that varies across samples. No result modality was stored.",
        ),
    ],
    ids=["flat_proteins", "flat_target"],
)
def test_the_correlation_refusal_in_full(tmp_path, flat, target, refusal):
    reply, _ = _call(
        tmp_path, _proteins(flat), "run_correlation_analysis", target_column=target
    )

    assert reply == refusal


def test_the_de_refusal_in_full_for_one_protein(tmp_path):
    reply, _ = _call(
        tmp_path,
        _proteins(flat=True)[:, :1].copy(),
        "find_differential_proteins",
        group_column="group",
        platform_type="affinity",
        method="t_test",
    )

    assert reply == (
        "**DE refused: no protein has a p-value.**\n\n"
        "Of 1 test, none has a p-value: every protein has the same value in every "
        f"sample compared. {FLAT_ADVICE}"
    )


@pytest.mark.parametrize(
    "tool, args",
    [
        (
            "find_differential_proteins",
            {"group_column": "group", "platform_type": "affinity", "method": "t_test"},
        ),
        ("run_time_course_analysis", {"time_column": "time"}),
        ("run_correlation_analysis", {"target_column": "dose"}),
    ],
    ids=["de", "time_course", "correlation"],
)
@pytest.mark.parametrize("flat", [True, False])
def test_a_refusal_logs_nothing(tmp_path, monkeypatch, tool, args, flat):
    logged = _logged(tmp_path, monkeypatch, _proteins(flat), tool, **args)

    assert logged == ([] if flat else [tool])
