"""train_cox_model reports what the real survival service returned."""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

pytest.importorskip("sksurv")

from lobster.agents.machine_learning.shared_tools import (  # noqa: E402
    create_survival_analysis_tools,
)
from lobster.core.runtime.data_manager import DataManagerV2  # noqa: E402
from lobster.services.analysis.survival_analysis_service import (  # noqa: E402
    SurvivalAnalysisService,
)


@pytest.fixture
def context(tmp_path):
    rng = np.random.default_rng(0)
    n, g = 120, 30
    x = rng.normal(size=(n, g)).astype("float32")
    risk = x[:, 0] * 1.2 + x[:, 1] * 0.8
    obs = pd.DataFrame(
        {
            "time": rng.exponential(np.exp(-risk) * 10) + 0.1,
            "event": (rng.random(n) < 0.7).astype(int),
        },
        index=[f"s{i}" for i in range(n)],
    )
    adata = AnnData(x, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(g)]))
    dm = DataManagerV2(workspace_path=tmp_path / "workspace")
    dm.store_modality("m", adata)
    tools = {t.name: t for t in create_survival_analysis_tools(dm)}
    return dm, tools["train_cox_model"]


def record(monkeypatch, **forced):
    returned = []
    original = SurvivalAnalysisService.train_cox_model

    def wrapper(self, *args, **kwargs):
        out = original(self, *args, **{**kwargs, **forced})
        returned.append(out)
        return out

    monkeypatch.setattr(SurvivalAnalysisService, "train_cox_model", wrapper)
    return returned


ARGS = {"modality_name": "m", "time_column": "time", "event_column": "event"}


def test_the_training_set_c_index_is_labelled_and_the_model_is_stored(
    context, monkeypatch
):
    dm, tool = context
    returned = record(monkeypatch)
    reply = tool.invoke(ARGS)
    stats = returned[-1][1]
    assert stats["c_index_source"] == "training_set"  # what the tool can produce today
    assert not reply.startswith("Error"), reply
    assert (
        f"- C-index: {stats['c_index_train']:.3f} (training set, optimistically biased)"
        in reply
    )
    assert f"- Concordant pairs: {stats['concordant_pairs']}" in reply
    assert f"- Discordant pairs: {stats['discordant_pairs']}" in reply
    assert f"- Features selected: {stats['n_features_selected']}" in reply
    for warning in stats["warnings"]:
        assert warning in reply
    assert "m_cox_model" in dm.list_modalities()


def test_a_cross_validated_c_index_has_no_pair_counts(context, monkeypatch):
    dm, tool = context
    returned = record(monkeypatch, regularized=True)
    reply = tool.invoke(ARGS)
    stats = returned[-1][1]
    assert stats["c_index_source"] == "cross_validation"
    assert not reply.startswith("Error"), reply
    assert f"- C-index: {stats['c_index_cv']:.3f} (cross-validation)" in reply
    assert "Concordant pairs" not in reply
    assert "m_cox_model" in dm.list_modalities()


def test_the_docstring_does_not_promise_regularization(context):
    _, tool = context
    text = tool.description.lower()
    assert "unregularized" in text and "training" in text
    assert "elastic net regularization" not in text


def _scored(context):
    dm, tool = context
    tools = {t.name: t for t in create_survival_analysis_tools(dm)}
    assert not tools["train_cox_model"].invoke(ARGS).startswith("Error")
    return dm, tools


def test_optimize_risk_threshold_runs_and_reports_the_service_numbers(
    context, monkeypatch
):
    dm, tools = _scored(context)
    returned = []
    original = SurvivalAnalysisService.optimize_threshold

    def wrapper(self, *args, **kwargs):
        out = original(self, *args, **kwargs)
        returned.append((kwargs, out))
        return out

    monkeypatch.setattr(SurvivalAnalysisService, "optimize_threshold", wrapper)
    reply = tools["optimize_risk_threshold"].invoke(
        {**ARGS, "modality_name": "m_cox_model", "n_bootstrap": 7}
    )
    assert not reply.startswith("Error"), reply
    kwargs, (_, stats, _) = returned[-1]
    assert kwargs["n_iterations"] == 7  # the tool's n_bootstrap reaches the service
    assert f"- MCC: {stats['mcc']:.3f} (+/- {stats['mcc_std']:.3f})" in reply
    assert f"- High risk: {stats['n_high_risk']}" in reply


def test_kaplan_meier_ungrouped_reply_matches_the_service(context, monkeypatch):
    dm, tools = _scored(context)
    returned = []
    original = SurvivalAnalysisService.kaplan_meier_analysis

    def wrapper(self, *args, **kwargs):
        out = original(self, *args, **kwargs)
        returned.append(out)
        return out

    monkeypatch.setattr(SurvivalAnalysisService, "kaplan_meier_analysis", wrapper)
    reply = tools["run_kaplan_meier"].invoke(ARGS)
    stats = returned[-1][1]
    assert not reply.startswith("Error"), reply
    assert stats["median_survival"] is not None
    assert f"- Median survival: {stats['median_survival']:.1f}" in reply
    assert f"- Max follow-up: {stats['max_time']:.1f}" in reply


def test_kaplan_meier_says_not_reached_when_there_is_no_median(context):
    dm, tool = context
    tools = {t.name: t for t in create_survival_analysis_tools(dm)}
    adata = dm.get_modality("m")
    adata.obs["event"] = 0
    adata.obs.iloc[:25, adata.obs.columns.get_loc("event")] = 1  # survival stays > 0.5
    dm.store_modality("few_events", adata.copy())
    reply = tools["run_kaplan_meier"].invoke({**ARGS, "modality_name": "few_events"})
    assert "- Median survival: Not reached" in reply


def test_the_survival_prompt_describes_the_tools_as_they_are():
    from lobster.agents.machine_learning import prompts

    text = prompts.create_survival_analysis_expert_prompt()
    for false_statement in (
        "elastic net regularization",
        "regularized=True",
        "test_adata",
        'obs["risk_score"]',
        'obs["risk_group"]',
        "exclude_early_censored",
        "subsample_fraction",
        "HR near",
        "_thresholded",
        "Report test set C-index",
        "ALWAYS use test set",
        "Use test set C-index",
        "Bootstrap-based",
    ):
        assert false_statement not in text
    assert 'obs["cox_risk_score"]' in text and 'obs["risk_category"]' in text
    assert "not hazard ratios or confidence intervals" in text
    assert "require validation on independent test data" in text


def test_the_ml_expert_prompt_does_not_promise_hazard_ratios():
    from lobster.agents.machine_learning import prompts

    text = prompts.create_ml_expert_prompt()
    assert '"hazard ratio"' in text  # routing trigger kept
    assert "not hazard ratios or confidence intervals" in text
