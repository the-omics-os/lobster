"""Each genomics tool's reply states the numbers the real service returned.

The service runs for real (a wrapper only records what it returned), so a key the agent
reads under a name the service never uses shows up as an error or a wrong number.
"""

import importlib

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

pytest.importorskip("sgkit")

from lobster.core.runtime.data_manager import DataManagerV2  # noqa: E402

agent_module = importlib.import_module("lobster.agents.genomics.genomics_expert")
gwas_module = importlib.import_module("lobster.services.analysis.gwas_service")


class Recorder:
    """Wrap a real service method and keep what it returned."""

    def __init__(self, monkeypatch, name):
        self.returned = []
        original = getattr(gwas_module.GWASService, name)

        def wrapper(service, *args, **kwargs):
            out = original(service, *args, **kwargs)
            self.returned.append(out)
            return out

        monkeypatch.setattr(gwas_module.GWASService, name, wrapper)

    @property
    def stats(self):
        assert self.returned, "the service was never called"
        return self.returned[-1][1]


@pytest.fixture
def context(tmp_path, monkeypatch):
    dm = DataManagerV2(workspace_path=tmp_path / "workspace")
    monkeypatch.setattr(agent_module, "create_llm", lambda *a, **k: object())
    monkeypatch.setattr(agent_module, "create_react_agent", lambda **kw: kw)
    switch = importlib.import_module("lobster.tools.code_tool_switch")
    monkeypatch.setattr(switch, "attach_code_tool", lambda tools, *a, **k: tools)
    graph = agent_module.genomics_expert(dm)
    dm.store_modality("g", genotypes())
    return dm, {tool.name: tool for tool in graph["tools"]}


def genotypes(n=300, v=60, seed=42):
    rng = np.random.default_rng(seed)
    g = rng.choice([0, 1, 2], size=(n, v), p=[0.49, 0.42, 0.09]).astype(float)
    g[1] = g[0]  # two identical samples: at least one related pair
    adata = AnnData(X=g.copy())
    adata.obs_names = [f"s{i}" for i in range(n)]
    adata.var = pd.DataFrame(
        {
            "CHROM": ["22"] * v,
            "POS": np.arange(16050000, 16050000 + v) * 10,
            "REF": ["A"] * v,
            "ALT": ["G"] * v,
        },
        index=[f"v{i}" for i in range(v)],
    )
    adata.obs["height"] = 170 + 8 * g[:, 3] + rng.normal(0, 3, n)
    adata.obs["age"] = rng.integers(20, 80, n)
    adata.obs["sex"] = rng.choice([1, 2], n)
    adata.layers["GT"] = g.copy()
    adata.uns["data_type"] = "genomics"
    adata.uns["modality"] = "wgs"
    return adata


def summary_of(dm, name):
    return {m["name"]: m for m in dm.list_modalities_with_lineage()}[name][
        "step_summary"
    ]


def test_run_gwas_step_summary_counts_the_significant_variants(context, monkeypatch):
    dm, tools = context
    rec = Recorder(monkeypatch, "run_gwas")
    reply = tools["run_gwas"].invoke(
        {"modality_name": "g", "phenotype": "height", "covariates": "age,sex"}
    )
    n = rec.stats["n_variants_significant"]
    assert n > 0  # a zero here would make the test vacuous
    assert f"- Significant variants (p < 5e-08): {n:,}" in reply
    assert summary_of(dm, "g_gwas") == f"GWAS: {n} significant hits"


def test_compute_kinship_reply_matches_the_service(context, monkeypatch):
    dm, tools = context
    rec = Recorder(monkeypatch, "compute_kinship")
    reply = tools["compute_kinship"].invoke({"modality_name": "g"})
    stats = rec.stats
    assert not reply.startswith("Unexpected error"), reply
    assert stats["n_related_pairs"] > 0
    assert f"- Samples: {stats['n_samples']:,}" in reply
    assert f"- Related pairs (kinship > 0.125): {stats['n_related_pairs']:,}" in reply
    assert f"- Mean kinship: {stats['mean_kinship']:.4f}" in reply
    # the pair list the service returns on the modality is shown, with its own values
    first = rec.returned[-1][0].uns["related_pairs"][0]
    assert f"{first['sample_i']} <-> {first['sample_j']}" in reply
    assert f"kinship={first['kinship_coeff']:.4f}" in reply


def test_clump_results_reply_matches_the_service(context, monkeypatch):
    dm, tools = context
    tools["run_gwas"].invoke(
        {"modality_name": "g", "phenotype": "height", "covariates": "age,sex"}
    )
    rec = Recorder(monkeypatch, "clump_gwas_results")
    reply = tools["clump_results"].invoke({"modality_name": "g_gwas"})
    stats = rec.stats
    assert not reply.startswith("Unexpected error"), reply
    assert stats["n_significant"] > 0
    assert f"- Significant variants (p < 5e-08): {stats['n_significant']:,}" in reply
    assert f"- Independent loci (clumps): {stats['n_clumps']:,}" in reply
    assert (
        summary_of(dm, "g_gwas_clumped")
        == f"Clumped: {stats['n_clumps']} independent loci from "
        f"{stats['n_significant']} significant variants"
    )
