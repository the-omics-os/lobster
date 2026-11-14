"""Each tool's reply states the numbers the real service returned.

The service runs for real; a wrapper only records what it returned, so a key the agent
reads under a name the service never uses shows up as a different number in the reply.
"""

import importlib
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from lobster.core.runtime.data_manager import DataManagerV2

parent_module = importlib.import_module(
    "lobster.agents.transcriptomics.transcriptomics_expert"
)


class Recorder:
    """Wrap a real service method and keep what it returned."""

    def __init__(self, monkeypatch, cls, name):
        self.returned = []
        original = getattr(cls, name)

        def wrapper(service, *args, **kwargs):
            out = original(service, *args, **kwargs)
            self.returned.append(out)
            return out

        monkeypatch.setattr(cls, name, wrapper)

    @property
    def stats(self):
        assert self.returned, "the service was never called"
        return self.returned[-1][1]


@pytest.fixture
def context(tmp_path, monkeypatch):
    dm = DataManagerV2(workspace_path=tmp_path / "workspace")
    settings = SimpleNamespace(get_agent_llm_params=lambda name: {})
    monkeypatch.setattr(parent_module, "get_settings", lambda: settings)
    monkeypatch.setattr(parent_module, "create_llm", lambda *a, **k: object())
    monkeypatch.setattr(parent_module, "create_react_agent", lambda **kw: kw)
    switch = importlib.import_module("lobster.tools.code_tool_switch")
    monkeypatch.setattr(switch, "attach_code_tool", lambda tools, *a, **k: tools)
    graph = parent_module.transcriptomics_expert(dm)
    return dm, {tool.name: tool for tool in graph["tools"]}


def single_cell(n_cells=300, n_genes=400, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.poisson(1.0, size=(n_cells, n_genes)).astype(np.float32)
    x[:150, :40] += 5  # two groups
    x[0:20] = x[0:20] + x[150:170]  # 20 synthetic doublets
    obs = pd.DataFrame(index=[f"c{i}" for i in range(n_cells)])
    var = pd.DataFrame(index=[f"G{i}" for i in range(n_genes)])
    return ad.AnnData(sparse.csr_matrix(x), obs=obs, var=var)


def bulk(n_samples=12, n_genes=500, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.poisson(30, size=(n_samples, n_genes)).astype(np.float32)
    x[:, :60] = 0  # genes the count filter must remove
    obs = pd.DataFrame(
        {
            "batch": ["a", "b"] * (n_samples // 2),
            "condition": ["ctrl"] * (n_samples // 2) + ["trt"] * (n_samples // 2),
        },
        index=[f"s{i}" for i in range(n_samples)],
    )
    var = pd.DataFrame(index=[f"G{i}" for i in range(n_genes)])
    adata = ad.AnnData(x, obs=obs, var=var)
    adata.layers["counts"] = adata.X.copy()
    return adata


def test_detect_doublets_reply_matches_the_service(context, monkeypatch):
    from lobster.services.analysis.enhanced_singlecell_service import (
        EnhancedSingleCellService,
    )

    dm, tools = context
    dm.store_modality("sc", single_cell())
    rec = Recorder(monkeypatch, EnhancedSingleCellService, "detect_doublets")
    reply = tools["detect_doublets"].invoke({"modality_name": "sc"})
    stats = rec.stats
    # Without scrublet installed the service takes its percentile fallback, and that is the
    # path this test runs; with scrublet installed the same assertions run on the scrublet
    # path (it may find 0 on synthetic data, so only the fallback carries the "> 0" check).
    if stats["detection_method"] != "scrublet":
        # the fallback cuts at a percentile, so it always finds some; a zero here
        # would make the test vacuous
        assert stats["n_doublets_detected"] > 0
    assert (
        f"**Doublets detected**: {stats['n_doublets_detected']} / 300 cells "
        f"({stats['actual_doublet_rate'] * 100:.1f}%)" in reply
    )
    assert f"**Method**: {stats['detection_method']}" in reply
    shown = "auto" if stats["threshold"] is None else stats["threshold"]
    assert f"**Threshold**: {shown}" in reply
    summary = dm.get_modality("sc_doublets_detected")
    assert int(summary.obs["predicted_doublet"].sum()) == stats["n_doublets_detected"]


def test_compute_trajectory_reply_matches_the_service(context, monkeypatch):
    import scanpy as sc

    from lobster.services.analysis.enhanced_singlecell_service import (
        EnhancedSingleCellService,
    )

    dm, tools = context
    adata = single_cell()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.pca(adata, n_comps=15)
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata, key_added="leiden")
    dm.store_modality("sc", adata)
    rec = Recorder(monkeypatch, EnhancedSingleCellService, "compute_trajectory")
    reply = tools["compute_trajectory"].invoke({"modality_name": "sc"})
    stats = rec.stats
    low, high = stats["pseudotime_range"]
    assert high > low  # a [0, 0] range would make the test vacuous
    assert f"index {stats['root_cell_index']}" in reply
    assert f"[{low:.3f}, {high:.3f}]" in reply
    assert "automatic (lowest first diffusion component)" in reply

    explicit = tools["compute_trajectory"].invoke(
        {"modality_name": "sc", "root_cell": 0}
    )
    assert rec.stats["root_cell_index"] == 0  # index 0 still counts as given
    assert "explicit root cell" in explicit
    grouped = tools["compute_trajectory"].invoke(
        {"modality_name": "sc", "root_group": "0"}
    )
    assert "root group '0'" in grouped


def test_import_bulk_counts_records_the_detected_tool(context, tmp_path):
    dm, tools = context
    root = tmp_path / "salmon"
    for sample in ("s1", "s2"):
        (root / sample).mkdir(parents=True)
        pd.DataFrame(
            {
                "Name": [f"T{i}" for i in range(30)],
                "Length": 1000,
                "EffectiveLength": 900.0,
                "TPM": 10.0,
                "NumReads": 100.0,
            }
        ).to_csv(root / sample / "quant.sf", sep="\t", index=False)
    reply = tools["import_bulk_counts"].invoke(
        {"modality_name": "quant", "file_path": str(root), "source": "auto"}
    )
    assert "- Source format: salmon" in reply
    activity = [
        a for a in dm.provenance.activities if a["type"] == "import_bulk_counts"
    ]
    assert activity[-1]["parameters"]["detected_source"] == "salmon"


def test_assess_bulk_sample_quality_reply_matches_the_service(context, monkeypatch):
    from lobster.services.analysis.bulk_preprocessing_service import (
        BulkPreprocessingService,
    )

    dm, tools = context
    adata = bulk()
    adata.X[3, 60:280] = 0  # one sample with a different composition
    adata.X[3, 280:] *= 8
    dm.store_modality("bulk", adata)
    rec = Recorder(monkeypatch, BulkPreprocessingService, "assess_sample_quality")
    reply = tools["assess_bulk_sample_quality"].invoke({"modality_name": "bulk"})
    stats = rec.stats
    assert stats["n_outliers"] > 0  # an empty list would make the test vacuous
    assert f"- Samples assessed: {stats['n_samples']}" in reply
    assert f"- Outlier samples: {stats['n_outliers']}" in reply
    assert f"- Median pairwise correlation: {stats['median_correlation']:.3f}" in reply
    for name in stats["outlier_names"]:
        assert str(name) in reply.split("Outlier names:")[1]


def test_filter_bulk_genes_reply_matches_the_service(context, monkeypatch):
    from lobster.services.analysis.bulk_preprocessing_service import (
        BulkPreprocessingService,
    )

    dm, tools = context
    dm.store_modality("bulk", bulk())
    rec = Recorder(monkeypatch, BulkPreprocessingService, "filter_genes")
    reply = tools["filter_bulk_genes"].invoke({"modality_name": "bulk"})
    stats = rec.stats
    assert stats["n_filtered"] > 0
    assert f"- Genes before: {stats['n_genes_before']:,}" in reply
    assert f"- Genes after: {stats['n_genes_after']:,}" in reply
    assert f"- Genes removed: {stats['n_filtered']:,}" in reply


def test_normalize_bulk_counts_reply_matches_the_service(context, monkeypatch):
    from lobster.services.analysis.bulk_preprocessing_service import (
        BulkPreprocessingService,
    )

    dm, tools = context
    adata = bulk()
    adata.X[0] = adata.X[0] * 3  # unequal library sizes, so the factors differ
    adata.layers["counts"] = adata.X.copy()
    dm.store_modality("bulk", adata[:, 60:].copy())
    rec = Recorder(monkeypatch, BulkPreprocessingService, "normalize_counts")
    reply = tools["normalize_bulk_counts"].invoke(
        {"modality_name": "bulk", "method": "deseq2"}
    )
    stats = rec.stats
    low, high = stats["size_factor_range"]
    assert high > low
    assert f"- Mean size factor: {stats['mean_size_factor']:.3f}" in reply
    assert f"- Size factor range: [{low:.3f}, {high:.3f}]" in reply


def test_detect_batch_effects_reply_matches_the_service(context, monkeypatch):
    from lobster.services.analysis.bulk_preprocessing_service import (
        BulkPreprocessingService,
    )

    dm, tools = context
    adata = bulk()
    adata.X[adata.obs["batch"] == "b"] += 40  # a strong batch shift
    dm.store_modality("bulk", adata)
    rec = Recorder(monkeypatch, BulkPreprocessingService, "detect_batch_effects")
    reply = tools["detect_batch_effects"].invoke(
        {"modality_name": "bulk", "batch_key": "batch", "condition_key": "condition"}
    )
    stats = rec.stats
    assert max(stats["batch_variance_pcs"].values()) > 0.5
    for pc, value in stats["batch_variance_pcs"].items():
        assert f"{pc} {value:.3f}" in reply
    for pc, value in stats["condition_variance_pcs"].items():
        assert f"{pc} {value:.3f}" in reply
    assert stats["recommendation"] in reply
    # the old reply claimed "minimal batch effects" from a value that was always 0
    assert "Minimal batch effects" not in reply


def test_a_missing_stats_key_raises_instead_of_reading_zero(context, monkeypatch):
    from lobster.services.analysis.enhanced_singlecell_service import (
        EnhancedSingleCellService,
    )

    dm, tools = context
    dm.store_modality("sc", single_cell())
    original = EnhancedSingleCellService.detect_doublets

    def without_rate(self, *args, **kwargs):
        adata, stats, ir = original(self, *args, **kwargs)
        stats = {k: v for k, v in stats.items() if k != "actual_doublet_rate"}
        return adata, stats, ir

    monkeypatch.setattr(EnhancedSingleCellService, "detect_doublets", without_rate)
    reply = tools["detect_doublets"].invoke({"modality_name": "sc"})
    assert reply.startswith("Unexpected error") and "actual_doublet_rate" in reply
    assert "Doublets detected" not in reply
