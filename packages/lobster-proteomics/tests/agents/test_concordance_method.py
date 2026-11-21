"""Cross-platform concordance takes only spearman or pearson; its cell names the test.

``method`` was not checked and the notebook cell rendered it raw, as
``stats.{{ method }}r(...)``: any text in it became code when the notebook ran.
Another method is now refused before anything is stored or logged, and the cell
calls ``spearmanr`` or ``pearsonr``, never the given text.
"""

from unittest.mock import MagicMock

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.agents.proteomics.shared_tools import create_shared_tools
from lobster.core.notebooks.written_templates import outdated_tool, template_hash
from lobster.testing.provenance_replay import RecordingDataManager

INJECTED = "pearson(v1, v2) if __import__('os').getcwd() else stats.pearson"


def _session():
    rng = np.random.default_rng(0)
    samples = [f"S{i}" for i in range(6)]
    proteins = [f"P{i}" for i in range(4)]
    x = rng.normal(size=(6, 4))

    def modality(values):
        return ad.AnnData(
            values,
            obs=pd.DataFrame(index=samples),
            var=pd.DataFrame(index=proteins),
        )

    dm = RecordingDataManager(
        {"olink": modality(x), "soma": modality(x + rng.normal(0, 0.1, x.shape))}
    )
    tools = create_shared_tools(dm, MagicMock(), MagicMock(), MagicMock())
    tool = next(t for t in tools if t.name == "assess_cross_platform_concordance")
    return dm, tool


@pytest.mark.parametrize("method", [INJECTED, "kendall", "Spearman"])
def test_another_method_is_refused_before_anything_is_stored(method):
    dm, tool = _session()
    before = dm.modalities["olink"]

    reply = tool.invoke(
        {"modality_name_1": "olink", "modality_name_2": "soma", "method": method}
    )

    assert reply == f"Unknown method {method!r}: use 'spearman' or 'pearson'.", reply
    assert dm.calls == []
    assert dm.modalities["olink"] is before


@pytest.mark.parametrize("method", ["spearman", "pearson"])
def test_the_cell_calls_the_test_of_its_method(method):
    dm, tool = _session()
    tool.invoke(
        {"modality_name_1": "olink", "modality_name_2": "soma", "method": method}
    )
    ir = dm.last_ir("assess_cross_platform_concordance")

    assert f"stats.{method}r(v1[valid], v2[valid])" in ir.render()
    assert "getcwd" not in ir.render(method=INJECTED)


@pytest.mark.parametrize("method", ["spearman", "pearson"])
def test_the_template_is_the_one_the_exporter_knows(method):
    dm, tool = _session()
    tool.invoke(
        {"modality_name_1": "olink", "modality_name_2": "soma", "method": method}
    )
    ir = dm.last_ir("assess_cross_platform_concordance")

    assert outdated_tool(ir) is None, (ir.operation, template_hash(ir.code_template))
