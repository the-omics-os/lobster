"""A step that reads two or more modalities is not replayed, and says so.

A replay starts from one input, and no cell binds the modalities a
concatenation (``sample_adatas``) or a cross-platform concordance (``adata1``,
``adata2``) reads: their cells stopped every replay with a NameError. Both steps
change data, so they are marked as not replayable with an incomplete replay: a
replay export is refused, and a default export notes it under the step.
"""

from unittest.mock import MagicMock

import anndata as ad
import nbformat
import numpy as np
import pandas as pd
import pytest

from lobster.agents.proteomics.shared_tools import create_shared_tools
from lobster.core.notebooks.exporter import NotebookExporter
from lobster.core.provenance.analysis_ir import AnalysisStep
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.services.data_management.concatenation_service import (
    ConcatenationService,
)
from lobster.testing.provenance_replay import RecordingDataManager

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _no_host_dependency_scan(monkeypatch):
    monkeypatch.setattr(NotebookExporter, "_snapshot_dependencies", lambda self: {})


def _concat(n=3):
    service = ConcatenationService.__new__(ConcatenationService)
    return service._create_concatenation_ir(n, "inner", "batch", True)


def _concordance():
    rng = np.random.default_rng(0)
    samples = [f"S{i}" for i in range(6)]
    x = rng.normal(size=(6, 4))

    def modality(values):
        return ad.AnnData(
            values,
            obs=pd.DataFrame(index=samples),
            var=pd.DataFrame(index=[f"P{i}" for i in range(4)]),
        )

    dm = RecordingDataManager({"olink": modality(x), "soma": modality(x + 0.1)})
    tools = create_shared_tools(dm, MagicMock(), MagicMock(), MagicMock())
    tool = next(t for t in tools if t.name == "assess_cross_platform_concordance")
    tool.invoke({"modality_name_1": "olink", "modality_name_2": "soma"})
    return dm.last_ir("assess_cross_platform_concordance")


STEPS = {"concat": (_concat, 3), "concordance": (_concordance, 2)}


@pytest.mark.parametrize("step", list(STEPS))
def test_the_step_is_not_replayable_and_its_replay_is_incomplete(step):
    ir = STEPS[step][0]()

    assert ir.replayable is False
    assert ir.execution_context["replay_incomplete"] is True
    assert ir.execution_context["reads_modalities"] == STEPS[step][1]


def _plain():
    return AnalysisStep(
        operation="n",
        tool_name="n",
        description="n",
        library="custom",
        code_template="x = 1",
        imports=[],
        parameters={},
        parameter_schema={},
    )


def _session(tmp_path, ir):
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    dm.provenance.create_activity("normalize", "a", description="n", ir=_plain())
    dm.provenance.create_activity(ir.tool_name, "a", description="step", ir=ir)
    return dm


@pytest.mark.parametrize("step", list(STEPS))
def test_a_replay_export_is_refused(tmp_path, step):
    dm = _session(tmp_path, STEPS[step][0]())

    with pytest.raises(ValueError, match="changed data but cannot be replayed"):
        dm.notebook_exporter.export("nb", filter_strategy="replay")


@pytest.mark.parametrize("step", list(STEPS))
def test_a_default_export_notes_it_under_the_step(tmp_path, step):
    build, n = STEPS[step]
    dm = _session(tmp_path, build())

    path = dm.notebook_exporter.export("nb", filter_strategy="successful")

    docs = [
        c.source
        for c in nbformat.read(path, as_version=4).cells
        if c.source.startswith("## Step ")
    ]
    note = (
        f"*Not replayed: this step reads {n} modalities, and a replay starts "
        "from one input.*"
    )
    assert note not in docs[0]
    assert docs[1].endswith(f"\n\n{note}"), docs[1]
