"""The proteomics QC cell runs every check the tool ran; no cell prints a free name.

``assess_proteomics_quality`` runs four checks (missing values, coefficient of
variation, contaminants, dynamic range) and stores the result of the last; its
cell ran only the first. Five templates printed ``{method}``, a name the cell
never sets: the notebook's own ``method`` (another step's) was printed instead.
"""

import ast
import inspect

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.agents.proteomics import proteomics_expert as pe
from lobster.core.provenance.analysis_ir import AnalysisStep
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.services.analysis import proteomics_analysis_service
from lobster.services.quality import proteomics_preprocessing_service
from lobster.testing.provenance_replay import factory_tools

pytestmark = pytest.mark.unit


def test_the_quality_cell_runs_every_check_the_tool_ran(tmp_path):
    rng = np.random.default_rng(5)
    X = rng.lognormal(10, 1, size=(12, 40))
    X[rng.random(X.shape) < 0.1] = np.nan
    source = tmp_path / "pr.h5ad"
    ad.AnnData(
        X=X.astype(np.float32),
        obs=pd.DataFrame(index=[f"S{i}" for i in range(12)]),
        var=pd.DataFrame(index=[f"P{i}" for i in range(40)]),
    ).write_h5ad(source)
    dm = DataManagerV2(workspace_path=tmp_path / "ws", auto_scan=False)
    dm.load_modality("pr", str(source), adapter="transcriptomics_bulk")
    tools = factory_tools(pe.proteomics_expert, pe.__name__, dm)
    tools["assess_proteomics_quality"].invoke(
        {"modality_name": "pr", "platform_type": "mass_spec"}
    )
    activity = dm.provenance.activities[-1]
    assert activity["type"] == "assess_proteomics_quality"
    code = AnalysisStep.from_dict(activity["ir"]).render()
    for check in (
        "assess_missing_value_patterns(",
        "assess_coefficient_variation(",
        "detect_contaminants(",
        "evaluate_dynamic_range(",
    ):
        assert check in code, code
    assert "cv_threshold=0.5" in code  # the mass-spec threshold, as a fraction


@pytest.mark.parametrize(
    "module", [proteomics_preprocessing_service, proteomics_analysis_service]
)
def test_no_template_prints_the_free_name_method(module):
    tree = ast.parse(inspect.getsource(module))
    templates = [
        node.value
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and getattr(call.func, "id", getattr(call.func, "attr", "")) == "AnalysisStep"
        for node in call.keywords
        if node.arg == "code_template" and isinstance(node.value, ast.Constant)
    ]
    assert templates
    assert not [t for t in templates if "{method}" in t.value]
