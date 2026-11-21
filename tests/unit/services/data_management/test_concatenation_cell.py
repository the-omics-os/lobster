"""The notebook cell of a sample concatenation runs and keeps sample names text.

The cell wrote ``join=inner``, a name the notebook does not define, so every
replay of a concatenation stopped with a NameError. The sample names, which are
modality names the model chooses, were put into the template with ``str()``:
Jinja then evaluated ``{{ ... }}`` inside a name, and its output could close the
quote and add code to the cell. The cell now takes both from its parameters.
"""

import ast
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.services.data_management.concatenation_service import (
    ConcatenationService,
)

pytestmark = pytest.mark.unit

HOSTILE = """{{ "', __import__('os').getpid(), '" }}"""


def _samples():
    out = []
    for i, genes in enumerate((["g0", "g1", "g2"], ["g1", "g2", "g3"])):
        x = np.arange(6, dtype=float).reshape(2, 3) + 10 * i
        obs = pd.DataFrame(index=[f"c{i}_{j}" for j in range(2)])
        out.append(ad.AnnData(x, obs=obs, var=pd.DataFrame(index=genes)))
    return out


def _service(tmp_path):
    return ConcatenationService(SimpleNamespace(cache_dir=tmp_path, console=None))


def _cell(sample_ids, join_type="inner"):
    ir = ConcatenationService.__new__(ConcatenationService)._create_concatenation_ir(
        n_samples=2,
        join_type=join_type,
        batch_key="batch",
        use_intersecting_genes_only=join_type == "inner",
        sample_ids=sample_ids,
    )
    return ir.render()


def _keys(cell):
    call = next(
        n
        for n in ast.walk(ast.parse(cell))
        if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "concat"
    )
    return next(k.value for k in call.keywords if k.arg == "keys")


@pytest.mark.parametrize("join_type", ["inner", "outer"])
@pytest.mark.parametrize("sample_ids", [["GSM1", "GSM2"], None])
def test_the_cell_runs_and_gives_the_session_result(tmp_path, join_type, sample_ids):
    samples = _samples()
    kwargs = {"use_intersecting_genes_only": join_type == "inner"}
    if sample_ids is not None:
        kwargs["sample_ids"] = sample_ids
    session, _, ir = _service(tmp_path).concatenate_samples(
        samples, batch_key="batch", **kwargs
    )

    namespace = {"ad": ad, "sample_adatas": samples}
    exec(ir.render(), namespace)
    cell = namespace["concatenated_adata"]

    assert list(cell.obs_names) == list(session.obs_names)
    assert list(cell.var_names) == list(session.var_names)
    assert list(cell.obs["batch"]) == list(session.obs["batch"])
    np.testing.assert_array_equal(np.asarray(cell.X), np.asarray(session.X))


@pytest.mark.parametrize("name", [HOSTILE, "{{ 7 * 7 }}", 'it\'s "GSM1"\nx'])
def test_a_sample_name_stays_text_in_the_cell(name):
    cell = _cell([name, "b"])

    assert ast.literal_eval(_keys(cell)) == [name, "b"]
    assert "__import__" not in {
        n.id for n in ast.walk(ast.parse(cell)) if isinstance(n, ast.Name)
    }
