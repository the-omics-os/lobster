"""``| tojson`` in IR templates writes Python literals, not JSON.

Exported notebook code is Python. JSON spells True, None and NaN as ``true``,
``null`` and ``NaN``, which are undefined names in a notebook cell; jinja's
built-in filter also writes NumPy integers as a render error and escapes
``< > & '``. Every IR template renders through one environment, where
``tojson`` now writes the same literal as ``repr``.
"""

import ast
import math

import numpy as np
import pytest

from lobster.core.provenance.analysis_ir import AnalysisStep, _python_literal

HOSTILE = "a<b>&c'd\"e\n{f}\\"


def _rendered(value):
    ir = AnalysisStep(
        operation="t",
        tool_name="t",
        description="t",
        library="t",
        code_template="x = {{ x | tojson }}",
        imports=[],
        parameters={"x": value},
        parameter_schema={},
    )
    code = ir.render()
    scope = {}
    exec(compile(code, "<ir>", "exec"), scope)  # nosec B102
    return code, scope["x"]


@pytest.mark.parametrize(
    "value",
    [True, False, None, 0, 1.5, HOSTILE, [HOSTILE, None], {"k": True, 2: None}],
    ids=repr,
)
def test_tojson_writes_a_python_literal(value):
    code, value_back = _rendered(value)

    assert code == "x = " + _python_literal(value)
    assert value_back == value
    assert type(value_back) is type(value)


def test_tojson_writes_nan_and_numpy_values_that_run():
    _, value_back = _rendered([float("nan"), np.int64(3), np.float64(0.5)])

    assert math.isnan(value_back[0])
    assert value_back[1:] == [3, 0.5]


def test_tojson_keeps_html_characters_as_they_are():
    code, _ = _rendered(HOSTILE)

    assert "<b>&c" in code
    assert "\\u00" not in code
    ast.parse(code)
