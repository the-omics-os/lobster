"""Rendering a cell's template never reaches Python objects outside its values.

A template is Python source written by Lobster, but text from a tool call or a
file name could reach it through an f-string; a Jinja expression in that text
ran in Lobster's own process, which holds the provider keys, when the notebook
was exported. The templates render in a sandbox: an expression that reaches for
an object's internals stops the render instead.
"""

import os
import time

import numpy as np
import pytest
from jinja2.sandbox import SecurityError

from lobster.core.provenance.analysis_ir import _JINJA_ENV, AnalysisStep


def _step(template, **parameters):
    return AnalysisStep(
        operation="test.step",
        tool_name="test_step",
        description="test",
        library="test",
        code_template=template,
        imports=[],
        parameters=parameters,
        parameter_schema={},
    )


@pytest.mark.parametrize(
    "expression",
    [
        "cycler.__init__.__globals__.os.getpid()",
        "lipsum.__globals__['os'].getpid()",
        "value.__class__.__mro__",
    ],
)
def test_an_expression_reaching_for_internals_stops_the_render(expression):
    step = _step("pid = {{ " + expression + " }}", value="x")

    with pytest.raises(ValueError, match="Template rendering failed") as stopped:
        print(step.render())  # the process id, when the expression ran

    assert str(os.getpid()) not in str(stopped.value)


def test_values_filters_and_branches_render_as_before():
    step = _step(
        "x = {{ value | repr }}\n"
        "{% for k, v in table.items() %}{{ k }}={{ v | repr }}\n{% endfor %}"
        "{% if flag %}y = {{ path | string | repr }}{% endif %}",
        value="a'b",
        table={"n": 1.5},
        flag=True,
        path="exports/x.csv",
    )

    assert step.render() == "x = \"a'b\"\nn=1.5\ny = 'exports/x.csv'"


@pytest.mark.parametrize(
    "expression",
    [
        "10**(10**10)",
        "10**10**10**10",  # Jinja reads ** left to right: (10**10)**10 is 10**100
        "2**10000",
        "2**big",
        "'a' * 10**9",
        "'a' * 10**6",
        "[1] * 10**9",
        "10**9 * [1]",
        "(10**60) * (10**60)",
    ],
)
def test_an_oversized_power_or_product_stops_before_it_is_computed(expression):
    started = time.monotonic()

    with pytest.raises(SecurityError):
        _JINJA_ENV.from_string("{{ " + expression + " }}").render(big=10**400)

    assert time.monotonic() - started < 1.0


def test_an_oversized_power_stops_the_cell_render():
    step = _step("x = {{ base ** exponent }}", base=10, exponent=10**10)

    with pytest.raises(ValueError, match="Template rendering failed"):
        step.render()


@pytest.mark.parametrize(
    "expression, rendered",
    [
        ("2**8", "256"),
        ("2**-1", "0.5"),
        ("1.5**2", "2.25"),
        ("10**100 == 10**100", "True"),
        ("3 * 4", "12"),
        ("'ab' * 3", "ababab"),
        ("2 * [0]", "[0, 0]"),
        ("'=' * 60", "=" * 60),
        ("0.5 * 3", "1.5"),
    ],
)
def test_small_powers_and_products_render_as_before(expression, rendered):
    assert _JINJA_ENV.from_string("{{ " + expression + " }}").render() == rendered


@pytest.mark.parametrize(
    "expression",
    ["'a' * n", "n * 'a'", "[1] * n", "n * [1]", "(1,) * n"],
)
@pytest.mark.parametrize("n", [np.int64(10**6), np.int32(10**6), np.uint64(10**6)])
def test_a_numpy_integer_count_is_held_to_the_same_limit(expression, n):
    started = time.monotonic()

    with pytest.raises(SecurityError):
        _JINJA_ENV.from_string("{{ " + expression + " }}").render(n=n)

    assert time.monotonic() - started < 1.0


@pytest.mark.parametrize(
    "expression, rendered",
    [("'ab' * n", "ababab"), ("n * [0]", "[0, 0, 0]"), ("n * 4", "12")],
)
def test_a_small_numpy_integer_count_renders_as_before(expression, rendered):
    out = _JINJA_ENV.from_string("{{ " + expression + " }}").render(n=np.int64(3))

    assert out == rendered
