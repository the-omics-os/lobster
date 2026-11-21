"""Rendering a cell's template never reaches Python objects outside its values.

A template is Python source written by Lobster, but text from a tool call or a
file name could reach it through an f-string; a Jinja expression in that text
ran in Lobster's own process, which holds the provider keys, when the notebook
was exported. The templates render in a sandbox: an expression that reaches for
an object's internals stops the render instead.
"""

import os

import pytest

from lobster.core.provenance.analysis_ir import AnalysisStep


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
