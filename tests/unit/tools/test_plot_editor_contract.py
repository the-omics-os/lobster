"""AQUADIF contracts for the actual editor tools returned to agents."""

from unittest.mock import MagicMock

import pytest

from lobster.config.aquadif import requires_provenance
from lobster.testing.contract_mixins import AgentContractTestMixin
from lobster.tools.plot_editor_tool import (
    create_figure_edit_tools,
    create_plot_editor_tools,
)


@pytest.fixture
def editor_tools():
    manager = MagicMock()
    return create_plot_editor_tools(manager) + create_figure_edit_tools(manager)


class _EditorContract(AgentContractTestMixin):
    agent_module = "lobster.tools.plot_editor_tool"
    factory_name = "create_plot_editor_tools/create_figure_edit_tools"

    def __init__(self, tools):
        self.tools = tools

    def _get_tools_from_factory(self):
        return self.tools


@pytest.mark.contract
@pytest.mark.parametrize(
    "check",
    [
        "test_tools_have_aquadif_metadata",
        "test_categories_are_valid",
        "test_categories_capped_at_three",
        "test_provenance_tools_have_flag",
        "test_metadata_objects_are_unique",
        "test_provenance_categories_not_buried",
        "test_provenance_ast_validation",
        "test_tool_params_no_bare_collections",
    ],
)
def test_editor_tools_pass_shared_contracts(editor_tools, check):
    getattr(_EditorContract(editor_tools), check)()


@pytest.mark.contract
def test_editor_tools_are_utilities_without_analysis_ir(editor_tools):
    assert {tool.name for tool in editor_tools} == {
        "list_editable_figures",
        "open_plot_editor",
        "inspect_saved_figure",
        "edit_saved_figure",
    }
    for tool in editor_tools:
        assert tool.metadata == {"categories": ["UTILITY"], "provenance": False}
        assert tool.tags == ["UTILITY"]
        assert requires_provenance(tool.metadata["categories"][0]) is False


@pytest.mark.contract
def test_edit_schema_accepts_nested_visual_updates(editor_tools):
    edit = next(tool for tool in editor_tools if tool.name == "edit_saved_figure")
    arguments = {
        "figure_ref": "f_aaaaaaaaaa@r_aaaaaaaaaa",
        "generation": 1,
        "request_id": "request" * 4,
        "layout_updates": {"xaxis": {"title": {"text": "Effect"}}},
        "trace_updates": [{"index": 0, "properties": {"marker": {"size": 12}}}],
        "explanation": "Label the x axis and enlarge points",
    }
    assert edit.args_schema.model_validate(arguments).model_dump() == arguments
