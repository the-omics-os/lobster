"""Provenance source discovery must inspect real sync or async implementations."""

import unittest
from types import SimpleNamespace

from langchain_core.tools import StructuredTool

from lobster.testing.contract_mixins import AgentContractTestMixin


def sync_logged():
    ir = object()
    data_manager.log_tool_usage(tool_name="example", ir=ir)


async def async_logged():
    ir = object()
    data_manager.log_tool_usage(tool_name="example", ir=ir)


def sync_missing_ir():
    data_manager.log_tool_usage(tool_name="example")


async def async_missing_ir():
    data_manager.log_tool_usage(tool_name="example")


def synthetic_sync_wrapper():
    return "delegates to async implementation"


async def async_no_logging():
    return "no provenance"


class TestContractCallableSelection(unittest.TestCase):
    def validate(self, func, coroutine=None, with_coroutine_attribute=True):
        attributes = {
            "name": "example",
            "func": func,
            "metadata": {"categories": ["ANALYZE"], "provenance": True},
        }
        if with_coroutine_attribute:
            attributes["coroutine"] = coroutine
        tool = SimpleNamespace(**attributes)
        self.validate_tool(tool)

    def validate_tool(self, tool):
        contract = AgentContractTestMixin()
        contract.agent_module = "synthetic_agent"
        contract._require_tools = lambda: [tool]
        contract.test_provenance_ast_validation()

    def test_sync_logging_detected_without_coroutine_attribute(self):
        self.validate(sync_logged, with_coroutine_attribute=False)

    def test_sync_logging_detected_with_none_coroutine(self):
        self.validate(sync_logged)

    def test_async_logging_detected_behind_sync_wrapper(self):
        self.validate(synthetic_sync_wrapper, async_logged)

    def test_actual_structured_tool_async_logging_detected(self):
        tool = StructuredTool.from_function(
            func=synthetic_sync_wrapper,
            coroutine=async_logged,
            name="example",
            description="Synthetic source-discovery fixture",
        )
        tool.metadata = {"categories": ["ANALYZE"], "provenance": True}
        self.validate_tool(tool)

    def test_actual_structured_tool_sync_logging_detected(self):
        tool = StructuredTool.from_function(
            func=sync_logged,
            name="example",
            description="Synthetic source-discovery fixture",
        )
        tool.metadata = {"categories": ["ANALYZE"], "provenance": True}
        self.validate_tool(tool)

    def test_missing_ir_rejected_for_sync_and_async(self):
        for func, coroutine in (
            (sync_missing_ir, None),
            (synthetic_sync_wrapper, async_missing_ir),
        ):
            with (
                self.subTest(coroutine=coroutine),
                self.assertRaisesRegex(AssertionError, "does NOT call log_tool_usage"),
            ):
                self.validate(func, coroutine)

    def test_no_logging_rejected_for_sync_and_async(self):
        for func, coroutine in (
            (synthetic_sync_wrapper, None),
            (synthetic_sync_wrapper, async_no_logging),
        ):
            with (
                self.subTest(coroutine=coroutine),
                self.assertRaisesRegex(AssertionError, "does NOT call log_tool_usage"),
            ):
                self.validate(func, coroutine)

    def test_async_invalid_cannot_hide_behind_valid_sync_callable(self):
        with self.assertRaisesRegex(AssertionError, "does NOT call log_tool_usage"):
            self.validate(sync_logged, async_no_logging)

    def test_parent_missing_minimum_categories_still_fails(self):
        contract = AgentContractTestMixin()
        contract.factory_name = "synthetic_parent"
        contract.is_parent_agent = True
        contract._require_tools = lambda: [
            SimpleNamespace(metadata={"categories": ["UTILITY"]})
        ]
        with self.assertRaisesRegex(
            AssertionError, "missing minimum viable categories"
        ):
            contract.test_minimum_viable_parent()

    def test_nonparent_aggregate_reaches_later_failing_provenance_check(self):
        contract = AgentContractTestMixin()
        contract.is_parent_agent = False
        # Earlier checks are irrelevant to this aggregate-control fixture.
        for name in (
            "test_factory_has_standard_params",
            "test_no_deprecated_handoff_tools",
            "test_agent_config_exists",
            "test_agent_config_has_name",
            "test_agent_config_has_tier_requirement",
            "test_tools_have_aquadif_metadata",
            "test_categories_are_valid",
            "test_categories_capped_at_three",
            "test_provenance_tools_have_flag",
            "test_metadata_objects_are_unique",
            "test_provenance_categories_not_buried",
        ):
            setattr(contract, name, lambda: None)

        reached = []

        def later_failure():
            reached.append(True)
            raise AssertionError("later provenance check executed")

        contract.test_provenance_ast_validation = later_failure
        try:
            with self.assertRaisesRegex(
                AssertionError, "later provenance check executed"
            ):
                contract.test_all_contract_requirements()
        finally:
            # A skip must fail this regression, not silently skip the regression.
            self.assertTrue(reached, "non-parent aggregate aborted before later checks")


if __name__ == "__main__":
    unittest.main()
