"""
Contract test mixins for validating plugin API compliance.

This module provides test mixins that agent package authors can inherit
to validate their packages follow the plugin contract established in Phase 2.

Usage:
    from lobster.testing import AgentContractTestMixin

    class TestMyAgent(AgentContractTestMixin):
        agent_module = 'my_package.agents.my_agent'
        factory_name = 'my_agent'

    # Then run with pytest
"""

import ast
import importlib
import inspect
import textwrap
from typing import Any, ClassVar, Optional, Set
from unittest.mock import MagicMock


class AgentContractTestMixin:
    """
    Test mixin for validating agent package plugin API compliance.

    Subclasses must define:
        agent_module: str - Module path containing AGENT_CONFIG (entry point module)
        factory_name: str - Name of the factory function

    Optional:
        factory_module: str - Module containing the factory (defaults to agent_module)
        expected_tier: str - Expected tier requirement (default: None, skips check)
        tools_required: bool - If True (default), empty tools = FAIL. If False, empty tools = skip.

    Example (same module for config and factory):
        class TestAnnotationExpert(AgentContractTestMixin):
            agent_module = 'lobster_transcriptomics.annotation_expert'
            factory_name = 'annotation_expert'

    Example (separate modules - package __init__ has config, nested module has factory):
        class TestTranscriptomicsExpert(AgentContractTestMixin):
            agent_module = 'lobster_transcriptomics'  # AGENT_CONFIG in __init__.py
            factory_module = 'lobster_transcriptomics.transcriptomics_expert'  # factory here
            factory_name = 'transcriptomics_expert'
    """

    # Required attributes - subclasses must define these
    agent_module: ClassVar[str]
    factory_name: ClassVar[str]

    # Optional attributes
    factory_module: ClassVar[Optional[str]] = (
        None  # Defaults to agent_module if not set
    )
    expected_tier: ClassVar[Optional[str]] = None
    is_parent_agent: ClassVar[bool] = (
        False  # Set True for parent agents with child agents
    )
    tools_required: ClassVar[bool] = True  # Set False for agents still being scaffolded

    # Standard factory parameters as defined in Phase 2 plugin contract
    STANDARD_PARAMS: ClassVar[Set[str]] = {
        "data_manager",
        "callback_handler",
        "agent_name",
        "delegation_tools",
        "workspace_path",
    }

    # Deprecated parameters that should not be used
    DEPRECATED_PARAMS: ClassVar[Set[str]] = {
        "handoff_tools",  # Renamed to delegation_tools in Phase 2
    }

    # Required AGENT_CONFIG fields
    REQUIRED_CONFIG_FIELDS: ClassVar[Set[str]] = {
        "name",
        "tier_requirement",
    }

    # Class-level cache for tool extraction (populated once per test class)
    _tools_cache: ClassVar[dict] = {}

    def _get_module(self) -> Any:
        """Import and return the config module (contains AGENT_CONFIG)."""
        return importlib.import_module(self.agent_module)

    def _get_factory_module(self) -> Any:
        """Import and return the factory module (contains factory function)."""
        module_path = self.factory_module if self.factory_module else self.agent_module
        return importlib.import_module(module_path)

    def _get_factory(self) -> Any:
        """Get the factory function from the factory module."""
        module = self._get_factory_module()
        return getattr(module, self.factory_name)

    def _get_factory_params(self) -> Set[str]:
        """Get the set of parameter names from the factory signature."""
        factory = self._get_factory()
        sig = inspect.signature(factory)
        return set(sig.parameters.keys())

    def _get_agent_config(self) -> Any:
        """Get the AGENT_CONFIG from the module."""
        module = self._get_module()
        return getattr(module, "AGENT_CONFIG", None)

    def _get_tools_from_factory(self) -> list:
        """
        Get list of tool objects from agent factory.

        Calls the factory with MagicMock dependencies and extracts
        tool objects. Raises on unexpected failures (fail-by-default).

        Known-benign failures (ImportError for optional deps like torch/scvi)
        raise ImportError which callers handle via pytest.skip().

        Results are cached per class to avoid redundant graph compilation.

        Returns:
            List of tool objects from the factory's compiled graph

        Raises:
            ImportError: When factory needs optional dependencies not installed
            Exception: All other errors propagate (fail-by-default)
        """
        import pytest

        cache_key = f"{self.agent_module}:{self.factory_name}"
        if cache_key in self.__class__._tools_cache:
            return self.__class__._tools_cache[cache_key]

        try:
            from unittest.mock import patch

            factory = self._get_factory()

            # Create mock dependencies for factory call
            mock_data_manager = MagicMock()
            mock_callback_handler = MagicMock()
            mock_delegation_tools = []

            # Patch LLM creation — factories need LLMs but contract tests
            # don't execute tools, they only inspect metadata.
            # Patch at both the definition site and the factory's module to
            # handle both `from X import create_llm` and `X.create_llm()`.
            factory_module_path = (
                self.factory_module if self.factory_module else self.agent_module
            )
            with (
                patch(
                    "lobster.config.llm_factory.create_llm",
                    return_value=MagicMock(),
                ),
                patch(
                    f"{factory_module_path}.create_llm",
                    return_value=MagicMock(),
                ),
            ):
                # Call factory to get CompiledStateGraph
                graph = factory(
                    data_manager=mock_data_manager,
                    callback_handler=mock_callback_handler,
                    agent_name="test",
                    delegation_tools=mock_delegation_tools,
                    workspace_path=None,
                )

            # Extract tools from the graph's tool node
            # LangGraph wraps ToolNode in PregelNode — check both levels
            tools = []
            if hasattr(graph, "nodes") and "tools" in graph.nodes:
                tools_node = graph.nodes["tools"]
                if hasattr(tools_node, "tools_by_name"):
                    tools = list(tools_node.tools_by_name.values())
                elif hasattr(tools_node, "bound") and hasattr(
                    tools_node.bound, "tools_by_name"
                ):
                    tools = list(tools_node.bound.tools_by_name.values())

            self.__class__._tools_cache[cache_key] = tools
            return tools

        except (ImportError, ModuleNotFoundError) as e:
            # Known-benign: optional heavy deps (torch, scvi-tools) not installed
            pytest.skip(
                f"Factory '{self.factory_name}' requires unavailable dependency: {e}"
            )

    def _has_log_tool_usage_with_ir(self, tree: ast.AST) -> bool:
        """
        Check if AST tree contains log_tool_usage call with ir= keyword argument.

        Delegates to the standalone ``has_provenance_call`` function in
        ``lobster.config.aquadif`` for reusability by ``lobster validate-plugin``.

        Args:
            tree: AST tree from parsed tool source code

        Returns:
            True if log_tool_usage(ir=ir) call found, False otherwise
        """
        from lobster.config.aquadif import has_provenance_call

        return has_provenance_call(tree)

    def _require_tools(self) -> list:
        """Get tools, failing or skipping based on tools_required setting."""
        import pytest

        tools = self._get_tools_from_factory()

        if not tools:
            if self.tools_required:
                raise AssertionError(
                    f"Factory '{self.factory_name}' returned no tools — "
                    f"check graph construction in '{self.agent_module}'. "
                    f"Set tools_required = False if this agent has no tools yet."
                )
            else:
                pytest.skip(
                    f"No tools from factory '{self.factory_name}' (tools_required=False)."
                )

        return tools

    def test_factory_has_standard_params(self) -> None:
        """
        Verify factory function has the standard parameters.

        The plugin contract requires these parameters:
        - data_manager: DataManagerV2 instance
        - callback_handler: Optional callback handler for LLM interactions
        - agent_name: Name identifier for the agent instance
        - delegation_tools: List of tools for delegating to child agents
        - workspace_path: Optional path to workspace directory
        """
        params = self._get_factory_params()
        missing = self.STANDARD_PARAMS - params

        assert not missing, (
            f"Factory '{self.factory_name}' in '{self.agent_module}' "
            f"is missing required parameters: {sorted(missing)}. "
            f"Factory must accept: {sorted(self.STANDARD_PARAMS)}"
        )

    def test_no_deprecated_handoff_tools(self) -> None:
        """
        Verify factory does NOT use deprecated handoff_tools parameter.

        The handoff_tools parameter was renamed to delegation_tools in Phase 2
        for semantic clarity. Factories should use delegation_tools instead.
        """
        params = self._get_factory_params()
        deprecated_used = self.DEPRECATED_PARAMS & params

        assert not deprecated_used, (
            f"Factory '{self.factory_name}' in '{self.agent_module}' "
            f"uses deprecated parameters: {sorted(deprecated_used)}. "
            f"Use 'delegation_tools' instead of 'handoff_tools'."
        )

    def test_agent_config_exists(self) -> None:
        """
        Verify AGENT_CONFIG exists at module top level.

        AGENT_CONFIG must be defined at the top of the module (before heavy imports)
        to enable fast entry point discovery (<50ms target).
        """
        config = self._get_agent_config()

        assert config is not None, (
            f"Module '{self.agent_module}' does not have AGENT_CONFIG defined. "
            f"Add: AGENT_CONFIG = AgentRegistryConfig(...) at module top, "
            f"before heavy imports."
        )

    def test_agent_config_has_name(self) -> None:
        """
        Verify AGENT_CONFIG has required 'name' field.

        The name field uniquely identifies the agent in the registry
        and is used for routing, configuration, and logging.
        """
        config = self._get_agent_config()

        assert (
            config is not None
        ), f"Cannot check name field: AGENT_CONFIG not found in '{self.agent_module}'"

        assert hasattr(
            config, "name"
        ), f"AGENT_CONFIG in '{self.agent_module}' is missing 'name' field."

        assert config.name, f"AGENT_CONFIG.name in '{self.agent_module}' is empty."

    def test_agent_config_has_tier_requirement(self) -> None:
        """
        Verify AGENT_CONFIG has required 'tier_requirement' field.

        The tier_requirement field controls runtime access based on
        user's subscription level. Valid values: "free", "premium", "enterprise".
        """
        config = self._get_agent_config()

        assert (
            config is not None
        ), f"Cannot check tier_requirement: AGENT_CONFIG not found in '{self.agent_module}'"

        assert hasattr(
            config, "tier_requirement"
        ), f"AGENT_CONFIG in '{self.agent_module}' is missing 'tier_requirement' field."

        valid_tiers = {"free", "premium", "enterprise"}
        tier = config.tier_requirement

        assert tier in valid_tiers, (
            f"AGENT_CONFIG.tier_requirement in '{self.agent_module}' has invalid value '{tier}'. "
            f"Must be one of: {sorted(valid_tiers)}"
        )

        # Optional: check expected tier if specified
        if self.expected_tier is not None:
            assert (
                tier == self.expected_tier
            ), f"AGENT_CONFIG.tier_requirement is '{tier}', expected '{self.expected_tier}'"

    # =========================================================================
    # AQUADIF Contract Tests (Phase 2)
    # =========================================================================

    def test_tools_have_aquadif_metadata(self) -> None:
        """
        Verify all tools have AQUADIF metadata (categories and provenance).

        Each tool must declare:
        - .metadata dict with "categories" and "provenance" keys
        - categories: non-empty list of AQUADIF category strings
        - provenance: boolean indicating if provenance tracking is required

        This validates TEST-02: Metadata presence.
        """
        tools = self._require_tools()

        for tool in tools:
            tool_name = getattr(tool, "name", str(tool))

            # Check metadata dict exists
            assert hasattr(tool, "metadata") and tool.metadata is not None, (
                f"Tool '{tool_name}' in '{self.agent_module}' is missing .metadata. "
                f"Add: tool.metadata = {{'categories': [...], 'provenance': True/False}}"
            )

            # Check categories key exists
            assert "categories" in tool.metadata, (
                f"Tool '{tool_name}' in '{self.agent_module}' has .metadata but missing 'categories' key. "
                f"Add: 'categories': ['PRIMARY_CATEGORY']"
            )

            # Check categories is non-empty list
            categories = tool.metadata["categories"]
            assert isinstance(categories, list) and len(categories) > 0, (
                f"Tool '{tool_name}' in '{self.agent_module}' has empty or invalid categories. "
                f"Must be non-empty list: ['CATEGORY']"
            )

            # Check provenance key exists
            assert "provenance" in tool.metadata, (
                f"Tool '{tool_name}' in '{self.agent_module}' has .metadata but missing 'provenance' key. "
                f"Add: 'provenance': True or False"
            )

    def test_categories_are_valid(self) -> None:
        """
        Verify all tool categories are valid AQUADIF categories.

        Categories must be from the 10-category AQUADIF taxonomy:
        IMPORT, QUALITY, FILTER, PREPROCESS, ANALYZE, ANNOTATE,
        DELEGATE, SYNTHESIZE, UTILITY, CODE_EXEC

        This validates TEST-03: Category validity.
        """
        from lobster.config.aquadif import AquadifCategory

        tools = self._require_tools()

        invalid_categories = []

        for tool in tools:
            tool_name = getattr(tool, "name", str(tool))
            categories = tool.metadata.get("categories", [])

            for category in categories:
                try:
                    # Validate by attempting to construct enum member
                    AquadifCategory(category)
                except ValueError:
                    invalid_categories.append((tool_name, category))

        assert not invalid_categories, (
            f"Invalid AQUADIF categories found in '{self.agent_module}':\n"
            + "\n".join(
                f"  - Tool '{tool}': invalid category '{cat}'"
                for tool, cat in invalid_categories
            )
            + f"\n\nValid categories: {', '.join(c.value for c in AquadifCategory)}"
        )

    def test_categories_capped_at_three(self) -> None:
        """
        Verify tools have at most 3 categories.

        Most tools need only 1 category. Multi-category is uncommon and should
        only be used when a tool has substantial functionality in multiple areas.

        This validates TEST-04: Category cap.
        """
        tools = self._require_tools()

        violations = []

        for tool in tools:
            tool_name = getattr(tool, "name", str(tool))
            categories = tool.metadata.get("categories", [])

            if len(categories) > 3:
                violations.append((tool_name, len(categories)))

        assert not violations, (
            f"Tools in '{self.agent_module}' exceed 3-category limit:\n"
            + "\n".join(
                f"  - Tool '{tool}': {count} categories (max is 3)"
                for tool, count in violations
            )
            + "\n\nGuidance: Most tools need 1 category; multi-category is uncommon."
        )

    def test_provenance_tools_have_flag(self) -> None:
        """
        Verify tools with provenance-required categories declare provenance: True.

        Categories requiring provenance (7 of 10):
        IMPORT, QUALITY, FILTER, PREPROCESS, ANALYZE, ANNOTATE, SYNTHESIZE

        Categories NOT requiring provenance (3 of 10):
        DELEGATE, UTILITY, CODE_EXEC

        The primary category (first in list) determines the provenance requirement.

        This validates TEST-05: Provenance flag compliance.
        """
        from lobster.config.aquadif import PROVENANCE_REQUIRED, AquadifCategory

        tools = self._require_tools()

        violations = []

        for tool in tools:
            tool_name = getattr(tool, "name", str(tool))
            categories = tool.metadata.get("categories", [])

            if not categories:
                continue  # Already caught by metadata presence test

            # Get primary category (first in list)
            primary_category_str = categories[0]
            try:
                primary_category = AquadifCategory(primary_category_str)
            except ValueError:
                continue  # Already caught by category validity test

            # Check if primary category requires provenance
            requires_prov = primary_category in PROVENANCE_REQUIRED
            declared_prov = tool.metadata.get("provenance", False)

            if requires_prov and not declared_prov:
                violations.append((tool_name, primary_category.value))

        assert not violations, (
            f"Tools in '{self.agent_module}' have provenance-required categories but provenance=False:\n"
            + "\n".join(
                f"  - Tool '{tool}': primary category '{cat}' requires provenance=True"
                for tool, cat in violations
            )
        )

    def test_metadata_objects_are_unique(self) -> None:
        """
        Verify each tool has its own metadata dict (no shared objects).

        Tools must not share metadata dict references. Common cause:
        metadata dict created outside the tool creation loop in the factory.

        This validates TEST-07: Metadata uniqueness.
        """
        tools = self._require_tools()

        metadata_ids = [
            id(tool.metadata) for tool in tools if hasattr(tool, "metadata")
        ]

        # Count occurrences of each id
        from collections import Counter

        id_counts = Counter(metadata_ids)
        shared_ids = {id_val: count for id_val, count in id_counts.items() if count > 1}

        assert not shared_ids, (
            f"Tools in '{self.agent_module}' share metadata dict objects ({len(shared_ids)} shared dicts). "
            f"Each tool must have its own metadata dict. "
            f"Common cause: metadata dict created outside loop in factory. "
            f"Fix: Create metadata dict inside the tool creation loop."
        )

    def test_provenance_categories_not_buried(self) -> None:
        """
        Verify provenance-required categories are not buried behind non-provenance primary.

        If ANY category in the list requires provenance, the PRIMARY (first) category
        must also require provenance. This prevents gaming by reordering categories
        to dodge provenance validation.

        Example violation: categories: ["UTILITY", "IMPORT"] — IMPORT requires provenance
        but UTILITY (primary) does not, so provenance checks would be bypassed.

        This validates TEST-09: Category ordering integrity.
        """
        from lobster.config.aquadif import PROVENANCE_REQUIRED, AquadifCategory

        tools = self._require_tools()

        violations = []

        for tool in tools:
            tool_name = getattr(tool, "name", str(tool))
            categories = tool.metadata.get("categories", [])

            if len(categories) < 2:
                continue  # Single-category tools can't have ordering issues

            # Parse all categories, skip if any are invalid (caught by other test)
            parsed = []
            for cat_str in categories:
                try:
                    parsed.append(AquadifCategory(cat_str))
                except ValueError:
                    break
            else:
                primary = parsed[0]
                secondary_cats = parsed[1:]

                primary_requires = primary in PROVENANCE_REQUIRED
                any_secondary_requires = any(
                    c in PROVENANCE_REQUIRED for c in secondary_cats
                )

                if any_secondary_requires and not primary_requires:
                    buried = [
                        c.value for c in secondary_cats if c in PROVENANCE_REQUIRED
                    ]
                    violations.append((tool_name, primary.value, buried))

        assert not violations, (
            f"Tools in '{self.agent_module}' have provenance-required categories buried behind non-provenance primary:\n"
            + "\n".join(
                f"  - Tool '{tool}': primary '{primary}' doesn't require provenance, "
                f"but secondary {buried} does. Reorder so a provenance-required category is first."
                for tool, primary, buried in violations
            )
        )

    def test_minimum_viable_parent(self) -> None:
        """
        Verify parent agents have minimum viable category set.

        Parent agents (domain experts with child agents) must provide:
        - IMPORT: Ability to load data
        - QUALITY: Ability to assess data quality
        - ANALYZE or DELEGATE: Either perform analysis or delegate to children

        This validates TEST-06: Minimum viable parent agent capabilities.
        """
        import pytest

        # Skip if not a parent agent
        if not self.is_parent_agent:
            pytest.skip("Not a parent agent")

        tools = self._require_tools()

        # Collect all unique categories across all tools
        all_categories = set()
        for tool in tools:
            categories = tool.metadata.get("categories", [])
            all_categories.update(categories)

        # Check for minimum viable set
        has_import = "IMPORT" in all_categories
        has_quality = "QUALITY" in all_categories
        has_analyze_or_delegate = (
            "ANALYZE" in all_categories or "DELEGATE" in all_categories
        )

        missing = []
        if not has_import:
            missing.append("IMPORT")
        if not has_quality:
            missing.append("QUALITY")
        if not has_analyze_or_delegate:
            missing.append("ANALYZE or DELEGATE")

        assert not missing, (
            f"Parent agent '{self.factory_name}' missing minimum viable categories. "
            f"Has: {sorted(all_categories)}. "
            f"Missing: {', '.join(missing)}. "
            f"Parent agents need: IMPORT + QUALITY + (ANALYZE or DELEGATE)"
        )

    def test_provenance_ast_validation(self) -> None:
        """
        Verify tools with provenance metadata call log_tool_usage(ir=ir).

        This is the CRITICAL test addressing the Phase 1 eval finding that
        100% of agents copied provenance boilerplate mechanically without
        adjusting based on metadata flags. This test catches metadata-runtime
        disconnect at test time by parsing tool source code.

        Rules:
        - If tool declares provenance=True: MUST call log_tool_usage(ir=ir)
        - If primary category requires provenance: tool SHOULD declare provenance=True
          (caught by test_provenance_tools_have_flag)

        This validates TEST-08: Provenance AST validation.
        """
        from lobster.config.aquadif import PROVENANCE_REQUIRED, AquadifCategory

        tools = self._require_tools()

        violations = []

        for tool in tools:
            tool_name = getattr(tool, "name", str(tool))
            categories = tool.metadata.get("categories", [])

            if not categories:
                continue  # Already caught by metadata presence test

            # Get primary category
            primary_category_str = categories[0]
            try:
                primary_category = AquadifCategory(primary_category_str)
            except ValueError:
                continue  # Already caught by category validity test

            # Check provenance requirements
            requires_prov = primary_category in PROVENANCE_REQUIRED
            declared_prov = tool.metadata.get("provenance", False)

            # Only validate if tool declares provenance OR category requires it
            if not (declared_prov or requires_prov):
                continue

            # Try to get tool source code
            try:
                source = inspect.getsource(tool.func)
                # Dedent in case source is indented (nested functions)
                source = textwrap.dedent(source)
            except (OSError, TypeError) as e:
                # Cannot get source (built-in, dynamically generated, etc.)
                import warnings

                warnings.warn(
                    f"Cannot validate provenance for tool '{tool_name}': "
                    f"source code not available ({e.__class__.__name__})"
                )
                continue

            # Parse source and check for log_tool_usage(ir=ir) call
            try:
                tree = ast.parse(source)
                has_ir_call = self._has_log_tool_usage_with_ir(tree)
            except SyntaxError as e:
                import warnings

                warnings.warn(f"Cannot parse source for tool '{tool_name}': {e}")
                continue

            # Validate: if provenance is declared, the call must exist
            if declared_prov and not has_ir_call:
                reason = "declares provenance=True"
                if requires_prov:
                    reason += f" (and primary category '{primary_category.value}' requires it)"
                reason += " but does NOT call log_tool_usage(ir=ir)"
                violations.append((tool_name, primary_category.value, reason))

        assert not violations, (
            f"Tools in '{self.agent_module}' have provenance metadata-runtime disconnect:\n"
            + "\n".join(
                f"  - Tool '{tool}' (category: {cat}): {reason}"
                for tool, cat, reason in violations
            )
            + "\n\nFix: Ensure tools with provenance=True call log_tool_usage(ir=ir) in their implementation."
        )

    def test_tool_params_no_bare_collections(self) -> None:
        """
        Verify no tool parameters use bare collection types (list, dict, tuple, set).

        Bare collection types produce JSON schemas without `items`/`properties`
        fields, which causes 400 INVALID_ARGUMENT on Gemini API. All collection
        type hints must be parameterized: list[str], dict[str, Any], etc.

        This validates TEST-10: Gemini-compatible tool schemas.
        """
        import types
        import typing

        tools = self._require_tools()

        violations = []
        # Bare builtin collection types that produce invalid JSON schema
        bare_builtins = {list, dict, tuple, set}
        # typing module aliases (unparameterized) — also produce bare schemas
        bare_typing = {typing.List, typing.Dict, typing.Tuple, typing.Set}
        bare_all = bare_builtins | bare_typing

        def _is_bare(hint) -> str | None:
            """Return type name if hint is a bare collection, else None."""
            if hint in bare_builtins:
                return hint.__name__
            if hint in bare_typing:
                # e.g. typing.List -> "List"
                return getattr(hint, "__name__", None) or str(hint).split(".")[-1]
            return None

        for tool in tools:
            tool_name = getattr(tool, "name", str(tool))

            # Get the underlying function
            func = getattr(tool, "func", None)
            if func is None:
                continue

            try:
                hints = typing.get_type_hints(func)
            except Exception:
                continue

            sig = inspect.signature(func)

            for param_name, param in sig.parameters.items():
                if param_name in ("self", "return"):
                    continue

                hint = hints.get(param_name)
                if hint is None:
                    continue

                # Unwrap Optional/Union to check the inner type
                actual = hint
                origin = typing.get_origin(actual)
                if origin is types.UnionType or origin is typing.Union:
                    # e.g. list | None -> check each arg
                    for arg in typing.get_args(actual):
                        if arg is type(None):
                            continue
                        name = _is_bare(arg)
                        if name:
                            violations.append((tool_name, param_name, name))
                else:
                    name = _is_bare(actual)
                    if name:
                        violations.append((tool_name, param_name, name))

        assert not violations, (
            f"Tools in '{self.agent_module}' have bare collection type hints "
            f"(breaks Gemini API — missing 'items' in JSON schema):\n"
            + "\n".join(
                f"  - Tool '{tool}', param '{param}': bare `{typ}` → "
                f"use `{typ}[str]` or appropriate parameterized type"
                for tool, param, typ in violations
            )
            + "\n\nFix: Replace `list` with `list[str]`, `dict` with `dict[str, Any]`, etc."
        )

    def test_all_contract_requirements(self) -> None:
        """
        Run all contract validation tests together.

        This is a convenience method that runs all validation checks
        in a single test, useful for quick verification.

        Includes both Phase 1 plugin contract tests and Phase 2 AQUADIF tests.
        """
        # Phase 1: Plugin contract tests
        self.test_factory_has_standard_params()
        self.test_no_deprecated_handoff_tools()
        self.test_agent_config_exists()
        self.test_agent_config_has_name()
        self.test_agent_config_has_tier_requirement()

        # Phase 2: AQUADIF contract tests (basic metadata)
        self.test_tools_have_aquadif_metadata()
        self.test_categories_are_valid()
        self.test_categories_capped_at_three()
        self.test_provenance_tools_have_flag()
        self.test_metadata_objects_are_unique()
        self.test_provenance_categories_not_buried()

        # Phase 2: AQUADIF contract tests (advanced)
        self.test_minimum_viable_parent()  # Skips if not parent agent
        self.test_provenance_ast_validation()

        # Phase 3: Schema compatibility tests
        self.test_tool_params_no_bare_collections()
