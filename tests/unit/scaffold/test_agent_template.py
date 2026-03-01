import ast

from lobster.scaffold.generators.agent import _render_template


class TestAgentTemplate:
    """Verify agent module template generates production-correct code."""

    CONTEXT = {
        "agent_name": "epigenomics_expert",
        "display_name": "Epigenomics Expert",
        "description": "Epigenomics analysis: bisulfite-seq, ATAC-seq, ChIP-seq",
        "domain": "epigenomics",
        "handoff_description": "Assign epigenomics tasks",
        "tier": "free",
        "has_children": False,
        "children": [],
        "state_class": "EpigenomicsExpertState",
        "package_name": "lobster-epigenomics",
    }

    def test_renders_valid_python(self):
        """Rendered agent module must be valid Python."""
        content = _render_template("agent.py.j2", self.CONTEXT)
        ast.parse(content)  # Raises SyntaxError if invalid

    def test_agent_config_before_heavy_imports(self):
        """AGENT_CONFIG must appear before any heavy imports."""
        content = _render_template("agent.py.j2", self.CONTEXT)
        config_pos = content.find("AGENT_CONFIG = AgentRegistryConfig(")
        langgraph_pos = content.find("from langgraph")
        langchain_pos = content.find("from langchain")
        assert config_pos > 0, "AGENT_CONFIG not found"
        assert config_pos < langgraph_pos, "AGENT_CONFIG must be before langgraph import"
        assert config_pos < langchain_pos, "AGENT_CONFIG must be before langchain import"

    def test_uses_prompt_not_state_modifier(self):
        """Factory must use prompt= parameter (not state_modifier=)."""
        content = _render_template("agent.py.j2", self.CONTEXT)
        assert "prompt=system_prompt" in content
        assert "state_modifier=" not in content

    def test_factory_has_standard_params(self):
        """Factory must have all standard parameters."""
        content = _render_template("agent.py.j2", self.CONTEXT)
        for param in [
            "data_manager",
            "callback_handler",
            "agent_name",
            "delegation_tools",
            "workspace_path",
            "provider_override",
            "model_override",
        ]:
            assert param in content, f"Missing factory parameter: {param}"

    def test_factory_returns_create_react_agent(self):
        """Factory must return create_react_agent() result."""
        content = _render_template("agent.py.j2", self.CONTEXT)
        assert "return create_react_agent(" in content

    def test_lazy_prompt_import(self):
        """Prompt import must be inside factory (lazy), not at module level."""
        content = _render_template("agent.py.j2", self.CONTEXT)
        tree = ast.parse(content)
        # Find the factory function
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "epigenomics_expert":
                # Check that prompt import is inside the function body
                body_source = ast.get_source_segment(content, node)
                assert "create_epigenomics_expert_prompt" in body_source
                break
        else:
            raise AssertionError("Factory function 'epigenomics_expert' not found")
