import tomllib

from lobster.scaffold.generators.agent import _render_template


class TestPyprojectTemplate:
    """Verify pyproject.toml template renders correctly."""

    CONTEXT = {
        "agent_name": "epigenomics_expert",
        "display_name": "Epigenomics Expert",
        "package_name": "lobster-epigenomics",
        "description": "Epigenomics analysis",
        "domain": "epigenomics",
        "tier": "free",
        "python_version": "3.12",
        "author_name": "Test Author",
        "author_email": "test@example.com",
        "state_class": "EpigenomicsExpertState",
        "children": [],
        "child_state_classes": [],
    }

    def test_renders_valid_toml(self):
        """Rendered pyproject.toml must be valid TOML."""
        content = _render_template("pyproject.toml.j2", self.CONTEXT)
        parsed = tomllib.loads(content)
        assert parsed["project"]["name"] == "lobster-epigenomics"

    def test_entry_point_format(self):
        """Entry point must match production pattern."""
        content = _render_template("pyproject.toml.j2", self.CONTEXT)
        parsed = tomllib.loads(content)
        agents_eps = parsed["project"]["entry-points"]["lobster.agents"]
        assert agents_eps["epigenomics_expert"] == (
            "lobster.agents.epigenomics.epigenomics_expert:AGENT_CONFIG"
        )

    def test_namespace_packages_enabled(self):
        """Must enable PEP 420 namespace packages."""
        content = _render_template("pyproject.toml.j2", self.CONTEXT)
        parsed = tomllib.loads(content)
        find_cfg = parsed["tool"]["setuptools"]["packages"]["find"]
        assert find_cfg["namespaces"] is True
        assert "lobster*" in find_cfg["include"]

    def test_lobster_ai_dependency(self):
        """Must depend on lobster-ai."""
        content = _render_template("pyproject.toml.j2", self.CONTEXT)
        parsed = tomllib.loads(content)
        deps = parsed["project"]["dependencies"]
        assert any("lobster-ai" in d for d in deps)

    def test_state_entry_point(self):
        """Must have state entry point."""
        content = _render_template("pyproject.toml.j2", self.CONTEXT)
        parsed = tomllib.loads(content)
        states_eps = parsed["project"]["entry-points"]["lobster.states"]
        assert "EpigenomicsExpertState" in states_eps
