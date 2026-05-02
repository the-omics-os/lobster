import ast

from lobster.scaffold.generators.agent import _render_template


class TestSharedToolsTemplate:
    """Verify shared_tools template generates AQUADIF-compliant tools."""

    CONTEXT = {
        "agent_name": "epigenomics_expert",
        "domain": "epigenomics",
        "display_name": "Epigenomics Expert",
    }

    def test_renders_valid_python(self):
        content = _render_template("shared_tools.py.j2", self.CONTEXT)
        ast.parse(content)

    def test_uses_string_literals_not_aquadif_import(self):
        """AQUADIF metadata must use string literals, not AquadifCategory import."""
        content = _render_template("shared_tools.py.j2", self.CONTEXT)
        assert (
            "AquadifCategory" not in content
        ), "Use string literals, not AquadifCategory import"
        assert '"IMPORT"' in content
        assert '"QUALITY"' in content
        assert '"ANALYZE"' in content
        assert '"UTILITY"' in content

    def test_tools_have_metadata_assignment(self):
        """Every tool must have .metadata and .tags assigned."""
        content = _render_template("shared_tools.py.j2", self.CONTEXT)
        assert ".metadata = {" in content
        assert ".tags = [" in content

    def test_provenance_tools_have_ir(self):
        """Tools with provenance must call log_tool_usage(ir=ir)."""
        content = _render_template("shared_tools.py.j2", self.CONTEXT)
        assert "ir=ir" in content

    def test_factory_returns_list(self):
        content = _render_template("shared_tools.py.j2", self.CONTEXT)
        assert "return [" in content

    def test_factory_takes_data_manager(self):
        content = _render_template("shared_tools.py.j2", self.CONTEXT)
        assert "data_manager: DataManagerV2" in content

    def test_provenance_categories_first_comment(self):
        """Delta 2: Template must have comment about provenance category ordering."""
        content = _render_template("shared_tools.py.j2", self.CONTEXT)
        assert "Provenance-required categories" in content
        assert "MUST be listed FIRST" in content
        assert "test_provenance_categories_not_buried" in content
