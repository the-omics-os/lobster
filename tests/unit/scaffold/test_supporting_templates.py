import ast

from lobster.scaffold.generators.agent import _render_template

CONTEXT = {
    "agent_name": "epigenomics_expert",
    "display_name": "Epigenomics Expert",
    "description": "Epigenomics analysis: bisulfite-seq, ATAC-seq, ChIP-seq",
    "domain": "epigenomics",
    "package_name": "lobster-epigenomics",
    "tier": "free",
    "has_children": False,
    "children": [],
    "state_class": "EpigenomicsExpertState",
    "agent_class_name": "EpigenomicsExpert",
}


class TestInitTemplate:
    """Verify __init__.py template."""

    def test_renders_valid_python(self):
        content = _render_template("__init__.py.j2", CONTEXT)
        ast.parse(content)

    def test_state_always_importable(self):
        content = _render_template("__init__.py.j2", CONTEXT)
        assert "from lobster.agents.epigenomics.state import" in content

    def test_no_try_except_import_error(self):
        content = _render_template("__init__.py.j2", CONTEXT)
        # Check actual Python code, not comments
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if (
                    node.type
                    and isinstance(node.type, ast.Name)
                    and node.type.id == "ImportError"
                ):
                    raise AssertionError("Template must not use try/except ImportError")

    def test_has_all_export(self):
        content = _render_template("__init__.py.j2", CONTEXT)
        assert "__all__" in content


class TestStateTemplate:
    """Verify state.py template."""

    def test_renders_valid_python(self):
        content = _render_template("state.py.j2", CONTEXT)
        ast.parse(content)

    def test_inherits_agent_state(self):
        content = _render_template("state.py.j2", CONTEXT)
        assert "AgentState" in content
        assert "class EpigenomicsExpertState(AgentState)" in content

    def test_has_next_field(self):
        content = _render_template("state.py.j2", CONTEXT)
        assert 'next: str = ""' in content


class TestConfigTemplate:
    """Verify config.py template."""

    def test_renders_valid_python(self):
        content = _render_template("config.py.j2", CONTEXT)
        ast.parse(content)

    def test_has_dataclass(self):
        content = _render_template("config.py.j2", CONTEXT)
        assert "@dataclass" in content

    def test_has_detection_function(self):
        content = _render_template("config.py.j2", CONTEXT)
        assert "def detect_platform_type" in content


class TestPromptsTemplate:
    """Verify prompts.py template."""

    def test_renders_valid_python(self):
        content = _render_template("prompts.py.j2", CONTEXT)
        ast.parse(content)

    def test_has_factory_function(self):
        content = _render_template("prompts.py.j2", CONTEXT)
        assert "def create_epigenomics_expert_prompt" in content

    def test_has_xml_sections(self):
        content = _render_template("prompts.py.j2", CONTEXT)
        assert "<Identity_And_Role>" in content
        assert "<Your_Environment>" in content
        assert "<Your_Responsibilities>" in content
        assert "<Your_Tools>" in content


class TestContractTestTemplate:
    """Verify test_contract.py template."""

    def test_renders_valid_python(self):
        content = _render_template("test_contract.py.j2", CONTEXT)
        ast.parse(content)

    def test_inherits_mixin(self):
        content = _render_template("test_contract.py.j2", CONTEXT)
        assert "AgentContractTestMixin" in content

    def test_has_contract_marker(self):
        """Delta 3: Must have @pytest.mark.contract decorator."""
        content = _render_template("test_contract.py.j2", CONTEXT)
        assert "@pytest.mark.contract" in content

    def test_sets_agent_module(self):
        content = _render_template("test_contract.py.j2", CONTEXT)
        assert (
            'agent_module = "lobster.agents.epigenomics.epigenomics_expert"' in content
        )


class TestConftestTemplate:
    """Verify conftest.py template."""

    def test_renders_valid_python(self):
        content = _render_template("conftest.py.j2", CONTEXT)
        ast.parse(content)

    def test_has_mock_data_manager(self):
        content = _render_template("conftest.py.j2", CONTEXT)
        assert "mock_data_manager" in content


class TestReadmeTemplate:
    """Verify README.md template."""

    def test_has_package_name(self):
        content = _render_template("README.md.j2", CONTEXT)
        assert "lobster-epigenomics" in content

    def test_no_bare_pip(self):
        """Generated README must use uv, not bare pip."""
        content = _render_template("README.md.j2", CONTEXT)
        assert "uv pip install" in content
        # Make sure we don't have bare "pip install" (without "uv " prefix)
        lines = content.split("\n")
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("pip install"):
                raise AssertionError(f"Bare pip found: {stripped}")
