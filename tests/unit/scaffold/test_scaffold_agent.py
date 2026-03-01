import ast
import tomllib
from pathlib import Path

from lobster.scaffold import scaffold_agent


class TestScaffoldAgent:
    """Integration test: scaffold_agent produces complete package."""

    def test_generates_all_required_files(self, tmp_path):
        """Scaffold must generate every file needed for a working plugin."""
        scaffold_agent(
            name="epigenomics_expert",
            display_name="Epigenomics Expert",
            description="Epigenomics analysis",
            tier="free",
            output_dir=tmp_path,
        )
        pkg = tmp_path / "lobster-epigenomics"
        assert (pkg / "pyproject.toml").exists()
        assert (pkg / "README.md").exists()
        assert (pkg / "lobster" / "agents" / "epigenomics" / "__init__.py").exists()
        assert (
            pkg / "lobster" / "agents" / "epigenomics" / "epigenomics_expert.py"
        ).exists()
        assert (pkg / "lobster" / "agents" / "epigenomics" / "shared_tools.py").exists()
        assert (pkg / "lobster" / "agents" / "epigenomics" / "state.py").exists()
        assert (pkg / "lobster" / "agents" / "epigenomics" / "config.py").exists()
        assert (pkg / "lobster" / "agents" / "epigenomics" / "prompts.py").exists()
        assert (pkg / "tests" / "test_contract.py").exists()
        assert (pkg / "tests" / "conftest.py").exists()
        # PEP 420: NO __init__.py at lobster/ or lobster/agents/ level
        assert not (pkg / "lobster" / "__init__.py").exists()
        assert not (pkg / "lobster" / "agents" / "__init__.py").exists()

    def test_pyproject_has_correct_entry_points(self, tmp_path):
        scaffold_agent(
            name="epigenomics_expert",
            display_name="Epigenomics Expert",
            description="Epigenomics analysis",
            tier="free",
            output_dir=tmp_path,
        )
        content = (tmp_path / "lobster-epigenomics" / "pyproject.toml").read_text()
        parsed = tomllib.loads(content)
        eps = parsed["project"]["entry-points"]["lobster.agents"]
        assert "epigenomics_expert" in eps

    def test_no_lobster_init_py(self, tmp_path):
        """PEP 420 compliance: no __init__.py at namespace boundaries."""
        scaffold_agent(
            name="test_expert",
            display_name="Test Expert",
            description="Test",
            tier="free",
            output_dir=tmp_path,
        )
        pkg = tmp_path / "lobster-test"
        assert not (pkg / "lobster" / "__init__.py").exists()
        assert not (pkg / "lobster" / "agents" / "__init__.py").exists()

    def test_children_flag(self, tmp_path):
        """--children flag generates child agent files."""
        scaffold_agent(
            name="epigenomics_expert",
            display_name="Epigenomics Expert",
            description="Epigenomics analysis",
            tier="free",
            children=["methylation_expert", "chromatin_expert"],
            output_dir=tmp_path,
        )
        pkg = tmp_path / "lobster-epigenomics"
        assert (
            pkg / "lobster" / "agents" / "epigenomics" / "methylation_expert.py"
        ).exists()
        assert (
            pkg / "lobster" / "agents" / "epigenomics" / "chromatin_expert.py"
        ).exists()

    def test_all_generated_python_is_valid(self, tmp_path):
        """Every .py file must be valid Python."""
        scaffold_agent(
            name="epigenomics_expert",
            display_name="Epigenomics Expert",
            description="Epigenomics analysis",
            tier="free",
            output_dir=tmp_path,
        )
        pkg = tmp_path / "lobster-epigenomics"
        for py_file in pkg.rglob("*.py"):
            content = py_file.read_text()
            if content.strip():  # Skip empty __init__.py
                ast.parse(content)

    def test_contract_test_has_marker(self, tmp_path):
        """Delta 3: Generated contract test must have @pytest.mark.contract."""
        scaffold_agent(
            name="epigenomics_expert",
            display_name="Epigenomics Expert",
            description="Epigenomics analysis",
            tier="free",
            output_dir=tmp_path,
        )
        content = (
            tmp_path / "lobster-epigenomics" / "tests" / "test_contract.py"
        ).read_text()
        assert "@pytest.mark.contract" in content

    def test_returns_package_path(self, tmp_path):
        result = scaffold_agent(
            name="test_expert",
            display_name="Test Expert",
            description="Test",
            tier="free",
            output_dir=tmp_path,
        )
        assert result == tmp_path / "lobster-test"
        assert result.is_dir()
