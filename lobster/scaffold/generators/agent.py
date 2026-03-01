"""Agent plugin package generator."""

import importlib.resources as pkg_resources
from pathlib import Path
from typing import List, Optional

import jinja2

# Use importlib.resources for robust template resolution across namespace
# packages, wheels, and zip installs — never rely on __file__ traversal.
_TEMPLATE_DIR = str(pkg_resources.files("lobster.scaffold").joinpath("templates"))

_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(_TEMPLATE_DIR),
    keep_trailing_newline=True,
    trim_blocks=True,
    lstrip_blocks=True,
)


def _render_template(template_name: str, context: dict) -> str:
    """Render a Jinja2 template with the given context."""
    template = _env.get_template(template_name)
    return template.render(**context)


def _derive_domain(agent_name: str) -> str:
    """Derive domain name from agent name (strip _expert suffix)."""
    return agent_name.removesuffix("_expert")


def _to_class_name(snake_name: str) -> str:
    """Convert snake_case to PascalCase (e.g., epigenomics_expert -> EpigenomicsExpert)."""
    return "".join(word.capitalize() for word in snake_name.split("_"))


def scaffold_agent(
    name: str,
    display_name: str,
    description: str,
    tier: str = "free",
    children: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    author_name: str = "Lobster AI Community",
    author_email: str = "community@lobsterbio.com",
    python_version: str = "3.12",
) -> Path:
    """Generate a complete agent plugin package.

    Args:
        name: Agent name in snake_case (e.g., "epigenomics_expert")
        display_name: Human-readable name (e.g., "Epigenomics Expert")
        description: Agent capabilities description
        tier: Subscription tier ("free", "premium", "enterprise")
        children: Optional list of child agent names
        output_dir: Where to create the package (default: current directory)
        author_name: Package author name
        author_email: Package author email
        python_version: Minimum Python version

    Returns:
        Path to the generated package directory
    """
    output_dir = output_dir or Path.cwd()
    children = children or []
    domain = _derive_domain(name)
    package_name = f"lobster-{domain}"
    agent_class_name = _to_class_name(name)
    state_class = f"{agent_class_name}State"

    # Build template context
    context = {
        "agent_name": name,
        "display_name": display_name,
        "description": description,
        "domain": domain,
        "package_name": package_name,
        "tier": tier,
        "has_children": len(children) > 0,
        "children": children,
        "agent_class_name": agent_class_name,
        "state_class": state_class,
        "child_state_classes": [f"{_to_class_name(c)}State" for c in children],
        "handoff_description": f"Assign {domain} tasks: {description}",
        "author_name": author_name,
        "author_email": author_email,
        "python_version": python_version,
    }

    # Create directory structure (PEP 420: no __init__.py at lobster/ or lobster/agents/)
    pkg_dir = output_dir / package_name
    agent_dir = pkg_dir / "lobster" / "agents" / domain
    test_dir = pkg_dir / "tests"
    agent_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Render and write files
    files = {
        pkg_dir / "pyproject.toml": "pyproject.toml.j2",
        pkg_dir / "README.md": "README.md.j2",
        agent_dir / "__init__.py": "__init__.py.j2",
        agent_dir / f"{name}.py": "agent.py.j2",
        agent_dir / "shared_tools.py": "shared_tools.py.j2",
        agent_dir / "state.py": "state.py.j2",
        agent_dir / "config.py": "config.py.j2",
        agent_dir / "prompts.py": "prompts.py.j2",
        test_dir / "__init__.py": None,  # Empty file
        test_dir / "test_contract.py": "test_contract.py.j2",
        test_dir / "conftest.py": "conftest.py.j2",
    }

    for filepath, template_name in files.items():
        if template_name is None:
            filepath.write_text("")
        else:
            content = _render_template(template_name, context)
            filepath.write_text(content)

    # Generate child agent files if any
    for child_name in children:
        child_class_name = _to_class_name(child_name)
        child_context = {
            **context,
            "agent_name": child_name,
            "display_name": child_class_name.replace("Expert", " Expert").strip(),
            "agent_class_name": child_class_name,
            "state_class": f"{child_class_name}State",
            "has_children": False,
            "children": [],
            "is_child": True,
        }
        child_content = _render_template("agent.py.j2", child_context)
        (agent_dir / f"{child_name}.py").write_text(child_content)

    return pkg_dir
