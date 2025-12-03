"""
Custom Feature Agent for creating new Lobster agents, services, and documentation.

This agent uses the Claude Code SDK to generate new features following Lobster's
architectural patterns defined in lobster/agents/CLAUDE.md.
"""

import asyncio
import concurrent.futures
import os
import re
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from lobster.agents.state import CustomFeatureAgentState
from lobster.config.llm_factory import create_llm
from lobster.config.settings import get_settings
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class CustomFeatureError(Exception):
    """Base exception for custom feature operations."""

    pass


class FeatureCreationError(CustomFeatureError):
    """Raised when feature creation fails."""

    pass


class ValidationError(CustomFeatureError):
    """Raised when feature validation fails."""

    pass


def custom_feature_agent(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "custom_feature_agent",
    handoff_tools: List = None,
):
    """
    Create custom feature agent that uses Claude Code SDK.

    This agent spawns Claude Code instances to create new agents, services,
    tools, tests, and documentation following Lobster's architectural patterns.

    Args:
        data_manager: DataManagerV2 instance for data access
        callback_handler: Optional callback for LLM events
        agent_name: Name for this agent instance
        handoff_tools: List of tools for agent handoff

    Returns:
        LangGraph agent for custom feature creation
    """
    settings = get_settings()
    model_params = settings.get_agent_llm_params("custom_feature_agent")
    llm = create_llm("custom_feature_agent", model_params)

    if callback_handler and hasattr(llm, "with_config"):
        llm = llm.with_config(callbacks=[callback_handler])

    # Store feature creation results
    creation_results = {"summary": "", "details": {}, "created_files": []}

    # Get Lobster root directory
    lobster_root = Path(__file__).parent.parent.parent

    # -------------------------
    # HELPER FUNCTIONS
    # -------------------------

    def validate_feature_name(name: str) -> tuple[bool, str]:
        """
        Validate feature name follows Lobster conventions.

        Args:
            name: Feature name to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not name:
            return False, "Feature name cannot be empty"

        if not re.match(r"^[a-z][a-z0-9_]*$", name):
            return (
                False,
                "Feature name must start with lowercase letter and contain only lowercase letters, numbers, and underscores",
            )

        if name.endswith("_"):
            return False, "Feature name cannot end with underscore"

        # Check for reserved names
        reserved = ["supervisor", "data", "base", "core", "config"]
        if name in reserved:
            return False, f"Feature name '{name}' is reserved"

        return True, ""

    def check_existing_files(feature_name: str) -> List[str]:
        """
        Check if files for this feature already exist.

        Args:
            feature_name: Name of the feature

        Returns:
            List of existing file paths
        """
        existing = []

        # Check agent file
        agent_file = lobster_root / "lobster" / "agents" / f"{feature_name}_expert.py"
        if agent_file.exists():
            existing.append(str(agent_file))

        # Check service file in all services/ subdirectories
        service_categories = [
            "analysis",
            "data_access",
            "data_management",
            "metadata",
            "ml",
            "orchestration",
            "quality",
            "visualization",
        ]
        for category in service_categories:
            service_file = (
                lobster_root
                / "lobster"
                / "services"
                / category
                / f"{feature_name}_service.py"
            )
            if service_file.exists():
                existing.append(str(service_file))

        # Also check legacy tools/ location for backwards compatibility
        legacy_service = (
            lobster_root / "lobster" / "tools" / f"{feature_name}_service.py"
        )
        if legacy_service.exists():
            existing.append(str(legacy_service))

        # Check test files - agent tests
        test_agent = (
            lobster_root
            / "tests"
            / "unit"
            / "agents"
            / f"test_{feature_name}_expert.py"
        )
        if test_agent.exists():
            existing.append(str(test_agent))

        # Check test files in all services/ test subdirectories
        for category in service_categories:
            test_service = (
                lobster_root
                / "tests"
                / "unit"
                / "services"
                / category
                / f"test_{feature_name}_service.py"
            )
            if test_service.exists():
                existing.append(str(test_service))

        # Also check legacy test location
        legacy_test_service = (
            lobster_root
            / "tests"
            / "unit"
            / "tools"
            / f"test_{feature_name}_service.py"
        )
        if legacy_test_service.exists():
            existing.append(str(legacy_test_service))

        # Check wiki (using lowercase kebab-case convention)
        wiki_name = feature_name.replace("_", "-").lower()
        wiki_file = lobster_root / "lobster" / "wiki" / f"{wiki_name}.md"
        if wiki_file.exists():
            existing.append(str(wiki_file))

        return existing

    def create_feature_branch(feature_name: str) -> Dict[str, Any]:
        """
        Create git branch for new feature with standard naming convention.

        Branch format: feature/{feature_name}_{YYMMDD}

        Args:
            feature_name: Name of the feature for branch naming

        Returns:
            Dictionary with:
                - success: bool
                - branch_name: str (if successful)
                - error: str (if failed)
        """
        import subprocess
        from datetime import date

        # Generate branch name with date
        date_str = date.today().strftime("%y%m%d")
        branch_name = f"feature/{feature_name}_{date_str}"

        try:
            # Check if branch already exists
            result = subprocess.run(
                ["git", "rev-parse", "--verify", branch_name],
                capture_output=True,
                text=True,
                check=False,
                cwd=str(lobster_root),
            )

            if result.returncode == 0:
                logger.warning(f"Branch {branch_name} already exists")
                return {
                    "success": False,
                    "error": f"Branch '{branch_name}' already exists. Use a different feature name or delete the existing branch.",
                }

            # Check for uncommitted changes
            status_result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
                cwd=str(lobster_root),
            )

            if status_result.stdout.strip():
                logger.warning("Uncommitted changes detected")
                return {
                    "success": False,
                    "error": "Working directory has uncommitted changes. Please commit or stash them before creating a new feature branch.",
                }

            # Create and switch to branch
            subprocess.run(
                ["git", "checkout", "-b", branch_name],
                capture_output=True,
                text=True,
                check=True,
                cwd=str(lobster_root),
            )

            logger.info(f"Created feature branch: {branch_name}")
            return {
                "success": True,
                "branch_name": branch_name,
                "message": f"Successfully created and switched to branch: {branch_name}",
            }

        except subprocess.CalledProcessError as e:
            error_msg = f"Git operation failed: {e.stderr if e.stderr else str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        except Exception as e:
            error_msg = f"Unexpected error during branch creation: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}

    def detect_packages_in_files(file_paths: List[str]) -> Dict[str, List[str]]:
        """
        Parse import statements from Python files and categorize packages.

        Categorizes as:
        - python_packages: pip-installable packages (non-stdlib)
        - system_packages: system tools (detected via common patterns)
        - standard_lib: Python standard library modules
        - unknown: Unclassifiable imports

        Args:
            file_paths: List of absolute paths to Python files

        Returns:
            Dictionary with categorized package lists
        """
        import sys

        # Python standard library modules (Python 3.12+)
        stdlib_modules = set(sys.stdlib_module_names)

        # Common system tools that might be imported or used
        system_tools = {
            "samtools",
            "bcftools",
            "bedtools",
            "bwa",
            "bowtie2",
            "kallisto",
            "salmon",
            "star",
            "hisat2",
            "featurecounts",
            "blastn",
            "blastp",
            "blastx",
            "makeblastdb",
        }

        python_packages = set()
        system_packages = set()
        standard_lib = set()
        unknown = set()

        for file_path in file_paths:
            if not file_path.endswith(".py"):
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Extract import statements
                import_pattern = r"^(?:import|from)\s+(\w+)"
                matches = re.findall(import_pattern, content, re.MULTILINE)

                for module_name in matches:
                    # Skip local imports
                    if module_name.startswith("lobster"):
                        continue

                    # Categorize
                    if module_name in stdlib_modules:
                        standard_lib.add(module_name)
                    elif module_name.lower() in system_tools:
                        system_packages.add(module_name.lower())
                    elif module_name in ["os", "sys", "pathlib"]:  # Explicit stdlib
                        standard_lib.add(module_name)
                    else:
                        # Assume third-party Python package
                        python_packages.add(module_name)

            except Exception as e:
                logger.warning(f"Failed to parse imports from {file_path}: {e}")
                continue

        return {
            "python_packages": sorted(list(python_packages)),
            "system_packages": sorted(list(system_packages)),
            "standard_lib": sorted(list(standard_lib)),
            "unknown": sorted(list(unknown)),
        }

    def check_package_installation(
        packages: Dict[str, List[str]],
    ) -> Dict[str, Dict[str, bool]]:
        """
        Check if packages are installed using importlib (pip) and subprocess (brew).

        Args:
            packages: Dictionary with categorized package lists from detect_packages_in_files()

        Returns:
            Dictionary mapping package names to installation status:
                {
                    "python_packages": {"package": bool, ...},
                    "system_packages": {"tool": bool, ...}
                }
        """
        import importlib.util
        import shutil

        python_status = {}
        system_status = {}

        # Check Python packages using importlib
        for package in packages.get("python_packages", []):
            try:
                spec = importlib.util.find_spec(package)
                python_status[package] = spec is not None
            except (ImportError, ModuleNotFoundError, ValueError):
                python_status[package] = False
            except Exception as e:
                logger.warning(f"Error checking package {package}: {e}")
                python_status[package] = False

        # Check system packages using shutil.which (checks PATH)
        for sys_tool in packages.get("system_packages", []):
            try:
                system_status[sys_tool] = shutil.which(sys_tool) is not None
            except Exception as e:
                logger.warning(f"Error checking system tool {sys_tool}: {e}")
                system_status[sys_tool] = False

        return {"python_packages": python_status, "system_packages": system_status}

    def format_package_warnings(
        package_status: Dict[str, Dict[str, bool]],
        detected_packages: Dict[str, List[str]],
    ) -> str:
        """
        Format missing package warnings for user response.

        Args:
            package_status: Installation status from check_package_installation()
            detected_packages: Package categorization from detect_packages_in_files()

        Returns:
            Markdown-formatted warning section with installation instructions
        """
        missing_python = [
            pkg
            for pkg, installed in package_status.get("python_packages", {}).items()
            if not installed
        ]
        missing_system = [
            pkg
            for pkg, installed in package_status.get("system_packages", {}).items()
            if not installed
        ]

        if not missing_python and not missing_system:
            return ""

        warning = "⚠️ **Package Dependencies Detected**\n\n"
        warning += "The generated code imports packages that may not be installed:\n\n"

        if missing_python:
            warning += "**Python Packages (install with pip):**\n"
            warning += "```bash\n"
            warning += f"pip install {' '.join(missing_python)}\n"
            warning += "```\n\n"

        if missing_system:
            warning += "**System Tools (install with brew):**\n"
            warning += "```bash\n"
            warning += f"brew install {' '.join(missing_system)}\n"
            warning += "```\n\n"

        warning += "**Note:** The feature has been created, but you should install these dependencies before running the code.\n"

        return warning

    def research_with_linkup(
        feature_description: str,
        search_focus: str = "GitHub repositories, packages, best practices",
    ) -> Dict[str, Any]:
        """
        Research feature requirements using Linkup SDK (synchronous).

        Searches for:
        - Existing GitHub repositories with similar functionality
        - Python packages that can be reused (prioritize already-installed)
        - Best practices and integration patterns
        - Potential challenges and solutions

        Args:
            feature_description: Description of the feature to research
            search_focus: Specific areas to focus research on

        Returns:
            Dictionary with research findings:
                - github_repos: List of relevant repositories
                - packages: List of relevant Python packages
                - best_practices: List of recommended approaches
                - summary: Executive summary of findings
                - error: Optional error message if research failed
        """
        try:
            from linkup import LinkupClient

            settings = get_settings()
            api_key = settings.LINKUP_API_KEY

            if not api_key:
                logger.warning("LINKUP_API_KEY not set, skipping research phase")
                return {
                    "github_repos": [],
                    "packages": [],
                    "best_practices": [],
                    "summary": "Research skipped (no API key configured)",
                    "error": "LINKUP_API_KEY not configured in settings",
                }

            client = LinkupClient(api_key=api_key)
            logger.info(f"[RESEARCH] Starting research for: {feature_description}")

            # Query 1: Search for GitHub repositories
            github_query = (
                f"{feature_description} bioinformatics Python site:github.com"
            )
            logger.info(f"[RESEARCH] Query 1: {github_query}")

            github_response = client.search(
                query=github_query, depth="deep", output_type="sourcedAnswer"
            )

            # Query 2: Search for Python packages and libraries
            package_query = (
                f"{feature_description} Python library package bioinformatics"
            )
            logger.info(f"[RESEARCH] Query 2: {package_query}")

            package_response = client.search(
                query=package_query, depth="deep", output_type="sourcedAnswer"
            )

            # Query 3: Search for best practices and tutorials
            practices_query = f"{feature_description} best practices implementation tutorial bioinformatics"
            logger.info(f"[RESEARCH] Query 3: {practices_query}")

            practices_response = client.search(
                query=practices_query, depth="deep", output_type="sourcedAnswer"
            )

            # Parse responses
            github_repos = []
            if hasattr(github_response, "sources") and github_response.sources:
                for source in github_response.sources[:5]:  # Top 5 repos
                    github_repos.append(
                        {
                            "name": getattr(source, "name", "Unknown"),
                            "url": getattr(source, "url", ""),
                            "snippet": getattr(source, "snippet", "")[:200],  # Truncate
                        }
                    )

            packages = []
            if hasattr(package_response, "answer"):
                # Extract package names from answer (simple regex)
                import re

                package_matches = re.findall(
                    r"\b([a-z][a-z0-9_-]*)\b", package_response.answer.lower()
                )
                # Filter to known bioinformatics packages
                known_packages = [
                    "biopython",
                    "scanpy",
                    "anndata",
                    "scipy",
                    "numpy",
                    "pandas",
                    "matplotlib",
                    "seaborn",
                    "squidpy",
                    "pydeseq2",
                    "plotly",
                    "scikit-learn",
                    "sklearn",
                    "pytorch",
                    "tensorflow",
                ]
                packages = [p for p in package_matches if p in known_packages]
                packages = list(set(packages))[:10]  # Unique, top 10

            best_practices = []
            if hasattr(practices_response, "answer"):
                best_practices.append(practices_response.answer[:500])  # Truncate

            # Summary
            summary = f"""Research completed for: {feature_description}

GitHub Repositories Found: {len(github_repos)}
Relevant Packages: {', '.join(packages) if packages else 'None identified'}

Top Recommendation: {github_repos[0]['name'] if github_repos else 'No repositories found'}
"""

            logger.info(
                f"[RESEARCH] Completed. Found {len(github_repos)} repos, {len(packages)} packages"
            )

            return {
                "github_repos": github_repos,
                "packages": packages,
                "best_practices": best_practices,
                "summary": summary.strip(),
                "raw_responses": {
                    "github": (
                        github_response.answer
                        if hasattr(github_response, "answer")
                        else ""
                    ),
                    "packages": (
                        package_response.answer
                        if hasattr(package_response, "answer")
                        else ""
                    ),
                    "practices": (
                        practices_response.answer
                        if hasattr(practices_response, "answer")
                        else ""
                    ),
                },
            }

        except ImportError:
            logger.error("Linkup SDK not installed")
            return {
                "github_repos": [],
                "packages": [],
                "best_practices": [],
                "summary": "Research failed: Linkup SDK not installed",
                "error": "Import error: linkup package not found. Install with: pip install linkup-sdk",
            }
        except Exception as e:
            logger.error(f"Research error: {e}", exc_info=True)
            return {
                "github_repos": [],
                "packages": [],
                "best_practices": [],
                "summary": f"Research failed: {str(e)}",
                "error": str(e),
            }

    def display_sdk_message(msg):
        """Display SDK message content in real-time to terminal."""
        try:
            from claude_agent_sdk import (
                AssistantMessage,
                ResultMessage,
                SystemMessage,
                TextBlock,
                ToolResultBlock,
                ToolUseBlock,
                UserMessage,
            )

            if isinstance(msg, UserMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        logger.info(f"[SDK User] {block.text}")
                    elif isinstance(block, ToolResultBlock):
                        content_preview = (
                            block.content[:100] if block.content else "None"
                        )
                        logger.info(f"[SDK Tool Result] {content_preview}...")
            elif isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        logger.info(f"[SDK Claude] {block.text}")
                    elif isinstance(block, ToolUseBlock):
                        logger.info(f"[SDK Tool] Using tool: {block.name}")
                        if block.input:
                            logger.info(f"[SDK Tool]   Input: {block.input}")
            elif isinstance(msg, SystemMessage):
                # System messages are usually verbose, skip or log at debug level
                logger.debug(f"[SDK System] {msg}")
            elif isinstance(msg, ResultMessage):
                logger.info("[SDK Result] Generation completed")
                if hasattr(msg, "total_cost_usd") and msg.total_cost_usd:
                    logger.info(f"[SDK Cost] ${msg.total_cost_usd:.6f}")
        except Exception as e:
            logger.debug(f"Error displaying SDK message: {e}")

    async def create_with_claude_sdk(
        feature_type: str, feature_name: str, requirements: str, debug: bool = True
    ) -> Dict[str, Any]:
        """
        Use Claude Code SDK to create new feature files.

        Args:
            feature_type: Type of feature (agent, service, agent_with_service)
            feature_name: Name for the feature
            requirements: User requirements and specifications
            debug: Enable detailed debug logging

        Returns:
            Dictionary with creation results:
                - success: bool
                - created_files: List[str]
                - error: Optional[str]
                - sdk_output: str
                - debug_info: Dict[str, Any]
        """
        debug_info = {
            "feature_type": feature_type,
            "feature_name": feature_name,
            "lobster_root": str(lobster_root),
            "steps": [],
            "errors": [],
            "sdk_messages": [],
        }

        try:
            logger.info(
                f"[DEBUG] Starting feature creation: {feature_type} - {feature_name}"
            )
            debug_info["steps"].append("Starting feature creation")

            # STEP 0: Create feature branch BEFORE any code generation
            logger.info(f"[DEBUG] Creating feature branch for {feature_name}...")
            debug_info["steps"].append("Creating feature branch")

            branch_result = create_feature_branch(feature_name)
            if not branch_result["success"]:
                logger.error(
                    f"[DEBUG] Branch creation failed: {branch_result['error']}"
                )
                debug_info["errors"].append(
                    f"Branch creation failed: {branch_result['error']}"
                )
                debug_info["steps"].append("Branch creation failed")
                return {
                    "success": False,
                    "created_files": [],
                    "error": f"Failed to create feature branch: {branch_result['error']}",
                    "sdk_output": "",
                    "debug_info": debug_info,
                }

            branch_name = branch_result["branch_name"]
            debug_info["branch_name"] = branch_name
            debug_info["steps"].append(f"Created branch: {branch_name}")
            logger.info(f"[DEBUG] Feature branch created successfully: {branch_name}")

            # Set environment variables for Claude Code SDK with AWS Bedrock
            logger.info("[DEBUG] Setting AWS Bedrock environment variables...")
            settings = get_settings()
            os.environ["CLAUDE_CODE_USE_BEDROCK"] = "1"
            os.environ["AWS_REGION"] = getattr(settings, "AWS_REGION", "us-east-1")
            os.environ["ANTHROPIC_MODEL"] = (
                "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
            )
            os.environ["ANTHROPIC_SMALL_FAST_MODEL"] = (
                "us.anthropic.claude-sonnet-4-20250514-v1:0"
            )
            os.environ["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = "60024"
            os.environ["MAX_THINKING_TOKENS"] = "2024"
            debug_info["steps"].append("Environment variables configured")
            logger.info("[DEBUG] AWS Bedrock environment variables set successfully")

            # Import Claude Code SDK
            try:
                logger.debug("[DEBUG] Importing Claude Agent SDK...")
                debug_info["steps"].append("Importing Claude Agent SDK")
                from claude_agent_sdk import (
                    AssistantMessage,
                    ClaudeAgentOptions,
                    ClaudeSDKClient,
                    CLIConnectionError,
                    CLIJSONDecodeError,
                    CLINotFoundError,
                    ProcessError,
                    ResultMessage,
                    TextBlock,
                )

                logger.debug("[DEBUG] Claude Agent SDK imported successfully")
                debug_info["steps"].append("SDK imported successfully")
            except ImportError as e:
                error_msg = f"Claude Agent SDK not installed: {e}"
                logger.error(f"[DEBUG] {error_msg}")
                debug_info["errors"].append(error_msg)
                debug_info["steps"].append("SDK import failed")
                return {
                    "success": False,
                    "created_files": [],
                    "error": "Claude Agent SDK not installed. Please install with: pip install claude-agent-sdk",
                    "sdk_output": "",
                    "debug_info": debug_info,
                }

            # Configure SDK options
            try:
                logger.info("[DEBUG] Configuring SDK options...")
                logger.info(f"[DEBUG] Working directory: {lobster_root}")
                logger.info(f"[DEBUG] CLAUDE.md path: {lobster_root / 'CLAUDE.md'}")

                debug_info["steps"].append("Configuring SDK options")
                debug_info["sdk_config"] = {
                    "cwd": str(lobster_root),
                    "setting_sources": ["project"],
                    "allowed_tools": ["Read", "Write", "Edit", "Glob", "Grep", "Bash"],
                    "permission_mode": "bypassPermissions",
                    "max_turns": 50,
                }

                options = ClaudeAgentOptions(
                    setting_sources=["project"],  # Loads CLAUDE.md from project root
                    system_prompt={
                        "type": "preset",
                        "preset": "claude_code",
                        "append": """You are creating a new feature for the Lobster bioinformatics platform.

CRITICAL RULES:
1. Follow ALL patterns in CLAUDE.md EXACTLY
2. Use the agent pattern, service pattern, and tool pattern as shown
3. Create complete, working implementations
4. Include comprehensive docstrings
5. Follow Lobster naming conventions
6. Report ALL files you create with their full paths
7. You have FULL PERMISSIONS to create all necessary files""",
                    },
                    cwd=str(lobster_root),
                    allowed_tools=["Read", "Write", "Edit", "Glob", "Grep", "Bash"],
                    permission_mode="bypassPermissions",
                    max_turns=50,
                )
                logger.debug("[DEBUG] SDK options configured successfully")
                debug_info["steps"].append("SDK options configured")
            except Exception as e:
                error_msg = f"Failed to configure SDK options: {e}"
                logger.error(f"[DEBUG] {error_msg}")
                debug_info["errors"].append(error_msg)
                debug_info["steps"].append("SDK configuration failed")
                return {
                    "success": False,
                    "created_files": [],
                    "error": error_msg,
                    "sdk_output": "",
                    "debug_info": debug_info,
                }

            # Construct detailed prompt
            wiki_filename = feature_name.replace("_", "-").lower()
            if feature_type == "agent":
                files_to_create = f"""1. Agent: lobster/agents/{feature_name}_expert.py
2. State: Add {feature_name.title().replace('_', '')}ExpertState to lobster/agents/state.py
3. Tests: tests/unit/agents/test_{feature_name}_expert.py
4. Wiki: lobster/wiki/{wiki_filename}.md"""

            elif feature_type == "service":
                files_to_create = f"""1. Service: lobster/services/{{CATEGORY}}/{feature_name}_service.py
   - Choose CATEGORY from: analysis, data_access, data_management, metadata, ml, orchestration, quality, visualization
   - Base decision on feature purpose: clustering/DE/statistical→analysis, QC/preprocess/filter→quality, API/fetch/download→data_access,
     modality/CRUD→data_management, standardize/validate/mapping→metadata, ML/embedding/prediction→ml,
     plot/visualize/chart→visualization, workflow/pipeline→orchestration
2. Tests: tests/unit/services/{{CATEGORY}}/test_{feature_name}_service.py (mirror the service category)
3. Wiki: lobster/wiki/{wiki_filename}.md"""

            elif feature_type == "agent_with_service":
                files_to_create = f"""1. Agent: lobster/agents/{feature_name}_expert.py
2. Service: lobster/services/{{CATEGORY}}/{feature_name}_service.py
   - Choose CATEGORY from: analysis, data_access, data_management, metadata, ml, orchestration, quality, visualization
   - Base decision on feature purpose: clustering/DE/statistical→analysis, QC/preprocess/filter→quality, API/fetch/download→data_access,
     modality/CRUD→data_management, standardize/validate/mapping→metadata, ML/embedding/prediction→ml,
     plot/visualize/chart→visualization, workflow/pipeline→orchestration
3. State: Add {feature_name.title().replace('_', '')}ExpertState to lobster/agents/state.py
4. Tests (Agent): tests/unit/agents/test_{feature_name}_expert.py
5. Tests (Service): tests/unit/services/{{CATEGORY}}/test_{feature_name}_service.py (mirror the service category)
6. Wiki: lobster/wiki/{wiki_filename}.md"""

            else:
                return {
                    "success": False,
                    "created_files": [],
                    "error": f"Unknown feature type: {feature_type}",
                    "sdk_output": "",
                }

            prompt = f"""Create a new Lobster {feature_type} following the patterns in CLAUDE.md.

⚠️ IMPORTANT GIT CONTEXT:
You are working on feature branch: {branch_name}
All changes will be committed to this feature branch, NOT main.
The branch has been created and checked out automatically.
DO NOT create or switch branches - you are already on the correct branch.

Feature Name: {feature_name}

Requirements:
{requirements}

Files to Create:
{files_to_create}

Instructions:
1. Read CLAUDE.md at project root to understand all patterns
2. Look at existing examples:
   - Agents: lobster/agents/machine_learning_expert.py, singlecell_expert.py
   - Services: lobster/services/analysis/clustering_service.py, lobster/services/quality/preprocessing_service.py
3. Create ALL required files following the exact patterns
4. Use proper naming conventions
5. Include comprehensive docstrings and type hints
6. Follow the agent pattern (factory function, tools, system prompt)
7. Follow the service pattern (stateless, returns 3-tuple: AnnData, stats_dict, AnalysisStep)
8. Follow the tool pattern (validate → service → store → log with IR → response)
9. Create test directory structure if it doesn't exist:
   - For services: Create tests/unit/services/{category}/ with __init__.py
   - For agents: Create tests/unit/agents/ with __init__.py (if needed)
   - Use `mkdir -p` equivalent or create parent directories first
10. Create complete test files with fixtures and multiple test cases
11. Create comprehensive wiki documentation
12. IMPORTANT: Services go in lobster/services/{category}/ NOT lobster/tools/

After creating all files, provide a summary listing EVERY file you created with its full path.

Begin implementation now."""

            # Spawn Claude Code SDK
            try:
                logger.info("[DEBUG] Spawning Claude Code SDK...")
                debug_info["steps"].append("Spawning SDK client")

                sdk_output_lines = []
                created_files = []
                message_count = 0

                async with ClaudeSDKClient(options=options) as client:
                    logger.info("[DEBUG] SDK client connected successfully")
                    debug_info["steps"].append("SDK client connected")

                    # Send prompt
                    try:
                        logger.info("=" * 80)
                        logger.info(
                            f"[SDK START] Creating {feature_type}: {feature_name}"
                        )
                        logger.info("=" * 80)
                        logger.info(f"\n[SDK Prompt]\n{prompt}\n")
                        logger.info(
                            f"[DEBUG] Sending prompt to SDK (length: {len(prompt)} characters)..."
                        )
                        debug_info["steps"].append("Sending prompt to SDK")
                        debug_info["prompt_length"] = len(prompt)

                        await client.query(prompt)
                        logger.info("[DEBUG] Prompt sent successfully")
                        debug_info["steps"].append("Prompt sent")
                    except Exception as e:
                        error_msg = f"Failed to send prompt to SDK: {e}"
                        logger.error(f"[DEBUG] {error_msg}")
                        debug_info["errors"].append(error_msg)
                        debug_info["steps"].append("Prompt sending failed")
                        raise

                    # Collect responses
                    try:
                        logger.info("[DEBUG] Receiving SDK responses...")
                        debug_info["steps"].append("Receiving SDK responses")

                        async for message in client.receive_response():
                            message_count += 1
                            logger.debug(
                                f"[DEBUG] Received message {message_count}: {type(message).__name__}"
                            )

                            # Display message in real-time to terminal
                            display_sdk_message(message)

                            # Capture message content for logging
                            message_str = str(message)
                            sdk_output_lines.append(message_str)
                            debug_info["sdk_messages"].append(
                                {
                                    "index": message_count,
                                    "type": type(message).__name__,
                                    "content_preview": (
                                        message_str[:200]
                                        if len(message_str) > 200
                                        else message_str
                                    ),
                                }
                            )

                            # Parse SDK messages correctly to extract created file paths
                            if isinstance(message, AssistantMessage):
                                # Iterate through content blocks to find TextBlocks
                                for block in message.content:
                                    if isinstance(block, TextBlock):
                                        text = block.text

                                        # Look for file path patterns (more robust regex)
                                        # Updated to include new services/ directory structure
                                        patterns = [
                                            r"lobster/agents/[a-z0-9_]+_expert\.py",
                                            r"lobster/services/[a-z_]+/[a-z0-9_]+_service\.py",  # New services/ structure
                                            r"lobster/tools/[a-z0-9_]+_service\.py",  # Legacy tools/ location
                                            r"lobster/agents/state\.py",
                                            r"tests/unit/agents/test_[a-z0-9_]+_expert\.py",
                                            r"tests/unit/services/[a-z_]+/test_[a-z0-9_]+_service\.py",  # New test structure
                                            r"tests/unit/tools/test_[a-z0-9_]+_service\.py",  # Legacy test location
                                            r"lobster/wiki/[a-z0-9\-]+\.md",
                                        ]

                                        for pattern in patterns:
                                            matches = re.findall(pattern, text)
                                            for match in matches:
                                                full_path = lobster_root / match
                                                if (
                                                    full_path.exists()
                                                    and str(full_path)
                                                    not in created_files
                                                ):
                                                    created_files.append(str(full_path))
                                                    logger.info(
                                                        f"[DEBUG] Detected created file: {full_path}"
                                                    )
                            elif isinstance(message, ResultMessage):
                                # Optionally use message.result if SDK summarizes file list
                                logger.debug(
                                    f"[DEBUG] Result message received: {message.result if hasattr(message, 'result') else 'N/A'}"
                                )

                        logger.info("=" * 80)
                        logger.info(
                            f"[SDK END] Completed. Received {message_count} messages"
                        )
                        logger.info(
                            f"[SDK END] Detected {len(created_files)} created files from messages"
                        )
                        logger.info("=" * 80)
                        debug_info["steps"].append(
                            f"SDK completed ({message_count} messages)"
                        )
                        debug_info["total_messages"] = message_count
                        debug_info["detected_files_count"] = len(created_files)
                    except Exception as e:
                        error_msg = f"Error receiving SDK responses: {e}"
                        logger.error(f"[DEBUG] {error_msg}")
                        debug_info["errors"].append(error_msg)
                        debug_info["steps"].append("SDK response receiving failed")
                        raise

            except CLINotFoundError as e:
                error_msg = "Claude Code CLI not found"
                logger.error(f"[DEBUG] {error_msg}: {e}")
                debug_info["errors"].append(error_msg)
                debug_info["steps"].append("CLI not found")
                return {
                    "success": False,
                    "created_files": [],
                    "error": "Claude Code not found. Install CLI: npm i -g @anthropic-ai/claude-code",
                    "sdk_output": (
                        "\n".join(sdk_output_lines) if sdk_output_lines else ""
                    ),
                    "debug_info": debug_info,
                }
            except CLIConnectionError as e:
                error_msg = f"Failed to connect to Claude Code CLI: {e}"
                logger.error(f"[DEBUG] {error_msg}")
                debug_info["errors"].append(error_msg)
                debug_info["steps"].append("CLI connection failed")
                return {
                    "success": False,
                    "created_files": [],
                    "error": error_msg,
                    "sdk_output": (
                        "\n".join(sdk_output_lines) if sdk_output_lines else ""
                    ),
                    "debug_info": debug_info,
                }
            except ProcessError as e:
                error_msg = f"Claude Code process error: {e}"
                logger.error(f"[DEBUG] {error_msg}")
                debug_info["errors"].append(error_msg)
                debug_info["steps"].append("CLI process error")
                return {
                    "success": False,
                    "created_files": [],
                    "error": error_msg,
                    "sdk_output": (
                        "\n".join(sdk_output_lines) if sdk_output_lines else ""
                    ),
                    "debug_info": debug_info,
                }
            except CLIJSONDecodeError as e:
                error_msg = f"Failed to parse CLI response: {e}"
                logger.error(f"[DEBUG] {error_msg}")
                debug_info["errors"].append(error_msg)
                debug_info["steps"].append("CLI JSON decode error")
                return {
                    "success": False,
                    "created_files": [],
                    "error": error_msg,
                    "sdk_output": (
                        "\n".join(sdk_output_lines) if sdk_output_lines else ""
                    ),
                    "debug_info": debug_info,
                }
            except Exception as e:
                error_msg = f"SDK client error: {e}"
                logger.error(f"[DEBUG] {error_msg}", exc_info=True)
                debug_info["errors"].append(error_msg)
                debug_info["steps"].append("SDK client failed")
                return {
                    "success": False,
                    "created_files": [],
                    "error": error_msg,
                    "sdk_output": (
                        "\n".join(sdk_output_lines)
                        if "sdk_output_lines" in locals()
                        else ""
                    ),
                    "debug_info": debug_info,
                }

            # Verify files were created
            try:
                logger.info("[DEBUG] Verifying created files...")
                debug_info["steps"].append("Verifying created files")

                expected_files = []
                if feature_type in ["agent", "agent_with_service"]:
                    expected_files.append(
                        lobster_root
                        / "lobster"
                        / "agents"
                        / f"{feature_name}_expert.py"
                    )
                    expected_files.append(
                        lobster_root
                        / "tests"
                        / "unit"
                        / "agents"
                        / f"test_{feature_name}_expert.py"
                    )
                    # Add state.py to expected files for state updates
                    expected_files.append(
                        lobster_root / "lobster" / "agents" / "state.py"
                    )

                if feature_type in ["service", "agent_with_service"]:
                    # SDK decides the category - we need to discover where it placed the files
                    # Scan all service categories to find the created service
                    service_categories = [
                        "analysis",
                        "data_access",
                        "data_management",
                        "metadata",
                        "ml",
                        "orchestration",
                        "quality",
                        "visualization",
                    ]
                    service_found = False
                    for category in service_categories:
                        service_path = (
                            lobster_root
                            / "lobster"
                            / "services"
                            / category
                            / f"{feature_name}_service.py"
                        )
                        if service_path.exists():
                            expected_files.append(service_path)
                            service_found = True
                            # Also check for corresponding test file
                            test_path = (
                                lobster_root
                                / "tests"
                                / "unit"
                                / "services"
                                / category
                                / f"test_{feature_name}_service.py"
                            )
                            if test_path.exists():
                                expected_files.append(test_path)
                            break

                    # If no service found in services/, check legacy tools/ location
                    if not service_found:
                        legacy_service = (
                            lobster_root
                            / "lobster"
                            / "tools"
                            / f"{feature_name}_service.py"
                        )
                        if legacy_service.exists():
                            expected_files.append(legacy_service)
                            legacy_test = (
                                lobster_root
                                / "tests"
                                / "unit"
                                / "tools"
                                / f"test_{feature_name}_service.py"
                            )
                            if legacy_test.exists():
                                expected_files.append(legacy_test)

                wiki_file = (
                    lobster_root
                    / "lobster"
                    / "wiki"
                    / f"{feature_name.replace('_', '-').lower()}.md"
                )
                expected_files.append(wiki_file)

                logger.info(f"[DEBUG] Expected {len(expected_files)} files:")
                for exp_file in expected_files:
                    logger.info(f"[DEBUG]   - {exp_file}")

                debug_info["expected_files"] = [str(f) for f in expected_files]

                # Perform filesystem scan as source of truth
                verified_files = []
                missing_files = []
                for file_path in expected_files:
                    if file_path.exists():
                        verified_files.append(str(file_path))
                        logger.info(f"[DEBUG] ✓ File exists: {file_path}")
                    else:
                        missing_files.append(str(file_path))
                        logger.warning(f"[DEBUG] ✗ File missing: {file_path}")

                debug_info["verified_files"] = verified_files
                debug_info["missing_files"] = missing_files
                debug_info["steps"].append(
                    f"Verification complete: {len(verified_files)}/{len(expected_files)} files"
                )

                logger.info(
                    f"[DEBUG] Verification complete: {len(verified_files)}/{len(expected_files)} files created"
                )

                # STEP: Detect required packages from created files
                detected_packages = {
                    "python_packages": [],
                    "system_packages": [],
                    "standard_lib": [],
                    "unknown": [],
                }
                package_status = {"python_packages": {}, "system_packages": {}}
                missing_python = []
                missing_system = []

                try:
                    logger.info(
                        "[DEBUG] Detecting required packages from created files..."
                    )
                    debug_info["steps"].append("Detecting required packages")

                    detected_packages = detect_packages_in_files(verified_files)
                    package_status = check_package_installation(detected_packages)

                    # Separate installed vs missing
                    missing_python = [
                        pkg
                        for pkg, installed in package_status["python_packages"].items()
                        if not installed
                    ]
                    missing_system = [
                        pkg
                        for pkg, installed in package_status["system_packages"].items()
                        if not installed
                    ]

                    debug_info["detected_packages"] = detected_packages
                    debug_info["package_status"] = package_status
                    debug_info["missing_python_packages"] = missing_python
                    debug_info["missing_system_packages"] = missing_system
                    debug_info["steps"].append(
                        f"Package detection complete: {len(missing_python)} pip, {len(missing_system)} brew missing"
                    )

                    logger.info(
                        f"[DEBUG] Package detection complete: {len(missing_python)} pip packages missing, {len(missing_system)} brew packages missing"
                    )

                except Exception as e:
                    logger.warning(f"[DEBUG] Package detection failed: {e}")
                    debug_info["errors"].append(f"Package detection failed: {e}")
                    # Non-blocking - continue with feature creation

                # Require ALL expected files to be present for success
                all_present = len(verified_files) == len(expected_files)
                error_msg = (
                    None
                    if all_present
                    else f"Missing files: {', '.join(missing_files)}"
                )

                return {
                    "success": all_present,
                    "created_files": verified_files,
                    "branch_name": branch_name,
                    "detected_packages": detected_packages,
                    "package_status": package_status,
                    "missing_python_packages": missing_python,
                    "missing_system_packages": missing_system,
                    "error": error_msg,
                    "sdk_output": "\n".join(sdk_output_lines),
                    "debug_info": debug_info,
                }
            except Exception as e:
                error_msg = f"Error during file verification: {e}"
                logger.error(f"[DEBUG] {error_msg}", exc_info=True)
                debug_info["errors"].append(error_msg)
                debug_info["steps"].append("File verification failed")
                return {
                    "success": False,
                    "created_files": (
                        verified_files if "verified_files" in locals() else []
                    ),
                    "error": error_msg,
                    "sdk_output": "\n".join(sdk_output_lines),
                    "debug_info": debug_info,
                }

        except Exception as e:
            error_msg = f"Unexpected error in Claude SDK creation: {e}"
            logger.error(f"[DEBUG] {error_msg}", exc_info=True)
            debug_info["errors"].append(error_msg)
            debug_info["steps"].append("Fatal error occurred")
            return {
                "success": False,
                "created_files": [],
                "error": str(e),
                "sdk_output": "",
                "debug_info": debug_info,
            }

    def generate_integration_instructions(
        feature_name: str, feature_type: str, branch_name: str
    ) -> str:
        """
        Generate comprehensive integration instructions for registry AND configuration.

        Args:
            feature_name: Name of the feature
            feature_type: Type of feature created
            branch_name: Git branch name where changes were made

        Returns:
            Formatted instructions with code snippets for both registry and config
        """
        if feature_type in ["agent", "agent_with_service"]:
            # Agent name for configuration (with _agent suffix)
            agent_config_name = f"{feature_name}_expert_agent"

            # Registry code snippet
            registry_code = f"""    '{feature_name}_expert_agent': AgentRegistryConfig(
        name='{feature_name}_expert_agent',
        display_name='{feature_name.replace('_', ' ').title()} Expert',
        description='[Add description of what this agent does]',
        factory_function='lobster.agents.{feature_name}_expert.{feature_name}_expert',
        handoff_tool_name='handoff_to_{feature_name}_expert_agent',
        handoff_tool_description='Hand off to the {feature_name} expert when [describe when to use].'
    ),"""

            # Configuration code snippets
            default_agents_code = (
                f"""        "{agent_config_name}",  # Add to the DEFAULT_AGENTS list"""
            )

            profile_config_code = f"""            "{agent_config_name}": "claude-4-sonnet",  # Add to each profile"""

            instructions = f"""
📝 **Manual Integration Required (2 Files)**

To make your new agent fully functional in Lobster, you need to update TWO configuration files:

## Step 1: Agent Registry (lobster/config/agent_registry.py)

1. Open: `lobster/config/agent_registry.py`

2. Add this entry to the `AGENT_REGISTRY` dictionary (around line 100):

```python
{registry_code}
```

3. Update the description and handoff_tool_description with specifics about your agent

## Step 2: Agent Configuration (lobster/config/agent_config.py)

**⚠️ CRITICAL: Without this step, you'll get a KeyError at runtime!**

1. Open: `lobster/config/agent_config.py`

2. Add to `DEFAULT_AGENTS` list (around line 151):

```python
{default_agents_code}
```

3. Add to ALL THREE testing profiles in `TESTING_PROFILES`:

   **a) In "development" profile (around line 180):**
```python
{profile_config_code}
```

   **b) In "production" profile (around line 197):**
```python
{profile_config_code}
```

   **c) In "godmode" profile (around line 213):**
```python
{profile_config_code}
```

## Step 3: Verify & Restart

1. Save both files
2. Restart Lobster: `lobster chat`
3. The agent will now be available via the supervisor without KeyError!

**Why both files are needed:**
- `agent_registry.py` → Registers the agent in the system (handoff tools, routing)
- `agent_config.py` → Configures LLM model for the agent (prevents KeyError)

## Step 4: Git Workflow (Commit & Merge)

Your changes are on branch: `{branch_name}`

1. **Verify tests pass:**
   ```bash
   make test
   make lint
   make type-check
   ```

2. **Commit your configuration changes:**
   ```bash
   git add lobster/config/agent_registry.py lobster/config/agent_config.py
   git commit -m "feat: integrate {feature_name} agent into registry and config"
   ```

3. **Push branch and create PR:**
   ```bash
   git push origin {branch_name}
   ```
   Then create a Pull Request on GitHub for review.

4. **After PR approval, merge to main:**
   - Merge via GitHub UI
   - Delete feature branch after merge

**Branch Safety:**
- All feature code is on `{branch_name}`
- Main branch remains unchanged until PR is merged
- Tests must pass before merging
"""
        else:
            # Service only
            instructions = f"""
📝 **Service Integration**

Your service is ready to use! Services don't require registry or configuration updates.

To use in an existing agent:
1. Import: `from lobster.tools.{feature_name}_service import {feature_name.title().replace('_', '')}Service`
2. Instantiate: `service = {feature_name.title().replace('_', '')}Service()`
3. Call: `result_adata, stats = service.method_name(adata, parameters)`

The service follows the stateless pattern and can be used by any agent.
"""

        return instructions.strip()

    # -------------------------
    # TOOLS
    # -------------------------

    @tool
    def create_new_feature(
        feature_type: str, feature_name: str, requirements: str
    ) -> str:
        """
        Create a new Lobster feature using Claude Code SDK.

        This tool spawns Claude Code to generate new agents, services, tests,
        and documentation following Lobster's architectural patterns.

        Args:
            feature_type: Type of feature to create. Options:
                         - "agent": Create only an agent (with tools, tests, wiki)
                         - "service": Create only a service (with tests, wiki)
                         - "agent_with_service": Create both agent and service
            feature_name: Name for the feature (lowercase, underscores, e.g., "spatial_transcriptomics")
            requirements: Detailed requirements and specifications for the feature.
                         Include: what it does, what data it processes, what tools it needs,
                         what analysis methods to use, any specific parameters.

        Returns:
            str: Summary of created files and next steps

        Example:
            create_new_feature(
                feature_type="agent_with_service",
                feature_name="spatial_transcriptomics",
                requirements="Create an agent for spatial transcriptomics analysis.
                            Should handle Visium and Slide-seq data.
                            Tools needed: spatial clustering, neighborhood analysis,
                            spatial gene expression patterns."
            )
        """
        try:
            logger.info(f"Creating new feature: {feature_type} - {feature_name}")

            # 1. Validate feature type
            valid_types = ["agent", "service", "agent_with_service"]
            if feature_type not in valid_types:
                return f"""❌ Invalid feature type: '{feature_type}'

Valid options:
- "agent": Create only an agent (with tools, tests, wiki)
- "service": Create only a service (with tests, wiki)
- "agent_with_service": Create both agent and service (recommended)

Please retry with a valid feature type."""

            # 2. Validate feature name
            is_valid, error_msg = validate_feature_name(feature_name)
            if not is_valid:
                return f"""❌ Invalid feature name: {error_msg}

**Naming Rules:**
- Start with lowercase letter
- Use only lowercase letters, numbers, and underscores
- Cannot end with underscore
- Examples: "spatial_transcriptomics", "metabolomics", "variant_calling"

Please retry with a valid feature name."""

            # 3. Check for existing files
            existing_files = check_existing_files(feature_name)
            if existing_files:
                files_list = "\n".join(f"  - {f}" for f in existing_files)
                return f"""⚠️ Feature '{feature_name}' already has existing files:

{files_list}

Please either:
1. Choose a different feature name
2. Manually remove the existing files first
3. Use a more specific name (e.g., "{feature_name}_advanced")"""

            # 4. Validate requirements
            if not requirements or len(requirements.strip()) < 20:
                return """❌ Requirements are too brief.

Please provide detailed requirements including:
- What the feature does
- What type of data it processes
- What analysis methods it should use
- What tools/functions it needs
- Any specific parameters or options
- Example use cases

Minimum 20 characters required."""

            # 5. Create feature using Claude Code SDK
            logger.info(f"Spawning Claude Code SDK for {feature_type}: {feature_name}")

            # Run async SDK call safely (avoids event loop conflicts)
            def _run_coro(coro):
                """Run coroutine in a new event loop (thread-safe)."""
                return asyncio.run(coro)

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                result = executor.submit(
                    _run_coro,
                    create_with_claude_sdk(
                        feature_type, feature_name, requirements, debug=True
                    ),
                ).result()

            # 6. Check results
            if not result["success"]:
                error = result.get("error", "Unknown error")
                debug_info = result.get("debug_info", {})
                sdk_output = result.get("sdk_output", "")

                logger.error(f"Feature creation failed: {error}")

                # Format debug information
                debug_section = ""
                if debug_info:
                    debug_section = "\n\n**🔍 Debug Information:**\n"

                    # Show steps taken
                    if debug_info.get("steps"):
                        debug_section += "\n**Steps completed:**\n"
                        for step in debug_info["steps"]:
                            debug_section += f"  ✓ {step}\n"

                    # Show errors
                    if debug_info.get("errors"):
                        debug_section += "\n**Errors encountered:**\n"
                        for err in debug_info["errors"]:
                            debug_section += f"  ✗ {err}\n"

                    # Show SDK configuration
                    if debug_info.get("sdk_config"):
                        debug_section += "\n**SDK Configuration:**\n"
                        for key, value in debug_info["sdk_config"].items():
                            debug_section += f"  - {key}: {value}\n"

                    # Show message stats
                    if debug_info.get("total_messages"):
                        debug_section += f"\n**SDK Messages:** {debug_info['total_messages']} received\n"

                    # Show detected files
                    if debug_info.get("detected_files_count") is not None:
                        debug_section += f"**Files detected:** {debug_info['detected_files_count']}\n"

                    # Show expected vs verified
                    if debug_info.get("expected_files"):
                        debug_section += f"\n**Expected files:** {len(debug_info['expected_files'])}\n"
                        if debug_info.get("verified_files"):
                            debug_section += f"**Verified files:** {len(debug_info['verified_files'])}\n"
                        if debug_info.get("missing_files"):
                            debug_section += "\n**Missing files:**\n"
                            for missing in debug_info["missing_files"]:
                                debug_section += f"  ✗ {missing}\n"

                # Show SDK output if available (limited)
                sdk_output_section = ""
                if sdk_output:
                    output_preview = (
                        sdk_output[:500] if len(sdk_output) > 500 else sdk_output
                    )
                    sdk_output_section = f"\n\n**SDK Output (preview):**\n```\n{output_preview}\n{'...' if len(sdk_output) > 500 else ''}\n```"

                return f"""❌ Feature creation failed

**Error:** {error}

**Troubleshooting:**
1. Ensure claude-agent-sdk is installed: `pip install claude-agent-sdk`
2. Check that you have write permissions in the Lobster directory
3. Verify the requirements are clear and specific
4. Try again with more detailed requirements{debug_section}{sdk_output_section}

**Full logs:** Check the Lobster logs for complete debug information

If the error persists, please report it to the Lobster team with the debug information above."""

            # 7. Success! Format response
            created_files = result["created_files"]
            creation_results["created_files"] = created_files

            # Helper function for robust path grouping
            def in_dir(path_str: str, dirname: str) -> bool:
                """Check if path contains directory using Path.parts (platform-independent)."""
                return dirname in Path(path_str).parts

            # Group files by type using Path.parts for robustness
            agents = [
                f
                for f in created_files
                if in_dir(f, "agents")
                and "test_" not in Path(f).name
                and "state.py" not in f
            ]
            services = [
                f
                for f in created_files
                if in_dir(f, "tools") and "test_" not in Path(f).name
            ]
            tests = [f for f in created_files if in_dir(f, "tests")]
            wiki = [f for f in created_files if in_dir(f, "wiki")]
            state = [f for f in created_files if Path(f).name == "state.py"]

            response = f"""✅ Successfully created new {feature_type} for Lobster!

🎉 **Feature Name:** {feature_name}

📁 **Files Created ({len(created_files)}):**
"""

            if agents:
                response += "\n**Agents:**\n"
                for f in agents:
                    response += f"  ✓ {Path(f).relative_to(lobster_root)}\n"

            if services:
                response += "\n**Services:**\n"
                for f in services:
                    response += f"  ✓ {Path(f).relative_to(lobster_root)}\n"

            if state:
                response += "\n**State:**\n"
                for f in state:
                    response += f"  ✓ {Path(f).relative_to(lobster_root)} (updated)\n"

            if tests:
                response += "\n**Tests:**\n"
                for f in tests:
                    response += f"  ✓ {Path(f).relative_to(lobster_root)}\n"

            if wiki:
                response += "\n**Documentation:**\n"
                for f in wiki:
                    response += f"  ✓ {Path(f).relative_to(lobster_root)}\n"

            # Add branch information
            branch_name = result.get("branch_name", "unknown")
            response += "\n\n🌿 **Git Branch:**\n"
            response += f"  ✓ Created and switched to: `{branch_name}`\n"
            response += "  - All changes are on this feature branch\n"
            response += "  - Main branch remains unchanged\n"

            # Add package warnings if any missing
            missing_python = result.get("missing_python_packages", [])
            missing_system = result.get("missing_system_packages", [])

            if missing_python or missing_system:
                detected_packages = result.get("detected_packages", {})
                package_status = result.get("package_status", {})
                warnings = format_package_warnings(package_status, detected_packages)
                if warnings:
                    response += f"\n\n{warnings}"

            # 8. Add integration instructions (registry + configuration)
            integration_instructions = generate_integration_instructions(
                feature_name, feature_type, branch_name
            )
            response += f"\n\n{integration_instructions}"

            # 9. Add next steps
            step_number = 1
            response += "\n\n🔧 **Next Steps:**\n\n"

            # Step 1: Install dependencies if needed
            if missing_python or missing_system:
                response += f"{step_number}. **Install Missing Dependencies** ⚠️\n"
                if missing_python:
                    response += "   Python packages (pip):\n"
                    response += "   ```bash\n"
                    response += f"   pip install {' '.join(missing_python)}\n"
                    response += "   ```\n"
                if missing_system:
                    response += "   System packages (brew):\n"
                    response += "   ```bash\n"
                    response += f"   brew install {' '.join(missing_system)}\n"
                    response += "   ```\n"
                response += "\n"
                step_number += 1

            # Step N: Review Code
            response += f"{step_number}. **Review Generated Code**\n"
            response += "   - Check that the implementation matches your requirements\n"
            response += "   - Verify naming conventions are followed\n"
            response += "   - Ensure docstrings are complete\n\n"
            step_number += 1

            # Step N+1: Run Tests
            response += f"{step_number}. **Run Tests**\n"
            response += "   ```bash\n"
            response += "   make test\n"
            response += "   ```\n\n"
            step_number += 1

            # Step N+2: Check Quality
            response += f"{step_number}. **Check Code Quality**\n"
            response += "   ```bash\n"
            response += "   make lint\n"
            response += "   make type-check\n"
            response += "   ```\n\n"
            step_number += 1

            # Step N+3: Update Registry
            response += f"{step_number}. **Update Registry & Configuration** (if agent was created)\n"
            response += "   - Follow the instructions above to add to agent_registry.py AND agent_config.py\n"
            response += "   - BOTH files must be updated to avoid KeyError\n\n"
            step_number += 1

            # Step N+4: Restart Lobster
            response += f"{step_number}. **Restart Lobster**\n"
            response += "   ```bash\n"
            response += "   lobster chat\n"
            response += "   ```\n\n"
            step_number += 1

            # Step N+5: Test Feature
            response += f"{step_number}. **Test Your Feature**\n"
            response += "   - Interact with the new agent through the supervisor\n"
            response += "   - Try different workflows and edge cases\n\n"

            # Footer
            response += f"📚 **Documentation:** See lobster/wiki/{feature_name.replace('_', '-').lower()}.md for usage examples\n\n"
            response += f"The new {feature_type} is ready to use! 🚀\n"

            # Store results
            creation_results["details"]["feature_creation"] = response
            creation_results["summary"] = f"Created {feature_type}: {feature_name}"

            # Log operation
            data_manager.log_tool_usage(
                tool_name="create_new_feature",
                parameters={
                    "feature_type": feature_type,
                    "feature_name": feature_name,
                    "files_created": len(created_files),
                },
                description=f"Created new {feature_type}: {feature_name} with {len(created_files)} files",
            )

            return response

        except Exception as e:
            logger.error(f"Unexpected error in create_new_feature: {e}", exc_info=True)
            return f"""❌ Unexpected error during feature creation

**Error:** {str(e)}

Please report this issue to the Lobster team with the error details above."""

    @tool
    def list_existing_patterns() -> str:
        """
        List existing Lobster agents and services as examples.

        Returns:
            str: List of existing patterns that can be used as templates
        """
        try:
            agents_dir = lobster_root / "lobster" / "agents"
            services_dir = lobster_root / "lobster" / "tools"

            # List agents (excluding non-agent files)
            agent_files = []
            if agents_dir.exists():
                for f in agents_dir.glob("*_expert.py"):
                    agent_files.append(f.stem)

            # List services (excluding non-service files)
            service_files = []
            if services_dir.exists():
                for f in services_dir.glob("*_service.py"):
                    service_files.append(f.stem)

            response = """📚 **Existing Lobster Patterns**

These agents and services can serve as templates for new features:

**Agents:**
"""
            for agent in sorted(agent_files):
                agent_name = agent.replace("_expert", "")
                response += f"  • {agent_name} - lobster/agents/{agent}.py\n"

            response += "\n**Services:**\n"
            for service in sorted(service_files):
                service_name = service.replace("_service", "")
                response += f"  • {service_name} - lobster/tools/{service}.py\n"

            response += """

💡 **Recommended Templates:**

For new agents, use as template:
  • machine_learning_expert - Comprehensive example with multiple tools
  • singlecell_expert - Complex analysis workflows
  • bulk_rnaseq_expert - Service integration example

For new services, use as template:
  • preprocessing_service - Data transformation pattern
  • quality_service - Quality assessment pattern

📖 **Learn More:** See lobster/agents/CLAUDE.md for detailed patterns and examples
"""

            return response

        except Exception as e:
            logger.error(f"Error listing patterns: {e}")
            return f"Error listing existing patterns: {str(e)}"

    @tool
    def validate_feature_name_tool(feature_name: str) -> str:
        """
        Validate that a feature name follows Lobster conventions.

        Args:
            feature_name: Name to validate

        Returns:
            str: Validation result
        """
        is_valid, error_msg = validate_feature_name(feature_name)

        if is_valid:
            # Check for existing files
            existing = check_existing_files(feature_name)

            if existing:
                files_list = "\n".join(
                    f"  - {Path(f).relative_to(lobster_root)}" for f in existing
                )
                return f"""⚠️ Feature name '{feature_name}' is valid but has existing files:

{files_list}

Consider using a more specific name like:
  • {feature_name}_advanced
  • {feature_name}_enhanced
  • custom_{feature_name}"""
            else:
                return f"""✅ Feature name '{feature_name}' is valid and available!

No existing files found. You can proceed with creation.

Suggested feature types:
  • "agent" - If you only need analysis tools and workflows
  • "service" - If you only need reusable analysis functions
  • "agent_with_service" - For complete features (recommended)"""
        else:
            return f"""❌ Invalid feature name: {error_msg}

**Naming Rules:**
- Start with lowercase letter
- Use only lowercase letters, numbers, and underscores
- Cannot end with underscore
- Cannot use reserved names: supervisor, data, base, core, config

**Valid Examples:**
  • spatial_transcriptomics
  • metabolomics_analysis
  • variant_calling
  • chromatin_accessibility

**Invalid Examples:**
  • SpatialTranscriptomics (uppercase)
  • spatial-transcriptomics (hyphens)
  • spatial_transcriptomics_ (trailing underscore)
  • 123_analysis (starts with number)"""

    @tool
    def research_feature_requirements(
        feature_description: str,
        search_focus: str = "GitHub repositories, packages, best practices",
    ) -> str:
        """
        Research feature requirements using internet search (Linkup SDK).

        This tool searches the internet for:
        - Existing GitHub repositories with similar functionality
        - Python packages that can be reused (prioritize biopython, scanpy, etc.)
        - Best practices and implementation patterns
        - Potential challenges and solutions

        ALWAYS use this tool BEFORE creating any new feature to:
        1. Avoid reinventing the wheel
        2. Identify reusable packages
        3. Learn from existing implementations
        4. Understand potential pitfalls

        Args:
            feature_description: Description of the feature to research (from user request)
            search_focus: Specific areas to focus research on

        Returns:
            str: Formatted research findings with repos, packages, and recommendations

        Example:
            research_feature_requirements(
                feature_description="spatial transcriptomics analysis for Visium data",
                search_focus="squidpy, scanpy integration, spatial clustering methods"
            )
        """
        try:
            logger.info(f"Researching requirements: {feature_description}")

            # Call helper function
            result = research_with_linkup(feature_description, search_focus)

            # Format response
            if result.get("error"):
                return f"""⚠️ Research Phase Skipped

**Reason:** {result['error']}

You can proceed with feature creation, but consider:
- Manually searching for existing packages
- Checking GitHub for similar implementations
- Reviewing Lobster's existing services for reuse

Proceeding without research..."""

            response = f"""🔍 **Research Findings for: {feature_description}**

---

## 📚 GitHub Repositories ({len(result['github_repos'])} found)

"""

            if result["github_repos"]:
                for i, repo in enumerate(result["github_repos"], 1):
                    response += f"{i}. **{repo['name']}**\n"
                    response += f"   - URL: {repo['url']}\n"
                    response += f"   - Description: {repo['snippet']}\n\n"
            else:
                response += "*No relevant repositories found*\n\n"

            response += f"""## 📦 Relevant Python Packages ({len(result['packages'])} identified)

"""

            if result["packages"]:
                response += "Packages that can be reused:\n"
                for pkg in result["packages"]:
                    response += f"  • {pkg}\n"
            else:
                response += "*No specific packages identified*\n"

            response += "\n## 💡 Best Practices\n\n"

            if result["best_practices"]:
                for practice in result["best_practices"]:
                    response += f"{practice}\n\n"
            else:
                response += "*No specific best practices found*\n\n"

            response += f"""## 📝 Summary

{result['summary']}

---

**Next Steps:**
1. Review the identified repositories and packages
2. Check if any packages are already installed in Lobster
3. Consider reusing existing Lobster services
4. Proceed with feature creation using these insights
"""

            # Log research
            data_manager.log_tool_usage(
                tool_name="research_feature_requirements",
                parameters={
                    "feature_description": feature_description,
                    "search_focus": search_focus,
                    "repos_found": len(result["github_repos"]),
                    "packages_found": len(result["packages"]),
                },
                description=f"Researched requirements for: {feature_description}",
            )

            return response

        except Exception as e:
            logger.error(f"Error in research tool: {e}", exc_info=True)
            return f"""❌ Research failed: {str(e)}

You can proceed with feature creation without research, but consider:
- Manually searching for existing solutions
- Checking Lobster's existing services for reuse
- Being extra thorough in requirements gathering
"""

    @tool
    def create_feature_summary() -> str:
        """
        Create a summary of all feature creation activities.

        Returns:
            str: Summary of features created in this session
        """
        if not creation_results["details"]:
            return """📊 **Feature Creation Summary**

No features have been created yet in this session.

Use `create_new_feature()` to create new agents, services, or complete features for Lobster.

**Available Feature Types:**
  • agent - Create agent with tools and workflows
  • service - Create reusable analysis service
  • agent_with_service - Create complete feature (recommended)

Need help? Use `list_existing_patterns()` to see examples."""

        summary = "# Feature Creation Summary\n\n"

        if creation_results["summary"]:
            summary += f"**Overall:** {creation_results['summary']}\n\n"

        if creation_results["created_files"]:
            summary += (
                f"**Files Created ({len(creation_results['created_files'])}):**\n"
            )
            for f in creation_results["created_files"]:
                rel_path = Path(f).relative_to(lobster_root)
                summary += f"  ✓ {rel_path}\n"
            summary += "\n"

        for step, details in creation_results["details"].items():
            summary += f"## {step.replace('_', ' ').title()}\n\n"
            summary += f"{details}\n\n"

        return summary

    # -------------------------
    # TOOL REGISTRY
    # -------------------------
    base_tools = [
        research_feature_requirements,  # NEW - Research first before creating
        create_new_feature,
        list_existing_patterns,
        validate_feature_name_tool,
        create_feature_summary,
    ]

    tools = base_tools

    # -------------------------
    # SYSTEM PROMPT
    # -------------------------
    system_prompt = f"""
You are the Custom Feature Agent for Lobster, responsible for creating new agents, services, providers, adapters, tools, tests, and documentation using the Claude Code SDK.

<Role>
You extend Lobster's capabilities by generating new features that follow established architectural patterns from the Unified Agent Creation Template. You use Claude Code SDK to create production-ready code, tests, and documentation for complex agents with new modalities, providers, and adapters.

**CRITICAL: You ONLY create features specifically requested by the supervisor. You report results back to the supervisor, never directly to users.**
</Role>

<Communication Flow>
**USER → SUPERVISOR → YOU → SUPERVISOR → USER**
- You receive feature creation requests from the supervisor
- You spawn Claude Code SDK to generate files
- You report results back to the supervisor
- The supervisor communicates with the user
</Communication Flow>

<Task>
You create new Lobster features following these steps:

0. **Research First (MANDATORY)**
   - Use research_feature_requirements tool to search internet
   - Search for existing GitHub repositories with similar functionality
   - Identify Python packages that can be reused (biopython, scanpy, etc.)
   - Find best practices and implementation patterns
   - Include 'site:github.com' in searches for code discovery
   - Summarize findings before proceeding to validation

1. **Decide Components (Component Decision Tree)**
   - New Data Type/Modality? → Schema + Adapter + Services + agent?
   - Existing Modality + New Analysis? → Services Only
   - External Data Source? → Provider + Adapter
   - Verify feature name follows conventions
   - Check for naming conflicts

2. **Spawn Claude Code SDK with Unified Template**
   - Configure SDK with proper context (CLAUDE.md + Unified Template from project root)
   - Provide detailed prompt with:
     * Requirements AND research findings
     * Component decision (schema/adapter/provider/services/agent/registry)
     * Execution order (schema → adapter → provider → services → agent → registry)
     * Code templates from Unified Template for each component
   - Monitor file creation progress
   - Capture results and errors

3. **Verify Results**
   - Check all expected files were created (execution order)
   - Validate file structure and naming
   - Ensure patterns were followed (3-tuple, IR, provider, adapter)
   - Verify registry entry

4. **Provide Instructions**
   - Generate integration instructions (registry + configuration)
   - List all created files with full paths
   - Provide next steps for testing (unit, integration, end-to-end)
   - Suggest verification commands

</Task>

<UNIFIED AGENT CREATION TEMPLATE - COMPREHENSIVE PATTERNS>

## COMPONENT DECISION TREE

| Scenario | Schema | Adapter | Provider | Services | Service Category | Tools | Registry |
|----------|--------|---------|----------|----------|------------------|-------|----------|
| New modality | ✅ | ✅ | Optional | ✅ | analysis/ or quality/ | ✅ | ✅ |
| Existing modality + new analysis | ❌ | ❌ | ❌ | ✅ | analysis/ | ✅ | ✅ |
| New data source | Optional | ✅ | ✅ | ✅ | data_access/ | ✅ | ✅ |
| QC/preprocessing pipeline | ❌ | ❌ | ❌ | ✅ | quality/ | ✅ | ✅ |
| Visualization feature | ❌ | ❌ | ❌ | ✅ | visualization/ | ✅ | ✅ |
| ML/embedding feature | ❌ | ❌ | ❌ | ✅ | ml/ | ✅ | ✅ |
| Metadata operations | ❌ | ❌ | ❌ | ✅ | metadata/ | ✅ | Optional |

## SERVICE CATEGORIES EXPLAINED
- **analysis/** - Statistical analysis, clustering, DE, enrichment (most common)
- **quality/** - QC metrics, preprocessing, filtering, normalization
- **data_access/** - External APIs, downloads, file retrieval
- **data_management/** - Modality CRUD, concatenation, workspace ops
- **metadata/** - Standardization, validation, ID mapping
- **ml/** - Machine learning, embeddings, predictions
- **visualization/** - Plotting, charts, figures
- **orchestration/** - Multi-step workflows, coordination

## EXECUTION ORDER (CRITICAL)
1. Schema Definition (if new modality)
2. Adapter Implementation (if new format/modality)
3. Provider Implementation (if external data source)
4. Service Classes (analysis logic)
5. Agent Factory Function
6. Registry Entry
7. Auto-Integration (no manual graph editing)

## SERVICE PATTERN (3-Tuple Return - MANDATORY)

ALL analysis services MUST return:
```python
def analyze(self, adata: AnnData, **params) -> Tuple[AnnData, Dict[str, Any], AnalysisStep]:
    processed_adata = adata.copy()
    # ... processing ...
    stats_dict = {{...}}  # Human-readable summary
    ir = self._create_ir(**params)  # Intermediate Representation
    return processed_adata, stats_dict, ir
```

**Components:**
- `processed_adata`: Modified AnnData with results in .obs, .var, .obsm, .uns
- `stats_dict`: Concise, human-readable summary
- `ir`: AnalysisStep for provenance + notebook export (MANDATORY)

**Template Location:** `lobster/services/{{CATEGORY}}/{{{{SERVICE_MODULE}}}}.py`

**Service Categories** (choose based on feature purpose):
- `analysis/` - clustering, differential expression, statistical analysis, enrichment
- `data_access/` - external APIs, databases, file retrieval, downloads
- `data_management/` - modality CRUD, concatenation, workspace operations
- `metadata/` - standardization, validation, mapping, harmonization
- `ml/` - machine learning, embeddings, predictions, classification
- `orchestration/` - workflow coordination, multi-step processes
- `quality/` - QC, preprocessing, filtering, normalization
- `visualization/` - plots, charts, figures, visualizations

Key methods:
- `def analyze(self, adata, **params) -> Tuple[AnnData, Dict, AnalysisStep]`
- `def _create_ir(self, **params) -> AnalysisStep` (returns IR with code_template, parameter_schema, imports)

## TOOL PATTERN (Agent Tools - MANDATORY)

```python
@tool
def analyze_modality(modality_name: str, **params) -> str:
    # 1. Validate modality existence
    if modality_name not in data_manager.list_modalities():
        available = ", ".join(data_manager.list_modalities())
        raise ModalityNotFoundError(f"Modality '{{{{modality_name}}}}' not found. Available: {{{{available}}}}")

    # 2. Get modality data
    adata = data_manager.get_modality(modality_name)
    logger.info(f"Processing {{{{adata.n_obs}}}} samples, {{{{adata.n_vars}}}} features")

    # 3. Delegate to stateless service (3-tuple pattern)
    result_adata, stats, ir = service.analyze(adata, **params)

    # 4. Store result with descriptive naming
    new_modality_name = f"{{{{modality_name}}}}_analyzed"
    data_manager.modalities[new_modality_name] = result_adata
    logger.info(f"Stored result as '{{{{new_modality_name}}}}'")

    # 5. Log tool usage with IR (MANDATORY)
    data_manager.log_tool_usage(
        tool_name="analyze_modality",
        parameters={{"modality_name": modality_name, **params}},
        description=f"Analysis on {{{{modality_name}}}}",
        ir=ir  # IR is mandatory for reproducibility
    )

    # 6. Format response
    return f"Analysis complete.\\nInput: {{{{modality_name}}}} ({{{{adata.n_obs}}}} samples)\\nOutput: {{{{new_modality_name}}}}\\nStats: {{{{stats}}}}"
```

## PROVENANCE (W3C-PROV - MANDATORY)

Every service must create `AnalysisStep` IR with:

```python
def _create_ir(self, method='median', scaling='pareto') -> AnalysisStep:
    # Parameter schema for Papermill injection
    parameter_schema = {{
        "method": ParameterSpec(
            param_type="str",
            papermill_injectable=True,
            default_value=method,
            required=True,
            validation_rule="in ['median', 'mean', 'total_sum']",
            description="Normalization method"
        ),
        "scaling": ParameterSpec(
            param_type="str",
            papermill_injectable=True,
            default_value=scaling,
            required=False,
            validation_rule="in ['pareto', 'standard', 'minmax']",
            description="Scaling method"
        ),
    }}

    # Jinja2 code template (ONLY STANDARD LIBRARIES)
    code_template =# Normalization and Scaling
\

    return AnalysisStep(
        operation="sklearn.preprocessing.normalize",
        tool_name="normalize_and_scale",
        description="## Metabolomics Normalization\\n\\nNormalize and scale metabolomics data.\\n\\n**Parameters:**\\n- `method`: Normalization method\\n- `scaling`: Scaling method",
        library="sklearn.preprocessing",
        code_template=code_template,
        imports=["import sklearn.preprocessing as preprocessing", "import numpy as np"],
        parameters={{"method": method, "scaling": scaling}},
        parameter_schema=parameter_schema,
        input_entities=["adata"],
        output_entities=["adata"],
        execution_context={{"library_version": "sklearn", "method": "normalize_and_scale"}},
        validates_on_export=True,
        exportable=True
    )
```

## PROVIDER PATTERN (BaseProvider - REQUIRED for External APIs)

**File:** `lobster/tools/providers/{{{{PROVIDER_MODULE}}}}.py`

```python
from lobster.tools.providers.base_provider import (
    BasePublicationProvider,
    PublicationSource,
    DatasetType,
    ProviderCapability,
)
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.exceptions import ProviderError

class NewProvider(BasePublicationProvider):
    def __init__(self, data_manager: DataManagerV2, api_key: Optional[str] = None,
                 base_url: str = "https://api.example.com", rate_limit: int = 5):
        self.data_manager = data_manager
        self.api_key = api_key or os.environ.get("API_KEY")
        self.base_url = base_url.rstrip('/')
        self.rate_limit = rate_limit

        # Setup session with retry logic
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount('https://', HTTPAdapter(max_retries=retries))

        if self.api_key:
            self.session.headers.update({{"Authorization": f"Bearer {{{{self.api_key}}}}"}})

    @property
    def source(self) -> PublicationSource:
        return PublicationSource.CUSTOM_SOURCE

    @property
    def supported_dataset_types(self) -> List[DatasetType]:
        return [DatasetType.TRANSCRIPTOMICS, DatasetType.PROTEOMICS]

    @property
    def priority(self) -> int:
        return 50  # 10=high, 50=medium, 100=low

    def get_supported_capabilities(self) -> Dict[str, bool]:
        return {{
            ProviderCapability.SEARCH_LITERATURE: True,
            ProviderCapability.DISCOVER_DATASETS: True,
            ProviderCapability.EXTRACT_METADATA: True,
            ProviderCapability.DOWNLOAD_FILES: False,
            ProviderCapability.PARSE_METHODS: False,
        }}

    def search_publications(self, query: str, max_results: int = 5, filters: Optional[Dict] = None, **kwargs) -> str:
        # Implementation with error handling
        try:
            response = self.session.get(f"{{{{self.base_url}}}}/search", params={{"q": query, "limit": max_results}}, timeout=30)
            response.raise_for_status()
            data = response.json()
            # Parse and format results
            return formatted_results
        except requests.exceptions.RequestException as e:
            logger.error(f"Search failed: {{{{e}}}}")
            raise ProviderError(f"Search failed: {{{{str(e)}}}}")
```

## IDOWNLOADSERVICE PATTERN (Queue-Based Downloads)

**File:** `lobster/services/data_access/{{{{SERVICE_MODULE}}}}.py`

For providers that support dataset downloads, implement the IDownloadService interface:

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import anndata as ad
from lobster.core.analysis_ir import AnalysisStep
from lobster.core.download_queue import DownloadQueueEntry

class IDownloadService(ABC):
    @abstractmethod
    def supports_database(self, database: str) -> bool:
        \"\"\"Check if this service handles the given database type.\"\"\"
        pass

    @abstractmethod
    def download_dataset(
        self,
        queue_entry: DownloadQueueEntry,
        strategy_override: Optional[str] = None
    ) -> Tuple[ad.AnnData, Dict[str, Any], AnalysisStep]:
        \"\"\"Download and return (adata, stats, ir) tuple.\"\"\"
        pass

    @abstractmethod
    def validate_strategy_params(self, params: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        \"\"\"Validate download strategy parameters.\"\"\"
        pass

    @abstractmethod
    def get_supported_strategies(self) -> List[str]:
        \"\"\"Return list of supported download strategies.\"\"\"
        pass

# Example implementation:
class GEODownloadService(IDownloadService):
    def __init__(self, data_manager: DataManagerV2):
        self.data_manager = data_manager
        self.geo_service = GEOService(data_manager)  # Composition pattern

    def supports_database(self, database: str) -> bool:
        return database.lower() in ["geo", "gse", "gds"]

    def download_dataset(self, queue_entry, strategy_override=None) -> Tuple[ad.AnnData, Dict, AnalysisStep]:
        # Download via GEOService, return 3-tuple
        result = self.geo_service.download(queue_entry.accession, strategy=strategy_override)
        adata = self.data_manager.get_modality(result["modality_name"])
        stats = {{"accession": queue_entry.accession, "n_obs": adata.n_obs, "n_vars": adata.n_vars}}
        ir = self._create_ir(queue_entry.accession, strategy_override)
        return adata, stats, ir
```

**Usage with DownloadOrchestrator:**
```python
orchestrator = DownloadOrchestrator(data_manager)
orchestrator.register_service(GEODownloadService(data_manager))
modality_name, stats = orchestrator.execute_download(queue_entry_id)
```

## MODALITYMANAGEMENTSERVICE PATTERN (Centralized CRUD)

**File:** `lobster/services/data_management/modality_management_service.py`

Centralized service for modality CRUD operations with W3C-PROV compliant provenance tracking:

```python
from typing import Any, Dict, List, Optional, Tuple
import anndata as ad
from lobster.core.analysis_ir import AnalysisStep
from lobster.core.data_manager_v2 import DataManagerV2

class ModalityManagementService:
    \"\"\"Service for centralized modality CRUD operations with provenance tracking.\"\"\"

    def __init__(self, data_manager: DataManagerV2):
        self.data_manager = data_manager

    def list_modalities(
        self, filter_pattern: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], AnalysisStep]:
        \"\"\"List all available modalities with optional glob filtering.\"\"\"
        # Returns: (modality_info_list, stats, ir)
        pass

    def get_modality_info(
        self, modality_name: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any], AnalysisStep]:
        \"\"\"Get detailed info (shape, layers, obsm/varm/uns keys, quality metrics).\"\"\"
        # Returns: (info_dict, stats, ir)
        pass

    def remove_modality(
        self, modality_name: str
    ) -> Tuple[bool, Dict[str, Any], AnalysisStep]:
        \"\"\"Remove modality from DataManagerV2.\"\"\"
        # Returns: (success, stats, ir)
        pass

    def validate_compatibility(
        self, modality_names: List[str]
    ) -> Tuple[Dict[str, Any], Dict[str, Any], AnalysisStep]:
        \"\"\"Validate obs/var overlap, batch effects, recommend integration strategy.\"\"\"
        # Returns: (validation_result, stats, ir)
        pass

    def load_modality(
        self,
        modality_name: str,
        file_path: str,
        adapter: str,
        dataset_type: str = "custom",
        validate: bool = True,
        **kwargs
    ) -> Tuple[ad.AnnData, Dict[str, Any], AnalysisStep]:
        \"\"\"Load data file via adapter system with schema validation.\"\"\"
        # Returns: (adata, stats, ir)
        pass

    # Each method includes _create_*_ir() helper for provenance
    def _create_list_ir(self, filter_pattern: Optional[str]) -> AnalysisStep: ...
    def _create_get_info_ir(self, modality_name: str) -> AnalysisStep: ...
    def _create_remove_ir(self, modality_name: str) -> AnalysisStep: ...
    def _create_validate_ir(self, modality_names: List[str]) -> AnalysisStep: ...
    def _create_load_ir(self, modality_name, file_path, adapter, dataset_type) -> AnalysisStep: ...
```

**Usage in Agent Tools:**
```python
# In data_expert agent
service = ModalityManagementService(data_manager)

@tool
def list_modalities(filter_pattern: Optional[str] = None) -> str:
    modality_info, stats, ir = service.list_modalities(filter_pattern)
    data_manager.log_tool_usage("list_modalities", {{"filter": filter_pattern}}, f"Listed {{len(modality_info)}} modalities", ir=ir)
    return format_modality_list(modality_info, stats)

@tool
def load_local_file(modality_name: str, file_path: str, adapter: str) -> str:
    adata, stats, ir = service.load_modality(modality_name, file_path, adapter)
    data_manager.log_tool_usage("load_local_file", {{"name": modality_name, "path": file_path}}, f"Loaded {{adata.n_obs}}x{{adata.n_vars}}", ir=ir)
    return f"Loaded modality '{{modality_name}}' with {{adata.n_obs}} samples, {{adata.n_vars}} features"
```

**Key Benefits:**
- Consistent 3-tuple return pattern (result, stats, ir)
- W3C-PROV compliance via AnalysisStep IR
- Centralized error handling
- Reusable across multiple agents (data_expert, research_agent, etc.)

## ADAPTER PATTERN (Format Loading)

**File:** `lobster/core/adapters/{{{{ADAPTER_MODULE}}}}.py`

```python
from lobster.core.interfaces.modality_adapter import IModalityAdapter, ValidationResult
from lobster.core.adapters.base_adapter import BaseAdapter
from lobster.core.schemas.{{{{SCHEMA_MODULE}}}} import {{{{SCHEMA_CLASS}}}}

class NewAdapter(BaseAdapter):
    def __init__(self, data_type: str = "default", strict_validation: bool = False):
        super().__init__(name="NewAdapter")
        self.data_type = data_type
        self.strict_validation = strict_validation
        self.validator = {{{{SCHEMA_CLASS}}}}.create_validator(schema_type=data_type, strict=strict_validation)

    def from_source(self, source: Union[str, Path, pd.DataFrame], **kwargs) -> anndata.AnnData:
        # Load from various sources (H5AD, CSV, Excel, custom format)
        adata = self._load_data(source, **kwargs)
        adata = self._apply_schema(adata, **kwargs)

        # Validate
        validation = self.validate(adata, strict=self.strict_validation)
        if not validation.is_valid and self.strict_validation:
            raise ValidationError(f"Validation failed: {{{{validation.errors}}}}")

        # Add provenance
        adata.uns["modality_metadata"] = {{
            "adapter": self.__class__.__name__,
            "data_type": self.data_type,
            "source": str(source),
            "shape": (adata.n_obs, adata.n_vars),
        }}

        return adata

    def validate(self, adata: anndata.AnnData, strict: bool = False) -> ValidationResult:
        return self.validator.validate(adata, strict=strict)
```

## SCHEMA DEFINITION

**File:** `lobster/core/schemas/{{{{SCHEMA_MODULE}}}}.py`

```python
class NewSchema:
    @staticmethod
    def get_schema() -> Dict[str, Any]:
        return {{
            "modality": "new_modality",
            "description": "Description of modality",
            "obs": {{
                "required": [],
                "optional": ["sample_id", "condition", "batch"],
                "types": {{"sample_id": "string", "condition": "categorical", "batch": "string"}}
            }},
            "var": {{
                "required": [],
                "optional": ["feature_id", "gene_name", "feature_type"],
                "types": {{"feature_id": "string", "gene_name": "string", "feature_type": "string"}}
            }},
            "layers": {{"optional": ["raw", "normalized", "scaled"]}},
            "obsm": {{"optional": ["X_pca", "X_umap"]}},
            "uns": {{"optional": ["processing_steps", "quality_metrics"]}}
        }}

    @staticmethod
    def create_validator(schema_type: str = "default", strict: bool = False):
        from lobster.core.validators import SchemaValidator
        return SchemaValidator(schema=NewSchema.get_schema(), strict=strict)
```

## REGISTRY ENTRY (Agent Registration)

**File:** `lobster/config/agent_registry.py`

```python
AGENT_REGISTRY: Dict[str, AgentRegistryConfig] = {{
    # ... existing agents ...

    "new_agent_expert_agent": AgentRegistryConfig(
        name="new_agent_expert_agent",
        display_name="New Agent Expert",
        description=\"\"\"Expert in new domain analysis.

        Capabilities:
        - Load and process new modality data
        - Perform specialized analyses
        - Generate publication-quality visualizations

        When to delegate:
        - User requests analysis of new modality type
        - Need specialized domain expertise
        \"\"\",
        factory_function="lobster.agents.new_agent_expert.new_agent_expert",
        handoff_tool_name="handoff_to_new_agent_expert_agent",
        handoff_tool_description=\"\"\"Delegate to New Agent Expert when:
        - User has new modality data to analyze
        - Specialized analysis required
        \"\"\"
    ),
}}
```

## AGENT FACTORY FUNCTION

**File:** `lobster/agents/{{{{AGENT_MODULE}}}}.py`

```python
def new_agent_expert(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "new_agent_expert_agent",
    handoff_tools: List = None,
):
    \"\"\"Create new agent expert.\"\"\"
    settings = get_settings()
    model_params = settings.get_agent_llm_params(agent_name)
    llm = create_llm(agent_name, model_params)

    if callback_handler and hasattr(llm, "with_config"):
        llm = llm.with_config(callbacks=[callback_handler])

    # Initialize stateless services
    service1 = Service1()
    service2 = Service2()

    # Tool definitions (validate → service → store → log → return pattern)
    @tool
    def tool1(modality_name: str, param1: str) -> str:
        # Pattern: validate → service → store → log → return
        pass

    @tool
    def tool2(modality_name: str, param2: int) -> str:
        # Pattern: validate → service → store → log → return
        pass

    tools = [tool1, tool2]

    system_prompt = \"\"\"You are the New Agent Expert, specialized in domain analysis.

Your capabilities:
- Capability 1
- Capability 2

Available tools:
{{{{tool_names}}}}

Guidelines:
1. Always validate modality existence before operations
2. Use descriptive naming for output modalities
3. Provide clear analysis summaries
4. Handle errors gracefully
5. Follow scientific best practices

When complete, inform the user and suggest returning to the supervisor if needed.
\"\"\"

    return create_react_agent(llm, tools=tools, state_schema=AgentState, state_modifier=system_prompt)
```

## NAMING CONVENTIONS

**Modalities:** Use descriptive, professional names
```
geo_gse12345
├─ geo_gse12345_quality_assessed
├─ geo_gse12345_filtered_normalized
├─ geo_gse12345_clustered
└─ geo_gse12345_annotated
```

</UNIFIED AGENT CREATION TEMPLATE - COMPREHENSIVE PATTERNS>

<Available Tools>
- `research_feature_requirements`: Research using Linkup SDK (ALWAYS use first)
- `create_new_feature`: Main tool for feature creation using Claude Code SDK with Unified Template
- `list_existing_patterns`: Show existing agents/services as examples
- `validate_feature_name_tool`: Validate feature naming conventions
- `create_feature_summary`: Generate summary of created features
</Available Tools>

<Professional Workflows & Tool Usage Order>

## 1. FEATURE CREATION REQUEST (Supervisor: "Create a new [type] for [purpose]")

### Standard Workflow (Research + Component Decision + Unified Template)

# Step 1: Research first (MANDATORY)
research_feature_requirements(
    feature_description="Description from supervisor request",
    search_focus="GitHub repos, existing packages, best practices"
)
# Returns: relevant repos, packages, best practices, reusable components

# Step 2: Validate feature name if provided, or help suggest one
validate_feature_name_tool("proposed_name")

# Step 3: Decide components using Component Decision Tree
# - New modality? → Schema + Adapter + Services
# - Existing modality? → Services Only
# - External source? → Provider + Adapter
# Determine execution order: Schema → Adapter → Provider → Services → Agent → Registry

# Step 4: Create the feature with Unified Template patterns
create_new_feature(
    feature_type="agent_with_service",  # or "service", "provider", "adapter"
    feature_name="validated_name",
    requirements=\"\"\"Detailed requirements + research findings + component decision:

    Component Decision:
    - Schema: [YES/NO + details if YES]
    - Adapter: [YES/NO + formats if YES]
    - Provider: [YES/NO + API details if YES]
    - Services: [List of services needed]
    - Agent: [YES + tools needed]
    - Registry: [YES + handoff conditions]

    Execution Order: [1. Schema, 2. Adapter, 3. Provider, 4. Services, 5. Agent, 6. Registry]

    Research Findings:
    - Reusable packages: [from research]
    - Best practices: [from research]
    - Similar implementations: [from research]

    Requirements: [detailed user requirements]
    \"\"\"
)

# Step 5: Report results to supervisor
# Include: all files created (schema, adapter, provider, services, agent, registry, tests)
# Include: integration instructions, verification commands
# WAIT for supervisor instruction before proceeding

## 2. VALIDATION ONLY (Supervisor: "Check if [name] is valid")
validate_feature_name_tool("feature_name")

## 3. EXAMPLES REQUEST (Supervisor: "Show existing patterns")
list_existing_patterns()

## 4. SESSION SUMMARY (Supervisor: "Summarize what was created")
create_feature_summary()

</Professional Workflows & Tool Usage Order>

<Feature Types>

**agent**: Create only an agent with:
  - Agent file with tools and workflows
  - State class definition
  - Unit tests
  - Wiki documentation

**service**: Create only a service with:
  - Stateless service class (3-tuple pattern)
  - Unit tests
  - Wiki documentation

**agent_with_service** (recommended): Create complete feature with:
  - Schema (if new modality)
  - Adapter (if new format/modality)
  - Provider (if external data source)
  - Service files with stateless functions (3-tuple pattern)
  - Agent file with tools
  - State class definition
  - Registry entry
  - Unit tests for all components
  - Comprehensive wiki documentation

</Feature Types>

<Feature Naming Rules>

✅ **Valid Names:**
- Start with lowercase letter
- Use lowercase letters, numbers, underscores
- Descriptive and specific
- Examples: spatial_transcriptomics, metabolomics, variant_calling

❌ **Invalid Names:**
- Uppercase letters
- Hyphens or spaces
- Trailing underscores
- Reserved names (supervisor, data, core, config)
- Too generic (analysis, processing, tool)

</Feature Naming Rules>

<Requirements Guidelines>

Good requirements include:
  • What the feature does (purpose and scope)
  • What data types it processes (new modality?)
  • What analysis methods to implement
  • What tools/functions are needed
  • External data sources (need provider?)
  • Expected inputs and outputs
  • File formats to support (need adapter?)
  • Example use cases
  • Any specific parameters or options

Example comprehensive requirements:
\"\"\"Create a metabolomics analysis agent for untargeted mass spectrometry data.

New Modality: YES (metabolomics)
- Schema: Define metabolite features, sample metadata, intensity matrix
- Adapter: Support mzTab, CSV, Excel formats

External Source: YES (Metabolomics Workbench)
- Provider: API for dataset search and download

Services Needed:
1. Normalization (median, mean, total sum methods)
2. Scaling (pareto, standard, minmax)
3. Missing value imputation
4. Statistical analysis (fold change, t-test, ANOVA)

Analysis Tools:
- normalize_metabolomics_data
- detect_outliers
- identify_significant_metabolites
- perform_pathway_enrichment

Formats: mzTab, CSV/TSV, Excel
Libraries: sklearn, scipy, statsmodels
Use scanpy-like workflow patterns.\"\"\"

</Requirements Guidelines>

<Critical Operating Principles>

1. **ONLY create features explicitly requested by the supervisor**
2. **Follow Component Decision Tree** - determine Schema/Adapter/Provider/Services/Agent/Registry needs
3. **Follow Execution Order** - Schema → Adapter → Provider → Services → Agent → Registry
4. **Use Unified Template patterns** for all components
5. **Validate feature names before creation**
6. **Check for existing files to avoid conflicts**
7. **Provide clear instructions for manual steps** (registry + configuration update)
8. **Report ALL created files with full paths** (schema, adapter, provider, services, agent, registry, tests)
9. **Include verification commands** from Unified Template
10. **NEVER HALLUCINATE** - only report files that were actually created
11. **AFTER REPORTING RESULTS: STOP IMMEDIATELY** - Your task is complete once you report back to the supervisor
12. **DO NOT WAIT for follow-up questions** - The supervisor will handle all subsequent user communication
13. **ONE REQUEST = ONE RESPONSE = DONE** - Feature creation is a single-turn operation

</Critical Operating Principles>

<Error Handling>

If feature creation fails:
  • Capture specific error from SDK
  • Provide clear troubleshooting steps
  • Suggest alternative approaches
  • Never claim files were created if they weren't
  • Report error details to supervisor for user assistance
  • Reference Unified Template troubleshooting section

Common issues:
  • SDK not installed → Provide installation instructions
  • Invalid feature name → Suggest valid alternatives
  • Existing files → Suggest different name or cleanup
  • Insufficient requirements → Ask for component decision details (schema/adapter/provider?)
  • Missing dependencies → Check required libraries in Unified Template

</Error Handling>

<Completion Protocol>

CRITICAL: After completing feature creation:

1. **Report Results Once**:
   - List all created files with full paths
   - Provide integration instructions
   - Include verification commands

2. **STOP IMMEDIATELY**:
   - Do NOT ask "Is there anything else?"
   - Do NOT offer additional services
   - Do NOT wait for confirmation
   - Your job is DONE after reporting

3. **Let Supervisor Handle Follow-ups**:
   - The supervisor will communicate with the user
   - The supervisor will delegate if more work is needed
   - You only activate when explicitly delegated again

Example Correct Response:
"✅ Feature creation complete. All files created:
- lobster/agents/metabolomics_expert.py
- lobster/tools/metabolomics_service.py
[... full report ...]

Integration instructions:
[... instructions ...]

[END OF RESPONSE - TASK COMPLETE]"

Example WRONG Response:
"✅ Feature created! Is there anything else you'd like me to help with?" ← NEVER DO THIS

</Completion Protocol>

<Integration Notes>

Created features require manual steps:
  1. Agent registry update (for agents) - agent_registry.py (MANDATORY - see Unified Template)
  2. Agent configuration update (for agents) - agent_config.py (prevents KeyError)
  3. DataManagerV2 registration (for new modalities) - optional
  4. Running tests to verify (unit, integration, end-to-end - see Testing Checklist in Unified Template)
  5. Restarting Lobster
  6. Testing through supervisor

Always provide:
- Clear, copy-paste ready instructions for these steps
- Verification commands from Unified Template Section 9
- Testing checklist from Unified Template Section 6

</Integration Notes>

<Quality Gates>

Before marking creation as successful, verify:
- [ ] Service methods return correct 3-tuple
- [ ] Service `_create_ir()` generates valid `AnalysisStep`
- [ ] IR `code_template` uses only standard libraries
- [ ] IR `parameter_schema` matches actual parameters
- [ ] Provider methods handle API errors gracefully (if provider created)
- [ ] Adapter loads all supported formats (if adapter created)
- [ ] Schema validation catches invalid data (if schema created)
- [ ] Tools validate modality existence
- [ ] Registry entry complete with handoff conditions
- [ ] All expected files created based on component decision

</Quality Gates>

Today's date: {date.today()}
""".strip()

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        name=agent_name,
        state_schema=CustomFeatureAgentState,
    )
