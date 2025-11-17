#!/usr/bin/env python3
"""
API Validation Script for Agent-Service Method Calls

Purpose:
    Detect mismatches between agent service calls and actual service methods.
    Prevents runtime AttributeErrors from API evolution without agent updates.

Usage:
    python scripts/validate_agent_service_apis.py              # validate all
    python scripts/validate_agent_service_apis.py --verbose    # detailed output
    python scripts/validate_agent_service_apis.py --agent research_agent  # specific agent

Exit Codes:
    0 - All validations passed
    1 - API mismatches found
    2 - Script execution error

CI/CD Integration:
    Add to .github/workflows/ci-basic.yml:
    ```yaml
    - name: Validate Agent APIs
      run: python scripts/validate_agent_service_apis.py
    ```
"""

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple


@dataclass
class ServiceCall:
    """Represents a service method call found in agent code."""
    agent_file: str
    line_number: int
    service_name: str
    method_name: str
    full_expression: str


@dataclass
class ServiceMethod:
    """Represents a method definition found in service code."""
    service_file: str
    line_number: int
    method_name: str
    is_private: bool


class AgentAPIValidator:
    """Validates agent service calls against actual service methods."""

    def __init__(self, project_root: Path, verbose: bool = False):
        self.project_root = project_root
        self.verbose = verbose
        self.agents_dir = project_root / "lobster" / "agents"
        self.tools_dir = project_root / "lobster" / "tools"

        # Service name patterns to file mapping
        self.service_file_mapping = {
            "content_service": "content_access_service.py",
            "content_access_service": "content_access_service.py",
            "geo_service": "geo_service.py",
            "bulk_rnaseq_service": "bulk_rnaseq_service.py",
            "quality_service": "quality_service.py",
            "preprocessing_service": "preprocessing_service.py",
            "clustering_service": "clustering_service.py",
            "singlecell_service": "enhanced_singlecell_service.py",
            "pseudobulk_service": "pseudobulk_service.py",
            "proteomics_quality_service": "proteomics_quality_service.py",
            "proteomics_preprocessing_service": "proteomics_preprocessing_service.py",
            "proteomics_analysis_service": "proteomics_analysis_service.py",
            "proteomics_differential_service": "proteomics_differential_service.py",
            "proteomics_visualization_service": "proteomics_visualization_service.py",
            "visualization_service": "visualization_service.py",
            "sample_mapping_service": "sample_mapping_service.py",
            "metadata_standardization_service": "metadata_standardization_service.py",
            "metadata_validation_service": "metadata_validation_service.py",
            "manual_annotation_service": "manual_annotation_service.py",
            "fetch_service": "protein_structure_fetch_service.py",
            "viz_service": "protein_structure_visualization_service.py",
            "analysis_service": "protein_structure_analysis_service.py",
        }

    def extract_service_calls_from_agent(self, agent_file: Path) -> List[ServiceCall]:
        """Extract all service method calls from an agent file using regex."""
        calls = []

        try:
            content = agent_file.read_text()

            # Pattern: service_name.method_name(...)
            # Matches: content_service.extract_methods(...)
            pattern = r'(\w+_service)\.(\w+)\('

            for line_num, line in enumerate(content.splitlines(), start=1):
                matches = re.finditer(pattern, line)
                for match in matches:
                    service_name = match.group(1)
                    method_name = match.group(2)

                    calls.append(ServiceCall(
                        agent_file=agent_file.name,
                        line_number=line_num,
                        service_name=service_name,
                        method_name=method_name,
                        full_expression=line.strip()
                    ))

        except Exception as e:
            print(f"Warning: Error parsing {agent_file.name}: {e}", file=sys.stderr)

        return calls

    def extract_methods_from_service(self, service_file: Path) -> List[ServiceMethod]:
        """Extract all method definitions from a service file using AST."""
        methods = []

        try:
            content = service_file.read_text()
            tree = ast.parse(content, filename=str(service_file))

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Get line number
                    line_num = node.lineno

                    # Check if method is private (starts with _)
                    is_private = node.name.startswith('_')

                    methods.append(ServiceMethod(
                        service_file=service_file.name,
                        line_number=line_num,
                        method_name=node.name,
                        is_private=is_private
                    ))

        except Exception as e:
            print(f"Warning: Error parsing {service_file.name}: {e}", file=sys.stderr)

        return methods

    def validate_service_call(
        self,
        call: ServiceCall,
        service_methods: Dict[str, List[ServiceMethod]]
    ) -> Tuple[bool, str]:
        """
        Validate a single service call against available methods.

        Returns:
            (is_valid, error_message)
        """
        # Map service name to service file
        service_file = self.service_file_mapping.get(call.service_name)

        if not service_file:
            return True, ""  # Unknown service, skip validation

        # Get methods for this service
        methods = service_methods.get(service_file, [])
        method_names = {m.method_name for m in methods}

        # Check if method exists
        if call.method_name not in method_names:
            return False, (
                f"Method '{call.method_name}' does not exist in {service_file}\n"
                f"  Called in: {call.agent_file}:{call.line_number}\n"
                f"  Expression: {call.full_expression}"
            )

        # Check if calling private method (code smell warning, not error)
        method = next(m for m in methods if m.method_name == call.method_name)
        if method.is_private:
            if self.verbose:
                print(
                    f"⚠️  Warning: Agent calling private method '{call.method_name}' in {service_file}\n"
                    f"  Called in: {call.agent_file}:{call.line_number}\n"
                    f"  Expression: {call.full_expression}",
                    file=sys.stderr
                )

        return True, ""

    def validate_all_agents(self, specific_agent: str = None) -> Tuple[bool, List[str]]:
        """
        Validate all agent files (or specific agent) against service definitions.

        Returns:
            (all_valid, error_messages)
        """
        errors = []

        # Step 1: Load all service methods
        if self.verbose:
            print("Loading service method definitions...")

        service_methods: Dict[str, List[ServiceMethod]] = {}
        for service_file_name in set(self.service_file_mapping.values()):
            service_path = self.tools_dir / service_file_name
            if service_path.exists():
                methods = self.extract_methods_from_service(service_path)
                service_methods[service_file_name] = methods
                if self.verbose:
                    print(f"  {service_file_name}: {len(methods)} methods")

        # Step 2: Extract agent service calls
        if self.verbose:
            print("\nExtracting agent service calls...")

        agent_files = []
        if specific_agent:
            agent_path = self.agents_dir / f"{specific_agent}.py"
            if agent_path.exists():
                agent_files = [agent_path]
            else:
                errors.append(f"Agent file not found: {specific_agent}.py")
                return False, errors
        else:
            agent_files = list(self.agents_dir.glob("*_agent.py"))
            agent_files.extend(self.agents_dir.glob("*_expert.py"))
            agent_files.extend(self.agents_dir.glob("*_assistant.py"))
            agent_files.append(self.agents_dir / "supervisor.py")

        all_calls = []
        for agent_file in agent_files:
            if agent_file.exists():
                calls = self.extract_service_calls_from_agent(agent_file)
                all_calls.extend(calls)
                if self.verbose:
                    print(f"  {agent_file.name}: {len(calls)} service calls")

        # Step 3: Validate each call
        if self.verbose:
            print(f"\nValidating {len(all_calls)} service calls...")

        for call in all_calls:
            is_valid, error_msg = self.validate_service_call(call, service_methods)
            if not is_valid:
                errors.append(error_msg)

        return len(errors) == 0, errors


def main():
    parser = argparse.ArgumentParser(
        description="Validate agent service API calls",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output"
    )
    parser.add_argument(
        "--agent", "-a",
        type=str,
        help="Validate specific agent only (e.g., 'research_agent')"
    )

    args = parser.parse_args()

    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Validate
    validator = AgentAPIValidator(project_root, verbose=args.verbose)

    try:
        all_valid, errors = validator.validate_all_agents(specific_agent=args.agent)

        if all_valid:
            print("✅ All agent service API calls are valid")
            return 0
        else:
            print("❌ API validation failed\n", file=sys.stderr)
            for error in errors:
                print(error, file=sys.stderr)
                print("", file=sys.stderr)

            print(f"Found {len(errors)} API mismatch(es)", file=sys.stderr)
            return 1

    except Exception as e:
        print(f"❌ Validation script error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
