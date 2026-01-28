#!/usr/bin/env python3
"""
Simple script to generate and display the full supervisor prompt.
Usage: python kevin_notes/supervisor_prompt_concat.py
"""

import sys
from pathlib import Path

# Add parent directory to path to import lobster modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from lobster.agents.supervisor import create_supervisor_prompt
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.config.supervisor_config import SupervisorConfig


def main():
    """Generate and print the full supervisor prompt."""

    # Create a temporary workspace for DataManagerV2
    workspace_path = Path("/tmp/lobster_supervisor_prompt_debug")
    workspace_path.mkdir(parents=True, exist_ok=True)

    # Initialize DataManagerV2
    data_manager = DataManagerV2(workspace_path=workspace_path)

    # Create supervisor config (uses environment defaults)
    config = SupervisorConfig.from_env()

    # Generate the full prompt
    print("=" * 80)
    print("SUPERVISOR PROMPT CONFIGURATION")
    print("=" * 80)
    print(f"Mode: {config.get_prompt_mode()}")
    print(f"Workflow Guidance Level: {config.workflow_guidance_level}")
    print(f"Show Agent Capabilities: {config.show_agent_capabilities}")
    print(f"Include Agent Tools: {config.include_agent_tools}")
    print(f"Summarize Expert Output: {config.summarize_expert_output}")
    print(f"Require Download Confirmation: {config.require_download_confirmation}")
    print("=" * 80)
    print()

    # Generate and print the full prompt
    full_prompt = create_supervisor_prompt(
        data_manager=data_manager,
        config=config,
        active_agents=None  # Will auto-discover from registry
    )

    print("=" * 80)
    print("FULL SUPERVISOR PROMPT")
    print("=" * 80)
    print(full_prompt)
    print()
    print("=" * 80)
    print(f"Total characters: {len(full_prompt)}")
    print(f"Total lines: {len(full_prompt.splitlines())}")
    print("=" * 80)


if __name__ == "__main__":
    main()
