"""
Minimal agent configuration defaults for Lobster AI.

Per-agent overrides are managed via:
- ``lobster config models`` (interactive)
- ``.lobster_workspace/config.toml`` [agent_settings] (manual)
- Environment variables (LOBSTER_{AGENT}_TEMPERATURE, LOBSTER_GLOBAL_THINKING)

This module replaces the legacy LobsterAgentConfigurator with a thin
defaults system that never fails for unknown agents.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_TEMPERATURE = 1.0


@dataclass
class ThinkingConfig:
    """Configuration for model thinking/reasoning feature."""

    enabled: bool = False
    budget_tokens: int = 2000
    type: str = "enabled"  # AWS Bedrock uses "enabled" as the type value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API parameters."""
        if not self.enabled:
            return {}
        return {
            "thinking": {
                "type": self.type,
                "budget_tokens": self.budget_tokens,
            }
        }


THINKING_PRESETS: Dict[str, ThinkingConfig] = {
    "disabled": ThinkingConfig(enabled=False),
    "light": ThinkingConfig(enabled=True, budget_tokens=1000),
    "standard": ThinkingConfig(enabled=True, budget_tokens=2000),
    "extended": ThinkingConfig(enabled=True, budget_tokens=5000),
    "deep": ThinkingConfig(enabled=True, budget_tokens=10000),
}


def get_agent_params(
    agent_name: str, workspace_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Get LLM parameters for an agent.

    Resolution order:
    1. Environment variable override (LOBSTER_{AGENT}_TEMPERATURE)
    2. Workspace config.toml agent_settings
    3. Defaults (temperature=1.0, thinking=disabled)

    Never raises for unknown agents — they get sensible defaults.

    Args:
        agent_name: Agent name (e.g., 'transcriptomics_expert')
        workspace_path: Path to .lobster_workspace/ directory

    Returns:
        Dict with 'temperature' and optionally 'additional_model_request_fields'
    """
    temperature = DEFAULT_TEMPERATURE
    thinking_preset = "disabled"

    # Layer 1: Read from workspace config.toml agent_settings
    try:
        from lobster.config.workspace_agent_config import WorkspaceAgentConfig

        if workspace_path is not None:
            config = WorkspaceAgentConfig.load(workspace_path)
            agent_settings = config.agent_settings.get(agent_name)
            if agent_settings:
                if agent_settings.temperature is not None:
                    temperature = agent_settings.temperature
                if agent_settings.thinking_preset:
                    thinking_preset = agent_settings.thinking_preset
    except Exception:
        pass  # Graceful fallback to defaults

    # Layer 2: Environment variable overrides (highest priority)
    env_temp = os.environ.get(f"LOBSTER_{agent_name.upper()}_TEMPERATURE")
    if env_temp:
        try:
            temperature = float(env_temp)
        except ValueError:
            pass

    # Global thinking override (lower priority)
    global_thinking = os.environ.get("LOBSTER_GLOBAL_THINKING")
    if global_thinking and global_thinking in THINKING_PRESETS:
        thinking_preset = global_thinking

    # Per-agent thinking override (highest priority, overrides global)
    env_thinking = os.environ.get(f"LOBSTER_{agent_name.upper()}_THINKING")
    if env_thinking and env_thinking in THINKING_PRESETS:
        thinking_preset = env_thinking

    # Build params dict
    params: Dict[str, Any] = {"temperature": temperature}

    if thinking_preset != "disabled" and thinking_preset in THINKING_PRESETS:
        thinking_config = THINKING_PRESETS[thinking_preset]
        thinking_params = thinking_config.to_dict()
        if thinking_params:
            params["additional_model_request_fields"] = thinking_params

    return params


def get_current_profile(workspace_path: Optional[Path] = None) -> str:
    """
    Get the current profile name for display purposes.

    Resolution: env var > workspace config > default.

    Args:
        workspace_path: Path to .lobster_workspace/ directory

    Returns:
        Profile name string (e.g., 'production', 'development')
    """
    # Environment variable takes precedence
    env_profile = os.environ.get("LOBSTER_PROFILE")
    if env_profile:
        return env_profile

    # Read from workspace provider config
    try:
        from lobster.config.workspace_config import WorkspaceProviderConfig

        if workspace_path is not None:
            config = WorkspaceProviderConfig.load(workspace_path)
            if config.profile:
                return config.profile
    except Exception:
        pass

    return "production"
