#!/usr/bin/env python3
"""
Configuration Manager CLI Tool for LOBSTER AI.

This tool provides command-line utilities to manage agent configurations,
view available models, switch profiles, and test different setups.
"""

import argparse
import sys
from pathlib import Path

from tabulate import tabulate

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from lobster.config.agent_defaults import get_current_profile  # noqa: E402
from lobster.config.providers.registry import ProviderRegistry  # noqa: E402


def print_colored(text: str, color: str = "white"):
    """Print colored text to terminal."""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m",
    }

    print(f"{colors.get(color, colors['white'])}{text}{colors['reset']}")


def list_available_models():
    """List all available model presets from registered providers."""
    print_colored("\n Available Models by Provider", "cyan")
    print_colored("=" * 60, "cyan")

    providers = ProviderRegistry.get_all()
    if not providers:
        print_colored("No providers registered.", "yellow")
        return

    table_data = []
    for provider in providers:
        try:
            models = provider.list_models() if hasattr(provider, "list_models") else []
            for model in models:
                table_data.append(
                    [
                        provider.name,
                        model.name,
                        getattr(model, "display_name", model.name),
                    ]
                )
        except Exception as e:
            table_data.append([provider.name, f"(error: {e})", ""])

    headers = ["Provider", "Model ID", "Display Name"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


def list_available_profiles():
    """List known agent profiles."""
    print_colored("\n Available Agent Profiles", "cyan")
    print_colored("=" * 60, "cyan")

    # Profiles are defined as named constants; list them explicitly.
    profiles = {
        "development": "Sonnet 4 — fastest, most affordable",
        "production": "Sonnet 4 + Sonnet 4.5 supervisor (recommended)",
        "performance": "Sonnet 4.5 — highest quality",
        "max": "Opus 4.5 supervisor — most capable, most expensive",
    }

    current = get_current_profile()
    table_data = []
    for name, description in profiles.items():
        marker = "*" if name == current else ""
        table_data.append([marker, name, description])

    headers = ["", "Profile", "Description"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print_colored(f"\nCurrent profile: {current}", "green")


def show_current_config(profile: str = None):
    """Show current configuration."""
    current = profile or get_current_profile()
    print_colored(f"\n Current Configuration (profile: {current})", "cyan")
    print_colored("=" * 60, "cyan")

    configured = ProviderRegistry.get_configured_providers()
    if configured:
        print_colored("\nConfigured providers:", "yellow")
        for p in configured:
            print(
                f"  {p.name}: {p.display_name if hasattr(p, 'display_name') else p.name}"
            )
    else:
        print_colored("\nNo providers configured.", "yellow")
        print("Run 'lobster init' to configure a provider.")

    print_colored(f"\nActive profile: {current}", "green")


def test_configuration(profile: str, agent: str = None):
    """Test a specific configuration."""
    try:
        from lobster.config.agent_defaults import get_agent_params

        if agent:
            try:
                params = get_agent_params(agent)
                print_colored(f"\n Agent '{agent}' configuration is valid", "green")
                print(f"   Temperature: {params.get('temperature')}")
                if "additional_model_request_fields" in params:
                    print(f"   Thinking: {params['additional_model_request_fields']}")
            except Exception as exc:
                print_colored(f"\n Agent '{agent}' configuration error: {exc}", "red")
                return False
        else:
            print_colored(f"\n Testing Profile: {profile}", "yellow")
            configured = ProviderRegistry.get_configured_providers()
            if configured:
                for p in configured:
                    print_colored(f"   {p.name}: available", "green")
                print_colored(
                    f"\n Profile '{profile}' uses configured providers.", "green"
                )
            else:
                print_colored(
                    f"\n No providers configured for profile '{profile}'.", "yellow"
                )
                return False

        return True

    except Exception as e:
        print_colored(f"\n Error testing configuration: {str(e)}", "red")
        return False


def create_custom_config():
    """Interactive creation of custom configuration (stub)."""
    print_colored("\n Custom configuration", "cyan")
    print_colored("=" * 50, "cyan")
    print_colored(
        "Per-agent overrides are managed via:\n"
        "  • lobster config models  (interactive)\n"
        "  • .lobster_workspace/config.toml [agent_settings]\n"
        "  • Environment variables (LOBSTER_{AGENT}_TEMPERATURE)",
        "yellow",
    )
    print_colored(
        "\nTo set up API credentials and provider, run: lobster init", "white"
    )


def generate_env_template():
    """Generate .env template with all available options."""
    template = """# LOBSTER AI Configuration Template
# Copy this file to .env and configure as needed

# =============================================================================
# API KEYS (Required)
# =============================================================================
AWS_BEDROCK_ACCESS_KEY="your-aws-access-key-here"
AWS_BEDROCK_SECRET_ACCESS_KEY="your-aws-secret-key-here"
NCBI_API_KEY="your-ncbi-api-key-here"

# =============================================================================
# LOBSTER CLOUD CONFIGURATION (Optional)
# =============================================================================
# Set these to use Lobster Cloud instead of local processing
# Get your API key from https://cloud.lobster.ai or contact info@omics-os.com

# LOBSTER_CLOUD_KEY="your-cloud-api-key-here"
# LOBSTER_ENDPOINT="https://api.lobster.omics-os.com"  # Optional: defaults to production

# When LOBSTER_CLOUD_KEY is set, all processing will be done in the cloud
# When not set, Lobster will run locally with full functionality

# =============================================================================
# AGENT CONFIGURATION (Professional System)
# =============================================================================

# Profile-based configuration (recommended)
# Available profiles: development, production, performance, max
LOBSTER_PROFILE=production

# OR use custom configuration file
# LOBSTER_CONFIG_FILE=config/custom_agent_config.json

# Per-agent model overrides (optional)
# Available models: claude-haiku, claude-sonnet, claude-sonnet-eu, claude-opus, claude-opus-eu, claude-3-7-sonnet, claude-3-7-sonnet-eu
# LOBSTER_SUPERVISOR_MODEL=claude-haiku
# LOBSTER_TRANSCRIPTOMICS_EXPERT_MODEL=claude-opus
# LOBSTER_METHOD_AGENT_MODEL=claude-sonnet
# LOBSTER_GENERAL_CONVERSATION_MODEL=claude-haiku

# Global model override (overrides all agents)
# LOBSTER_GLOBAL_MODEL=claude-sonnet

# Per-agent temperature overrides
# LOBSTER_SUPERVISOR_TEMPERATURE=0.5
# LOBSTER_TRANSCRIPTOMICS_EXPERT_TEMPERATURE=0.7
# LOBSTER_METHOD_AGENT_TEMPERATURE=0.3

# =============================================================================
# APPLICATION SETTINGS
# =============================================================================

# Server configuration
PORT=8501
HOST=0.0.0.0
DEBUG=False

# Data processing
LOBSTER_MAX_FILE_SIZE_MB=500
LOBSTER_CLUSTER_RESOLUTION=0.5
LOBSTER_CACHE_DIR=data/cache

# =============================================================================
# EXAMPLE CONFIGURATIONS
# =============================================================================

# Example 1: Development setup (Claude Haiku 4.5 - fastest, most affordable)
# LOBSTER_PROFILE=development

# Example 2: Production setup (Claude Sonnet 4 - balanced quality & speed)
# LOBSTER_PROFILE=production

# Example 3: Performance setup (Claude Sonnet 4.5 - highest quality)
# LOBSTER_PROFILE=performance

# Example 4: Max setup (Claude Opus 4.5 - experimental, most expensive)
# LOBSTER_PROFILE=max
"""

    with open(".env.template", "w") as f:
        f.write(template)

    print_colored(" Environment template saved to: .env.template", "green")
    print_colored("Copy this file to .env and configure your API keys", "yellow")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LOBSTER AI Configuration Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s list-models              # Show available models
  %(prog)s list-profiles            # Show available profiles
  %(prog)s show-config              # Show current configuration
  %(prog)s show-config -p development  # Show specific profile
  %(prog)s test -p production       # Test a profile
  %(prog)s test -p production -a supervisor  # Test specific agent
  %(prog)s create-custom           # Show custom config instructions
  %(prog)s generate-env            # Generate .env template
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List models command
    subparsers.add_parser("list-models", help="List available model presets")

    # List profiles command
    subparsers.add_parser("list-profiles", help="List available testing profiles")

    # Show config command
    show_parser = subparsers.add_parser(
        "show-config", help="Show current configuration"
    )
    show_parser.add_argument("-p", "--profile", help="Profile to show")

    # Test command
    test_parser = subparsers.add_parser("test", help="Test configuration")
    test_parser.add_argument("-p", "--profile", required=True, help="Profile to test")
    test_parser.add_argument("-a", "--agent", help="Specific agent to test")

    # Create custom command
    subparsers.add_parser(
        "create-custom", help="Show custom configuration instructions"
    )

    # Generate env command
    subparsers.add_parser("generate-env", help="Generate .env template file")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == "list-models":
            list_available_models()
        elif args.command == "list-profiles":
            list_available_profiles()
        elif args.command == "show-config":
            show_current_config(args.profile)
        elif args.command == "test":
            test_configuration(args.profile, args.agent)
        elif args.command == "create-custom":
            create_custom_config()
        elif args.command == "generate-env":
            generate_env_template()

    except KeyboardInterrupt:
        print_colored("\n\n Goodbye!", "yellow")
    except Exception as e:
        print_colored(f"\n Error: {str(e)}", "red")
        sys.exit(1)


if __name__ == "__main__":
    main()
