# Configuration Guide

This guide covers all aspects of configuring Lobster AI, from basic API key setup to advanced model customization and cloud integration.

## Quick Start

The easiest way to configure Lobster AI is using the interactive wizard:

```bash
# Launch interactive configuration wizard
lobster init

# Test your configuration
lobster config test

# View your configuration (secrets masked)
lobster config show
```

For advanced configuration options, continue reading below.

## Table of Contents

- [Quick Start](#quick-start)
- [Environment Variables](#environment-variables)
- [API Key Management](#api-key-management)
- [Model Profiles](#model-profiles)
- [Supervisor Configuration](#supervisor-configuration)
- [Cloud vs Local Configuration](#cloud-vs-local-configuration)
- [Other Settings](#other-settings)
- [Configuration Management](#configuration-management)
- [Security Best Practices](#security-best-practices)
- [Troubleshooting Configuration](#troubleshooting-configuration)

## Environment Variables

Lobster AI uses environment variables for configuration. These can be set in a `.env` file in your project root directory, or as system environment variables.

### Required Variables

You must provide API keys for at least one Large Language Model (LLM) provider.

- `ANTHROPIC_API_KEY`: For using Claude models via Anthropic.
- `AWS_BEDROCK_ACCESS_KEY` and `AWS_BEDROCK_SECRET_ACCESS_KEY`: For using models via AWS Bedrock.

See the [API Key Management](#api-key-management) section for more details.

### Optional Variables

Most other settings are controlled via environment variables that follow these patterns:

- `LOBSTER_*`: For core application and model configuration.
- `SUPERVISOR_*`: For controlling the behavior of the supervisor agent.

Details on these variables are provided in the sections below.

## API Key Management

Lobster AI supports **three LLM providers**: two cloud-based and one local. Choose the provider that best fits your needs:

### Ollama (Local) - NEW! 🏠

**Best for**: Privacy, zero API costs, offline work, development without cloud dependencies.

**Requirements**: 8-48GB RAM depending on model size.

**Setup:**
```bash
# 1. Install Ollama (one-time)
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull a model (one-time)
ollama pull llama3:8b-instruct

# 3. Configure Lobster
lobster init  # Select option 3 (Ollama)
# Or manually:
export LOBSTER_LLM_PROVIDER=ollama
```

**Configuration:**
```env
LOBSTER_LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434  # Optional: default
OLLAMA_DEFAULT_MODEL=llama3:8b-instruct  # Optional: default
```

**Model Recommendations:**
- `llama3:8b-instruct` - Fast, good for testing (8GB RAM)
- `mixtral:8x7b-instruct` - Better quality (26GB RAM)
- `llama3:70b-instruct` - Maximum quality (48GB VRAM, requires GPU)

### Claude API (Cloud)

**Best for**: Quick testing, simple setup, best quality.

**Configuration:**
```env
ANTHROPIC_API_KEY=sk-ant-api03-xxxxx
```

### AWS Bedrock (Cloud)

**Best for**: Production use, enterprise compliance, higher rate limits.

**Configuration:**
```env
AWS_BEDROCK_ACCESS_KEY=AKIA...
AWS_BEDROCK_SECRET_ACCESS_KEY=abc123...
AWS_REGION=us-east-1  # Optional: defaults to us-east-1
```

### Provider Auto-Detection

Lobster AI automatically detects which provider to use based on available configuration in this priority order:

1. **Explicit override**: `LOBSTER_LLM_PROVIDER` environment variable
2. **Ollama detection**: If Ollama server is running (http://localhost:11434)
3. **Anthropic API**: If `ANTHROPIC_API_KEY` is set
4. **AWS Bedrock**: If `AWS_BEDROCK_ACCESS_KEY` and `AWS_BEDROCK_SECRET_ACCESS_KEY` are set

**Force a specific provider:**
```env
LOBSTER_LLM_PROVIDER=ollama      # Local LLM
LOBSTER_LLM_PROVIDER=anthropic   # Claude API
LOBSTER_LLM_PROVIDER=bedrock     # AWS Bedrock
```

### NCBI API Key (Optional)

**Benefits**: Enhanced literature search with higher rate limits.

**Configuration:**
```env
NCBI_API_KEY=your-ncbi-api-key-here
```

## Model Profiles

Lobster AI uses a profile-based system to manage agent and model configurations. You can set the active profile using the `LOBSTER_PROFILE` environment variable.

### Available Profiles

-   **`production` (Default)**: Supervisor uses Claude 4.5 Sonnet, expert agents use Claude 4 Sonnet, assistant uses Claude 3.7 Sonnet. Recommended for production deployments with optimal coordination and balanced analysis.
-   **`development`**: Supervisor and expert agents use Claude 4 Sonnet, assistant uses Claude 3.7 Sonnet. Ideal for development and testing with consistent expert-tier performance.
-   **`godmode`**: All agents (supervisor, experts, and assistant) use Claude 4.5 Sonnet. Maximum performance and capability for demanding analyses.

**Set the profile in your `.env` file:**
```env
LOBSTER_PROFILE=production    # Default
LOBSTER_PROFILE=development   # Development/testing
LOBSTER_PROFILE=godmode       # Maximum performance
```

### Available Models

The following models are available in Lobster AI:

-   **`claude-3-7-sonnet`**: Claude 3.7 Sonnet - Development and worker tier model
-   **`claude-4-sonnet`**: Claude 4 Sonnet - Production tier model (balanced performance)
-   **`claude-4-5-sonnet`**: Claude 4.5 Sonnet - Highest performance model for demanding tasks

### Custom Model Configuration

You can override the model for all agents or for specific agents using environment variables.

```env
# Override the model for ALL agents
LOBSTER_GLOBAL_MODEL=claude-4-5-sonnet

# Override the model for a specific agent
LOBSTER_SINGLECELL_EXPERT_AGENT_MODEL=claude-4-sonnet

# Override the temperature for a specific agent (0.0 to 1.0)
LOBSTER_SUPERVISOR_TEMPERATURE=0.3
```

### "Thinking" Configuration

For models that support it, you can enable a "thinking" feature, which allows the model to perform a chain-of-thought before answering.

```env
# Set a global thinking preset for all agents (light, standard, extended, deep)
LOBSTER_GLOBAL_THINKING=standard

# Enable or disable thinking for a specific agent
LOBSTER_SUPERVISOR_THINKING_ENABLED=true

# Set the token budget for thinking
LOBSTER_SUPERVISOR_THINKING_BUDGET=2000
```

## Supervisor Configuration

The supervisor agent's behavior can be fine-tuned using `SUPERVISOR_*` environment variables.

```env
# Interaction Settings
SUPERVISOR_ASK_QUESTIONS=true               # Ask clarification questions (default: true)
SUPERVISOR_MAX_QUESTIONS=2                  # Max clarification questions (default: 2)
SUPERVISOR_REQUIRE_CONFIRMATION=true        # Require download confirmation (default: true)
SUPERVISOR_REQUIRE_PREVIEW=true             # Preview metadata before download (default: true)

# Response Settings
SUPERVISOR_AUTO_SUGGEST=true               # Suggest next steps (default: true)
SUPERVISOR_VERBOSE=true                    # Verbose delegation explanations (default: true)
SUPERVISOR_INCLUDE_EXPERT_OUTPUT=true      # Include full expert output (default: true)
SUPERVISOR_SUMMARIZE_OUTPUT=false          # Summarize expert output (default: false)

# Context Settings
SUPERVISOR_INCLUDE_DATA=true              # Include data context (default: true)
SUPERVISOR_INCLUDE_WORKSPACE=true         # Include workspace status (default: true)
SUPERVISOR_INCLUDE_SYSTEM=false           # Include system info (default: false)
SUPERVISOR_INCLUDE_MEMORY=false           # Include memory stats (default: false)

# Workflow Guidance
SUPERVISOR_WORKFLOW_GUIDANCE=detailed      # minimal, standard, detailed (default: detailed)
SUPERVISOR_DELEGATION_STRATEGY=auto        # auto, conservative, aggressive (default: auto)
SUPERVISOR_ERROR_HANDLING=informative      # silent, informative, verbose (default: informative)

# Agent Discovery
SUPERVISOR_AUTO_DISCOVER=true             # Auto-discover agents (default: true)
SUPERVISOR_INCLUDE_AGENT_TOOLS=true       # List agent tools (default: true)
SUPERVISOR_MAX_TOOLS_PER_AGENT=20         # Tools shown per agent (default: 20)
```

## Cloud vs Local Configuration

Lobster AI can run in local mode (default) or cloud mode.

### Local Mode (Default)

-   **Trigger**: No `LOBSTER_CLOUD_KEY` is set.
-   **Processing**: Runs entirely on your local machine.
-   **Requires**: Local compute resources and API keys.

### Cloud Mode

-   **Trigger**: `LOBSTER_CLOUD_KEY` is set.
-   **Processing**: Occurs on the Lobster Cloud infrastructure.
-   **Requires**: A valid `LOBSTER_CLOUD_KEY`.

```env
# Cloud API key enables cloud mode
LOBSTER_CLOUD_KEY=your-cloud-api-key-here

# Optional: custom endpoint for development or enterprise
LOBSTER_ENDPOINT=https://api.lobster.omics-os.com
```

## Other Settings

These variables control other aspects of the application.

```env
# --- Data Processing ---
# Maximum file size for uploads in MB
LOBSTER_MAX_FILE_SIZE_MB=500
# Default resolution for clustering algorithms
LOBSTER_CLUSTER_RESOLUTION=0.5
# Directory for caching data
LOBSTER_CACHE_DIR=./lobster/data/cache

# --- Web Server ---
# Port for the Streamlit web interface
PORT=8501
# Host address for the web interface
HOST=0.0.0.0
# Enable or disable debug mode
DEBUG=False

# --- SSL/HTTPS ---
# Verify SSL certificates for outgoing requests
LOBSTER_SSL_VERIFY=true
# Path to a custom SSL certificate bundle
LOBSTER_SSL_CERT_PATH=
```

## Configuration Management

### Interactive Setup

The recommended way to configure Lobster AI:

```bash
# Launch interactive configuration wizard
lobster init

# The wizard will:
# 1. Prompt you to choose LLM provider (Claude API or AWS Bedrock)
# 2. Securely collect your API keys (input is masked)
# 3. Optionally configure NCBI API key
# 4. Save configuration to .env file in current directory
```

### Configuration Commands

Use the `lobster config` commands to manage and verify your configuration:

```bash
# Test API connectivity and validate configuration
lobster config test

# Display current configuration with masked secrets
lobster config show
```

### Advanced Options

```bash
# Reconfigure (creates timestamped backup of existing .env)
lobster init --force

# Non-interactive mode for CI/CD and automation
lobster init --non-interactive \
  --anthropic-key=sk-ant-xxx

# Non-interactive with AWS Bedrock
lobster init --non-interactive \
  --bedrock-access-key=AKIA... \
  --bedrock-secret-key=xxx

# Add NCBI API key in non-interactive mode
lobster init --non-interactive \
  --anthropic-key=sk-ant-xxx \
  --ncbi-key=your-ncbi-key
```

### Manual Configuration

For advanced users, you can manually edit the `.env` file in your working directory. See the [Environment Variables](#environment-variables) and [API Key Management](#api-key-management) sections for details on available settings.

## Security Best Practices

-   **Never commit `.env` files** to version control.
-   Use system environment variables or a secrets management tool for production.
-   Rotate API keys regularly.

## Troubleshooting Configuration

-   Use `lobster config show` to see your current configuration with masked secrets.
-   Use `lobster config test` to validate API connectivity and test your configuration.
-   Use `lobster init --force` to reconfigure (creates a backup of your existing .env file).
-   Run `lobster chat --debug` for verbose configuration loading information.
-   If you see "No configuration found" errors, run `lobster init` to create your .env file.
