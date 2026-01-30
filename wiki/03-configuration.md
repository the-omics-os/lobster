# Configuration Guide

This guide covers all aspects of configuring Lobster AI, from basic API key setup to advanced model customization and cloud integration.

## Quick Start

The easiest way to configure Lobster AI is using the interactive wizard:

```bash
# Workspace-specific configuration (default)
lobster init

# Global configuration for all workspaces (v0.4+)
lobster init --global

# Test your configuration
lobster config test

# View your configuration (secrets masked)
lobster config show
```

### What's New in v0.4: External Workspaces

You can now work with data in **any directory** without per-directory setup:

```bash
# Set global defaults once
lobster init --global

# Use any workspace - just works!
lobster chat --workspace ~/Documents/project1
lobster chat --workspace ~/Desktop/quick_analysis
lobster query "cluster cells" --workspace /tmp/test_data
```

**Before v0.4:** Each workspace needed its own `.env` file
**After v0.4:** Global config (`~/.config/lobster/providers.json`) provides defaults

For advanced configuration options, continue reading below.

## Table of Contents

- [Quick Start](#quick-start)
- [Environment Variables](#environment-variables)
- [API Key Management](#api-key-management)
  - [Ollama (Local)](#ollama-local---new-)
  - [Claude API (Cloud)](#claude-api-cloud)
  - [AWS Bedrock (Cloud)](#aws-bedrock-cloud)
  - [Google Gemini (Cloud)](#google-gemini-cloud-)
  - [Azure AI (Cloud)](#azure-ai-cloud-)
  - [Provider Auto-Detection](#provider-auto-detection)
  - [Running Multiple Sessions with Different Providers](#running-multiple-sessions-with-different-providers)
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

You must configure at least one Large Language Model (LLM) provider:

**Cloud Providers (require API keys):**
- `ANTHROPIC_API_KEY`: For using Claude models via Anthropic Direct API
- `AWS_BEDROCK_ACCESS_KEY` and `AWS_BEDROCK_SECRET_ACCESS_KEY`: For using models via AWS Bedrock
- `GOOGLE_API_KEY`: For using Gemini models via Google AI Studio
- `AZURE_AI_ENDPOINT` and `AZURE_AI_CREDENTIAL`: For using Azure AI Foundry models

**Local Provider (no API keys needed):**
- `LOBSTER_LLM_PROVIDER=ollama`: For using local models via Ollama (requires Ollama installation)

See the [API Key Management](#api-key-management) section for detailed setup instructions.

### Optional Variables

Most other settings are controlled via environment variables that follow these patterns:

- `LOBSTER_*`: For core application and model configuration.
- `SUPERVISOR_*`: For controlling the behavior of the supervisor agent.

Details on these variables are provided in the sections below.

## API Key Management

Lobster AI supports **five LLM providers**: four cloud-based and one local. Choose the provider that best fits your needs:

### Ollama (Local) - NEW! 🏠

**Best for**: Privacy, zero API costs, offline work, development without cloud dependencies.

**Requirements**: 8-48GB RAM depending on model size.

**Setup:**
```bash
# 1. Install Ollama (one-time)
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull a model (one-time)
ollama pull gpt-oss:20b

# 3. Configure Lobster
lobster init  # Select option 3 (Ollama)
# Or manually:
export LOBSTER_LLM_PROVIDER=ollama
```

**Configuration:**
```env
LOBSTER_LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434  # Optional: default
OLLAMA_DEFAULT_MODEL=gpt-oss:20b  # Optional: default
```

**Model Recommendations:**
- `gpt-oss:20b` - **Recommended** for Lobster (supports tools, 16GB RAM)
- `mixtral:8x7b-instruct` - Better quality (26GB RAM)
- `llama3:70b-instruct` - Maximum quality (48GB VRAM, requires GPU)

**Note**: llama3:8b models do NOT support tool calling and will fail with Lobster. Use gpt-oss:20b or larger models.

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

### Google Gemini (Cloud) ✨

**Best for**: Long context windows, multimodal capabilities, free tier available.

**Configuration:**
```env
GOOGLE_API_KEY=your-key-here
LOBSTER_LLM_PROVIDER=gemini
```

**Get your API key**: https://aistudio.google.com/apikey

**Available Models:**
- `gemini-3-pro-preview` - Best balance ($2.00 input / $12.00 output per million tokens)
- `gemini-3-flash-preview` - Fastest, free tier available ($0.50 input / $3.00 output)

**Note**: Gemini 3.0+ models require `temperature=1.0` (lower values can cause issues).

### Azure AI (Cloud) 🔷

**Best for**: Enterprise customers with existing Azure infrastructure, Azure compliance requirements, multi-model access.

**Configuration:**
```env
AZURE_AI_ENDPOINT=https://your-project.inference.ai.azure.com/
AZURE_AI_CREDENTIAL=your-api-key
AZURE_AI_API_VERSION=2024-05-01-preview  # Optional
LOBSTER_LLM_PROVIDER=azure
```

**Get your credentials**: https://ai.azure.com/
1. Create/open an Azure AI Foundry project
2. Deploy a model (GPT-4o, DeepSeek R1, Cohere, Phi, Mistral)
3. Copy endpoint URL and API key from deployment details

**Available Models:**
- `gpt-4o` - OpenAI GPT-4o (recommended default) ($5.00 input / $15.00 output per million tokens)
- `deepseek-r1` - DeepSeek R1 reasoning model ($0.55 input / $2.19 output)
- `gpt-4-turbo` - OpenAI GPT-4 Turbo ($10.00 input / $30.00 output)
- `cohere-command-r-plus` - Cohere Command R+ ($3.00 input / $15.00 output)
- `phi-4` - Microsoft Phi-4 small model ($0.07 input / $0.14 output)
- `mistral-large` - Mistral Large ($4.00 input / $12.00 output)

**Key Features:**
- Access to multiple model providers through single Azure account
- Enterprise compliance (HIPAA, SOC2, ISO 27001)
- Data stays within your Azure tenant
- Supports custom model deployments

**Legacy Environment Variables** (backward compatibility):
```env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
```

### Configuration Resolution Priority (v0.4+)

Lobster AI uses a **5-layer priority hierarchy** for provider configuration:

1. **Runtime CLI flags**: `--provider` (highest priority, overrides everything)
2. **Workspace config**: `.lobster_workspace/provider_config.json` (project-specific)
3. **Global user config**: `~/.config/lobster/providers.json` (user-wide defaults)
4. **Environment variable**: `LOBSTER_LLM_PROVIDER` (temporary override)
5. **FAIL**: Requires explicit configuration (no auto-detection)

**Force a specific provider:**
```bash
# Runtime override (highest priority)
lobster chat --provider anthropic

# Global defaults (applies to all workspaces)
lobster init --global

# Environment variable (temporary override)
export LOBSTER_LLM_PROVIDER=ollama

# Workspace-specific (project defaults)
lobster init  # Creates .env + provider_config.json
```

**Model resolution** follows the same hierarchy plus provider defaults.

### Running Multiple Sessions with Different Providers

You can run multiple Lobster sessions simultaneously, each using a different LLM provider. This is useful for:
- **A/B Testing**: Compare analysis quality between providers
- **Development vs Production**: Use local for dev, cloud for production
- **Cost Optimization**: Use local for exploratory work, cloud for final analyses
- **Privacy Control**: Use local for sensitive data, cloud for general analyses

#### Method 1: Different Terminal Sessions (Current)

Each terminal maintains its own environment variables:

```bash
# Terminal 1: Local development with Ollama
export LOBSTER_LLM_PROVIDER=ollama
cd ~/project-dev
lobster chat

# Terminal 2: Production with Claude (simultaneously)
export LOBSTER_LLM_PROVIDER=anthropic
cd ~/project-prod
lobster chat

# Terminal 3: Enterprise with Bedrock
export LOBSTER_LLM_PROVIDER=bedrock
cd ~/project-enterprise
lobster chat
```

**How it works:**
- Environment variables are process-specific (don't interfere between terminals)
- Each session is completely independent
- Can run unlimited simultaneous sessions

#### Method 2: Shell Aliases (Convenience)

Create aliases for quick provider switching:

```bash
# Add to ~/.bashrc or ~/.zshrc
alias lobster-local='LOBSTER_LLM_PROVIDER=ollama lobster'
alias lobster-cloud='LOBSTER_LLM_PROVIDER=anthropic lobster'
alias lobster-bedrock='LOBSTER_LLM_PROVIDER=bedrock lobster'
alias lobster-gemini='LOBSTER_LLM_PROVIDER=gemini lobster'

# Usage
lobster-local chat     # Always uses Ollama
lobster-cloud query "analyze data"  # Always uses Claude
lobster-bedrock chat   # Always uses Bedrock
lobster-gemini chat    # Always uses Gemini
```

#### Method 3: Per-Command Inline (Quick Tests)

```bash
# One-off command with specific provider
LOBSTER_LLM_PROVIDER=ollama lobster query "cluster my data"
LOBSTER_LLM_PROVIDER=anthropic lobster query "cluster my data"

# Compare results side-by-side
```

#### Method 4: CLI Flag (Coming Soon)

**Future enhancement** - provider flag per command:
```bash
lobster chat --provider ollama
lobster query --provider anthropic "analyze data"
```

#### Method 5: Workspace-Specific Config (Coming Soon)

**Future enhancement** - each workspace remembers its provider:
```bash
# project1/.lobster_workspace/config.json
{"llm_provider": "ollama"}

# project2/.lobster_workspace/config.json
{"llm_provider": "anthropic"}

cd project1 && lobster chat  # Auto-uses Ollama
cd project2 && lobster chat  # Auto-uses Claude
```

#### Provider Selection Priority

When multiple configurations exist, Lobster uses this resolution order:

```
1. Runtime CLI flag (--provider)                           [✅ v0.4+]
2. Workspace config (.lobster_workspace/provider_config.json)  [✅ Current]
3. Global user config (~/.config/lobster/providers.json)      [✅ v0.4+]
4. Environment variable (LOBSTER_LLM_PROVIDER)                [✅ Current]
5. FAIL with diagnostic message                               [✅ v0.4+]
```

**Key improvements in v0.4:**
- Added global user config for user-wide defaults
- External workspaces now inherit from global config
- Better error diagnostics showing what was checked

#### Practical Example: Development Workflow

```bash
# Setup: Configure both providers once
cat > ~/.env << EOF
ANTHROPIC_API_KEY=sk-ant-xxx
LOBSTER_LLM_PROVIDER=anthropic  # Default to cloud
EOF

# Day-to-day usage:
# Quick local test (Terminal 1)
LOBSTER_LLM_PROVIDER=ollama lobster chat

# Production analysis (Terminal 2, simultaneously)
lobster chat  # Uses default (anthropic)

# Both sessions run independently!
```

### NCBI API Key (Optional)

**Benefits**: Enhanced literature search with higher rate limits.

**Configuration:**
```env
NCBI_API_KEY=your-ncbi-api-key-here
```

## Deployment Patterns

Lobster supports flexible deployment configurations combining execution environments, LLM providers, and data sources. Choose a pattern based on your privacy, quality, and scale requirements.

### Pattern 1: Local + Ollama (Zero-Cost Stack)

**Best for**: Individual researchers, privacy-sensitive data, unlimited usage, offline work

**Setup:**
```bash
# 1. Install Ollama (one-time)
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull model
ollama pull gpt-oss:20b

# 3. Install Lobster
uv pip install lobster-ai

# 4. Run
lobster chat
# Ollama auto-detected, no API keys needed
```

**Characteristics:**
- ✅ **Zero cost**: No API charges
- ✅ **Full privacy**: All data stays on your machine
- ✅ **Offline capable**: Works without internet
- ✅ **Unlimited usage**: No rate limits
- ✅ **Tool support**: gpt-oss:20b supports multi-agent tool calling
- ⚠️ **Hardware dependent**: Requires 16-48GB RAM depending on model
- ⚠️ **Quality varies**: Model-dependent (gpt-oss:20b < mixtral < llama3:70b)

**Configuration:**
```env
LOBSTER_LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434  # Optional: default
OLLAMA_DEFAULT_MODEL=gpt-oss:20b  # Optional: default
```

---

### Pattern 2: Local + Anthropic (Quality-First)

**Best for**: High-quality analysis, quick start, flexible execution, development

**Setup:**
```bash
# 1. Get API key from console.anthropic.com

# 2. Install Lobster
uv pip install lobster-ai

# 3. Configure
export ANTHROPIC_API_KEY=sk-ant-api03-...

# 4. Run
lobster chat
```

**Characteristics:**
- ✅ **Best quality**: Claude Sonnet 4.5, highest accuracy
- ✅ **Quick setup**: Just API key, no infrastructure
- ✅ **Local execution**: Your hardware, your control
- ✅ **Flexible**: Switch to other providers anytime
- ⚠️ **API costs**: ~$0.50/analysis
- ⚠️ **Rate limits**: ~50 requests/min for new accounts
- ⚠️ **Requires internet**: Online-only

**Configuration:**
```env
ANTHROPIC_API_KEY=sk-ant-api03-xxxxx
LOBSTER_LLM_PROVIDER=anthropic  # Optional: auto-detected
```

---

### Pattern 3: Cloud + Bedrock (Enterprise Scale)

**Best for**: Team collaboration, production workloads, compliance requirements, high throughput

**Setup:**
```bash
# 1. Configure AWS Bedrock access
export LOBSTER_CLOUD_KEY=your-cloud-key
export AWS_BEDROCK_ACCESS_KEY=AKIA...
export AWS_BEDROCK_SECRET_ACCESS_KEY=...

# 2. Install Lobster
uv pip install lobster-ai

# 3. Run
lobster chat
# Cloud mode + Bedrock auto-configured
```

**Characteristics:**
- ✅ **Enterprise SLA**: Production-grade reliability
- ✅ **High rate limits**: No throttling for production use
- ✅ **Team collaboration**: Shared cloud infrastructure
- ✅ **Compliance ready**: HIPAA, SOC2, GDPR support
- ✅ **Scalable**: Handles large datasets automatically
- ⚠️ **Cost**: $6K-$30K/year (volume-based)
- ⚠️ **Setup complexity**: Requires AWS configuration

**Configuration:**
```env
# Cloud execution
LOBSTER_CLOUD_KEY=your-cloud-api-key
LOBSTER_ENDPOINT=https://api.lobster.omics-os.com  # Optional

# AWS Bedrock LLM
AWS_BEDROCK_ACCESS_KEY=AKIA...
AWS_BEDROCK_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1  # Optional: default
LOBSTER_LLM_PROVIDER=bedrock  # Optional: auto-detected
```

---

### Comparison Matrix

| Aspect | Pattern 1 (Ollama) | Pattern 2 (Anthropic) | Pattern 3 (Bedrock) |
|--------|-------------------|----------------------|---------------------|
| **Cost** | Free | ~$0.50/analysis | $6K-$30K/year |
| **Setup** | Ollama install | API key | AWS + Cloud key |
| **Quality** | Model-dependent | Highest (Claude 4.5) | Highest (Claude 4.5) |
| **Privacy** | 100% local | Cloud LLM, local data | Cloud execution |
| **Rate Limits** | None | 50 req/min | Enterprise (high) |
| **Offline** | ✅ Yes | ❌ No | ❌ No |
| **Scalability** | Hardware-limited | Hardware-limited | Cloud-managed |
| **Best For** | Privacy, learning | Quality, development | Production, teams |

### Switching Between Patterns

You can switch patterns anytime or run multiple sessions with different patterns simultaneously:

```bash
# Terminal 1: Privacy-focused with Ollama
export LOBSTER_LLM_PROVIDER=ollama
cd ~/private-project
lobster chat

# Terminal 2: Quality-focused with Anthropic (simultaneously)
export LOBSTER_LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-ant-xxx
cd ~/research-project
lobster chat

# Terminal 3: Production with Bedrock
export LOBSTER_CLOUD_KEY=xxx
export LOBSTER_LLM_PROVIDER=bedrock
cd ~/production-project
lobster chat
```

Each session is completely independent with its own workspace, configuration, and execution environment.

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
# Workspace-specific configuration (default)
lobster init

# Global configuration for all workspaces (v0.4+)
lobster init --global

# The wizard will:
# 1. Prompt you to choose LLM provider:
#    - Option 1: Claude API (Anthropic)
#    - Option 2: AWS Bedrock
#    - Option 3: Ollama (Local)
#    - Option 4: Google Gemini
# 2. Guide you through provider-specific setup:
#    - Cloud: Securely collect API keys (input is masked)
#    - Ollama: Check installation, list available models
#    - Gemini: Collect Google API key, select model (Pro/Flash)
# 3. Optionally configure NCBI API key
# 4. Save configuration:
#    - Workspace mode: .env + .lobster_workspace/provider_config.json
#    - Global mode: ~/.config/lobster/providers.json
```

**When to use `--global`:**
- Set user-wide defaults that apply to all workspaces
- Enable seamless use of external workspaces without per-workspace setup
- Ideal for users who want consistent settings across projects

**Global config location (platform-specific):**
- **Linux/macOS**: `~/.config/lobster/providers.json` (CLI convention)
- **Windows**: `%APPDATA%\lobster\providers.json` (e.g., `C:\Users\Name\AppData\Roaming\lobster\providers.json`)

### External Workspaces (v0.4+)

External workspaces allow you to work with data in any directory without per-directory configuration:

```bash
# Step 1: Set global defaults (one-time)
lobster init --global

# Step 2: Use any workspace seamlessly
lobster chat --workspace ~/Documents/project1
lobster chat --workspace ~/Desktop/analysis
lobster query "analyze data" --workspace /tmp/quick_test

# All workspaces inherit from your global config!
```

**How it works:**
1. You run `lobster init --global` to set user-wide provider defaults
2. When you use `--workspace` with a directory that has no config:
   - Lobster checks `.lobster_workspace/provider_config.json` (not found)
   - Falls back to global config (platform-specific location)
   - Uses your defaults seamlessly

**Global config locations:**
- Linux/macOS: `~/.config/lobster/providers.json`
- Windows: `%APPDATA%\lobster\providers.json`

**Override for specific workspace:**
```bash
cd ~/special_project
lobster init  # Creates workspace-specific config
lobster chat  # Uses workspace config (overrides global)
```

**Best practices:**
- Use global config for your typical setup (e.g., Ollama for privacy)
- Use workspace config only when a project needs different settings
- Store API keys in environment variables (not in config files)

### Configuration Commands

Use the `lobster config` commands to manage and verify your configuration:

```bash
# Test API connectivity and validate configuration
lobster config test

# Display current configuration with masked secrets (simple view)
lobster config show

# Display detailed runtime configuration (shows per-agent models)
lobster config show-config

# List available providers
lobster config provider

# View available models for current provider
lobster config model
```

**New in v0.4.0:** The `lobster config show-config` command now displays actual runtime configuration using ConfigResolver and ProviderRegistry, showing:
- Active provider and configuration source
- Per-agent model assignments (see which model each agent uses)
- Profile information
- Configuration files status
- License tier and available agents

### Testing Your Configuration

The `lobster config test` command validates your LLM provider connectivity:

```bash
# Auto-detect and test current provider
lobster config test

# Test specific configuration profile
lobster config test --profile production

# Test specific agent in a profile
lobster config test --profile production --agent transcriptomics_expert
```

**Auto-detection behavior (when no `--profile` is specified):**

1. Detects your currently configured provider from:
   - `LOBSTER_LLM_PROVIDER` environment variable, or
   - Auto-detection (Ollama → Anthropic → Bedrock → Gemini)

2. Tests basic connectivity with a simple API call

3. Displays clear success/failure message with provider name

**Profile-based testing (when `--profile` is specified):**

- Tests all agents configured in that profile
- Validates model configurations and API access
- Useful for testing custom profile setups

### Advanced Options

```bash
# Reconfigure (creates timestamped backup of existing .env)
lobster init --force
lobster init --global --force

# Non-interactive mode for CI/CD and automation

# Option 1: Claude API (workspace)
lobster init --non-interactive \
  --anthropic-key=sk-ant-xxx

# Option 1b: Claude API (global defaults)
lobster init --global --non-interactive \
  --anthropic-key=sk-ant-xxx

# Option 2: AWS Bedrock
lobster init --non-interactive \
  --bedrock-access-key=AKIA... \
  --bedrock-secret-key=xxx

# Option 3: Ollama (Local) - Global
lobster init --global --non-interactive \
  --use-ollama

# Ollama with custom model
lobster init --non-interactive \
  --use-ollama \
  --ollama-model=mixtral:8x7b-instruct

# Option 4: Google Gemini - Global
lobster init --global --non-interactive \
  --gemini-key=your-google-api-key

# Gemini with specific model
lobster init --non-interactive \
  --gemini-key=your-google-api-key \
  --gemini-model=gemini-3-flash-preview

# With NCBI API key
lobster init --non-interactive \
  --anthropic-key=sk-ant-xxx \
  --ncbi-key=your-ncbi-key
```

**Global vs Workspace Configuration:**
- `lobster init` (default): Creates `.env` + workspace config in current directory
- `lobster init --global`: Creates `~/.config/lobster/providers.json` for all workspaces
- Global config is ideal for users who want consistent settings across all projects
- Workspace config overrides global config (project-specific needs)

### Manual Configuration

For advanced users, you can manually edit the `.env` file in your working directory. See the [Environment Variables](#environment-variables) and [API Key Management](#api-key-management) sections for details on available settings.

## Configuration Architecture (Advanced)

For developers extending Lobster AI or understanding the configuration system internals, this section documents the architecture patterns used for configuration management.

### Single Source of Truth Pattern

As of v0.4.0+, Lobster AI uses a **constants module** as the single source of truth for valid providers and profiles. This eliminates code duplication and ensures consistency across the codebase.

**File**: `lobster/config/constants.py`

```python
from typing import Final, List

# Valid LLM providers (single source of truth)
VALID_PROVIDERS: Final[List[str]] = ["anthropic", "bedrock", "ollama", "gemini"]

# Valid model profiles
VALID_PROFILES: Final[List[str]] = ["development", "production", "ultra", "godmode", "hybrid"]

# Provider display names for user interfaces
PROVIDER_DISPLAY_NAMES: Final[dict] = {
    "anthropic": "Anthropic Direct API",
    "bedrock": "AWS Bedrock",
    "ollama": "Ollama (Local)",
    "gemini": "Google Gemini",
}
```

**Benefits:**
- **No duplication**: Adding a new provider requires updating only `constants.py`
- **Type safety**: `Final[List[str]]` ensures immutability
- **Centralized**: All consumers import from same location
- **Maintainable**: Changes propagate automatically to all config classes

### Abstract Base Class Pattern

Configuration classes inherit from `ProviderConfigBase`, which provides shared validation logic and abstract properties.

**File**: `lobster/config/base_config.py`

```python
import abc
from pydantic import BaseModel, model_validator
from lobster.config.constants import VALID_PROVIDERS, VALID_PROFILES

class ProviderConfigBase(BaseModel, abc.ABC):
    """Abstract base for WorkspaceProviderConfig and GlobalProviderConfig."""

    @property
    @abc.abstractmethod
    def provider_field_name(self) -> str:
        """Name of the provider field (e.g., 'global_provider', 'default_provider')."""
        pass

    @property
    @abc.abstractmethod
    def model_field_suffix(self) -> str:
        """Suffix for model fields (e.g., '_model', '_default_model')."""
        pass

    @model_validator(mode="before")
    @classmethod
    def validate_providers_and_profiles(cls, data):
        """Shared validation for provider and profile fields."""
        # Validates global_provider, default_provider, profile, per_agent_providers
        # Uses VALID_PROVIDERS and VALID_PROFILES from constants.py
        ...

    def get_model_for_provider(self, provider: str) -> Optional[str]:
        """Get model name for provider (e.g., 'anthropic' -> 'anthropic_model')."""
        field_name = f"{provider}{self.model_field_suffix}"
        return getattr(self, field_name, None)
```

**Benefits:**
- **Shared validation**: Pydantic `model_validator` ensures consistency
- **Explicit contracts**: Abstract properties enforce implementation
- **DRY principle**: ~120 lines of duplicated code removed
- **Extensible**: Adding validation logic applies to all config classes

### Configuration Classes

**WorkspaceProviderConfig** (`lobster/config/workspace_config.py`):
```python
from lobster.config.base_config import ProviderConfigBase

class WorkspaceProviderConfig(ProviderConfigBase):
    """Workspace-specific provider configuration."""

    @property
    def provider_field_name(self) -> str:
        return "global_provider"

    @property
    def model_field_suffix(self) -> str:
        return "_model"

    # Fields: global_provider, anthropic_model, bedrock_model, ollama_model, gemini_model, etc.
```

**GlobalProviderConfig** (`lobster/config/global_config.py`):
```python
from lobster.config.base_config import ProviderConfigBase

class GlobalProviderConfig(ProviderConfigBase):
    """Global provider configuration."""

    @property
    def provider_field_name(self) -> str:
        return "default_provider"

    @property
    def model_field_suffix(self) -> str:
        return "_default_model"

    # Fields: default_provider, anthropic_default_model, bedrock_default_model, etc.
```

### Configuration Priority System (v0.4+)

Lobster AI uses a **5-layer priority system** for configuration resolution:

```
1. Runtime CLI flags (highest priority)
   ↓  --provider, --model
2. Workspace config
   ↓  .lobster_workspace/provider_config.json
3. Global user config
   ↓  ~/.config/lobster/providers.json
4. Environment variable
   ↓  LOBSTER_LLM_PROVIDER
5. FAIL with diagnostic message (lowest priority)
   ↓  No auto-detection, no silent defaults
```

**Implementation**: `lobster/core/config_resolver.py`

**Design Philosophy:**
- **No auto-detection**: Prevents unexpected costs from accidental API usage
- **Explicit configuration**: Users must consciously choose a provider
- **Diagnostic errors**: Shows exactly what was checked when config is missing
- **Multiple layers**: Supports workspace overrides and global defaults

**Example diagnostic error:**
```
❌ No provider configured.

Checked (in priority order):
  ✗ Runtime flag: --provider not provided
  ✗ Workspace config: .lobster_workspace/provider_config.json (not found)
  ✗ Global config: ~/.config/lobster/providers.json (not found)
  ✗ Environment: LOBSTER_LLM_PROVIDER (not set)

Quick Setup:
  lobster init              # Configure this workspace
  lobster init --global     # Set global defaults for all workspaces
```

### Adding a New Provider (Developer Guide)

To add a new LLM provider (e.g., "openai"):

1. **Update constants** (`lobster/config/constants.py`):
   ```python
   VALID_PROVIDERS: Final[List[str]] = ["anthropic", "bedrock", "ollama", "gemini", "openai"]
   PROVIDER_DISPLAY_NAMES["openai"] = "OpenAI API"
   ```

2. **Add provider class** (`lobster/config/providers/openai_provider.py`):
   ```python
   class OpenAIProvider(BaseProvider):
       """OpenAI provider implementation."""
       ...
   ```

3. **Register provider** (`lobster/config/providers/registry.py`):
   ```python
   PROVIDER_REGISTRY.register("openai", OpenAIProvider)
   ```

4. **Update config fields**:
   - `workspace_config.py`: Add `openai_model: Optional[str] = None`
   - `global_config.py`: Add `openai_default_model: Optional[str] = None`
   - Update `reset()` and `set_model_for_provider()` methods

5. **Update CLI** (`lobster/cli.py`):
   - Add OpenAI option to `lobster init` wizard
   - Add provider setup logic in `provider_setup.py`

6. **Regenerate allowlist**:
   ```bash
   python scripts/generate_allowlist.py --write
   ```

The constants pattern ensures that provider validation automatically includes the new provider across all configuration classes without additional changes.

## Security Best Practices

-   **Never commit `.env` files** to version control.
-   Use system environment variables or a secrets management tool for production.
-   Rotate API keys regularly.

## Troubleshooting Configuration

### Diagnostic Error Messages (v0.4+)

When provider configuration is missing, Lobster now shows **exactly what was checked**:

```
❌ No provider configured.

Checked (in priority order):
  ✗ Runtime flag: --provider not provided
  ✗ Workspace config: /path/to/workspace/provider_config.json (not found)
  ✗ Global config: ~/.config/lobster/providers.json (not found)
  ✗ Environment: LOBSTER_LLM_PROVIDER (not set)

Quick Setup:
  lobster init              # Configure this workspace
  lobster init --global     # Set global defaults for all workspaces

Or set environment variable:
  export LOBSTER_LLM_PROVIDER=anthropic
```

This makes troubleshooting much easier - you can see which configuration sources were checked and why they failed.

### Common Issues

**Issue: "No provider configured" in external workspace**
```bash
# Solution: Set global defaults (one-time)
lobster init --global
# Now all external workspaces work automatically
```

**Issue: Wrong provider being used**
```bash
# Check priority order - workspace overrides global
lobster config show-config  # See which config is active

# Force specific provider for this session
lobster chat --provider anthropic
```

**Issue: Invalid environment variable**
```bash
# v0.4+ raises explicit error for invalid values
export LOBSTER_LLM_PROVIDER=typo
lobster chat
# Error: Invalid provider 'typo' in LOBSTER_LLM_PROVIDER
```

### Configuration Commands

-   Use `lobster config show` to see your current configuration with masked secrets.
-   Use `lobster config show-config` to see runtime resolution (provider, models, sources).
-   Use `lobster config test` to validate API connectivity and test your configuration.
-   Use `lobster init --force` to reconfigure (creates a backup of your existing .env file).
-   Use `lobster init --global` to set user-wide defaults for all workspaces.
-   Run `lobster chat --debug` for verbose configuration loading information.
-   If you see "No configuration found" errors, run `lobster init` to create your .env file.
