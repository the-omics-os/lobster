# Getting Started with Lobster AI

Welcome to Lobster AI! This quick start guide will have you analyzing bioinformatics data in minutes.

## Quick Start (5 minutes)

### 1. Prerequisites
- **Python 3.11+** (Python 3.12+ recommended)
- **Git** for cloning the repository
- **LLM Provider** (choose one):
  - [Claude API Key](https://console.anthropic.com/) (Recommended - simpler setup)
  - [AWS Bedrock Access](https://console.aws.amazon.com/) (For AWS users)
- **NCBI API Key** (optional, for enhanced literature search)

### 2. Choose Installation Method

**Quick Decision:**
- Want `lobster` command everywhere? → **Global Installation**
- Working on specific project? → **Local Installation**

#### Global Installation (Recommended for Most Users)

```bash
# Install uv if needed
# macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Lobster globally
uv tool install lobster-ai
```

This makes `lobster` command available system-wide.

#### Local Installation (Project-Specific)

```bash
# In your project directory
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install lobster-ai
```

This installs Lobster in your project's virtual environment only.

### 3. Configure API Keys

Run the interactive configuration wizard:

```bash
# For local installation, activate virtual environment first
source .venv/bin/activate  # Skip if using global installation

# Launch configuration wizard
lobster init
```

The wizard will guide you through:
- ✅ Choosing your LLM provider (Anthropic, AWS Bedrock, or Ollama)
- ✅ Entering your API keys securely (input is masked)
- ✅ Optionally configuring NCBI API key for enhanced literature search
- ✅ Creating two config files:
  - `provider_config.json` - Provider selection (safe to commit)
  - `.env` - API keys and secrets (never commit)

**Configuration management:**
```bash
# Test your configuration
lobster config test

# View current configuration (secrets masked)
lobster config show

# Reconfigure (creates backup)
lobster init --force
```

**Manual configuration** (advanced users only):
If you prefer, create two files manually:

**1. `.lobster_workspace/provider_config.json` (provider selection):**
```json
{
  "global_provider": "anthropic",
  "anthropic_model": "claude-sonnet-4-20250514",
  "profile": "production"
}
```

**2. `.env` (API keys - never commit to git):**
```env
# Option 1: Anthropic API
ANTHROPIC_API_KEY=sk-ant-api03-xxxxx

# Option 2: AWS Bedrock
AWS_BEDROCK_ACCESS_KEY=your-aws-access-key
AWS_BEDROCK_SECRET_ACCESS_KEY=your-aws-secret-key

# Option 3: Ollama (local)
OLLAMA_BASE_URL=http://localhost:11434

# Optional: NCBI API key
NCBI_API_KEY=your-ncbi-api-key-here
```

### 4. Test Installation

**Global installation:**
```bash
# Command works from anywhere
lobster --help
lobster chat
```

**Local installation:**
```bash
# Activate virtual environment first
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Then use Lobster
lobster --help
lobster chat
```

### 5. Your First Analysis

Try this example in the chat interface:

```
🦞 You: Download and analyze GSE109564 from GEO database

🦞 Lobster: I'll download and analyze this single-cell RNA-seq dataset for you...

[Downloads data and performs quality control, filtering, clustering, and marker gene analysis]
```

## What Can Lobster Do?

### Single-Cell RNA-seq Analysis
- Quality control and filtering
- Normalization and scaling
- Clustering (Leiden algorithm)
- Marker gene identification
- Cell type annotation
- Trajectory analysis

### Bulk RNA-seq Analysis
- Differential expression with pyDESeq2
- Complex experimental design support
- Batch effect correction
- Pathway enrichment analysis

### Proteomics Analysis
- Mass spectrometry data processing
- Missing value pattern analysis
- Statistical testing with FDR control
- Protein network analysis
- Affinity proteomics (Olink panels)

### Multi-Omics Integration
- Cross-platform data integration
- Joint clustering and analysis
- Correlation network analysis
- Integrated visualization

### Literature Mining
- Automatic parameter extraction from publications
- PubMed literature search
- GEO dataset discovery
- Method validation against published results

## Essential Commands

Once in the chat interface (`lobster chat`), try these commands:

| Command | Description |
|---------|-------------|
| `/help` | Show all available commands |
| `/status` | Display system status and health |
| `/files` | List files in your workspace |
| `/data` | Show current datasets |
| `/plots` | List generated visualizations |
| `/read <file>` | Display file contents |
| `/export` | Export complete analysis |

## Example Workflows

### Quick Dataset Analysis
```
🦞 You: Load my_data.csv as single-cell RNA-seq data and perform standard analysis
```

### Literature-Guided Analysis
```
🦞 You: Find optimal clustering parameters for my single-cell data based on recent publications
```

### Multi-Dataset Integration
```
🦞 You: Load GSE12345 and GSE67890, then compare their cell populations
```

### Custom Analysis
```
🦞 You: Perform differential expression between clusters 2 and 5 using a negative binomial model
```

## Getting Help

- **Type `/help`** in the chat for command reference
- **Check `/status`** to verify system health
- **Use `/files`** to see what data is available
- **Try `/dashboard`** for system overview

## Next Steps

Regardless of installation method (global or local), these guides work the same:

1. **[Installation Guide](02-installation.md)** - Detailed installation options
2. **[Configuration Guide](03-configuration.md)** - Advanced configuration
3. **[Main Documentation](README.md)** - Complete feature overview
4. **[Discord Community](https://discord.gg/HDTRbWJ8omicsos)** - Get help from the community

## Troubleshooting Quick Fixes

**Installation fails?**
```bash
make clean-install  # Clean installation
```

**API errors?**
```bash
lobster config test  # Test API connectivity
```

**Memory issues?**
```bash
export LOBSTER_PROFILE=development  # Use lighter resource profile
```

**Need help?**
```bash
lobster chat  # Then type /help
```

## Uninstalling

### Remove Lobster

**Global installation:**
```bash
uv tool uninstall lobster-ai
```

**Local installation:**
```bash
source .venv/bin/activate
pip uninstall lobster-ai
deactivate
rm -rf .venv  # Optional: remove virtual environment
```

### Remove User Data (Optional)

⚠️ Deletes all analysis history!

```bash
rm -rf ~/.lobster ~/.lobster_workspace
```

For more details, see the [Installation Guide](02-installation.md#uninstalling-lobster-ai).

---

**Ready to dive deeper?** Check out the [comprehensive installation guide](02-installation.md) for advanced setup options, Docker deployment, and development installation.