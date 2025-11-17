# Comprehensive Installation Guide

This guide covers all installation methods for Lobster AI, from quick setup to advanced development configurations.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation Methods](#installation-methods)
- [Platform-Specific Instructions](#platform-specific-instructions)
- [Verification](#verification)
- [Development Installation](#development-installation)
- [Optional Dependencies](#optional-dependencies)
  - [PyMOL (Protein Structure Visualization)](#pymol-protein-structure-visualization)
  - [Docling (Advanced PDF Parsing)](#docling-advanced-pdf-parsing)
  - [AWS Bedrock (Enhanced Setup)](#aws-bedrock-enhanced-setup)
  - [Cloud Mode Configuration](#cloud-mode-configuration)
- [Docker Deployment](#docker-deployment)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

**Minimum Requirements:**
- **Python**: 3.12+
- **Memory**: 4GB RAM (8GB+ recommended for large datasets)
- **Storage**: 2GB free space (more for data analysis)
- **Network**: Internet connection for API access and data downloads

**Recommended Setup:**
- **Python**: 3.12+
- **Memory**: 16GB+ RAM
- **Storage**: 10GB+ free space
- **CPU**: Multi-core processor for parallel analysis

### Package Manager Recommendations

Lobster AI automatically detects and uses the best available package manager:

1. **uv** (fastest, recommended): `pip install uv`
2. **pip3** (macOS default)
3. **pip** (fallback)

### Required API Keys

Choose ONE of the following LLM providers:

1. **Claude API Key** (Recommended for most users)

   ⚠️ **Important: Rate Limits** - Anthropic applies conservative rate limits to new accounts. For production use or heavy workloads, we recommend AWS Bedrock. If you encounter rate limit errors, see [Troubleshooting Guide](28-troubleshooting.md).

   - Visit [Anthropic Console](https://console.anthropic.com/)
   - Create account and generate API key
   - Add to `.env`: `ANTHROPIC_API_KEY=sk-ant-...`
   - **Recommended for**: Quick testing, development with small datasets
   - **Not recommended for**: Production deployments, large-scale analysis

2. **AWS Bedrock Access** (Recommended for Production)

   ✅ **Best for production** - AWS Bedrock provides enterprise-grade rate limits and reliability. Recommended for heavy workloads and production deployments.

   - AWS account with Bedrock access
   - Create IAM user with Bedrock permissions
   - Add to `.env`:
     ```
     AWS_BEDROCK_ACCESS_KEY=...
     AWS_BEDROCK_SECRET_ACCESS_KEY=...
     ```
   - **Recommended for**: Production deployments, large-scale analysis, enterprise use
   - **Benefits**: Higher rate limits, better reliability, enterprise SLA

3. **NCBI API Key** (Optional)
   - Visit [NCBI E-utilities](https://ncbiinsights.ncbi.nlm.nih.gov/2017/11/02/new-api-keys-for-the-e-utilities/)
   - Enhances literature search capabilities

## Installation Methods

### Method 1: Quick Install (Recommended)

The easiest and most reliable installation method:

```bash
# Clone repository
git clone https://github.com/the-omics-os/lobster-local.git
cd lobster-local

# One-command installation
make install
```

**What this does:**
1. Verifies Python 3.12+ installation
2. Creates virtual environment at `.venv`
3. Installs all dependencies automatically
4. Sets up configuration files
5. Provides activation instructions

### Method 2: Development Install

For contributors and developers:

```bash
git clone https://github.com/the-omics-os/lobster-local.git
cd lobster-local

# Install with development dependencies
make dev-install
```

**Additional features:**
- Testing framework (pytest, coverage)
- Code quality tools (black, isort, pylint, mypy)
- Pre-commit hooks
- Documentation tools

### Method 3: Manual Installation

For full control over the installation process:

```bash
git clone https://github.com/the-omics-os/lobster-local.git
cd lobster-local
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Upgrade pip and install build tools
pip install --upgrade pip build wheel

# Install Lobster AI
pip install -e .
```

### Method 4: Global Installation

Install the `lobster` command globally (Unix/macOS):

```bash
# First, install locally
make install

# Then install globally
make install-global
```

This creates a symlink in `/usr/local/bin/lobster` allowing you to run `lobster` from anywhere.

### Method 5: Package Manager Installation (Future)

The `lobster` package is not yet available on PyPI. Once it is published, you will be able to install it using pip.

**For now, please install the package directly from the repository by following Method 1 (Quick Install) or Method 3 (Manual Installation).**

When available on PyPI, the commands will be:

```bash
# Standard installation
pip install lobster

# Development installation
pip install lobster[dev]

# All extras
pip install lobster[all]
```

## Platform-Specific Instructions

### macOS

**Homebrew Setup (Recommended):**
```bash
# Install Python 3.12+
brew install python@3.12

# Optional: Install uv for faster package management
brew install uv

git clone https://github.com/the-omics-os/lobster-local.git
cd lobster-local
make install
```

**Issues with System Python:**
If using system Python causes issues:
```bash
# Use Homebrew Python explicitly
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Linux (Ubuntu/Debian)

**Install Dependencies:**
```bash
# Update package list
sudo apt update

# Install Python 3.12+
sudo apt install python3.12 python3.12-venv python3.12-dev

# Clone and install
git clone https://github.com/the-omics-os/lobster-local.git
cd lobster-local
make install
```

**CentOS/RHEL/Fedora:**
```bash
# Install Python 3.12+
sudo dnf install python3.12 python3.12-devel

# Install development tools
sudo dnf groupinstall "Development Tools"

    # Clone and install
    git clone https://github.com/the-omics-os/lobster-local.git
    cd lobster-localmake install
```

### Windows

**Using PowerShell:**
```powershell
# Install Python 3.12+ from python.org
# Ensure Python is in PATH

# Clone repository
git clone https://github.com/the-omics-os/lobster-local.git
cd lobster-local

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install
pip install --upgrade pip wheel
pip install -e .
```

**Using Windows Subsystem for Linux (WSL):**
Follow the Linux installation instructions within WSL.

### Python Version Considerations

**Python 3.12+ Requirements:**
- **pyproject.toml specifies**: `>=3.12`
- **Makefile enforces**: `>=3.12`
- **Recommendation**: Use Python 3.12+ for best performance

**Installing Specific Python Version:**
```bash
# macOS with pyenv
brew install pyenv
pyenv install 3.12.0
pyenv local 3.12.0

# Ubuntu with deadsnakes PPA
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.12 python3.12-venv
```

## Verification

### Test Installation

After installation, verify everything works:

```bash
# Activate environment
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Test CLI
lobster --help

# Test imports
python -c "import lobster; print('✅ Lobster imported successfully')"

# Run verification script
python verify_installation.py
```

### Check System Status

```bash
# Start Lobster and check status
lobster chat

# In the chat interface, type:
/status
```

Expected output:
```
✅ System Status: Healthy
✅ Environment: Virtual environment active
✅ Dependencies: All packages installed
✅ Configuration: .env file present
⚠️  API Keys: Configure in .env file
```

### Verify API Connectivity

```bash
# Test API keys (after configuration)
lobster config test
```

## Development Installation

### Full Development Setup

```bash
git clone https://github.com/the-omics-os/lobster-local.git
cd lobster-local

# Development installation
make dev-install

# This installs:
# - All runtime dependencies
# - Testing framework (pytest, pytest-cov, pytest-xdist)
# - Code quality (black, isort, flake8, pylint, mypy)
# - Security tools (bandit)
# - Documentation (mkdocs)
# - Pre-commit hooks
```

### Development Commands

```bash
# Run tests
make test

# Fast parallel testing
make test-fast

# Code formatting
make format

# Linting
make lint

# Type checking
make type-check

# Clean installation
make clean-install
```

### Pre-commit Hooks

Development installation automatically sets up pre-commit hooks:

```bash
# Manual setup if needed
make setup-pre-commit

# Run on all files
pre-commit run --all-files
```

## Optional Dependencies

These optional components enhance Lobster AI with advanced features. Install based on your analysis needs.

### PyMOL (Protein Structure Visualization)

PyMOL enables 3D protein structure visualization and analysis (v2.4+).

**Automated Installation (macOS):**
```bash
cd lobster
make install-pymol
```

**Manual Installation:**

#### macOS
```bash
# Via Homebrew
brew install brewsci/bio/pymol

# Verify installation
pymol -c -Q
```

#### Linux (Ubuntu/Debian)
```bash
# Via apt
sudo apt-get update
sudo apt-get install pymol

# Verify installation
which pymol
pymol -c -Q
```

#### Linux (Fedora/RHEL)
```bash
# Via DNF
sudo dnf install pymol

# Verify installation
pymol -c -Q
```

#### Docker
PyMOL is pre-installed in the Docker image - no additional setup needed.

**Usage:**
```bash
# In Lobster chat
🦞 You: "Fetch protein structure 1AKE"
🦞 You: "Visualize 1AKE with PyMOL mode=interactive style=cartoon"
🦞 You: "Link protein structures to my RNA-seq data"
```

**Troubleshooting:**
If PyMOL is not found, check installation:
```bash
which pymol
pymol --version
```

See [Protein Structure Visualization Guide](40-protein-structure-visualization.md) for complete usage details.

### Docling (Advanced PDF Parsing)

Docling provides professional-grade PDF parsing for extracting methods from scientific publications (v2.4+).

**Installation:**
```bash
# Basic Docling
pip install docling

# Full installation with all features
pip install "docling[all]"

# With table extraction
pip install "docling[table]"

# With OCR support
pip install "docling[ocr]"
```

**Verify Installation:**
```bash
python -c "from docling.document_converter import DocumentConverter; print('✓ Docling installed')"
```

**Benefits:**
- **>90% Methods section detection** (vs 30% with PyPDF2 fallback)
- **Table extraction** from scientific papers
- **Formula recognition** in publications
- **Better structure detection** for complex PDFs

**Usage:**
```bash
# Docling is used automatically by ContentAccessService
🦞 You: "Extract methods from PMID:38448586"
🦞 You: "Read full publication PMID:35042229"
```

**Fallback Behavior:**
If Docling is not installed, Lobster automatically falls back to PyPDF2 with reduced functionality.

**Troubleshooting:**
```bash
# Test Docling functionality
python -c "import docling; print(docling.__version__)"

# Check dependencies
pip list | grep docling
```

See [Publication Intelligence Guide](37-publication-intelligence-deep-dive.md) for technical details.

### AWS Bedrock (Enhanced Setup)

Detailed AWS Bedrock configuration for production deployments.

**Step 1: Create AWS Account**
1. Visit [AWS Console](https://console.aws.amazon.com/)
2. Create account or sign in
3. Navigate to AWS Bedrock service

**Step 2: Request Model Access**
```bash
# Navigate to: AWS Bedrock → Model Access
# Request access to: Claude 3.5 Sonnet, Claude 3 Opus
# Approval typically takes 1-2 business days
```

**Step 3: Create IAM User**
```bash
# AWS Console → IAM → Users → Create User
# User name: lobster-ai-user
# Access type: Programmatic access

# Attach policy: AmazonBedrockFullAccess
# OR create custom policy (recommended):
```

**Custom IAM Policy (Least Privilege):**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream",
        "bedrock:ListFoundationModels"
      ],
      "Resource": "*"
    }
  ]
}
```

**Step 4: Configure Credentials**
```bash
# Option 1: AWS CLI configuration (recommended)
aws configure
# Enter: Access Key ID, Secret Access Key, Region (us-east-1), Output format (json)

# Option 2: Environment variables
export AWS_BEDROCK_ACCESS_KEY=AKIAIOSFODNN7EXAMPLE
export AWS_BEDROCK_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
export AWS_DEFAULT_REGION=us-east-1

# Option 3: .env file (for Lobster)
cat >> .env << EOF
AWS_BEDROCK_ACCESS_KEY=AKIAIOSFODNN7EXAMPLE
AWS_BEDROCK_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
AWS_DEFAULT_REGION=us-east-1
EOF
```

**Step 5: Verify Access**
```bash
# Test Bedrock connectivity
aws bedrock list-foundation-models --region us-east-1

# Test in Lobster
lobster chat
> /status
# Should show: "Model: AWS Bedrock (Claude)"
```

**Troubleshooting AWS Bedrock:**
```bash
# Check credentials
aws sts get-caller-identity

# Test model access
aws bedrock list-foundation-models --region us-east-1 | grep Claude

# Common issues:
# 1. Model access not approved → Wait for approval or request again
# 2. Wrong region → Bedrock availability varies by region
# 3. IAM permissions → Verify user has bedrock:InvokeModel permission
```

**Regional Availability:**
AWS Bedrock Claude models are available in:
- `us-east-1` (US East, N. Virginia) - Recommended
- `us-west-2` (US West, Oregon)
- `eu-west-1` (Europe, Ireland)
- `ap-southeast-1` (Asia Pacific, Singapore)

See [AWS Bedrock Regions](https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-regions.html) for current availability.

### Cloud Mode Configuration

Enable cloud processing for large-scale analyses (v2.4+).

**Setup:**
```bash
# 1. Request cloud API key
# Email: info@omics-os.com
# Subject: "Lobster Cloud API Key Request"
# Include: Organization name, use case, expected usage

# 2. Configure API key
export LOBSTER_CLOUD_KEY="your-cloud-api-key-here"

# 3. Start Lobster in cloud mode
lobster chat

# 4. Verify cloud mode active
> /status
# Should show: "Cloud mode: active"
```

**Benefits:**
- **Scalable compute** for datasets >100K cells
- **No local memory limits** for large datasets
- **Faster processing** with distributed infrastructure
- **Automatic resource management**

**Usage:**
```bash
# Cloud mode is automatic when LOBSTER_CLOUD_KEY is set
🦞 You: "Download GSE123456 and analyze with cloud resources"
🦞 You: "Process this large dataset using cloud infrastructure"

# Switch back to local mode
unset LOBSTER_CLOUD_KEY
lobster chat
```

**Cost Structure:**
- Free tier: 10 analyses/month
- Pro tier: $6K-$18K/year (based on usage)
- Enterprise: Custom pricing

**Troubleshooting Cloud Mode:**
```bash
# Check API key is set
echo $LOBSTER_CLOUD_KEY

# Test cloud connectivity
lobster chat
> /status

# Common issues:
# 1. API key not set → Export LOBSTER_CLOUD_KEY
# 2. Key expired → Request new key from info@omics-os.com
# 3. Network timeout → Check firewall/proxy settings
```

See [Configuration Guide](03-configuration.md) for complete cloud setup details.

## Docker Deployment

Lobster supports Docker for both CLI and FastAPI server modes. For comprehensive deployment guides, see [Docker Deployment Guide](42-docker-deployment-guide.md).

### Quick Start with Docker

```bash
# 1. Build images
make docker-build

# 2. Run CLI interactively
make docker-run-cli

# 3. Or run FastAPI server
make docker-run-server
```

### Build Docker Images

```bash
# Build both CLI and server images
make docker-build

# Or manually
docker build -t omicsos/lobster:latest -f Dockerfile .
docker build -t omicsos/lobster:server -f Dockerfile.server .
```

### Run CLI with Docker

```bash
# Using Makefile (recommended)
make docker-run-cli

# Or manually with environment file
docker run -it --rm \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  -v lobster-workspace:/app/.lobster_workspace \
  omicsos/lobster:latest chat

# Single query mode (automation)
docker run --rm \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  omicsos/lobster:latest query "download GSE12345"
```

### Run FastAPI Server with Docker

```bash
# Using Makefile (recommended)
make docker-run-server

# Or manually
docker run -d \
  --name lobster-api \
  -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  omicsos/lobster:server

# Check server health
curl http://localhost:8000/health

# Stop server
docker stop lobster-api
```

### Docker Compose

```bash
# Run CLI interactively
make docker-compose-cli

# Start FastAPI server in background
make docker-compose-up

# View logs
docker-compose logs -f lobster-server

# Stop all services
make docker-compose-down
```

**docker-compose.yml** supports both CLI and server modes. See [Docker Deployment Guide](42-docker-deployment-guide.md) for full configuration details.

## Troubleshooting

### Common Installation Issues

#### Python Version Problems

**Error**: `Python 3.12+ is required`

**Solutions:**
```bash
# Check Python version
python --version
python3 --version

# Install Python 3.12+
# macOS: brew install python@3.12
# Ubuntu: sudo apt install python3.12
# Windows: Download from python.org

# Use specific Python version
python3.12 -m venv .venv
```

#### Virtual Environment Issues

**Error**: `Failed to create virtual environment`

**Solutions:**
```bash
# Install venv module (Ubuntu/Debian)
sudo apt install python3.12-venv

# Clear existing environment
rm -rf .venv

# Create manually
python3 -m venv .venv --clear

# Alternative method
python3 -m venv .venv --without-pip
source .venv/bin/activate
curl https://bootstrap.pypa.io/get-pip.py | python
```

#### Dependency Installation Failures

**Error**: `Failed building wheel for [package]`

**Solutions:**
```bash
# Install development headers (Linux)
sudo apt install python3.12-dev build-essential

# Update pip and setuptools
pip install --upgrade pip setuptools wheel

# Clear pip cache
pip cache purge

# Install with no cache
pip install --no-cache-dir -e .

# Use uv for faster, more reliable installs
pip install uv
uv pip install -e .
```

#### Permission Errors

**Error**: `Permission denied`

**Solutions:**
```bash
# Don't use sudo with pip in virtual environment
# Instead, ensure virtual environment ownership
chown -R $USER:$USER .venv

# For global installation (Unix only)
sudo make install-global
```

#### Memory Issues During Installation

**Error**: `Killed` or memory-related errors

**Solutions:**
```bash
# Increase swap space (Linux)
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Install with limited parallelism
pip install -e . --no-build-isolation

# Use development profile for lighter resource usage
export LOBSTER_PROFILE=development
make install
```

### Runtime Issues

#### API Key Problems

**Error**: API key not found or invalid

**Solutions:**
```bash
# Check environment variables
echo $ANTHROPIC_API_KEY    # For Claude API
echo $AWS_BEDROCK_ACCESS_KEY  # For AWS Bedrock
source .env  # Load from file

# Test API connectivity
lobster config test

# Regenerate API keys if needed
```

#### Import Errors

**Error**: `ModuleNotFoundError: No module named 'lobster'`

**Solutions:**
```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Reinstall in development mode
pip install -e .

# Check PYTHONPATH
python -c "import sys; print(sys.path)"
```

#### Memory Issues During Analysis

**Solutions:**
```bash
# Use development profile (lighter than production)
export LOBSTER_PROFILE=development

# Reduce file size limits
export LOBSTER_MAX_FILE_SIZE_MB=100

# Monitor memory usage
htop  # Linux/macOS
# Task Manager on Windows
```

### Getting Additional Help

#### Check System Health
```bash
lobster chat
/dashboard  # Comprehensive system overview
/status     # Quick status check
```

#### Enable Debug Mode
```bash
# Verbose logging
lobster chat --debug --verbose

# Show reasoning
lobster chat --reasoning
```

#### Log Files
```bash
# Check logs in workspace
ls .lobster_workspace/logs/

# Enable detailed logging
export LOBSTER_LOG_LEVEL=DEBUG
```

#### Community Support

- **GitHub Issues**: [Report bugs](https://github.com/the-omics-os/lobster/issues)
- **Discord**: [Join community](https://discord.gg/HDTRbWJ8omicsos)
- **Email**: [Direct support](mailto:info@omics-os.com)
- **Documentation**: [Full docs](README.md)

### Clean Reinstallation

If all else fails, perform a clean reinstallation:

```bash
# Remove everything
make uninstall      # Remove virtual environment
make clean          # Remove build artifacts
rm -rf .lobster_workspace  # Remove workspace (optional)

# Fresh installation
make clean-install

# Or manually
rm -rf .venv
git clean -fdx  # Warning: removes all untracked files
make install
```

---

**Next Steps**: Once installation is complete, see the [Configuration Guide](03-configuration.md) to set up API keys and customize your Lobster AI environment.