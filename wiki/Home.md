# 🦞 Lobster AI Documentation

[![License: AGPL-3.0-or-later](https://img.shields.io/badge/License-AGPL%203.0--or--later-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Documentation: CC BY 4.0](https://img.shields.io/badge/Documentation-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Welcome to the comprehensive documentation for **Lobster AI** - the AI-powered multi-omics bioinformatics analysis platform. This documentation provides everything you need to use, develop, and extend Lobster AI.

## 📚 Documentation Structure

### 🚀 **Getting Started**
Start here if you're new to Lobster AI
- [**01 - Getting Started**](01-getting-started) - Quick 5-minute setup guide
- [**02 - Installation**](02-installation) - Comprehensive installation instructions
- [**03 - Configuration**](03-configuration) - API keys, environment setup, and model profiles

### 👤 **User Guide**
Learn how to use Lobster AI for your research
- [**04 - User Guide Overview**](04-user-guide-overview) - Understanding how Lobster AI works
- [**05 - CLI Commands**](05-cli-commands) - Complete command reference with examples
- [**06 - Data Analysis Workflows**](06-data-analysis-workflows) - Step-by-step analysis guides
- [**07 - Data Formats**](07-data-formats) - Supported input/output formats

### 💻 **Developer Guide**
Extend and contribute to Lobster AI
- [**08 - Developer Overview**](08-developer-overview) - Architecture and development setup
- [**09 - Creating Agents**](09-creating-agents) - Build new specialized AI agents
- [**10 - Creating Services**](10-creating-services) - Implement analysis services
- [**11 - Creating Adapters**](11-creating-adapters) - Add support for new data formats
- [**12 - Testing Guide**](12-testing-guide) - Writing and running tests
- [**49 - Custom Feature Agent**](49-custom-feature-agent-automated-generation) 🆕 - AI-powered automated feature generation with Claude Code SDK ✨

### 📖 **API Reference**
Complete API documentation
- [**13 - API Overview**](13-api-overview) - API organization and conventions
- [**14 - Core API**](14-core-api) - DataManagerV2 and client interfaces
- [**15 - Agents API**](15-agents-api) - Agent tools and capabilities
- [**16 - Services API**](16-services-api) - Analysis service interfaces
- [**17 - Interfaces API**](17-interfaces-api) - Abstract interfaces and contracts

### 🏗️ **Architecture & Internals**
Deep dive into system design
- [**18 - Architecture Overview**](18-architecture-overview) - System design and components
- [**19 - Agent System**](19-agent-system) - Multi-agent coordination architecture
- [**20 - Data Management**](20-data-management) - DataManagerV2 and modality system
- [**21 - Cloud/Local Architecture**](21-cloud-local-architecture) - Hybrid deployment design
- [**22 - Performance Optimization**](22-performance-optimization) - Memory and speed optimizations

### 🔬 **Advanced Features & Internals**
Deep dives into specialized capabilities and system internals (v0.2+)

**Agent Enhancements:**
- [**31 - Data Expert Agent Enhancements**](31-data-expert-agent-enhancements) - Workspace restoration and session continuity
- [**32 - Agent-Guided Formula Construction**](32-agent-guided-formula-construction) - Interactive formula design for DE analysis
- [**36 - Supervisor Configuration**](36-supervisor-configuration) - Dynamic agent registry and auto-discovery
- [**45 - Agent Customization Advanced**](45-agent-customization-advanced) - Advanced agent development patterns

**Content & Publication Intelligence:**
- [**37 - Publication Intelligence Deep Dive**](37-publication-intelligence-deep-dive) 🆕 - Docling integration & PDF parsing ✨
- [**38 - Workspace Content Service**](38-workspace-content-service) - Type-safe caching for research content

**Infrastructure & Performance:**
- [**35 - Download Queue System**](35-download-queue-system) 🆕 - Robust multi-step data acquisition with JSONL persistence ✨
- [**39 - Two-Tier Caching Architecture**](39-two-tier-caching-architecture) - 30-50x speedup on repeat content access
- [**43 - Docker Deployment Guide**](43-docker-deployment-guide) - Production containerization strategies
- [**47 - Fix #7: HTTPS GEO Download**](47-fix7-https-geo-download) 🆕 - 20x reliability improvement (91% → <5% corruption) ✨

**Specialized Features:**
- [**40 - Protein Structure Visualization**](40-protein-structure-visualization) 🆕 - PyMOL integration for 3D protein analysis ✨
- [**43 - S3 Backend Guide**](43-s3-backend-guide) - Cloud storage integration
- [**46 - Multi-Omics Integration**](46-multiomics-integration) - Cross-platform analysis workflows

**Migration & Maintenance:**
- [**41 - Migration Guides**](41-migration-guides) - Upgrade paths and breaking changes
- [**44 - Maintaining Documentation**](44-maintaining-documentation) - Documentation workflows and standards

### 🎯 **Tutorials & Examples**
Learn by doing with practical tutorials
- [**23 - Single-Cell RNA-seq Tutorial**](23-tutorial-single-cell) - Complete workflow with real data
- [**24 - Bulk RNA-seq Tutorial**](24-tutorial-bulk-rnaseq) - Differential expression analysis
- [**26 - Custom Agent Tutorial**](26-tutorial-custom-agent) - Create your own agent
- [**27 - Examples Cookbook**](27-examples-cookbook) - Code recipes and patterns

### 🔧 **Support & Reference**
Help and additional resources
- [**28 - Troubleshooting**](28-troubleshooting) - Common issues and solutions
- [**29 - FAQ**](29-faq) - Frequently asked questions
- [**30 - Glossary**](30-glossary) - Bioinformatics and technical terms

## 🎯 Quick Navigation by Task

### **"I want to..."**

#### **Get Started Quickly**
- [Install Lobster AI in 5 minutes](01-getting-started.md)
- [Configure my API keys](03-configuration.md)
- [Run my first analysis](01-getting-started.md)

#### **Analyze My Data**
- [Analyze single-cell RNA-seq data](23-tutorial-single-cell.md)
- [Perform bulk RNA-seq differential expression](24-tutorial-bulk-rnaseq.md)
- [Download and analyze GEO datasets](06-data-analysis-workflows.md)

#### **Understand the System**
- [Learn about the architecture](18-architecture-overview.md)
- [Understand how agents work](19-agent-system.md)
- [See supported data formats](07-data-formats.md)

#### **Extend Lobster AI**
- [Create a new agent](09-creating-agents.md)
- [Add a new analysis service](10-creating-services.md)
- [Support a new data format](11-creating-adapters.md)
- [Contribute to the project](08-developer-overview.md)

#### **Solve Problems**
- [Fix installation issues](28-troubleshooting.md)
- [Resolve data loading errors](28-troubleshooting.md)
- [Debug analysis failures](28-troubleshooting.md)

#### **Master Advanced Features**
- [Understand the two-tier caching system](39-two-tier-caching-architecture.md)
- [Implement custom download workflows](35-download-queue-system.md)
- [Optimize publication content access](37-publication-intelligence-deep-dive.md)
- [Visualize protein structures with PyMOL](40-protein-structure-visualization.md)
- [Deploy with Docker in production](43-docker-deployment-guide.md)

## 🌟 Key Features

### **🤖 AI-Powered Analysis**
- Natural language interface for complex bioinformatics
- 8+ specialized AI agents for different analysis domains
- Intelligent workflow coordination and parameter optimization

### **🧬 Scientific Capabilities**
- **Single-Cell RNA-seq**: QC, clustering, annotation, trajectory analysis
- **Bulk RNA-seq**: pyDESeq2 differential expression with complex designs
- **Multi-Omics**: Integrated cross-platform analysis

### **☁️ Deployment Flexibility**
- **Local Mode**: Full privacy with data on your machine
- **Cloud Mode**: Scalable computing with managed infrastructure
- **Hybrid**: Automatic switching between modes

### **📊 Professional Features**
- Publication-ready visualizations
- W3C-PROV compliant provenance tracking
- Comprehensive quality control metrics
- Batch effect detection and correction

## 📈 Version Highlights

> **Current Release**: v0.2 is the first public release of Lobster AI. See the [comprehensive documentation](41-migration-guides.md) for features and upgrade information.

### **Current Features (v0.2)** ✨

**Content Intelligence & Publications:**
- 🧬 **Protein Structure Visualization** - PyMOL integration for 3D protein visualization and analysis ([Details](40-protein-structure-visualization.md))
- 🔌 **ContentAccessService** - Unified publication/dataset access with 5 specialized providers ([Details](37-publication-intelligence-deep-dive.md))
- 📄 **Docling PDF Parsing** - Structure-aware Methods section extraction with >90% hit rate ([Details](37-publication-intelligence-deep-dive.md))
- 📊 **Table Extraction** - Parameter tables from scientific publications
- 🧮 **Formula Preservation** - Mathematical formulas in LaTeX format

**Data Management:**
- 📥 **Download Queue System** - Robust multi-step data acquisition with JSONL persistence ([Details](35-download-queue-system.md))
- ⚡ **Enhanced Two-Tier Caching** - 30-50x speedup on repeat content access (0.2-0.5s cached)
- 🔄 **Workspace Restoration** - Seamless session continuity ([Details](31-data-expert-agent-enhancements.md))
- 📂 **Pattern-based Dataset Loading** - Smart memory management
- 💾 **Session Persistence** - Automatic state tracking
- 💾 **WorkspaceContentService** - Type-safe caching for research content ([Details](38-workspace-content-service.md))

**Analysis & Workflows:**
- 🧪 **Formula-Based Differential Expression** - Complex experimental designs with R-style formulas ([Details](32-agent-guided-formula-construction.md))
- 🤖 **Enhanced Data Expert Agent** - New restoration tools and workflows

**Infrastructure:**
- 🏗️ **Provider Infrastructure** - Modular, extensible architecture for content retrieval
- 🏗️ **Agent Registry Auto-Discovery** - Dynamic agent configuration ([Details](36-supervisor-configuration.md))
- ⌨️ **Enhanced CLI** - Arrow navigation and command history
- 🎨 **Rich Interface** - Professional orange branding
- ⚡ **Performance** - Optimized startup and processing

## 🗂️ Feature Availability Matrix

Quick reference for feature availability across deployment modes.

### Core Features by Deployment Mode

| Feature | Local | Cloud |
|---------|:-----:|:-----:|
| **Content Intelligence** |
| Docling structure-aware parsing | ✅ | ✅ |
| Two-tier publication access | ✅ | ✅ |
| ContentAccessService | ✅ | ✅ |
| Provider infrastructure (5 providers) | ✅ | ✅ |
| **Analysis Capabilities** |
| Simple DE (two-group) | ✅ | ✅ |
| Formula-based DE | ✅ | ✅ |
| Agent-guided formulas | ✅ | ✅ |
| Protein visualization (batch) | ✅ | ✅ |
| Protein visualization (interactive) | ✅ | ⚠️ |
| **Data Management** |
| Basic workspace | ✅ | ✅ |
| WorkspaceContentService | ✅ | ✅ |
| Download queue (JSONL) | ✅ | ✅ |
| Two-tier caching | ✅ | ✅ |
| **Infrastructure** |
| Auto agent discovery | ✅ | ✅ |
| FTP retry logic | ✅ | ✅ |

**Legend:**
- ✅ Full support
- ⚠️ Partial support (see notes below)

> **Note:** Interactive PyMOL visualization requires local GUI support. Cloud mode supports batch image generation only.

For detailed feature documentation, see the [Migration Guide](41-migration-guides.md).

## 🔗 Quick Links

- **GitHub Repository**: [github.com/the-omics-os/lobster](https://github.com/the-omics-os/lobster)
- **Issue Tracker**: [Report bugs or request features](https://github.com/the-omics-os/lobster/issues)
- **Discord Community**: [Join our community](https://discord.gg/HDTRbWJ8omicsos)
- **Enterprise Support**: [info@omics-os.com](mailto:info@omics-os.com)

## 📝 Documentation Standards

This documentation follows these principles:
- **Progressive Disclosure**: Start simple, dive deeper as needed
- **Task-Oriented**: Organized by what you want to accomplish
- **Example-Rich**: Real datasets and practical code examples
- **Cross-Referenced**: Links between related topics
- **Maintained**: Regular updates with each release

## 🤝 Contributing to Documentation

Found an issue or want to improve the documentation?
1. Check our [developer overview](08-developer-overview.md)
2. Submit a pull request to the `docs/wiki` directory
3. Follow our [code style guidelines](08-developer-overview.md)

---

*Documentation for Lobster AI v0.2+ | Last updated: 2025*

*Made with ❤️ by [Omics-OS](https://omics-os.com)*