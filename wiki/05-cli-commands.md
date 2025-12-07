# CLI Commands Reference

## Overview

Lobster AI provides a rich command-line interface with enhanced features including Tab completion, command history, and context-aware suggestions. The CLI supports both slash commands for system operations and natural language for analysis tasks.

## Getting Started

### Starting Lobster AI

```bash
# Start interactive chat mode
lobster chat

# Start with custom workspace
lobster chat --workspace /path/to/my/workspace

# Enable detailed agent reasoning
lobster chat --reasoning

# Enable verbose output for debugging
lobster chat --verbose

# Start with all debugging features
lobster chat --reasoning --verbose --debug
```

### Single Query Mode

```bash
# Execute a single query and exit
lobster query "Analyze my single-cell data"

# Save output to file
lobster query "Generate QC report" --output results.md

# Use custom workspace
lobster query "Load data.h5ad" --workspace /my/data
```

### Dashboard Mode

```bash
# Launch interactive Textual-based dashboard
lobster dashboard

# With custom workspace
lobster dashboard --workspace /path/to/workspace
```

The dashboard provides a cockpit-style interface with:
- Real-time agent activity monitoring
- Live handoff visualization
- Query input with streaming responses
- Token usage and system status panels

### API Server Mode

```bash
# Start API server for web interfaces
lobster serve

# Custom host and port
lobster serve --host 0.0.0.0 --port 8080
```

## Interactive Features

### Enhanced Input Capabilities

**Arrow Key Navigation** (requires `prompt-toolkit`):
- **←/→**: Navigate within your input text
- **↑/↓**: Browse command history
- **Ctrl+R**: Reverse search through history
- **Home/End**: Jump to beginning/end of line

**Tab Completion**:
- **Commands**: Type `/` and press Tab to see all commands
- **Files**: Tab completion after `/read`, `/plot`, `/open`
- **Context-Aware**: Smart suggestions based on current context
- **Cloud Integration**: Works with both local and cloud clients

**Command History**:
- **Persistent**: Commands saved between sessions
- **Search**: Use Ctrl+R to find previous commands
- **Edit**: Recall and modify previous commands

### Installation for Enhanced Features

```bash
# Install optional dependency for full features
pip install prompt-toolkit
```

## System Commands

### Help and Information

#### `/help`
Display comprehensive help with all available commands.

```
/help
```

Shows categorized list of commands with descriptions and examples.

#### `/status`
Show current system status including session info, loaded data, and agent configurations.

```
/status
```

**Output includes**:
- Session ID and mode
- Loaded data summary
- Memory usage
- Workspace location

#### `/input-features`
Display available input features and navigation capabilities.

```
/input-features
```

Shows status of Tab completion, arrow navigation, and command history.

### Workspace Management

#### `/workspace`
Show comprehensive workspace information.

```
/workspace
```

**Displays**:
- Workspace path and configuration
- Loaded modalities and backends
- Directory structure and usage

#### `/workspace list`
List all available datasets in workspace without loading them.

```
/workspace list
```

Shows datasets with:
- **Index Number** (#) - Use for quick loading with `/workspace load` or `/workspace info`
- **Status** - ✓ (loaded) or ○ (available)
- **Name** - Intelligently truncated with middle-ellipsis for long names (max 60 chars)
- **Size** - Dataset size in MB
- **Shape** - Observations × variables
- **Modified** - Last modification date

**Features** (v0.2+):
- Numbered index for each dataset (1, 2, 3...) enables index-based loading
- Smart truncation preserves start and end of long dataset names
- Example: `geo_gse155698_quality_assess...ted_clustered_markers`
- Contextual help footer: "Use '/workspace info <#>' to see full details"
- Fixed column widths for professional table formatting

#### `/workspace info <#|pattern>`
Show detailed information for specific dataset(s) (v0.2+).

```
/workspace info 1                      # Show details for first dataset (index)
/workspace info gse12345              # Show details by name pattern
/workspace info *clustered*           # Show details for matching datasets
```

**Input Options**:
- **Index number**: Use # from `/workspace list` (e.g., `1`, `5`, `10`)
- **Name pattern**: Full or partial dataset name
- **Glob pattern**: Wildcards for multiple matches (e.g., `*liver*`, `geo_*`)

**Detailed Output**:
- Full dataset name (no truncation)
- Load status (✓ Loaded / ○ Not Loaded)
- Complete file path
- Precise size in MB
- Shape with formatted numbers (e.g., 50,000 observations × 20,000 variables)
- File type (H5AD, MuData, etc.)
- Modification timestamp
- Detected processing stages (quality, filter, normal, doublet, cluster, marker, annot, pseudobulk)

**Features**:
- Index-based selection for convenience (no typing long names)
- Pattern matching with wildcards for flexibility
- Multiple datasets displayed when pattern matches many
- Automatic lineage detection from dataset naming convention

**Example**:
```
/workspace info 1

Dataset #1 Details:
────────────────────────────────────────────────────────────
Name:        geo_gse155698_quality_assessed_filtered_normalized_doublets_detected_clustered_markers
Status:      ✓ Loaded
Path:        /workspace/geo_gse155698_quality_assessed_filtered_normalized_doublets_detected_clustered_markers.h5ad
Size:        287.4 MB
Shape:       94,371 observations × 32,738 variables
Type:        H5AD
Modified:    2025-01-10 14:23:45
Stages:      quality → filter → normal → doublet → cluster → marker
```

#### `/workspace load <#|pattern>`
Load specific dataset(s) from workspace by index or pattern (v0.2+).

```
/workspace load 1                     # Load first dataset (index-based)
/workspace load 5                     # Load fifth dataset
/workspace load my_dataset            # Load by name
/workspace load *clustered*           # Load all matching pattern
```

**Input Options**:
- **Index number**: Use # from `/workspace list` - fast and convenient
- **Name pattern**: Full or partial dataset name for targeted loading
- **Glob pattern**: Wildcards for loading multiple related datasets

**Features**:
- **Index-based loading** (v0.2+): No need to type long dataset names
- **Pattern matching**: Load multiple datasets matching criteria
- **Progress tracking**: Shows loading progress for each dataset
- **Automatic validation**: Data quality checks during load
- **Smart caching**: Efficient memory usage

**When to use**:
- Use `/workspace load <#>` for loading single datasets by index (fastest)
- Use `/workspace load <pattern>` for loading specific datasets by name
- Use `/restore` for session continuation and bulk loading workflows

#### `/restore [pattern]`
Restore datasets from workspace based on pattern matching.

```
/restore                    # Restore recent datasets (default)
/restore recent            # Same as above
/restore all               # Restore all available datasets
/restore my_dataset        # Restore specific dataset by name
/restore *liver*           # Restore datasets matching pattern
/restore geo_*             # Restore all GEO datasets
```

**Features**:
- Tab completion for dataset names
- Flexible pattern matching support
- Shows loading progress with detailed summaries
- Intelligent memory management
- Session continuation support
- **Works WITHOUT prior `/save` command** - uses automatic session tracking

**How It Works**:
- **`recent` mode**: Reads from `.session.json` (automatically updated whenever you load data or perform operations)
- **`all` mode**: Scans workspace directory for all `.h5ad` files
- **Pattern mode**: Uses glob matching against workspace files

**Pattern Options**:
- `recent` - Load most recently used datasets from automatic session tracking (default)
- `all` or `*` - Load all available datasets from workspace scan
- `<dataset_name>` - Load specific dataset by exact name
- `<partial_name>*` - Load datasets matching partial name pattern

> **Note**: Use `/restore` for session continuation and bulk loading workflows. Use `/workspace load` (v0.2+) for targeted single-dataset loading by index or specific pattern.

**Relationship with `/save`**:
- `/restore` does **NOT** require prior `/save` - session tracking is automatic
- `/save` creates explicit backup snapshots (with `_autosave` suffix)
- `/restore recent` loads from your original working files, not autosaves
- Use `/save` before risky operations, `/restore` for session continuation

### File Operations

#### `/files`
List all files in workspace organized by category.

```
/files
```

**Categories**:
- **Data**: Analysis datasets and input files
- **Exports**: Generated output files
- **Cache**: Temporary and cached files

#### `/tree`
Show directory tree view of current location and workspace.

```
/tree
```

Displays nested folder structure with file counts and sizes.

#### `/read <file>`
Load and analyze files from workspace or current directory.

```
/read data.h5ad                    # Load single file
/read *.h5ad                       # Load all H5AD files
/read data/*.csv                   # Load CSVs from data folder
/read sample_*.h5ad                # Pattern matching
```

**Supported Patterns**:
- `*`: Match any characters
- `?`: Match single character
- `[abc]`: Match any of a, b, or c
- `**`: Recursive directory matching

**Features**:
- Tab completion for file names
- Automatic format detection
- Batch loading with progress tracking
- Format conversion on-the-fly

#### `/archive <file>`
Load data from compressed archives containing bioinformatics data.

```
/archive GSE155698_RAW.tar         # Load 10X Genomics samples
/archive kallisto_results.tar.gz   # Load Kallisto quantification
/archive salmon_quant.zip          # Load Salmon quantification
```

**Supported Archive Formats**:
- TAR (`.tar`, `.tar.gz`, `.tar.bz2`)
- ZIP (`.zip`)

**Supported Data Formats**:
- **10X Genomics**: Both V2 (`genes.tsv`) and V3 (`features.tsv`) chemistry
  - Handles compressed and uncompressed files
  - Automatic sample detection and concatenation
- **Kallisto Quantification**: Multiple samples with `abundance.tsv` or `abundance.h5`
- **Salmon Quantification**: Multiple samples with `quant.sf`
- **GEO RAW Files**: GSM-prefixed expression files

**Features**:
- Smart content detection without full extraction
- Automatic format identification
- Memory-efficient processing
- Handles nested archive structures
- Sample concatenation for multi-sample archives
- Compressed file support (`.gz`, `.bz2`)

**Example Workflow**:
```
/archive /path/to/GSE155698_RAW.tar
# Automatically detects:
# - 17 10X Genomics samples (V2 and V3 mixed)
# - Loads and concatenates all samples
# - Result: 94,371 cells × 32,738 genes
```

**When to Use `/archive` vs `/read`**:
- Use `/archive` for: Compressed archives with multiple samples or nested structures
- Use `/read` for: Individual data files (H5AD, CSV, Excel)

#### `/open <file>`
Open file or folder in system default application.

```
/open results.pdf                  # Open in default PDF viewer
/open plots/                       # Open directory in file manager
/open .                            # Open current directory
```

Works with workspace files, absolute paths, and relative paths.

### Data Management

#### `/data`
Show comprehensive summary of currently loaded data.

```
/data
```

**For Single Modality**:
- Shape (observations × variables)
- Data type and memory usage
- Quality metrics
- Metadata columns
- Processing history

**For Multiple Modalities**:
- Individual modality summaries
- Combined statistics
- Cross-modality information

#### `/metadata`
Show detailed metadata information including cached GEO data.

```
/metadata
```

**Displays**:
- **Metadata Store**: Cached GEO and external datasets
- **Current Data Metadata**: Active dataset information
- **Validation Results**: Data quality assessments

#### `/modalities`
Show detailed information for each loaded modality.

```
/modalities
```

**For Each Modality**:
- Observation and variable columns
- Data layers and embeddings
- Unstructured annotations
- Shape and memory information

### Visualization

#### `/plots`
List all generated plots with metadata.

```
/plots
```

Shows plot ID, title, source, and creation time for all generated visualizations.

#### `/plot [ID]`
Open plots directory or specific plot.

```
/plot                              # Open plots directory
/plot plot_1                       # Open specific plot by ID
/plot "Quality Control"            # Open plot by title (partial match)
```

**Features**:
- Opens HTML version preferentially (interactive)
- Falls back to PNG if HTML unavailable
- Tab completion for plot IDs and titles

### Session Management

#### `/save`
Save current state including all loaded data and generated plots.

```
/save
```

**Saves**:
- All loaded modalities as H5AD files (with `_autosave` suffix)
- Generated plots in HTML and PNG formats
- Processing log and tool usage history
- Session metadata

**Important Notes**:
- **Explicit `/save` is NOT required for `/restore` to work** - Lobster automatically tracks your session via `.session.json`
- Use `/save` when you want to:
  - Create explicit backup snapshots before risky operations
  - Export data with specific naming for archival purposes
  - Preserve a clean checkpoint state
- Session tracking happens automatically whenever you load data or perform operations
- Autosaved files are named `<modality_name>_autosave.h5ad` to distinguish from working files

#### `/export`
Export complete session data as a comprehensive package.

```
/export
```

Creates timestamped ZIP file with all data, plots, metadata, and analysis history.

#### `/reset`
Reset conversation and clear loaded data (with confirmation).

```
/reset
```

Prompts for confirmation before clearing:
- Conversation history
- Loaded modalities
- Generated plots
- Analysis state

### Configuration

#### `/modes`
List available operation modes with descriptions.

```
/modes
```

**Available Modes**:
- `development`: Claude 4.5 Sonnet supervisor, Claude 3.7 Sonnet workers - fast development with balanced performance
- `production`: Claude 4.5 Sonnet for all agents - production-ready quality across all agents

#### `/mode <name>`
Change operation mode and agent configurations.

```
/mode production                   # Switch to production mode (Claude 4.5 Sonnet all agents)
/mode development                  # Use development profile (mixed models for cost efficiency)
```

**Effects**:
- Updates all agent model configurations
- Adjusts performance and cost parameters
- Maintains current data and session state

### Dashboard and Monitoring

#### `/dashboard`
Switch to the interactive Textual-based dashboard.

```
/dashboard
```

Launches a full-screen interactive terminal UI with:
- Multi-panel cockpit layout for real-time monitoring
- Live agent activity tracking and handoff visualization
- Query input with streaming responses
- Token usage and system status panels

Press ESC to quit, ^P for command palette.

**Note**: You can also launch the dashboard directly with `lobster dashboard` from the command line.

#### `/status-panel`
Show comprehensive system health dashboard (Rich panels in terminal).

```
/status-panel
```

**Includes**:
- Core system status
- Resource utilization
- Agent status and health

#### `/workspace-info`
Show detailed workspace overview with recent activity.

```
/workspace-info
```

**Displays**:
- Workspace configuration and paths
- Recent files and data access
- Data loading statistics

#### `/analysis-dash`
Show analysis monitoring dashboard.

```
/analysis-dash
```

**Tracks**:
- Active analysis operations
- Generated visualizations
- Processing performance metrics

#### `/progress`
Show multi-task progress monitor for concurrent operations.

```
/progress
```

Displays active background operations with progress bars and status.

### Utility Commands

#### `/clear`
Clear the terminal screen.

```
/clear
```

#### `/exit`
Exit Lobster AI (with confirmation prompt).

```
/exit
```

## Shell Integration

Lobster AI supports common shell commands directly without the `/` prefix:

### Directory Navigation

```bash
cd /path/to/data                  # Change directory
pwd                               # Print working directory
ls                                # List directory contents with metadata
ls /path/to/folder               # List specific directory
```

### File Operations

```bash
mkdir new_folder                  # Create directory
touch new_file.txt               # Create file
cp source.txt dest.txt           # Copy file
mv old_name.txt new_name.txt     # Move/rename file
rm unwanted_file.txt             # Remove file
```

### File Viewing

```bash
cat data.csv                     # View file with syntax highlighting
open results/                    # Open in file manager (same as /open)
```

**Enhanced Features**:
- **Syntax Highlighting**: Automatic language detection
- **Structured Output**: Tables and formatted displays
- **Rich Metadata**: File sizes, modification dates, types

## Configuration Commands

### Agent Configuration

```bash
# List available model presets
lobster config list-models

# List available testing profiles
lobster config list-profiles

# Show current configuration
lobster config show-config

# Test specific configuration
lobster config test --profile production

# Test specific agent
lobster config test --profile production --agent transcriptomics_expert

# Create custom configuration interactively
lobster config create-custom

# Generate environment template
lobster config generate-env
```

## Usage Examples

### Common Workflows

#### Starting a New Analysis

```bash
# Start Lobster
lobster chat

# Check existing workspace
/workspace list

# Load previous work by index (v0.2+)
/workspace load 1

# Or restore recent session
/restore recent

# Or load new data
/read my_data.h5ad

# Check data status
/data

# Begin analysis
"Analyze this single-cell RNA-seq data and identify cell types"
```

#### Dataset Browsing and Selection (v0.2+)

```bash
# List all datasets with numbered index
/workspace list

# Get detailed info for first dataset
/workspace info 1

# Load that dataset by index
/workspace load 1

# Or get info about specific pattern
/workspace info *liver*

# Check full details before loading
/workspace info geo_gse12345

# Load by pattern
/workspace load *clustered*
```

#### Data Exploration

```bash
# Quick data overview
/data

# View metadata
/metadata

# Check file structure
/tree

# Explore analysis options
"What analysis can I do with this data?"
```

#### Visualization Management

```bash
# List all plots
/plots

# Open specific plot
/plot plot_3

# Open plots folder
/plot

# Save current state
/save
```

#### Session Management

```bash
# Check system status
/status

# View workspace info
/workspace

# Export everything
/export

# Clean restart if needed
/reset
```

### Advanced Usage

#### Batch Operations

```bash
# Load multiple files
/read *.h5ad

# Pattern-based restoration
/restore *experiment_2*

# Dataset loading operations
/restore batch_*
```

#### Configuration Switching

```bash
# Check available modes
/modes

# Switch for production analysis
/mode production

# Verify change
/status
```

#### Debugging and Monitoring

```bash
# Start with verbose debugging
lobster chat --verbose --debug

# Monitor system resources
/dashboard

# Track analysis progress
/progress

# View detailed workspace info
/workspace-info
```

## Troubleshooting Commands

### Diagnostic Information

```bash
# System status
/status

# Input capabilities
/input-features

# Workspace health
/workspace

# Data validation
/metadata
```

### Recovery Operations

```bash
# List available data
/workspace list

# Restore from backup
/restore all

# Clear and restart
/reset

# Export before major changes
/export
```

### Performance Optimization

```bash
# Check resource usage
/dashboard

# Switch to development mode for lighter resource usage
/mode development

# Monitor active operations
/progress
```

This comprehensive CLI reference covers all available commands and their usage patterns. For analysis-specific workflows, see the Data Analysis Workflows section.