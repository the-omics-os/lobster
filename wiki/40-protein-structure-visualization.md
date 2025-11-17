# Protein Structure Visualization Expert Agent

**Since v2.4** - Protein structure analysis with PyMOL visualization and BioPython integration

**Agent Name**: `protein_structure_visualization_expert_agent`
**Display Name**: Protein Structure Visualization Expert
**Factory Function**: `lobster.agents.protein_structure_visualization_expert.protein_structure_visualization_expert`

## Overview

The Protein Structure Visualization Expert is a specialized agent for fetching, visualizing, and analyzing 3D protein structures from the RCSB Protein Data Bank (PDB). It integrates PyMOL (open-source) for high-quality molecular visualizations and BioPython for structural analysis, enabling seamless linking between protein structures and omics datasets.

**Version Note**: This agent requires Lobster v2.4+ and is fully supported in both local and cloud modes (with limited interactive visualization in cloud).

### Key Features

- **PDB Structure Fetching**: Download protein structures by PDB ID with comprehensive metadata
- **PyMOL Integration**: Generate professional 3D visualizations with customizable styles and colors
- **Structural Analysis**: Calculate RMSD, secondary structure, geometry, and residue contacts
- **Omics Integration**: Link protein structures to gene expression and proteomics data
- **Structure Comparison**: Compare multiple protein structures and calculate structural similarity
- **Provenance Tracking**: Full W3C-PROV compliant logging with Intermediate Representation (IR)

---

## Architecture

### Services (Stateless, 3-Tuple Pattern)

#### 1. ProteinStructureFetchService

**Location**: `lobster/tools/protein_structure_fetch_service.py`

Handles fetching protein structures from RCSB PDB with caching and metadata extraction.

**Methods**:
- `fetch_structure(pdb_id, format='cif', cache_dir, extract_metadata)` → Tuple[Dict, Dict, AnalysisStep]
- `link_structures_to_genes(adata, gene_column, organism, max_structures_per_gene)` → Tuple[AnnData, Dict, AnalysisStep]

**Features**:
- PDB ID format validation (4-character alphanumeric)
- Automatic caching to avoid redundant downloads
- BioPython-based structure parsing
- Metadata extraction (resolution, organism, experiment method)
- Gene-to-structure mapping via PDB search API

#### 2. PyMOLVisualizationService

**Location**: `lobster/tools/pymol_visualization_service.py`

Creates high-quality 3D visualizations using PyMOL (open-source).

**Methods**:
- `visualize_structure(structure_file, mode, style, color_by, output_image, width, height, execute_commands)` → Tuple[Dict, Dict, AnalysisStep]
- `check_pymol_installation()` → Dict[str, Any]

**Features**:
- Multiple representation styles: cartoon, surface, sticks, spheres, ribbon, lines
- Multiple coloring schemes: chain, secondary_structure, bfactor, element
- Interactive and batch modes (GUI or headless image generation)
- PyMOL command script generation (`.pml` files)
- Automatic PyMOL installation detection
- Graceful fallback when PyMOL is not installed
- High-resolution image export (customizable dimensions)
- Non-blocking GUI launch for interactive exploration

#### 3. StructureAnalysisService

**Location**: `lobster/tools/structure_analysis_service.py`

Performs structural analysis using BioPython.

**Methods**:
- `analyze_structure(structure_file, analysis_type, chain_id)` → Tuple[Dict, Dict, AnalysisStep]
- `calculate_rmsd(structure_file1, structure_file2, chain_id1, chain_id2, align)` → Tuple[Dict, Dict, AnalysisStep]

**Features**:
- Secondary structure analysis (DSSP integration with fallback)
- Geometric properties (center of mass, radius of gyration)
- Residue contact analysis (spatial proximity)
- RMSD calculation with optional superposition alignment
- BioPython Superimposer for structural alignment

---

## Agent Tools

### 1. fetch_protein_structure

**Purpose**: Download protein structure from RCSB PDB

**Parameters**:
- `pdb_id` (str, required): PDB identifier (e.g., '1AKE', '4HHB')
- `format` (str, default='cif'): File format ('pdb' or 'cif')

**Returns**: Summary with metadata, file paths, and structural properties

**Example**:
```python
fetch_protein_structure("1AKE")
fetch_protein_structure("4HHB", format="pdb")
```

**Output Includes**:
- PDB ID, title, organism
- Experiment method and resolution
- Number of chains, residues, atoms
- File path and size
- Publication DOI and citation

---

### 2. link_to_expression_data

**Purpose**: Link gene expression data to protein structures

**Parameters**:
- `modality_name` (str, required): Name of modality with gene/protein data
- `gene_column` (str, default='gene_symbol'): Column in adata.var with gene symbols
- `organism` (str, default='Homo sapiens'): Source organism for structure search
- `max_structures_per_gene` (int, default=5): Maximum structures per gene

**Returns**: Summary of structure links created

**Example**:
```python
link_to_expression_data("rna_seq_normalized")
link_to_expression_data("proteomics_data", gene_column="protein_name", organism="Mus musculus")
```

**Output Includes**:
- Genes searched and genes with structures found
- Total structures found and average per gene
- New modality name with structure links
- Columns added: `pdb_structures` (comma-separated PDB IDs), `has_structure` (boolean)

---

### 3. visualize_with_pymol

**Purpose**: Create high-quality 3D visualization using PyMOL

**Parameters**:
- `pdb_id` (str, required): PDB ID of structure (must be fetched first)
- `mode` (str, default='interactive'): Execution mode
  - Options: 'interactive' (launch GUI for exploration), 'batch' (save PNG and exit)
- `style` (str, default='cartoon'): Representation style
  - Options: 'cartoon', 'surface', 'sticks', 'spheres', 'ribbon', 'lines'
- `color_by` (str, default='chain'): Coloring scheme
  - Options: 'chain', 'secondary_structure', 'bfactor', 'element'
- `width` (int, default=1920): Image width in pixels
- `height` (int, default=1080): Image height in pixels
- `execute` (bool, default=True): Execute PyMOL commands if installed
- `highlight_residues` (str, optional): Residues to highlight (e.g., "15,42,89" or "A:15-20,B:42")
- `highlight_color` (str, default='red'): Color for highlighted residues
- `highlight_style` (str, default='sticks'): Visualization style for highlights
- `highlight_groups` (str, optional): Multiple highlight groups (format: "residues|color|style;...")

**Returns**: Visualization metadata with file paths and execution status

**Examples**:
```python
# Basic visualization
visualize_with_pymol("1AKE")  # Interactive mode by default
visualize_with_pymol("4HHB", mode="batch", style="surface", color_by="bfactor")
visualize_with_pymol("1AKE", mode="interactive")  # Launch GUI for exploration

# Residue highlighting - Single group
visualize_with_pymol("1AKE", highlight_residues="15,42,89", highlight_color="red", highlight_style="sticks")

# Residue highlighting - Chain-specific
visualize_with_pymol("4HHB", highlight_residues="A:15-20,B:30-35", highlight_color="yellow")

# Residue highlighting - Multiple groups
visualize_with_pymol("1AKE", highlight_groups="15,42|red|sticks;100-120|blue|surface;200,215|green|spheres")
```

**Output Includes**:
- Visualization settings (mode, style, color scheme, dimensions)
- Command script path (`.pml` file)
- Output image path (`.png` file)
- Execution status and PyMOL installation info
- Process ID (PID) for interactive mode

---

### 4. analyze_protein_structure

**Purpose**: Analyze protein structure properties

**Parameters**:
- `pdb_id` (str, required): PDB ID of structure (must be fetched first)
- `analysis_type` (str, default='secondary_structure'): Type of analysis
  - Options: 'secondary_structure', 'geometry', 'residue_contacts'
- `chain_id` (str, optional): Specific chain to analyze (None for all chains)

**Returns**: Analysis results with structural properties

**Example**:
```python
analyze_protein_structure("1AKE")
analyze_protein_structure("4HHB", analysis_type="geometry")
analyze_protein_structure("1AKE", analysis_type="residue_contacts", chain_id="A")
```

**Analysis Types**:

#### Secondary Structure
- Helix, sheet, coil percentages
- Per-residue secondary structure assignments
- Requires DSSP binary (with fallback)

#### Geometry
- Total atoms and chains
- Center of mass
- Radius of gyration
- Per-chain geometric properties

#### Residue Contacts
- Total residue-residue contacts (default cutoff: 8 Å)
- Average contacts per residue
- Contact distance matrix

---

### 5. compare_structures

**Purpose**: Compare two protein structures by RMSD

**Parameters**:
- `pdb_id1` (str, required): First PDB ID (must be fetched)
- `pdb_id2` (str, required): Second PDB ID (must be fetched)
- `align` (bool, default=True): Align structures before RMSD calculation
- `chain_id1` (str, optional): Specific chain in first structure
- `chain_id2` (str, optional): Specific chain in second structure

**Returns**: RMSD and structural comparison results

**Example**:
```python
compare_structures("1AKE", "4AKE")
compare_structures("1AKE", "4AKE", align=False)
compare_structures("4HHB", "2HHB", chain_id1="A", chain_id2="A")
```

**RMSD Interpretation**:
- **< 1.0 Å**: Nearly identical structures
- **1-2 Å**: Very similar (close homologs, small conformational changes)
- **2-3 Å**: Similar (homologs, moderate conformational changes)
- **3-5 Å**: Moderately similar (distant homologs, domain movements)
- **> 5 Å**: Different structures (large conformational changes)

---

## Workflows

### Basic Workflow: Fetch and Visualize

```plaintext
1. User: "Visualize protein structure 1AKE"
2. Supervisor → Protein Structure Visualization Expert
3. Agent: fetch_protein_structure("1AKE")
4. Agent: visualize_with_pymol("1AKE", mode="interactive", style="cartoon")
5. Agent → Supervisor: Results with visualization paths
6. Supervisor → User: Visualization complete
```

### Advanced Workflow: Link Structures to Expression Data

```plaintext
1. User: "Link structures to my RNA-seq data"
2. Supervisor → Protein Structure Visualization Expert
3. Agent: link_to_expression_data("rna_seq_normalized", organism="Homo sapiens")
4. Agent creates new modality with structure mappings
5. Agent → Supervisor: Linking results (e.g., "50 genes linked to 75 structures")
6. Supervisor → User: Structure links created
```

### Comparative Workflow: RMSD Analysis

```plaintext
1. User: "Compare structures 1AKE and 4AKE"
2. Supervisor → Protein Structure Visualization Expert
3. Agent: fetch_protein_structure("1AKE")
4. Agent: fetch_protein_structure("4AKE")
5. Agent: compare_structures("1AKE", "4AKE", align=True)
6. Agent → Supervisor: RMSD results (e.g., "RMSD = 1.2 Å, very similar")
7. Supervisor → User: Comparison complete
```

---

## PyMOL Installation

### Why PyMOL?

PyMOL is a professional open-source molecular visualization tool that provides:
- High-quality molecular graphics
- Publication-ready images
- Comprehensive visualization commands
- Python API for automation
- Interactive GUI mode for real-time exploration
- Active open-source community

### Automated Installation (Recommended)

#### Docker Container (Cloud Deployments)

PyMOL is **pre-installed** in the Lobster Docker image. No action required.

**Verify installation**:
```bash
docker run -it omicsos/lobster:latest pymol -c -Q
```

#### Local Development (macOS/Linux)

Install PyMOL via Makefile target:

```bash
# Install PyMOL automatically
make install-pymol
```

This command will:
- Detect your operating system (macOS or Linux)
- Install PyMOL via the appropriate package manager
- Verify the installation

**What it does**:
- **macOS**: Uses Homebrew with brewsci/bio tap
- **Linux**: Uses apt-get (Ubuntu/Debian) or dnf (Fedora/RHEL)
- **Homebrew on Linux**: Fallback if native package manager unavailable

**Requirements**:
- macOS: Homebrew must be installed
- Linux: sudo access for package installation

**Installation output**:
```bash
$ make install-pymol
🔬 Installing PyMOL for protein structure visualization...
🍎 macOS detected - Installing via Homebrew...
📦 Installing PyMOL...
✅ PyMOL installed successfully!

🎉 PyMOL installation complete!
💡 Test with: pymol -c -Q
```

### Manual Installation (Fallback)

If automated installation is not available or fails, you can install PyMOL manually.

#### macOS
```bash
# Install via Homebrew (recommended)
brew install brewsci/bio/pymol

# Or download from official website
# https://pymol.org/

# After installation, PyMOL is automatically added to PATH
```

#### Linux
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install pymol

# Or via Homebrew on Linux
brew install brewsci/bio/pymol

# Arch Linux
sudo pacman -S pymol

# Fedora/CentOS/RHEL
sudo dnf install pymol
```

#### Windows
```plaintext
1. Download installer from https://pymol.org/
2. Run installer and follow instructions
3. PyMOL executable will be added to Start Menu
4. Optionally add to PATH via System Environment Variables
```

### Manual Execution Without Installation

Even without PyMOL installed, the agent generates `.pml` command scripts that can be:
- Executed manually when PyMOL is installed
- Modified for custom visualizations
- Used as templates for batch processing

**Examples**:
```bash
# Interactive mode (with GUI)
pymol 1AKE_cartoon_chain_commands.pml

# Batch mode (headless, save image and exit)
pymol -c 1AKE_cartoon_chain_commands.pml
```

---

## Integration with Omics Workflows

### Single-Cell RNA-seq Integration

Link protein structures to highly expressed genes:

```plaintext
1. Run single-cell analysis (clustering, DE analysis)
2. Identify top expressed genes
3. Use link_to_expression_data() to find structures
4. Visualize structures for key marker genes
```

### Proteomics Integration

Link structures to identified proteins:

```plaintext
1. Run proteomics analysis (quantification, DE)
2. Identify significantly changing proteins
3. Use link_to_expression_data() with protein_name column
4. Compare structures of protein variants
```

### Multi-Omics Integration

Cross-reference structures across modalities:

```plaintext
1. Link structures to both RNA-seq and proteomics
2. Identify genes/proteins with structures in both datasets
3. Visualize structures colored by expression levels
4. Compare structural features with functional changes
```

---

## Performance and Caching

### Structure Caching

- **First fetch**: Downloads from PDB, stores in `protein_structures/` directory
- **Subsequent fetches**: Uses cached file (instant)
- **Cache location**: Workspace directory or current directory
- **Cache benefits**: Avoids redundant downloads, faster workflow iterations

### PDB Provider Rate Limits

- **Rate limit**: 5 requests/second (RCSB PDB API limit)
- **No authentication**: Public PDB API requires no API key
- **Batch operations**: Use link_to_expression_data() for efficient batch queries

### PyMOL Performance

- **Command scripts**: Generated instantly (no execution delay)
- **Interactive mode**: GUI launches in 2-5 seconds (non-blocking)
- **Batch mode (image generation)**: 5-30 seconds per structure (if PyMOL is installed)
- **Headless mode**: PyMOL runs without GUI for automation (use `pymol -c`)
- **Parallel execution**: Multiple structures can be visualized in parallel

---

## Error Handling

### Common Errors and Solutions

#### 1. Invalid PDB ID
**Error**: `Invalid PDB ID format: XYZ. Must be 4 alphanumeric characters.`

**Solution**: Ensure PDB ID is exactly 4 characters (e.g., '1AKE', not '1AK' or '1AKEE')

#### 2. Structure Not Found
**Error**: `Failed to download structure 1XYZ from PDB`

**Solution**: Verify PDB ID exists at https://www.rcsb.org/structure/1XYZ

#### 3. PyMOL Not Installed
**Error**: `PyMOL not found. Install with: brew install brewsci/bio/pymol`

**Solution**: Install PyMOL or use generated command scripts manually

#### 4. Gene Column Not Found
**Error**: `Gene column 'gene_symbol' not found in adata.var`

**Solution**: Check available columns with `adata.var.columns` and specify correct column name

#### 5. DSSP Not Available
**Warning**: `DSSP not available. Using simplified analysis.`

**Solution**: Install DSSP for secondary structure analysis:
```bash
conda install -c salilab dssp
```

---

## API Reference

### ProteinStructureFetchService

```python
from lobster.tools.protein_structure_fetch_service import ProteinStructureFetchService

service = ProteinStructureFetchService()

# Fetch structure
structure_data, stats, ir = service.fetch_structure(
    pdb_id="1AKE",
    format="cif",
    cache_dir=Path("protein_structures"),
    extract_metadata=True,
    data_manager=data_manager
)

# Link structures to genes
adata_linked, stats, ir = service.link_structures_to_genes(
    adata=adata,
    gene_column="gene_symbol",
    organism="Homo sapiens",
    max_structures_per_gene=5,
    data_manager=data_manager
)
```

### PyMOLVisualizationService

```python
from lobster.tools.pymol_visualization_service import PyMOLVisualizationService

service = PyMOLVisualizationService()

# Check installation
install_status = service.check_pymol_installation()

# Create visualization (batch mode - save PNG)
viz_data, stats, ir = service.visualize_structure(
    structure_file=Path("1AKE.cif"),
    mode="batch",
    style="cartoon",
    color_by="chain",
    output_image=Path("output.png"),
    width=1920,
    height=1080,
    execute_commands=True
)

# Or interactive mode (launch GUI)
viz_data, stats, ir = service.visualize_structure(
    structure_file=Path("1AKE.cif"),
    mode="interactive",
    style="cartoon",
    color_by="chain",
    execute_commands=True
)
```

### StructureAnalysisService

```python
from lobster.tools.structure_analysis_service import StructureAnalysisService

service = StructureAnalysisService()

# Analyze structure
analysis_results, stats, ir = service.analyze_structure(
    structure_file=Path("1AKE.cif"),
    analysis_type="secondary_structure",
    chain_id="A"
)

# Calculate RMSD
rmsd_results, stats, ir = service.calculate_rmsd(
    structure_file1=Path("1AKE.cif"),
    structure_file2=Path("4AKE.cif"),
    align=True
)
```

---

## Best Practices

### 1. PDB ID Validation
Always use uppercase 4-character PDB IDs:
```python
# Good
fetch_protein_structure("1AKE")

# Bad
fetch_protein_structure("1ake")  # Works but not consistent
fetch_protein_structure("1AK")   # Error: too short
```

### 2. Structure Caching
Leverage caching for iterative workflows:
```python
# First run: downloads structure
fetch_protein_structure("1AKE")

# Subsequent runs: uses cache (instant)
visualize_with_pymol("1AKE", mode="interactive", style="cartoon")
visualize_with_pymol("1AKE", mode="batch", style="surface")  # No re-download
```

### 3. PyMOL Fallback
Generate scripts even without PyMOL:
```python
# Script generation always works
visualize_with_pymol("1AKE", execute=False)

# Execute manually later when PyMOL is installed
# Interactive mode: pymol 1AKE_commands.pml
# Batch mode: pymol -c 1AKE_commands.pml
```

### 4. Gene-Structure Linking
Search by organism for better results:
```python
# Specific organism
link_to_expression_data("adata", organism="Homo sapiens")

# Mouse data
link_to_expression_data("adata", organism="Mus musculus")
```

### 5. RMSD Interpretation
Use alignment for meaningful comparisons:
```python
# With alignment (recommended)
compare_structures("1AKE", "4AKE", align=True)

# Without alignment (only for pre-aligned structures)
compare_structures("1AKE", "4AKE", align=False)
```

---

## Provenance and Reproducibility

All structure operations generate Intermediate Representation (IR) with:
- **Operation**: Specific operation performed (e.g., 'pdb.fetch_structure')
- **Parameters**: All parameters used (pdb_id, format, style, etc.)
- **Code Template**: Jinja2 template for notebook export
- **Imports**: Required Python imports
- **Parameter Schema**: Papermill-injectable parameters with validation

**Notebook Export**:
```python
# Export pipeline to Jupyter notebook
data_manager.export_notebook("protein_structure_pipeline.ipynb")

# Execute notebook with different PDB ID
papermill protein_structure_pipeline.ipynb output.ipynb -p pdb_id "4HHB"
```

---

## Troubleshooting

### Issue: "Structure file not found"
- Ensure structure was fetched first with `fetch_protein_structure()`
- Check cache directory permissions
- Verify file path in structure_data dictionary

### Issue: "PyMOL execution timed out"
- Large structures may take longer to render (batch mode)
- Increase timeout in service configuration
- Use `execute=False` to generate script without execution
- For interactive mode, the GUI may take 2-5 seconds to launch

### Issue: "No structures found for genes"
- Check organism name (use Latin names: "Homo sapiens", "Mus musculus")
- Verify gene symbols are standard (HGNC for human, MGI for mouse)
- Try reducing `max_structures_per_gene` for faster queries

### Issue: "RMSD calculation failed"
- Ensure both structures have been fetched
- Check chain IDs exist in structures
- Verify structures have matching residues (homologs, not random proteins)

---

## Examples

### Example 1: Basic Structure Visualization

```python
# Fetch and visualize adenylate kinase
fetch_protein_structure("1AKE")
visualize_with_pymol("1AKE", mode="interactive", style="cartoon", color_by="secondary_structure")
```

### Example 2: Comparative Analysis

```python
# Compare open and closed conformations of adenylate kinase
fetch_protein_structure("1AKE")  # Open form
fetch_protein_structure("4AKE")  # Closed form
compare_structures("1AKE", "4AKE", align=True)
# Output: RMSD = 1.2 Å (moderate conformational change)
```

### Example 3: RNA-seq Integration

```python
# After RNA-seq analysis
link_to_expression_data("rna_seq_normalized", organism="Homo sapiens")

# Visualize top expressed genes with structures
# Filter: adata[adata.var['has_structure']]
```

### Example 4: Protein Family Analysis

```python
# Fetch multiple family members
fetch_protein_structure("1AKE")
fetch_protein_structure("2AKE")
fetch_protein_structure("3AKE")

# Pairwise RMSD comparisons
compare_structures("1AKE", "2AKE")
compare_structures("1AKE", "3AKE")
compare_structures("2AKE", "3AKE")
```

### Example 5: Residue Highlighting for Disease Mutations and Functional Sites

```python
# Fetch structure
fetch_protein_structure("1AKE")

# Example 1: Highlight disease mutation sites in red
# Single residue group - useful for showing known pathogenic variants
visualize_with_pymol(
    "1AKE",
    mode="batch",
    style="cartoon",
    color_by="chain",
    highlight_residues="15,42,89",
    highlight_color="red",
    highlight_style="sticks"
)

# Example 2: Chain-specific highlighting for protein-protein interfaces
# Highlight interface residues in hemoglobin subunits
fetch_protein_structure("4HHB")
visualize_with_pymol(
    "4HHB",
    highlight_residues="A:15-20,A:42,B:30-35,B:50",
    highlight_color="yellow",
    highlight_style="sticks"
)

# Example 3: Multiple highlight groups for complex functional annotation
# Show binding site (red), catalytic residues (blue), and allosteric site (green)
visualize_with_pymol(
    "1AKE",
    mode="interactive",  # Launch GUI for interactive exploration
    highlight_groups="15,42,89|red|sticks;100-120|blue|surface;200,215,230|green|spheres"
)

# Example 4: Combining with different color schemes
# Highlight active site residues while showing B-factors for the rest
visualize_with_pymol(
    "1AKE",
    style="cartoon",
    color_by="bfactor",  # Color by temperature factors
    highlight_residues="100-120",  # Active site region
    highlight_color="red",
    highlight_style="sticks"
)
```

**Use Cases for Residue Highlighting**:
- **Disease Mutations**: Highlight known pathogenic variants from ClinVar or GWAS studies
- **Binding Sites**: Show ligand or substrate binding pockets
- **Active Sites**: Emphasize catalytic residues (e.g., catalytic triad in proteases)
- **Post-Translational Modifications**: Highlight phosphorylation, methylation, or acetylation sites
- **Protein-Protein Interfaces**: Show interaction residues in multi-chain complexes
- **Conservation Analysis**: Highlight evolutionarily conserved residues

---

## Related Documentation

- [Agent System Overview](19-agent-system.md)
- [Creating Agents](09-creating-agents.md)
- [Creating Services](10-creating-services.md)
- [Data Formats](07-data-formats.md)
- [Testing Guide](12-testing-guide.md)

---

## References

- **RCSB PDB**: https://www.rcsb.org
- **PDB REST API**: https://data.rcsb.org/redoc/index.html
- **PyMOL**: https://pymol.org/
- **PyMOL Wiki**: https://pymolwiki.org/
- **BioPython**: https://biopython.org/wiki/The_Biopython_Structural_Bioinformatics_FAQ
- **DSSP**: https://swift.cmbi.umcn.nl/gv/dssp/

---

**Last Updated**: 2025-01-15
**Version**: 1.0.0
**Maintainer**: Lobster Development Team
