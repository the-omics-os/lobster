# lobster-structural-viz

Protein structure visualization and analysis with PyMOL and ChimeraX integration.

## Installation

```bash
pip install lobster-structural-viz
```

## Agents

| Agent | Description |
|-------|-------------|
| `protein_structure_visualization_expert` | Structural biology specialist. PDB fetching, 3D visualization, structural analysis, and omics data integration. |

## Services

| Service | Purpose |
|---------|---------|
| ProteinStructureFetchService | Download structures from RCSB PDB with metadata extraction |
| PyMOLVisualizationService | Generate 3D visualizations using PyMOL |
| ChimeraXVisualizationService | Alternative visualizations with UCSF ChimeraX (ALPHA) |
| StructureAnalysisService | Secondary structure, geometry, and RMSD calculations |

## Features

### Structure Fetching
- Download from RCSB PDB database with automatic caching
- Comprehensive metadata extraction (organism, method, resolution)
- Support for PDB, mmCIF, and biological assembly formats
- Batch download for structure comparison workflows

### 3D Visualization (PyMOL)
- Interactive mode for GUI-based exploration
- Batch mode for automated PNG image generation
- Multiple representation styles: cartoon, surface, sticks, spheres, ribbon
- Color schemes: chain, secondary_structure, bfactor, element, custom
- Residue highlighting for disease mutations, binding sites, active sites
- Ray-traced high-quality rendering for publication figures

### Structural Analysis
- Secondary structure distribution analysis (DSSP)
- Geometric properties: radius of gyration, chain length, surface area
- Residue contact maps and distance matrices
- B-factor analysis for flexibility assessment

### Structure Comparison
- RMSD calculation with optional structural alignment
- Biological interpretation of similarity scores
- Chain-specific comparisons for multi-chain complexes
- Sequence identity and structural coverage metrics

### Data Integration
- Link gene expression levels to PDB structures
- Filter differentially expressed genes by structure availability
- Cross-reference proteomics data with structural annotations
- Annotate structures with functional data from omics analysis

## Requirements

- Python 3.12+
- lobster-ai >= 1.0.0
- PyMOL (optional, for visualization execution)

## PyMOL Installation

For visualization execution, install PyMOL:

```bash
# macOS/Linux via Homebrew
brew install brewsci/bio/pymol

# Or download from https://pymol.org/
```

If PyMOL is not installed, the agent generates command scripts that can be executed manually.

## Documentation

Full documentation: [docs.omics-os.com/docs/agents/structural-viz](https://docs.omics-os.com/docs/agents/structural-viz)

## License

AGPL-3.0-or-later
