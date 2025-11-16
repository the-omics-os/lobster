"""
Protein structure analysis services for fetching, visualizing, and analyzing 3D structures.

This module provides stateless services for protein structure operations following
the 3-tuple pattern: (AnnData, stats_dict, IR).
"""

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import anndata
import numpy as np

from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.providers.pdb_provider import PDBProvider

logger = logging.getLogger(__name__)


class ProteinStructureServiceError(Exception):
    """Base exception for protein structure service errors."""
    pass


class StructureFetchService:
    """
    Service for fetching protein structures from PDB.

    This service handles downloading structures from RCSB PDB and
    converting them to AnnData format using the adapter.
    """

    def __init__(self, data_manager: DataManagerV2):
        """
        Initialize structure fetch service.

        Args:
            data_manager: DataManagerV2 instance for workspace management
        """
        self.data_manager = data_manager
        self.pdb_provider = PDBProvider(data_manager=data_manager)

    def fetch_structure(
        self,
        pdb_id: str,
        format: str = "cif",
        include_hetero: bool = True,
        include_waters: bool = False,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Fetch protein structure from PDB and convert to AnnData.

        Args:
            pdb_id: PDB identifier (4 characters)
            format: File format ('pdb' or 'cif')
            include_hetero: Whether to include HETATM records (ligands)
            include_waters: Whether to include water molecules

        Returns:
            Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
                - Loaded structure as AnnData
                - Fetch statistics
                - IR for provenance

        Raises:
            ProteinStructureServiceError: If fetch or parsing fails
        """
        logger.info(f"Fetching structure {pdb_id} from PDB (format={format})")

        try:
            # Validate PDB ID
            if not self.pdb_provider.validate_pdb_id(pdb_id):
                raise ProteinStructureServiceError(
                    f"Invalid or non-existent PDB ID: {pdb_id}"
                )

            # Get structure metadata
            metadata = self.pdb_provider.get_structure_metadata(pdb_id)
            if metadata is None:
                raise ProteinStructureServiceError(
                    f"Could not retrieve metadata for PDB ID: {pdb_id}"
                )

            # Download structure file
            output_path = self.pdb_provider.download_structure(
                pdb_id=pdb_id,
                format=format,
            )

            if output_path is None or not output_path.exists():
                raise ProteinStructureServiceError(
                    f"Failed to download structure file for {pdb_id}"
                )

            # Load structure using adapter
            from lobster.core.adapters.protein_structure_adapter import (
                ProteinStructureAdapter,
            )

            adapter = ProteinStructureAdapter(
                include_hetero=include_hetero,
                include_waters=include_waters,
            )

            adata = adapter.from_source(
                output_path,
                pdb_id=pdb_id,
            )

            # Add metadata from PDB provider
            adata.uns["pdb_metadata"] = {
                "title": metadata.title,
                "experiment_method": metadata.experiment_method,
                "resolution": metadata.resolution,
                "organism": metadata.organism,
                "chains": metadata.chains,
                "ligands": metadata.ligands,
                "deposition_date": metadata.deposition_date,
                "release_date": metadata.release_date,
                "authors": metadata.authors,
                "publication_doi": metadata.publication_doi,
            }

            # Prepare stats
            stats = {
                "pdb_id": pdb_id,
                "n_atoms": adata.n_obs,
                "n_chains": len(metadata.chains),
                "chains": metadata.chains,
                "n_ligands": len(metadata.ligands),
                "ligands": metadata.ligands,
                "experiment_method": metadata.experiment_method,
                "resolution": metadata.resolution,
                "format": format,
                "file_path": str(output_path),
            }

            # Create IR
            ir = self._create_fetch_ir(
                pdb_id=pdb_id,
                format=format,
                include_hetero=include_hetero,
                include_waters=include_waters,
            )

            logger.info(
                f"Successfully fetched {pdb_id}: {adata.n_obs} atoms, "
                f"{len(metadata.chains)} chains"
            )

            return adata, stats, ir

        except Exception as e:
            logger.error(f"Failed to fetch structure {pdb_id}: {e}")
            raise ProteinStructureServiceError(
                f"Structure fetch failed for {pdb_id}: {e}"
            ) from e

    def _create_fetch_ir(
        self,
        pdb_id: str,
        format: str,
        include_hetero: bool,
        include_waters: bool,
    ) -> AnalysisStep:
        """Create IR for structure fetch operation."""
        parameter_schema = {
            "pdb_id": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=pdb_id,
                required=True,
                validation_rule="len(pdb_id) == 4 and pdb_id.isalnum()",
                description="PDB identifier (4 characters)",
            ),
            "format": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=format,
                required=False,
                validation_rule="format in ['pdb', 'cif']",
                description="File format (pdb or cif)",
            ),
            "include_hetero": ParameterSpec(
                param_type="bool",
                papermill_injectable=True,
                default_value=include_hetero,
                required=False,
                description="Include HETATM records (ligands)",
            ),
            "include_waters": ParameterSpec(
                param_type="bool",
                papermill_injectable=True,
                default_value=include_waters,
                required=False,
                description="Include water molecules",
            ),
        }

        code_template = """# Fetch protein structure from PDB
# PDB ID: {{ pdb_id }}
# Format: {{ format }}

from pathlib import Path
import requests

# Download structure from PDB
pdb_id = "{{ pdb_id }}"
format_ext = "{{ format }}"
url = f"https://files.rcsb.org/download/{pdb_id}.{format_ext}"

output_dir = Path("structures")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / f"{pdb_id}.{format_ext}"

print(f"Downloading {pdb_id} from PDB...")
response = requests.get(url, timeout=30)
response.raise_for_status()

with open(output_path, 'wb') as f:
    f.write(response.content)

print(f"Downloaded to: {output_path}")

# Load structure using BioPython
from Bio.PDB import PDBParser, MMCIFParser
import numpy as np
import pandas as pd

if format_ext == "cif":
    parser = MMCIFParser(QUIET=True)
else:
    parser = PDBParser(QUIET=True)

structure = parser.get_structure(pdb_id, str(output_path))
print(f"Structure loaded: {pdb_id}")

# Extract atom data
model = list(structure.get_models())[0]
atom_data = []
coordinates = []

for chain in model:
    for residue in chain:
        {% if not include_waters %}
        if residue.id[0] == "W":
            continue
        {% endif %}
        {% if not include_hetero %}
        if residue.id[0] not in [" ", "W"]:
            continue
        {% endif %}

        for atom in residue:
            atom_data.append({
                'atom_name': atom.name,
                'residue_name': residue.resname,
                'chain_id': chain.id,
                'residue_number': residue.id[1],
                'element': atom.element,
                'b_factor': atom.bfactor,
            })
            coordinates.append(atom.coord)

print(f"Extracted {len(atom_data)} atoms from structure")
"""

        return AnalysisStep(
            operation="pdb.fetch_structure",
            tool_name="fetch_structure",
            description=f"Fetch protein structure {pdb_id} from PDB",
            library="biopython",
            code_template=code_template,
            imports=[
                "from Bio.PDB import PDBParser, MMCIFParser",
                "import requests",
                "import numpy as np",
                "import pandas as pd",
                "from pathlib import Path",
            ],
            parameters={
                "pdb_id": pdb_id,
                "format": format,
                "include_hetero": include_hetero,
                "include_waters": include_waters,
            },
            parameter_schema=parameter_schema,
            input_entities=[],
            output_entities=["structure_adata"],
            execution_context={
                "operation_type": "data_fetch",
                "source": "RCSB PDB",
            },
            validates_on_export=True,
            requires_validation=False,
        )


class StructureVisualizationService:
    """
    Service for generating protein structure visualizations.

    This service generates ChimeraX commands for structure visualization.
    Note: ChimeraX must be installed separately to execute the commands.
    """

    def visualize_structure(
        self,
        adata: anndata.AnnData,
        style: str = "cartoon",
        color_by: str = "chain",
        show_ligands: bool = True,
        show_waters: bool = False,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Generate ChimeraX visualization commands for structure.

        Args:
            adata: AnnData with protein structure
            style: Visualization style (cartoon, ribbon, sphere, stick, surface)
            color_by: Coloring scheme (chain, element, bfactor, residue)
            show_ligands: Whether to show ligands
            show_waters: Whether to show water molecules

        Returns:
            Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
                - AnnData with visualization metadata
                - Visualization stats
                - IR for provenance

        Raises:
            ProteinStructureServiceError: If visualization setup fails
        """
        logger.info(f"Generating visualization commands (style={style}, color_by={color_by})")

        try:
            # Validate inputs
            if "pdb_id" not in adata.uns:
                raise ProteinStructureServiceError(
                    "AnnData missing 'pdb_id' in uns - not a valid structure"
                )

            pdb_id = adata.uns["pdb_id"]

            # Generate ChimeraX commands
            commands = self._generate_chimerax_commands(
                pdb_id=pdb_id,
                style=style,
                color_by=color_by,
                show_ligands=show_ligands,
                show_waters=show_waters,
            )

            # Store visualization settings
            if "visualization_settings" not in adata.uns:
                adata.uns["visualization_settings"] = {}

            adata.uns["visualization_settings"]["chimerax"] = {
                "style": style,
                "color_by": color_by,
                "show_ligands": show_ligands,
                "show_waters": show_waters,
                "commands": commands,
            }

            # Prepare stats
            stats = {
                "pdb_id": pdb_id,
                "style": style,
                "color_by": color_by,
                "n_commands": len(commands),
                "commands_preview": commands[:3] if len(commands) > 3 else commands,
            }

            # Create IR
            ir = self._create_visualization_ir(
                style=style,
                color_by=color_by,
                show_ligands=show_ligands,
                show_waters=show_waters,
            )

            logger.info(f"Generated {len(commands)} ChimeraX commands")

            return adata, stats, ir

        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
            raise ProteinStructureServiceError(
                f"Visualization failed: {e}"
            ) from e

    def _generate_chimerax_commands(
        self,
        pdb_id: str,
        style: str,
        color_by: str,
        show_ligands: bool,
        show_waters: bool,
    ) -> List[str]:
        """Generate list of ChimeraX commands for visualization."""
        commands = [
            f"# ChimeraX visualization commands for {pdb_id}",
            "",
            f"# Open structure from PDB",
            f"open {pdb_id}",
            "",
        ]

        # Hide everything initially
        commands.append("hide all")

        # Apply style
        style_map = {
            "cartoon": "cartoon",
            "ribbon": "ribbon",
            "sphere": "sphere",
            "stick": "stick",
            "surface": "surface",
        }
        chimerax_style = style_map.get(style, "cartoon")
        commands.append(f"show {chimerax_style}")

        # Apply coloring
        if color_by == "chain":
            commands.append("color bychain")
        elif color_by == "element":
            commands.append("color byatom")
        elif color_by == "bfactor":
            commands.append("color bfactor")
        elif color_by == "residue":
            commands.append("color byattribute residue")

        # Handle ligands
        if show_ligands:
            commands.extend([
                "",
                "# Show ligands",
                "show ligands style stick",
                "color ligands byatom",
            ])
        else:
            commands.append("hide ligands")

        # Handle waters
        if show_waters:
            commands.extend([
                "",
                "# Show waters",
                "show solvent style sphere",
                "color solvent lightblue",
            ])
        else:
            commands.append("hide solvent")

        # Final commands
        commands.extend([
            "",
            "# Adjust view",
            "view",
            "lighting soft",
        ])

        return commands

    def _create_visualization_ir(
        self,
        style: str,
        color_by: str,
        show_ligands: bool,
        show_waters: bool,
    ) -> AnalysisStep:
        """Create IR for visualization operation."""
        parameter_schema = {
            "style": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=style,
                required=False,
                validation_rule="style in ['cartoon', 'ribbon', 'sphere', 'stick', 'surface']",
                description="Visualization style",
            ),
            "color_by": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=color_by,
                required=False,
                validation_rule="color_by in ['chain', 'element', 'bfactor', 'residue']",
                description="Coloring scheme",
            ),
            "show_ligands": ParameterSpec(
                param_type="bool",
                papermill_injectable=True,
                default_value=show_ligands,
                required=False,
                description="Show bound ligands",
            ),
            "show_waters": ParameterSpec(
                param_type="bool",
                papermill_injectable=True,
                default_value=show_waters,
                required=False,
                description="Show water molecules",
            ),
        }

        code_template = """# Generate ChimeraX visualization commands
# Style: {{ style }}
# Color by: {{ color_by }}

# ChimeraX commands (save as script.cxc and run in ChimeraX)
commands = [
    "open structure.pdb",  # Replace with your structure file
    "hide all",
    "show {{ style }}",
    {% if color_by == "chain" %}
    "color bychain",
    {% elif color_by == "element" %}
    "color byatom",
    {% elif color_by == "bfactor" %}
    "color bfactor",
    {% else %}
    "color byattribute residue",
    {% endif %}
    {% if show_ligands %}
    "show ligands style stick",
    "color ligands byatom",
    {% endif %}
    {% if show_waters %}
    "show solvent style sphere",
    "color solvent lightblue",
    {% endif %}
    "view",
    "lighting soft",
]

# Write commands to file
with open("visualization_commands.cxc", "w") as f:
    for cmd in commands:
        f.write(cmd + "\\n")

print("ChimeraX commands saved to: visualization_commands.cxc")
print("Run in ChimeraX: chimerax visualization_commands.cxc")
"""

        return AnalysisStep(
            operation="chimerax.visualize",
            tool_name="visualize_structure",
            description=f"Generate ChimeraX visualization (style={style})",
            library="chimerax",
            code_template=code_template,
            imports=[],
            parameters={
                "style": style,
                "color_by": color_by,
                "show_ligands": show_ligands,
                "show_waters": show_waters,
            },
            parameter_schema=parameter_schema,
            input_entities=["structure_adata"],
            output_entities=["visualization_commands"],
            execution_context={
                "operation_type": "visualization",
                "tool": "ChimeraX",
            },
            validates_on_export=True,
            requires_validation=False,
        )


class StructureAnalysisService:
    """
    Service for protein structure analysis (secondary structure, RMSD, etc.).

    This service provides analysis operations on protein structures.
    """

    def analyze_secondary_structure(
        self,
        adata: anndata.AnnData,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Analyze secondary structure composition.

        Note: Full DSSP analysis requires DSSP software installation.
        This method provides basic secondary structure statistics.

        Args:
            adata: AnnData with protein structure

        Returns:
            Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
                - AnnData with secondary structure annotations
                - Analysis stats
                - IR for provenance

        Raises:
            ProteinStructureServiceError: If analysis fails
        """
        logger.info("Analyzing secondary structure")

        try:
            # Basic secondary structure analysis from residue names
            # More sophisticated analysis would require DSSP

            if "residue_name" not in adata.obs.columns:
                raise ProteinStructureServiceError(
                    "No residue information in structure"
                )

            # Count residues
            residue_counts = adata.obs["residue_name"].value_counts()

            # Basic classification
            helix_residues = 0  # Would need DSSP for accurate classification
            sheet_residues = 0
            loop_residues = adata.n_obs

            # Store summary
            adata.uns["secondary_structure_summary"] = {
                "total_residues": len(adata.obs["residue_number"].unique()),
                "total_atoms": adata.n_obs,
                "residue_composition": residue_counts.to_dict(),
            }

            stats = {
                "total_atoms": adata.n_obs,
                "total_residues": len(adata.obs["residue_number"].unique()),
                "unique_residue_types": len(residue_counts),
                "most_common_residue": residue_counts.index[0] if len(residue_counts) > 0 else None,
            }

            ir = self._create_secondary_structure_ir()

            logger.info(
                f"Secondary structure analyzed: {stats['total_residues']} residues"
            )

            return adata, stats, ir

        except Exception as e:
            logger.error(f"Secondary structure analysis failed: {e}")
            raise ProteinStructureServiceError(
                f"Analysis failed: {e}"
            ) from e

    def calculate_rmsd(
        self,
        adata1: anndata.AnnData,
        adata2: anndata.AnnData,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Calculate RMSD between two structures.

        Args:
            adata1: First structure
            adata2: Second structure

        Returns:
            Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
                - Comparison AnnData
                - RMSD statistics
                - IR for provenance

        Raises:
            ProteinStructureServiceError: If RMSD calculation fails
        """
        logger.info("Calculating RMSD between structures")

        try:
            # Validate structures have same number of atoms
            if adata1.n_obs != adata2.n_obs:
                logger.warning(
                    f"Structures have different atom counts: "
                    f"{adata1.n_obs} vs {adata2.n_obs}. Using minimum."
                )
                n_atoms = min(adata1.n_obs, adata2.n_obs)
            else:
                n_atoms = adata1.n_obs

            # Get coordinates
            coords1 = adata1.X[:n_atoms]
            coords2 = adata2.X[:n_atoms]

            # Calculate RMSD
            diff = coords1 - coords2
            squared_diff = diff ** 2
            mean_squared_diff = np.mean(squared_diff)
            rmsd = np.sqrt(mean_squared_diff)

            # Per-atom deviations
            atom_deviations = np.sqrt(np.sum(squared_diff, axis=1))

            # Create result AnnData
            result_adata = adata1.copy()
            result_adata.obs["rmsd_deviation"] = atom_deviations[:n_atoms]

            # Store comparison metadata
            pdb_id1 = adata1.uns.get("pdb_id", "unknown")
            pdb_id2 = adata2.uns.get("pdb_id", "unknown")

            result_adata.uns["rmsd_comparison"] = {
                "structure1": pdb_id1,
                "structure2": pdb_id2,
                "overall_rmsd": float(rmsd),
                "n_atoms_compared": n_atoms,
                "max_deviation": float(np.max(atom_deviations)),
                "mean_deviation": float(np.mean(atom_deviations)),
            }

            stats = {
                "structure1": pdb_id1,
                "structure2": pdb_id2,
                "rmsd": float(rmsd),
                "n_atoms": n_atoms,
                "max_deviation": float(np.max(atom_deviations)),
                "mean_deviation": float(np.mean(atom_deviations)),
            }

            ir = self._create_rmsd_ir()

            logger.info(f"RMSD calculated: {rmsd:.3f} Å")

            return result_adata, stats, ir

        except Exception as e:
            logger.error(f"RMSD calculation failed: {e}")
            raise ProteinStructureServiceError(
                f"RMSD calculation failed: {e}"
            ) from e

    def _create_secondary_structure_ir(self) -> AnalysisStep:
        """Create IR for secondary structure analysis."""
        code_template = """# Analyze secondary structure composition
# Note: For full DSSP analysis, install DSSP software

# Count residue types
residue_counts = adata.obs["residue_name"].value_counts()

print("Residue composition:")
for residue, count in residue_counts.head(10).items():
    print(f"  {residue}: {count}")

# Store in uns
adata.uns["secondary_structure_summary"] = {
    "total_residues": len(adata.obs["residue_number"].unique()),
    "residue_composition": residue_counts.to_dict(),
}
"""

        return AnalysisStep(
            operation="structure.analyze_secondary",
            tool_name="analyze_secondary_structure",
            description="Analyze secondary structure composition",
            library="biopython",
            code_template=code_template,
            imports=["import numpy as np"],
            parameters={},
            parameter_schema={},
            input_entities=["structure_adata"],
            output_entities=["structure_adata"],
            execution_context={
                "operation_type": "structure_analysis",
            },
            validates_on_export=True,
            requires_validation=False,
        )

    def _create_rmsd_ir(self) -> AnalysisStep:
        """Create IR for RMSD calculation."""
        code_template = """# Calculate RMSD between two structures
import numpy as np

# Get coordinates
coords1 = adata1.X
coords2 = adata2.X

# Ensure same size
n_atoms = min(coords1.shape[0], coords2.shape[0])
coords1 = coords1[:n_atoms]
coords2 = coords2[:n_atoms]

# Calculate RMSD
diff = coords1 - coords2
squared_diff = diff ** 2
mean_squared_diff = np.mean(squared_diff)
rmsd = np.sqrt(mean_squared_diff)

print(f"RMSD: {rmsd:.3f} Å")
print(f"Compared {n_atoms} atoms")

# Per-atom deviations
atom_deviations = np.sqrt(np.sum(squared_diff, axis=1))
print(f"Max deviation: {np.max(atom_deviations):.3f} Å")
print(f"Mean deviation: {np.mean(atom_deviations):.3f} Å")
"""

        return AnalysisStep(
            operation="structure.calculate_rmsd",
            tool_name="calculate_rmsd",
            description="Calculate RMSD between structures",
            library="numpy",
            code_template=code_template,
            imports=["import numpy as np"],
            parameters={},
            parameter_schema={},
            input_entities=["adata1", "adata2"],
            output_entities=["rmsd_result"],
            execution_context={
                "operation_type": "structure_comparison",
            },
            validates_on_export=True,
            requires_validation=False,
        )
