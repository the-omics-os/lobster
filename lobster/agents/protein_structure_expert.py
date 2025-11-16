"""
Protein Structure Expert Agent for specialized 3D protein structure analysis.

This agent focuses exclusively on protein structure analysis using PDB data,
structural visualization with ChimeraX, and structural comparisons.
"""

from typing import List

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from lobster.config.llm_factory import create_llm
from lobster.config.settings import get_settings
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.protein_structure_service import (
    ProteinStructureServiceError,
    StructureAnalysisService,
    StructureFetchService,
    StructureVisualizationService,
)
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class ProteinStructureAgentError(Exception):
    """Base exception for protein structure agent operations."""
    pass


class ModalityNotFoundError(ProteinStructureAgentError):
    """Raised when requested modality doesn't exist."""
    pass


def protein_structure_expert(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "protein_structure_expert_agent",
    handoff_tools: List = None,
):
    """
    Create protein structure expert agent using DataManagerV2 and modular services.

    This agent handles:
    - Fetching protein structures from PDB
    - Generating ChimeraX visualizations
    - Analyzing secondary structure
    - Comparing structures (RMSD)
    - Searching PDB database

    Args:
        data_manager: DataManagerV2 instance for modality management
        callback_handler: Optional callback for streaming
        agent_name: Name of the agent
        handoff_tools: Optional list of handoff tools from supervisor

    Returns:
        LangGraph agent configured for protein structure analysis
    """
    settings = get_settings()
    model_params = settings.get_agent_llm_params("protein_structure_expert_agent")
    llm = create_llm("protein_structure_expert_agent", model_params)

    if callback_handler and hasattr(llm, "with_config"):
        llm = llm.with_config(callbacks=[callback_handler])

    # Initialize stateless services
    fetch_service = StructureFetchService(data_manager=data_manager)
    visualization_service = StructureVisualizationService()
    analysis_service = StructureAnalysisService()

    analysis_results = {"summary": "", "details": {}}

    # -------------------------
    # DATA STATUS TOOLS
    # -------------------------
    @tool
    def check_structure_status(modality_name: str = "") -> str:
        """
        Check if protein structure data is loaded and ready for analysis.

        Args:
            modality_name: Optional specific modality to check, empty string checks all

        Returns:
            str: Status of protein structure data
        """
        try:
            if modality_name == "":
                modalities = data_manager.list_modalities()
                if not modalities:
                    return "No modalities loaded. Use fetch_protein_structure() to download a structure from PDB."

                # Filter for protein structure modalities
                structure_modalities = [
                    mod for mod in modalities
                    if "protein" in mod.lower() or "structure" in mod.lower()
                    or data_manager._detect_modality_type(mod) == "protein_structure"
                ]

                if not structure_modalities:
                    response = f"Available modalities ({len(modalities)}) but none appear to be protein structures:\n"
                    for mod_name in modalities:
                        adata = data_manager.get_modality(mod_name)
                        response += f"- **{mod_name}**: {adata.n_obs} obs × {adata.n_vars} vars\n"
                    response += "\nUse fetch_protein_structure() to download a structure."
                else:
                    response = f"Protein structure modalities found ({len(structure_modalities)}):\n"
                    for mod_name in structure_modalities:
                        adata = data_manager.get_modality(mod_name)
                        pdb_id = adata.uns.get("pdb_id", "unknown")
                        response += f"- **{mod_name}**: {adata.n_obs} atoms (PDB: {pdb_id})\n"

                return response

            else:
                # Check specific modality
                if modality_name not in data_manager.list_modalities():
                    return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"

                adata = data_manager.get_modality(modality_name)

                # Validate it's a protein structure
                if "pdb_id" not in adata.uns:
                    return f"Modality '{modality_name}' doesn't appear to be a protein structure (missing pdb_id)"

                pdb_id = adata.uns["pdb_id"]
                response = f"Protein structure '{modality_name}' ready for analysis:\n"
                response += f"- PDB ID: {pdb_id}\n"
                response += f"- Atoms: {adata.n_obs:,}\n"
                response += f"- Coordinates: {adata.n_vars}D\n"

                # Add metadata if available
                if "pdb_metadata" in adata.uns:
                    meta = adata.uns["pdb_metadata"]
                    if meta.get("title"):
                        response += f"- Title: {meta['title']}\n"
                    if meta.get("experiment_method"):
                        response += f"- Method: {meta['experiment_method']}\n"
                    if meta.get("resolution"):
                        response += f"- Resolution: {meta['resolution']} Å\n"
                    if meta.get("chains"):
                        response += f"- Chains: {', '.join(meta['chains'])}\n"
                    if meta.get("ligands"):
                        response += f"- Ligands: {', '.join(meta['ligands'])}\n"

                analysis_results["details"]["structure_status"] = response
                return response

        except Exception as e:
            logger.error(f"Error checking structure status: {e}")
            return f"Error checking structure status: {str(e)}"

    # -------------------------
    # STRUCTURE FETCH TOOLS
    # -------------------------
    @tool
    def fetch_protein_structure(
        pdb_id: str,
        format: str = "cif",
        include_ligands: bool = True,
        include_waters: bool = False,
    ) -> str:
        """
        Fetch protein structure from PDB database.

        Args:
            pdb_id: PDB identifier (4 characters, e.g., '6FQF', '1AKI')
            format: File format ('pdb' or 'cif'). CIF is recommended.
            include_ligands: Whether to include bound ligands (HETATM records)
            include_waters: Whether to include water molecules

        Returns:
            str: Fetch results and structure information

        Example:
            fetch_protein_structure(pdb_id="6FQF", format="cif")
        """
        try:
            logger.info(f"Fetching structure {pdb_id} from PDB")

            # Fetch structure
            adata, stats, ir = fetch_service.fetch_structure(
                pdb_id=pdb_id.upper(),
                format=format,
                include_hetero=include_ligands,
                include_waters=include_waters,
            )

            # Store in data manager
            modality_name = f"protein_{pdb_id.upper()}"
            data_manager.modalities[modality_name] = adata

            # Log operation
            data_manager.log_tool_usage(
                tool_name="fetch_protein_structure",
                parameters={
                    "pdb_id": pdb_id,
                    "format": format,
                    "include_ligands": include_ligands,
                    "include_waters": include_waters,
                },
                description=f"Fetched protein structure {pdb_id} from PDB",
                ir=ir,
            )

            # Format response
            response = f"""Protein Structure Fetched Successfully!

📦 **Structure: {pdb_id.upper()}**
- Atoms: {stats['n_atoms']:,}
- Chains: {stats['n_chains']} ({', '.join(stats['chains'])})
- Ligands: {stats['n_ligands']} ({', '.join(stats['ligands']) if stats['ligands'] else 'none'})
- Method: {stats['experiment_method']}
- Resolution: {stats['resolution']} Å

💾 **Modality created**: '{modality_name}'
📁 **File saved**: {stats['file_path']}

You can now:
- Visualize with visualize_protein_structure()
- Analyze with analyze_secondary_structure()
- Compare with other structures using compare_structures()"""

            analysis_results["details"]["fetch"] = response
            return response

        except ProteinStructureServiceError as e:
            logger.error(f"Structure fetch error: {e}")
            return f"Failed to fetch structure {pdb_id}: {str(e)}"
        except Exception as e:
            logger.error(f"Error fetching structure: {e}")
            return f"Error fetching structure {pdb_id}: {str(e)}"

    # -------------------------
    # VISUALIZATION TOOLS
    # -------------------------
    @tool
    def visualize_protein_structure(
        modality_name: str,
        style: str = "cartoon",
        color_by: str = "chain",
        show_ligands: bool = True,
        show_waters: bool = False,
    ) -> str:
        """
        Generate ChimeraX visualization commands for protein structure.

        Args:
            modality_name: Name of the structure modality
            style: Visualization style ('cartoon', 'ribbon', 'sphere', 'stick', 'surface')
            color_by: Coloring scheme ('chain', 'element', 'bfactor', 'residue')
            show_ligands: Whether to show bound ligands
            show_waters: Whether to show water molecules

        Returns:
            str: Visualization results with ChimeraX commands

        Example:
            visualize_protein_structure(
                modality_name="protein_6FQF",
                style="cartoon",
                color_by="chain"
            )

        Note:
            ChimeraX must be installed separately to execute the commands.
            Commands are saved in the structure's metadata.
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                return f"Modality '{modality_name}' not found. Use check_structure_status() to see available structures."

            # Get structure
            adata = data_manager.get_modality(modality_name)

            # Generate visualization
            adata_viz, stats, ir = visualization_service.visualize_structure(
                adata=adata,
                style=style,
                color_by=color_by,
                show_ligands=show_ligands,
                show_waters=show_waters,
            )

            # Update modality
            data_manager.modalities[modality_name] = adata_viz

            # Log operation
            data_manager.log_tool_usage(
                tool_name="visualize_protein_structure",
                parameters={
                    "modality_name": modality_name,
                    "style": style,
                    "color_by": color_by,
                    "show_ligands": show_ligands,
                },
                description=f"Generated ChimeraX visualization for {modality_name}",
                ir=ir,
            )

            # Get commands
            commands = stats["commands_preview"]

            response = f"""ChimeraX Visualization Commands Generated!

🎨 **Visualization Settings:**
- Style: {stats['style']}
- Color by: {stats['color_by']}
- Structure: {stats['pdb_id']}

📜 **ChimeraX Commands (first 3):**
"""
            for cmd in commands:
                response += f"   {cmd}\n"

            response += f"""
💾 **Commands stored** in modality's visualization_settings

🖥️ **To visualize:**
1. Install ChimeraX: https://www.cgl.ucsf.edu/chimerax/
2. Use commands from visualization_settings
3. Or run: chimerax {stats['pdb_id']}

Total {stats['n_commands']} commands generated."""

            analysis_results["details"]["visualization"] = response
            return response

        except ProteinStructureServiceError as e:
            logger.error(f"Visualization error: {e}")
            return f"Visualization failed: {str(e)}"
        except Exception as e:
            logger.error(f"Error generating visualization: {e}")
            return f"Error generating visualization: {str(e)}"

    # -------------------------
    # ANALYSIS TOOLS
    # -------------------------
    @tool
    def analyze_secondary_structure(modality_name: str) -> str:
        """
        Analyze secondary structure composition of protein.

        Args:
            modality_name: Name of the structure modality

        Returns:
            str: Secondary structure analysis results

        Example:
            analyze_secondary_structure(modality_name="protein_6FQF")

        Note:
            Full DSSP analysis requires DSSP software installation.
            This provides basic residue composition analysis.
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                return f"Modality '{modality_name}' not found. Use check_structure_status() to see available structures."

            # Get structure
            adata = data_manager.get_modality(modality_name)

            # Analyze
            adata_analyzed, stats, ir = analysis_service.analyze_secondary_structure(
                adata=adata
            )

            # Update modality
            modality_name_analyzed = f"{modality_name}_analyzed"
            data_manager.modalities[modality_name_analyzed] = adata_analyzed

            # Log operation
            data_manager.log_tool_usage(
                tool_name="analyze_secondary_structure",
                parameters={"modality_name": modality_name},
                description=f"Analyzed secondary structure for {modality_name}",
                ir=ir,
            )

            response = f"""Secondary Structure Analysis Complete!

📊 **Structure Composition:**
- Total atoms: {stats['total_atoms']:,}
- Total residues: {stats['total_residues']:,}
- Unique residue types: {stats['unique_residue_types']}
- Most common residue: {stats['most_common_residue']}

💾 **New modality created**: '{modality_name_analyzed}'

Note: For full DSSP secondary structure assignment (helices, sheets, loops),
install DSSP software and use BioPython's DSSP module."""

            analysis_results["details"]["secondary_structure"] = response
            return response

        except ProteinStructureServiceError as e:
            logger.error(f"Secondary structure analysis error: {e}")
            return f"Secondary structure analysis failed: {str(e)}"
        except Exception as e:
            logger.error(f"Error analyzing secondary structure: {e}")
            return f"Error analyzing secondary structure: {str(e)}"

    @tool
    def compare_structures(modality_name1: str, modality_name2: str) -> str:
        """
        Compare two protein structures and calculate RMSD.

        Args:
            modality_name1: Name of first structure modality
            modality_name2: Name of second structure modality

        Returns:
            str: RMSD comparison results

        Example:
            compare_structures(
                modality_name1="protein_6FQF",
                modality_name2="protein_1AKI"
            )
        """
        try:
            # Validate both modalities exist
            for mod_name in [modality_name1, modality_name2]:
                if mod_name not in data_manager.list_modalities():
                    return f"Modality '{mod_name}' not found. Use check_structure_status() to see available structures."

            # Get structures
            adata1 = data_manager.get_modality(modality_name1)
            adata2 = data_manager.get_modality(modality_name2)

            # Calculate RMSD
            adata_comparison, stats, ir = analysis_service.calculate_rmsd(
                adata1=adata1,
                adata2=adata2,
            )

            # Store comparison
            comparison_name = f"{modality_name1}_vs_{modality_name2}_comparison"
            data_manager.modalities[comparison_name] = adata_comparison

            # Log operation
            data_manager.log_tool_usage(
                tool_name="compare_structures",
                parameters={
                    "modality_name1": modality_name1,
                    "modality_name2": modality_name2,
                },
                description=f"Compared structures {modality_name1} vs {modality_name2}",
                ir=ir,
            )

            response = f"""Structure Comparison Complete!

📐 **RMSD Analysis:**
- Structure 1: {stats['structure1']}
- Structure 2: {stats['structure2']}
- Overall RMSD: {stats['rmsd']:.3f} Å
- Atoms compared: {stats['n_atoms']:,}
- Max deviation: {stats['max_deviation']:.3f} Å
- Mean deviation: {stats['mean_deviation']:.3f} Å

💾 **Comparison modality created**: '{comparison_name}'

Interpretation:
- RMSD < 2.0 Å: Very similar structures
- RMSD 2.0-4.0 Å: Moderately similar
- RMSD > 4.0 Å: Significantly different

Per-atom deviations stored in comparison modality's obs['rmsd_deviation']."""

            analysis_results["details"]["comparison"] = response
            return response

        except ProteinStructureServiceError as e:
            logger.error(f"Structure comparison error: {e}")
            return f"Structure comparison failed: {str(e)}"
        except Exception as e:
            logger.error(f"Error comparing structures: {e}")
            return f"Error comparing structures: {str(e)}"

    # -------------------------
    # SEARCH TOOLS
    # -------------------------
    @tool
    def search_pdb_database(query: str, max_results: int = 10) -> str:
        """
        Search RCSB PDB database for protein structures.

        Args:
            query: Search query (protein name, keywords, organism)
            max_results: Maximum number of results to return (1-100)

        Returns:
            str: Search results with PDB IDs and metadata

        Example:
            search_pdb_database(query="lysozyme", max_results=5)
            search_pdb_database(query="kinase homo sapiens", max_results=10)
        """
        try:
            from lobster.tools.providers.pdb_provider import PDBProvider

            pdb_provider = PDBProvider(data_manager=data_manager)

            # Search PDB
            results = pdb_provider.search_publications(
                query=query,
                max_results=max_results,
            )

            if not results:
                return f"No structures found for query: '{query}'"

            response = f"""PDB Search Results for '{query}'

Found {len(results)} structures:

"""
            for i, result in enumerate(results[:max_results], 1):
                pdb_id = result.uid
                title = result.title[:80] + "..." if len(result.title) > 80 else result.title
                method = result.keywords[0] if result.keywords else "Unknown"
                response += f"{i}. **{pdb_id}** - {title}\n"
                response += f"   Method: {method}\n"
                if result.published:
                    response += f"   Released: {result.published}\n"
                response += "\n"

            response += f"""
To fetch a structure, use:
fetch_protein_structure(pdb_id="<PDB_ID>")"""

            analysis_results["details"]["search"] = response
            return response

        except Exception as e:
            logger.error(f"PDB search error: {e}")
            return f"PDB search failed: {str(e)}"

    # -------------------------
    # AGENT SYSTEM PROMPT
    # -------------------------
    system_prompt = """You are the **Protein Structure Expert**, specialized in 3D protein structure analysis.

Your expertise:
- Fetching structures from RCSB Protein Data Bank (PDB)
- Generating ChimeraX visualizations
- Analyzing secondary structure composition
- Calculating structural similarity (RMSD)
- Searching PDB database

**Available Tools:**
1. **check_structure_status()** - Check loaded structures
2. **fetch_protein_structure(pdb_id, format)** - Download from PDB
3. **visualize_protein_structure(modality_name, style, color_by)** - Generate ChimeraX commands
4. **analyze_secondary_structure(modality_name)** - Analyze composition
5. **compare_structures(modality_name1, modality_name2)** - Calculate RMSD
6. **search_pdb_database(query, max_results)** - Search PDB

**Workflow:**
1. Search PDB → 2. Fetch structure → 3. Visualize → 4. Analyze

**Key Guidelines:**
- Always validate PDB IDs (4 alphanumeric characters)
- Recommend CIF format over legacy PDB format
- ChimeraX commands are generated, not executed (user needs ChimeraX installed)
- RMSD < 2Å indicates very similar structures
- Include ligands by default (biologically relevant)
- Explain structural biology concepts clearly

**PDB ID Examples:**
- 6FQF (kinase), 1AKI (lysozyme), 1CRN (crambin), 1HHO (hemoglobin)

Always fetch structures before visualization or analysis."""

    # -------------------------
    # CREATE AGENT
    # -------------------------
    tools = [
        check_structure_status,
        fetch_protein_structure,
        visualize_protein_structure,
        analyze_secondary_structure,
        compare_structures,
        search_pdb_database,
    ]

    # Add handoff tools if provided
    if handoff_tools:
        tools.extend(handoff_tools)

    agent = create_react_agent(
        llm,
        tools=tools,
        state_schema=None,  # Use default AgentState
        state_modifier=system_prompt,
    )

    return agent
