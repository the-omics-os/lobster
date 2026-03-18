"""
Tool factory for the cheminformatics expert child agent.

9 tools for molecular analysis, descriptor calculation, fingerprint similarity,
ADMET prediction, 3D structure preparation, binding site identification, and
compound comparison.

All tools follow the Lobster AI tool pattern:
- Accept string parameters (LLM-friendly)
- Call stateless services returning 3-tuples
- Log with ir=ir for provenance
- Return human-readable summary strings

Cloud visualization integration:
- prepare_molecule_3d writes .sdf files to workspace/data/ for Mol* rendering
- fingerprint_similarity creates Plotly heatmaps via plot_manager
"""

import hashlib
from pathlib import Path
from typing import Callable, List

from langchain_core.tools import tool

from lobster.core.runtime.data_manager import DataManagerV2
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


def create_cheminformatics_tools(
    data_manager: DataManagerV2,
    molecular_analysis_service,
    admet_service,
    pubchem_service,
    compound_prep_service,
) -> List[Callable]:
    """
    Create cheminformatics tools for the cheminformatics expert child agent.

    Args:
        data_manager: DataManagerV2 instance for modality management.
        molecular_analysis_service: MolecularAnalysisService for descriptors,
            Lipinski, fingerprints, 3D prep, and comparison.
        admet_service: ADMETPredictionService for ADMET property prediction.
        pubchem_service: PubChemService for similarity search.
        compound_prep_service: CompoundPreparationService for CAS-to-SMILES
            conversion and binding site identification.

    Returns:
        List of 9 tool functions.
    """

    @tool
    def calculate_descriptors(smiles: str) -> str:
        """Calculate molecular descriptors from a SMILES string. Returns MW, LogP, TPSA, HBD, HBA, rotatable bonds, stereocenters, aromatic rings, formula. Args: smiles - SMILES representation of the molecule."""
        try:
            _, stats, ir = molecular_analysis_service.calculate_descriptors(smiles)

            if "error" in stats:
                return f"Error: {stats['error']}"

            data_manager.log_tool_usage(
                "calculate_descriptors",
                {"smiles": smiles},
                stats,
                ir=ir,
            )

            return (
                f"Descriptors for '{smiles}':\n"
                f"  Formula: {stats.get('formula', 'N/A')}\n"
                f"  MW: {stats.get('molecular_weight', 'N/A')} Da\n"
                f"  LogP: {stats.get('logp', 'N/A')}\n"
                f"  TPSA: {stats.get('tpsa', 'N/A')} A^2\n"
                f"  HBD: {stats.get('hbd', 'N/A')}\n"
                f"  HBA: {stats.get('hba', 'N/A')}\n"
                f"  Rotatable bonds: {stats.get('rotatable_bonds', 'N/A')}\n"
                f"  Stereocenters: {stats.get('stereocenters', 'N/A')}\n"
                f"  Aromatic rings: {stats.get('aromatic_rings', 'N/A')}\n"
                f"  Heavy atoms: {stats.get('num_heavy_atoms', 'N/A')}"
            )
        except Exception as e:
            return f"Error calculating descriptors: {e}"

    calculate_descriptors.metadata = {"categories": ["ANALYZE"], "provenance": True}
    calculate_descriptors.tags = ["ANALYZE"]

    @tool
    def lipinski_check(smiles: str) -> str:
        """Check Lipinski Rule of Five compliance for oral bioavailability. A molecule passes if it violates at most 1 of 4 rules (MW<=500, LogP<=5, HBD<=5, HBA<=10). Args: smiles - SMILES representation of the molecule."""
        try:
            _, stats, ir = molecular_analysis_service.lipinski_check(smiles)

            if "error" in stats:
                return f"Error: {stats['error']}"

            data_manager.log_tool_usage(
                "lipinski_check",
                {"smiles": smiles},
                stats,
                ir=ir,
            )

            verdict = "PASS" if stats.get("overall_pass") else "FAIL"
            violations = stats.get("violations", 0)
            classification = stats.get("classification", "unknown")
            checks = stats.get("checks", {})

            lines = [
                f"Lipinski Rule of Five: {verdict} ({violations} violation(s), {classification})"
            ]
            for rule_name, check in checks.items():
                rule_pass = "PASS" if check.get("pass") else "FAIL"
                lines.append(
                    f"  {rule_name}: {check.get('value', 'N/A')} "
                    f"(threshold: {check.get('threshold', 'N/A')}) [{rule_pass}]"
                )

            return "\n".join(lines)
        except Exception as e:
            return f"Error running Lipinski check: {e}"

    lipinski_check.metadata = {"categories": ["QUALITY"], "provenance": True}
    lipinski_check.tags = ["QUALITY"]

    @tool
    def fingerprint_similarity(
        smiles_list: str, fingerprint: str = "morgan", radius: int = 2
    ) -> str:
        """Compute pairwise Tanimoto similarity matrix using molecular fingerprints. Args: smiles_list - comma-separated SMILES strings (at least 2), fingerprint - fingerprint type (default 'morgan'), radius - Morgan fingerprint radius (default 2 = ECFP4)."""
        try:
            parsed = [s.strip() for s in smiles_list.split(",") if s.strip()]
            if len(parsed) < 2:
                return "Error: Provide at least 2 comma-separated SMILES strings."

            _, stats, ir = molecular_analysis_service.fingerprint_similarity(
                parsed, fingerprint=fingerprint, radius=radius
            )

            if "error" in stats:
                return f"Error: {stats['error']}"

            data_manager.log_tool_usage(
                "fingerprint_similarity",
                {
                    "smiles_list": parsed,
                    "fingerprint": fingerprint,
                    "radius": radius,
                },
                stats,
                ir=ir,
            )

            # Create Plotly heatmap for cloud UI canvas visualization
            matrix = stats.get("similarity_matrix")
            plot_manager = getattr(data_manager, "plot_manager", None)
            if matrix and plot_manager:
                try:
                    import plotly.graph_objects as go

                    labels = stats.get(
                        "compound_ids",
                        [f"Mol {i + 1}" for i in range(len(matrix))],
                    )
                    fig = go.Figure(
                        data=go.Heatmap(
                            z=matrix,
                            x=labels,
                            y=labels,
                            colorscale="Viridis",
                            text=[[f"{v:.3f}" for v in row] for row in matrix],
                            texttemplate="%{text}",
                            hovertemplate=(
                                "Similarity(%{x}, %{y}) = %{z:.3f}<extra></extra>"
                            ),
                        )
                    )
                    fig.update_layout(
                        title=f"Tanimoto Similarity ({fingerprint.title()} Fingerprints)",
                        width=600,
                        height=500,
                    )
                    plot_manager.add_plot(
                        fig,
                        title=f"Fingerprint Similarity ({len(parsed)} compounds)",
                        source="cheminformatics_expert",
                    )
                except Exception as plot_err:
                    logger.warning(f"Failed to create similarity heatmap: {plot_err}")

            n = stats.get("n_compounds", 0)
            mean_sim = stats.get("mean_similarity", 0)
            max_sim = stats.get("max_similarity", 0)
            min_sim = stats.get("min_similarity", 0)

            return (
                f"Fingerprint similarity ({fingerprint}, radius={radius}):\n"
                f"  Compounds: {n}\n"
                f"  Mean similarity: {mean_sim:.4f}\n"
                f"  Max similarity: {max_sim:.4f}\n"
                f"  Min similarity: {min_sim:.4f}\n"
                f"  Matrix computed ({n}x{n} Tanimoto distances)"
            )
        except Exception as e:
            return f"Error computing fingerprint similarity: {e}"

    fingerprint_similarity.metadata = {"categories": ["ANALYZE"], "provenance": True}
    fingerprint_similarity.tags = ["ANALYZE"]

    @tool
    def predict_admet(smiles: str) -> str:
        """Predict ADMET properties (absorption, distribution, metabolism, excretion, toxicity) using RDKit-based heuristics. These are computational predictions, not experimental data. Args: smiles - SMILES representation of the molecule."""
        try:
            _, stats, ir = admet_service.predict_admet(smiles)

            if "error" in stats:
                return f"Error: {stats['error']}"

            data_manager.log_tool_usage(
                "predict_admet",
                {"smiles": smiles},
                stats,
                ir=ir,
            )

            predictions = stats.get("predictions", {})
            overall = stats.get("overall_assessment", "unknown")
            n_flags = stats.get("n_flags", 0)
            flags = stats.get("overall_flags", [])

            lines = [
                f"ADMET Prediction for SMILES '{smiles[:50]}...':",
                f"  Overall: {overall} ({n_flags} flag(s))",
            ]

            if flags:
                lines.append(f"  Flags: {', '.join(flags)}")

            # Absorption
            abs_data = predictions.get("absorption", {})
            lines.append(
                f"  Absorption: oral={'likely' if abs_data.get('oral_absorption_likely') else 'unlikely'}, "
                f"Caco-2={abs_data.get('caco2_permeability_class', 'N/A')}, "
                f"solubility={abs_data.get('solubility_class', 'N/A')}"
            )

            # Distribution
            dist_data = predictions.get("distribution", {})
            lines.append(
                f"  Distribution: BBB={'permeable' if dist_data.get('bbb_permeable_likely') else 'limited'}, "
                f"Vd={dist_data.get('vd_class', 'N/A')}, "
                f"PPB={dist_data.get('ppb_estimate', 'N/A')}"
            )

            # Metabolism
            met_data = predictions.get("metabolism", {})
            lines.append(
                f"  Metabolism: stability={met_data.get('metabolic_stability_estimate', 'N/A')}, "
                f"CYP alerts={met_data.get('n_cyp_alerts', 0)}"
            )

            # Excretion
            exc_data = predictions.get("excretion", {})
            lines.append(
                f"  Excretion: route={exc_data.get('primary_excretion_route', 'N/A')}, "
                f"half-life={exc_data.get('halflife_class', 'N/A')}"
            )

            # Toxicity
            tox_data = predictions.get("toxicity", {})
            lines.append(
                f"  Toxicity: risk={tox_data.get('toxicity_risk_class', 'N/A')}, "
                f"PAINS={tox_data.get('n_pains_alerts', 0)}, "
                f"Brenk={tox_data.get('n_brenk_alerts', 0)}"
            )

            lines.append(
                "\nNote: These are computational predictions based on RDKit "
                "heuristics, not experimental measurements."
            )

            return "\n".join(lines)
        except Exception as e:
            return f"Error predicting ADMET: {e}"

    predict_admet.metadata = {"categories": ["ANALYZE"], "provenance": True}
    predict_admet.tags = ["ANALYZE"]

    @tool
    def prepare_molecule_3d(smiles: str, n_conformers: int = 1) -> str:
        """Generate 3D molecular conformation from SMILES using ETKDG embedding and MMFF94 force field optimization. Args: smiles - SMILES representation of the molecule, n_conformers - number of conformers to generate (best is selected, default 1)."""
        try:
            _, stats, ir = molecular_analysis_service.prepare_molecule_3d(
                smiles, n_conformers=n_conformers
            )

            if "error" in stats:
                return f"Error: {stats['error']}"

            data_manager.log_tool_usage(
                "prepare_molecule_3d",
                {"smiles": smiles, "n_conformers": n_conformers},
                stats,
                ir=ir,
            )

            # Write .sdf file to workspace for Mol* visualization in cloud UI
            sdf_file_info = ""
            mol_block = stats.get("mol_block", "")
            ws_path = getattr(data_manager, "workspace_path", None)
            if mol_block and ws_path:
                try:
                    smiles_hash = hashlib.md5(smiles.encode()).hexdigest()[:8]
                    safe_name = f"molecule_{smiles_hash}"
                    sdf_filename = f"{safe_name}.sdf"
                    data_dir = Path(ws_path) / "data"
                    data_dir.mkdir(parents=True, exist_ok=True)
                    sdf_path = data_dir / sdf_filename
                    sdf_path.write_text(mol_block)
                    sdf_file_info = f"\n  Structure file: data/{sdf_filename}"
                    logger.info(f"Wrote 3D structure to {sdf_path}")
                except Exception as write_err:
                    logger.warning(f"Failed to write SDF file: {write_err}")

            energy = stats.get("energy_kcal_mol")
            energy_str = f"{energy:.3f} kcal/mol" if energy is not None else "N/A"
            n_generated = stats.get("n_conformers_generated", 0)

            return (
                f"3D structure prepared for '{smiles[:50]}...':\n"
                f"  Atoms: {stats.get('num_atoms', 'N/A')} "
                f"(heavy: {stats.get('num_heavy_atoms', 'N/A')})\n"
                f"  Conformers generated: {n_generated}\n"
                f"  Best conformer energy: {energy_str}\n"
                f"  MOL block generated ({len(mol_block)} characters)"
                f"{sdf_file_info}"
            )
        except Exception as e:
            return f"Error preparing 3D structure: {e}"

    prepare_molecule_3d.metadata = {"categories": ["PREPROCESS"], "provenance": True}
    prepare_molecule_3d.tags = ["PREPROCESS"]

    @tool
    def cas_to_smiles(cas_numbers: str) -> str:
        """Convert CAS registry numbers to canonical SMILES via PubChem REST API. Args: cas_numbers - comma-separated CAS numbers (e.g. '50-78-2, 64-17-5')."""
        try:
            parsed = [c.strip() for c in cas_numbers.split(",") if c.strip()]
            if not parsed:
                return "Error: Provide at least one CAS number."

            _, stats, ir = compound_prep_service.cas_to_smiles(parsed)

            data_manager.log_tool_usage(
                "cas_to_smiles",
                {"cas_numbers": parsed},
                stats,
                ir=ir,
            )

            n_resolved = stats.get("n_resolved", 0)
            n_requested = stats.get("n_requested", 0)
            mapping = stats.get("cas_to_smiles", {})
            errors = stats.get("errors", {})

            lines = [
                f"CAS-to-SMILES conversion: {n_resolved}/{n_requested} resolved"
            ]
            for cas, smi in mapping.items():
                lines.append(f"  {cas} -> {smi}")
            for cas, err in errors.items():
                lines.append(f"  {cas} -> FAILED: {err}")

            return "\n".join(lines)
        except Exception as e:
            return f"Error converting CAS numbers: {e}"

    cas_to_smiles.metadata = {"categories": ["ANNOTATE"], "provenance": True}
    cas_to_smiles.tags = ["ANNOTATE"]

    @tool
    def search_similar_compounds(
        smiles: str, threshold: int = 85, limit: int = 20
    ) -> str:
        """Search PubChem for structurally similar compounds using Tanimoto similarity. Args: smiles - SMILES of query compound, threshold - similarity threshold in percent (default 85), limit - max results (default 20)."""
        try:
            _, stats, ir = pubchem_service.search_by_similarity(
                smiles, threshold=threshold, limit=limit
            )

            if "error" in stats:
                return f"Error: {stats['error']}"

            data_manager.log_tool_usage(
                "search_similar_compounds",
                {"smiles": smiles, "threshold": threshold, "limit": limit},
                stats,
                ir=ir,
            )

            n_results = stats.get("n_results", 0)
            compounds = stats.get("compounds", [])

            lines = [
                f"PubChem similarity search (>={threshold}% Tanimoto): "
                f"{n_results} compounds found"
            ]
            for i, comp in enumerate(compounds[:10], 1):
                mw = comp.get("molecular_weight", "N/A")
                xlogp = comp.get("xlogp", "N/A")
                cid = comp.get("cid", "N/A")
                lipinski = comp.get("lipinski", {})
                lip_status = (
                    "pass" if lipinski.get("compliant") else "fail"
                ) if lipinski else "N/A"
                lines.append(
                    f"  {i}. CID={cid}: MW={mw}, XLogP={xlogp}, Lipinski={lip_status}"
                )

            if n_results > 10:
                lines.append(f"  ... and {n_results - 10} more compounds")

            return "\n".join(lines)
        except Exception as e:
            return f"Error searching similar compounds: {e}"

    search_similar_compounds.metadata = {"categories": ["UTILITY"], "provenance": False}
    search_similar_compounds.tags = ["UTILITY"]

    @tool
    def identify_binding_site(
        modality_name: str,
        center_mode: str = "geometric",
        radius: float = 8.0,
    ) -> str:
        """Identify binding site residues from PDB content stored as a modality. Finds residues within a radius of the center point. Args: modality_name - name of modality containing PDB content in uns['pdb_content'], center_mode - center calculation: 'geometric' (mean of CA atoms) or 'ligand' (mean of HETATM), radius - search radius in Angstroms (default 8.0)."""
        try:
            adata = data_manager.get_modality(modality_name)
            if adata is None:
                return (
                    f"Error: Modality '{modality_name}' not found. "
                    "Use list_modalities or check_drug_discovery_status to see available datasets."
                )

            # Get PDB content from uns
            pdb_content = adata.uns.get("pdb_content", "")
            if not pdb_content:
                return (
                    f"Error: Modality '{modality_name}' does not contain PDB content "
                    "in uns['pdb_content']. Load a PDB file first."
                )

            _, stats, ir = compound_prep_service.identify_binding_site(
                pdb_content=pdb_content,
                center_mode=center_mode,
                radius=radius,
            )

            data_manager.log_tool_usage(
                "identify_binding_site",
                {
                    "modality_name": modality_name,
                    "center_mode": center_mode,
                    "radius": radius,
                },
                stats,
                ir=ir,
            )

            n_residues = stats.get("n_residues", 0)
            center = stats.get("center", [])
            residues = stats.get("residues", [])
            res_counts = stats.get("residue_type_counts", {})

            lines = [
                f"Binding site identified in '{modality_name}':",
                f"  Center: [{', '.join(f'{c:.1f}' for c in center)}] "
                f"(mode: {stats.get('center_mode', center_mode)})",
                f"  Radius: {radius} A",
                f"  Residues found: {n_residues}",
            ]

            if res_counts:
                top_res = sorted(
                    res_counts.items(), key=lambda x: x[1], reverse=True
                )[:5]
                lines.append(
                    f"  Top residue types: "
                    + ", ".join(f"{r}({c})" for r, c in top_res)
                )

            for res in residues[:10]:
                lines.append(
                    f"  {res.get('chain_id', '?')}:{res.get('res_name', '?')}"
                    f"{res.get('res_seq', '?')} "
                    f"(dist={res.get('distance_to_center', 0):.1f} A)"
                )
            if n_residues > 10:
                lines.append(f"  ... and {n_residues - 10} more residues")

            return "\n".join(lines)
        except Exception as e:
            return f"Error identifying binding site: {e}"

    identify_binding_site.metadata = {"categories": ["ANALYZE"], "provenance": True}
    identify_binding_site.tags = ["ANALYZE"]

    @tool
    def compare_molecules(smiles_a: str, smiles_b: str) -> str:
        """Side-by-side comparison of two molecules' molecular properties and Tanimoto similarity. Args: smiles_a - SMILES of first molecule, smiles_b - SMILES of second molecule."""
        try:
            _, stats, ir = molecular_analysis_service.compare_molecules(
                smiles_a, smiles_b
            )

            if "error" in stats:
                return f"Error: {stats['error']}"

            data_manager.log_tool_usage(
                "compare_molecules",
                {"smiles_a": smiles_a, "smiles_b": smiles_b},
                stats,
                ir=ir,
            )

            tanimoto = stats.get("tanimoto_similarity", 0)
            comparison = stats.get("comparison_table", [])

            lines = [
                f"Molecule Comparison (Tanimoto similarity: {tanimoto:.4f}):",
                f"  {'Property':<25} {'Molecule A':>12} {'Molecule B':>12} {'Difference':>12}",
                f"  {'-' * 61}",
            ]

            for row in comparison:
                prop = row.get("property", "")
                val_a = row.get("molecule_a", "N/A")
                val_b = row.get("molecule_b", "N/A")
                diff = row.get("difference")
                diff_str = f"{diff:+.3f}" if diff is not None else "N/A"

                if isinstance(val_a, float):
                    val_a = f"{val_a:.3f}"
                if isinstance(val_b, float):
                    val_b = f"{val_b:.3f}"

                lines.append(
                    f"  {prop:<25} {str(val_a):>12} {str(val_b):>12} {diff_str:>12}"
                )

            return "\n".join(lines)
        except Exception as e:
            return f"Error comparing molecules: {e}"

    compare_molecules.metadata = {"categories": ["ANALYZE"], "provenance": True}
    compare_molecules.tags = ["ANALYZE"]

    return [
        calculate_descriptors,
        lipinski_check,
        fingerprint_similarity,
        predict_admet,
        prepare_molecule_3d,
        cas_to_smiles,
        search_similar_compounds,
        identify_binding_site,
        compare_molecules,
    ]
