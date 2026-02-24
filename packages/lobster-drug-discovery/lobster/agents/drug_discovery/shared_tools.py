"""
Shared tools for drug discovery analysis (target identification, compound profiling).

This module provides 10 tools for the drug discovery parent agent. Tools call
stateless services (ChEMBL, Open Targets, PubChem, TargetScoring) and log
provenance via data_manager.log_tool_usage() with ir=ir.

Following the same factory pattern as proteomics shared_tools.py.
"""

import json
from typing import Any, Callable, Dict, List

import httpx
from langchain_core.tools import tool

from lobster.agents.drug_discovery.config import (
    CHEMBL_API_BASE,
    OPENTARGETS_GRAPHQL,
    PUBCHEM_API_BASE,
)
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.drug_discovery.chembl_service import ChEMBLService
from lobster.services.drug_discovery.opentargets_service import OpenTargetsService
from lobster.services.drug_discovery.pubchem_service import PubChemService
from lobster.services.drug_discovery.target_scoring_service import TargetScoringService
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# SHARED TOOL FACTORY
# =============================================================================


def create_shared_tools(
    data_manager: DataManagerV2,
    chembl_service: ChEMBLService,
    opentargets_service: OpenTargetsService,
    pubchem_service: PubChemService,
    target_scoring_service: TargetScoringService,
) -> List[Callable]:
    """
    Create shared drug discovery tools for the parent agent.

    These tools wrap stateless services for target identification, compound
    search, bioactivity retrieval, and target scoring. Each tool logs
    provenance via data_manager.log_tool_usage() with ir=ir.

    Args:
        data_manager: DataManagerV2 instance for modality management and logging.
        chembl_service: ChEMBLService for compound search and bioactivity.
        opentargets_service: OpenTargetsService for target-disease evidence.
        pubchem_service: PubChemService for compound properties.
        target_scoring_service: TargetScoringService for druggability scoring.

    Returns:
        List of 10 tool functions to be added to the parent agent.
    """

    # -------------------------------------------------------------------------
    # TOOL 1: search_drug_targets
    # -------------------------------------------------------------------------
    @tool
    def search_drug_targets(
        query: str,
        disease_context: str = "",
        limit: int = 25,
    ) -> str:
        """
        Search Open Targets for drug targets by Ensembl gene ID.

        Retrieves target-disease association evidence including overall scores,
        datatype breakdowns (genetic_association, known_drug, literature, etc.),
        and associated disease names.

        Args:
            query: Ensembl gene ID (e.g., 'ENSG00000157764' for BRAF).
            disease_context: Optional EFO disease ID to filter associations (e.g., 'EFO_0000311').
            limit: Maximum number of disease associations to return (default 25).
        """
        try:
            disease_id = disease_context if disease_context else None
            _, stats, ir = opentargets_service.get_target_disease_evidence(
                ensembl_id=query,
                disease_id=disease_id,
                limit=limit,
            )

            data_manager.log_tool_usage(
                tool_name="search_drug_targets",
                parameters={
                    "query": query,
                    "disease_context": disease_context,
                    "limit": limit,
                },
                description=f"Searched Open Targets for target {query}",
                ir=ir,
            )

            if "error" in stats:
                return (
                    f"Error searching Open Targets for '{query}': {stats['error']}\n\n"
                    "Ensure the query is a valid Ensembl gene ID (e.g., ENSG00000157764)."
                )

            symbol = stats.get("approved_symbol", query)
            name = stats.get("approved_name", "")
            n_assoc = stats.get("n_associations", 0)
            total = stats.get("total_associated_diseases", 0)

            response = f"**Target: {symbol}** ({name})\n"
            response += f"Total associated diseases: {total}\n"
            response += f"Showing top {n_assoc} associations:\n\n"

            associations = stats.get("associations", [])
            for i, assoc in enumerate(associations[:10], start=1):
                disease_name = assoc.get("disease_name", "Unknown")
                overall = assoc.get("overall_score", 0.0)
                datatype_scores = assoc.get("datatype_scores", {})
                dt_summary = ", ".join(
                    f"{k}={v:.2f}"
                    for k, v in sorted(
                        datatype_scores.items(), key=lambda x: -x[1]
                    )[:3]
                )
                response += (
                    f"{i}. **{disease_name}** (score: {overall:.3f})"
                )
                if dt_summary:
                    response += f" [{dt_summary}]"
                response += "\n"

            if n_assoc > 10:
                response += f"\n... and {n_assoc - 10} more associations.\n"

            return response

        except Exception as e:
            logger.error(f"Error in search_drug_targets: {e}")
            return f"Error searching drug targets: {str(e)}"

    # -------------------------------------------------------------------------
    # TOOL 2: score_drug_target
    # -------------------------------------------------------------------------
    @tool
    def score_drug_target(
        gene_symbol: str,
        evidence: str = "",
    ) -> str:
        """
        Score a drug target using a weighted evidence-based druggability model.

        If evidence JSON is provided (keys: genetic_association, known_drug,
        expression_specificity, pathogenicity, literature, each 0.0-1.0),
        computes a composite score. Otherwise returns guidance on required input.

        Args:
            gene_symbol: Gene symbol for identification (e.g., 'BRAF', 'EGFR').
            evidence: Optional JSON string with evidence scores per category.
        """
        try:
            if not evidence:
                return (
                    f"To score target '{gene_symbol}', provide evidence as a JSON string "
                    "with scores (0.0-1.0) for these categories:\n"
                    "- genetic_association (weight 0.30): GWAS/Mendelian evidence\n"
                    "- known_drug (weight 0.25): existing drugs against this target\n"
                    "- expression_specificity (weight 0.20): tissue-specific expression\n"
                    "- pathogenicity (weight 0.15): pathogenic variant evidence\n"
                    "- literature (weight 0.10): publication evidence density\n\n"
                    'Example: {"genetic_association": 0.8, "known_drug": 0.6, '
                    '"expression_specificity": 0.5, "pathogenicity": 0.3, "literature": 0.7}\n\n'
                    "Tip: Use search_drug_targets first to retrieve evidence from Open Targets, "
                    "then extract scores from the datatype breakdown."
                )

            # Parse evidence JSON
            try:
                evidence_dict = json.loads(evidence)
            except json.JSONDecodeError as parse_err:
                return (
                    f"Invalid JSON in evidence parameter: {parse_err}\n"
                    'Expected format: {{"genetic_association": 0.8, "known_drug": 0.6, ...}}'
                )

            if not isinstance(evidence_dict, dict):
                return "Evidence must be a JSON object mapping category names to float scores (0.0-1.0)."

            # Validate values are numeric and in range
            for key, val in evidence_dict.items():
                if not isinstance(val, (int, float)):
                    return f"Evidence value for '{key}' must be a number, got {type(val).__name__}."
                if not (0.0 <= val <= 1.0):
                    return f"Evidence value for '{key}' must be in [0.0, 1.0], got {val}."

            _, result, ir = target_scoring_service.score_target(evidence_dict)

            data_manager.log_tool_usage(
                tool_name="score_drug_target",
                parameters={
                    "gene_symbol": gene_symbol,
                    "evidence": evidence_dict,
                },
                description=f"Scored target {gene_symbol} for druggability",
                ir=ir,
            )

            overall = result.get("overall_score", 0.0)
            classification = result.get("classification", "unknown")
            strongest = result.get("strongest_evidence", "N/A")
            weakest = result.get("weakest_evidence", "N/A")
            missing = result.get("evidence_missing", [])

            response = f"**Target Druggability Score for {gene_symbol}**\n\n"
            response += f"Overall score: **{overall:.3f}** ({classification})\n\n"
            response += "**Component Breakdown:**\n"

            component_scores = result.get("component_scores", {})
            for category, detail in component_scores.items():
                raw = detail.get("raw_score", 0.0)
                weight = detail.get("weight", 0.0)
                weighted = detail.get("weighted_score", 0.0)
                response += f"- {category}: {raw:.2f} x {weight:.2f} = {weighted:.3f}\n"

            response += f"\nStrongest evidence: {strongest}\n"
            response += f"Weakest evidence: {weakest}\n"
            if missing:
                response += f"Missing categories: {', '.join(missing)}\n"

            if classification == "high_confidence":
                response += "\nVerdict: Strong drug target candidate."
            elif classification == "medium_confidence":
                response += "\nVerdict: Moderate potential, consider gathering more evidence."
            else:
                response += "\nVerdict: Weak evidence, additional validation needed."

            return response

        except Exception as e:
            logger.error(f"Error in score_drug_target: {e}")
            return f"Error scoring target '{gene_symbol}': {str(e)}"

    # -------------------------------------------------------------------------
    # TOOL 3: rank_targets
    # -------------------------------------------------------------------------
    @tool
    def rank_targets(
        gene_list: str,
        disease_context: str = "",
    ) -> str:
        """
        Rank multiple drug targets by composite druggability score.

        For each gene, fetches evidence from Open Targets and computes a
        weighted druggability score, then ranks all targets from best to worst.

        Args:
            gene_list: Comma-separated Ensembl gene IDs (e.g., 'ENSG00000157764,ENSG00000146648').
            disease_context: Optional disease context for informational display.
        """
        try:
            genes = [g.strip() for g in gene_list.split(",") if g.strip()]
            if not genes:
                return "Error: provide at least one Ensembl gene ID in gene_list (comma-separated)."

            if len(genes) > 50:
                return (
                    f"Error: too many genes ({len(genes)}). "
                    "Limit to 50 or fewer for performance."
                )

            # Collect evidence for each gene via Open Targets
            targets_evidence = []
            errors = []

            for gene_id in genes:
                _, stats, _ = opentargets_service.score_target(
                    ensembl_id=gene_id,
                )

                if "error" in stats:
                    errors.append(f"{gene_id}: {stats['error']}")
                    continue

                # Extract component scores from the OT scoring result
                symbol = stats.get("approved_symbol", gene_id)
                component_scores = stats.get("component_scores", {})

                # Convert OT component scores to evidence dict for the
                # target_scoring_service (values are already in the right format)
                evidence_dict = {}
                if component_scores:
                    evidence_dict = {
                        k: v for k, v in component_scores.items()
                        if isinstance(v, (int, float)) and k in {
                            "genetic_association",
                            "known_drug",
                            "expression_specificity",
                            "pathogenicity",
                            "literature",
                        }
                    }
                else:
                    # Fallback: use the overall druggability_score as genetic_association proxy
                    score = stats.get("druggability_score", 0.0)
                    evidence_dict = {"genetic_association": score}

                targets_evidence.append((symbol, evidence_dict))

            if not targets_evidence:
                error_msg = "No valid targets to rank."
                if errors:
                    error_msg += "\nErrors:\n" + "\n".join(f"- {e}" for e in errors)
                return error_msg

            # Rank via target scoring service
            _, result, ir = target_scoring_service.rank_targets(targets_evidence)

            data_manager.log_tool_usage(
                tool_name="rank_targets",
                parameters={
                    "gene_list": gene_list,
                    "disease_context": disease_context,
                    "n_targets": len(targets_evidence),
                },
                description=f"Ranked {len(targets_evidence)} targets by druggability",
                ir=ir,
            )

            # Format response
            ranked = result.get("ranked_targets", [])
            mean_score = result.get("mean_score", 0.0)
            summary = result.get("summary", {})

            response = "**Drug Target Ranking**\n"
            if disease_context:
                response += f"Disease context: {disease_context}\n"
            response += f"Targets ranked: {len(ranked)}\n"
            response += f"Mean score: {mean_score:.3f}\n\n"

            for entry in ranked:
                rank = entry.get("rank", "?")
                symbol = entry.get("gene_symbol", "?")
                score = entry.get("overall_score", 0.0)
                classification = entry.get("classification", "unknown")
                response += f"{rank}. **{symbol}**: {score:.3f} ({classification})\n"

            response += "\n**Summary:**\n"
            response += f"- High confidence: {summary.get('high_confidence', 0)}\n"
            response += f"- Medium confidence: {summary.get('medium_confidence', 0)}\n"
            response += f"- Low confidence: {summary.get('low_confidence', 0)}\n"

            if errors:
                response += f"\n**Warnings** ({len(errors)} targets could not be scored):\n"
                for err in errors[:5]:
                    response += f"- {err}\n"

            return response

        except Exception as e:
            logger.error(f"Error in rank_targets: {e}")
            return f"Error ranking targets: {str(e)}"

    # -------------------------------------------------------------------------
    # TOOL 4: search_compounds
    # -------------------------------------------------------------------------
    @tool
    def search_compounds(
        query: str,
        source: str = "chembl",
        limit: int = 20,
    ) -> str:
        """
        Search ChEMBL for compounds by name, synonym, or SMILES fragment.

        Returns compound identifiers, names, molecular properties, max clinical
        phase, and Ro5 violation counts.

        Args:
            query: Compound name, synonym, or SMILES fragment to search.
            source: Database to search (currently 'chembl' supported).
            limit: Maximum number of results (default 20).
        """
        try:
            if source.lower() != "chembl":
                return (
                    f"Source '{source}' not supported. Currently only 'chembl' is available. "
                    "Use get_compound_properties for PubChem lookups."
                )

            _, stats, ir = chembl_service.search_compounds(
                query=query,
                limit=limit,
            )

            data_manager.log_tool_usage(
                tool_name="search_compounds",
                parameters={
                    "query": query,
                    "source": source,
                    "limit": limit,
                },
                description=f"Searched ChEMBL for compounds matching '{query}'",
                ir=ir,
            )

            if "error" in stats:
                return f"Error searching ChEMBL for '{query}': {stats['error']}"

            compounds = stats.get("compounds", [])
            n_results = stats.get("n_results", 0)
            has_more = stats.get("has_more", False)

            if n_results == 0:
                return f"No compounds found in ChEMBL matching '{query}'."

            response = f"**ChEMBL Search Results for '{query}'**: {n_results} compounds found\n\n"

            for i, compound in enumerate(compounds[:15], start=1):
                chembl_id = compound.get("chembl_id", "")
                pref_name = compound.get("pref_name", "") or "N/A"
                mol_type = compound.get("molecule_type", "")
                max_phase = compound.get("max_phase", 0)
                mw = compound.get("molecular_weight")
                logp = compound.get("alogp")
                ro5 = compound.get("ro5_violations")
                smiles = compound.get("canonical_smiles", "")

                response += f"{i}. **{chembl_id}** - {pref_name}\n"
                response += f"   Type: {mol_type}, Max phase: {max_phase}\n"
                if mw is not None:
                    response += f"   MW: {mw}"
                    if logp is not None:
                        response += f", LogP: {logp}"
                    if ro5 is not None:
                        response += f", Ro5 violations: {ro5}"
                    response += "\n"
                if smiles:
                    response += f"   SMILES: {smiles[:80]}{'...' if len(smiles) > 80 else ''}\n"
                response += "\n"

            if has_more:
                response += f"More results available (showing first {min(15, n_results)}).\n"

            return response

        except Exception as e:
            logger.error(f"Error in search_compounds: {e}")
            return f"Error searching compounds: {str(e)}"

    # -------------------------------------------------------------------------
    # TOOL 5: get_compound_bioactivity
    # -------------------------------------------------------------------------
    @tool
    def get_compound_bioactivity(
        chembl_id: str,
        target_chembl_id: str = "",
    ) -> str:
        """
        Get bioactivity data (IC50, Ki, EC50, etc.) for a compound from ChEMBL.

        Retrieves activity measurements including standard type, value, units,
        pChEMBL values, and target information.

        Args:
            chembl_id: ChEMBL molecule identifier (e.g., 'CHEMBL25').
            target_chembl_id: Optional target ChEMBL ID to filter results.
        """
        try:
            target_filter = target_chembl_id if target_chembl_id else None
            _, stats, ir = chembl_service.get_bioactivity(
                chembl_id=chembl_id,
                target_chembl_id=target_filter,
            )

            data_manager.log_tool_usage(
                tool_name="get_compound_bioactivity",
                parameters={
                    "chembl_id": chembl_id,
                    "target_chembl_id": target_chembl_id,
                },
                description=f"Retrieved bioactivity for {chembl_id}",
                ir=ir,
            )

            if "error" in stats:
                return f"Error getting bioactivity for {chembl_id}: {stats['error']}"

            activities = stats.get("activities", [])
            n_activities = stats.get("n_activities", 0)
            type_counts = stats.get("activity_type_counts", {})

            if n_activities == 0:
                return (
                    f"No bioactivity data found for {chembl_id}"
                    + (f" against target {target_chembl_id}" if target_chembl_id else "")
                    + "."
                )

            response = f"**Bioactivity for {chembl_id}**: {n_activities} records\n"
            if target_chembl_id:
                response += f"Filtered to target: {target_chembl_id}\n"
            response += "\n"

            # Show activity type distribution
            if type_counts:
                response += "**Activity types:** "
                response += ", ".join(
                    f"{k}: {v}" for k, v in sorted(type_counts.items(), key=lambda x: -x[1])
                )
                response += "\n\n"

            # Show top activities
            response += "**Top activities:**\n"
            for i, act in enumerate(activities[:15], start=1):
                std_type = act.get("standard_type", "")
                std_value = act.get("standard_value")
                std_units = act.get("standard_units", "")
                pchembl = act.get("pchembl_value")
                target_name = act.get("target_pref_name", "")
                organism = act.get("target_organism", "")

                line = f"{i}. {std_type}"
                if std_value is not None:
                    line += f" = {std_value} {std_units}"
                if pchembl is not None:
                    line += f" (pChEMBL: {pchembl})"
                if target_name:
                    line += f" | {target_name}"
                    if organism:
                        line += f" ({organism})"
                response += line + "\n"

            if n_activities > 15:
                response += f"\n... and {n_activities - 15} more records.\n"

            return response

        except Exception as e:
            logger.error(f"Error in get_compound_bioactivity: {e}")
            return f"Error getting bioactivity for {chembl_id}: {str(e)}"

    # -------------------------------------------------------------------------
    # TOOL 6: get_target_compounds
    # -------------------------------------------------------------------------
    @tool
    def get_target_compounds(
        target_chembl_id: str,
        activity_type: str = "IC50",
        limit: int = 50,
    ) -> str:
        """
        Find compounds tested against a specific ChEMBL target.

        Returns unique compounds aggregated by best (most potent) activity value,
        sorted by potency.

        Args:
            target_chembl_id: ChEMBL target identifier (e.g., 'CHEMBL202').
            activity_type: Activity measurement type filter (default 'IC50'). Options: IC50, Ki, EC50, Kd.
            limit: Maximum number of activity records to retrieve (default 50).
        """
        try:
            _, stats, ir = chembl_service.get_target_compounds(
                target_chembl_id=target_chembl_id,
                activity_type=activity_type,
                limit=limit,
            )

            data_manager.log_tool_usage(
                tool_name="get_target_compounds",
                parameters={
                    "target_chembl_id": target_chembl_id,
                    "activity_type": activity_type,
                    "limit": limit,
                },
                description=f"Found compounds for target {target_chembl_id}",
                ir=ir,
            )

            if "error" in stats:
                return f"Error finding compounds for {target_chembl_id}: {stats['error']}"

            compounds = stats.get("compounds", [])
            n_unique = stats.get("n_unique_compounds", 0)
            n_activities = stats.get("n_activities", 0)
            has_more = stats.get("has_more", False)

            if n_unique == 0:
                return (
                    f"No compounds with {activity_type} data found for "
                    f"target {target_chembl_id}."
                )

            response = (
                f"**Compounds for target {target_chembl_id}** "
                f"({activity_type}): {n_unique} unique compounds "
                f"from {n_activities} measurements\n\n"
            )

            for i, comp in enumerate(compounds[:20], start=1):
                mol_id = comp.get("molecule_chembl_id", "")
                mol_name = comp.get("molecule_pref_name", "") or "N/A"
                best_val = comp.get("best_value")
                best_units = comp.get("best_units", "")
                best_pchembl = comp.get("best_pchembl")
                n_meas = comp.get("n_measurements", 0)

                line = f"{i}. **{mol_id}** ({mol_name})"
                if best_val is not None:
                    line += f" - Best {activity_type}: {best_val} {best_units}"
                if best_pchembl is not None:
                    line += f" (pChEMBL: {best_pchembl})"
                line += f" [{n_meas} measurements]"
                response += line + "\n"

            if has_more:
                response += f"\nMore compounds available beyond the {limit}-record limit.\n"

            return response

        except Exception as e:
            logger.error(f"Error in get_target_compounds: {e}")
            return f"Error finding compounds for {target_chembl_id}: {str(e)}"

    # -------------------------------------------------------------------------
    # TOOL 7: get_compound_properties
    # -------------------------------------------------------------------------
    @tool
    def get_compound_properties(
        identifier: str,
        id_type: str = "name",
    ) -> str:
        """
        Get molecular properties for a compound from PubChem.

        Retrieves molecular formula, weight, XLogP, TPSA, H-bond donors/acceptors,
        rotatable bonds, heavy atom count, and Lipinski Rule of Five evaluation.

        Args:
            identifier: Compound name, CID, SMILES, or InChIKey.
            id_type: Identifier type - one of 'name', 'cid', 'smiles', or 'inchikey' (default 'name').
        """
        try:
            valid_types = {"name", "cid", "smiles", "inchikey"}
            if id_type not in valid_types:
                return (
                    f"Invalid id_type '{id_type}'. "
                    f"Must be one of: {', '.join(sorted(valid_types))}."
                )

            _, stats, ir = pubchem_service.get_compound_properties(
                identifier=identifier,
                id_type=id_type,
            )

            data_manager.log_tool_usage(
                tool_name="get_compound_properties",
                parameters={
                    "identifier": identifier,
                    "id_type": id_type,
                },
                description=f"Retrieved PubChem properties for {id_type}='{identifier}'",
                ir=ir,
            )

            if "error" in stats:
                return f"Error getting properties for {id_type}='{identifier}': {stats['error']}"

            cid = stats.get("cid", "N/A")
            formula = stats.get("molecular_formula", "")
            mw = stats.get("molecular_weight")
            xlogp = stats.get("xlogp")
            tpsa = stats.get("tpsa")
            hbd = stats.get("hbond_donor_count")
            hba = stats.get("hbond_acceptor_count")
            rot_bonds = stats.get("rotatable_bond_count")
            heavy_atoms = stats.get("heavy_atom_count")
            lipinski = stats.get("lipinski", {})

            response = f"**PubChem Properties for '{identifier}'** (CID: {cid})\n\n"
            response += f"- Molecular formula: {formula}\n"
            response += f"- Molecular weight: {mw} Da\n" if mw is not None else ""
            response += f"- XLogP: {xlogp}\n" if xlogp is not None else ""
            response += f"- TPSA: {tpsa} A^2\n" if tpsa is not None else ""
            response += f"- H-bond donors: {hbd}\n" if hbd is not None else ""
            response += f"- H-bond acceptors: {hba}\n" if hba is not None else ""
            response += f"- Rotatable bonds: {rot_bonds}\n" if rot_bonds is not None else ""
            response += f"- Heavy atoms: {heavy_atoms}\n" if heavy_atoms is not None else ""

            if lipinski:
                compliant = lipinski.get("compliant", False)
                n_violations = lipinski.get("n_violations", 0)
                rules = lipinski.get("rules", {})
                response += "\n**Lipinski Rule of Five:**\n"
                response += f"- Compliant: {'Yes' if compliant else 'No'} ({n_violations} violations)\n"
                for rule_name, passed in rules.items():
                    status = "PASS" if passed else ("FAIL" if passed is False else "N/A")
                    response += f"  - {rule_name}: {status}\n"

            return response

        except Exception as e:
            logger.error(f"Error in get_compound_properties: {e}")
            return f"Error getting compound properties: {str(e)}"

    # -------------------------------------------------------------------------
    # TOOL 8: get_drug_indications
    # -------------------------------------------------------------------------
    @tool
    def get_drug_indications(
        chembl_id: str,
    ) -> str:
        """
        Get known therapeutic indications and clinical trial phase for a drug.

        Retrieves indications, max clinical trial phase, mechanisms of action,
        and phase distribution from Open Targets.

        Args:
            chembl_id: ChEMBL drug identifier (e.g., 'CHEMBL25' for aspirin).
        """
        try:
            _, stats, ir = opentargets_service.get_drug_indications(
                chembl_id=chembl_id,
            )

            data_manager.log_tool_usage(
                tool_name="get_drug_indications",
                parameters={"chembl_id": chembl_id},
                description=f"Retrieved drug indications for {chembl_id}",
                ir=ir,
            )

            if "error" in stats:
                return f"Error getting indications for {chembl_id}: {stats['error']}"

            drug_name = stats.get("drug_name", chembl_id)
            drug_type = stats.get("drug_type", "")
            max_phase = stats.get("max_clinical_trial_phase", "N/A")
            n_indications = stats.get("n_indications", 0)
            total = stats.get("total_indications", 0)
            phase_dist = stats.get("phase_distribution", {})
            mechanisms = stats.get("mechanisms_of_action", [])
            indications = stats.get("indications", [])

            response = f"**Drug: {drug_name}** ({chembl_id})\n"
            response += f"Type: {drug_type}\n"
            response += f"Max clinical trial phase: {max_phase}\n"
            response += f"Total indications: {total}\n\n"

            if mechanisms:
                response += "**Mechanisms of Action:**\n"
                for moa in mechanisms[:5]:
                    mechanism = moa.get("mechanism", "")
                    targets = moa.get("targets", [])
                    target_str = ", ".join(
                        t.get("symbol", "") for t in targets if t.get("symbol")
                    )
                    response += f"- {mechanism}"
                    if target_str:
                        response += f" (targets: {target_str})"
                    response += "\n"
                response += "\n"

            if indications:
                response += "**Indications:**\n"
                for i, ind in enumerate(indications[:15], start=1):
                    disease_name = ind.get("disease_name", "Unknown")
                    ind_phase = ind.get("max_phase", 0)
                    response += f"{i}. {disease_name} (phase {ind_phase})\n"

                if n_indications > 15:
                    response += f"\n... and {n_indications - 15} more indications.\n"

            if phase_dist:
                response += "\n**Phase Distribution:** "
                response += ", ".join(
                    f"Phase {k}: {v}" for k, v in sorted(phase_dist.items())
                )
                response += "\n"

            return response

        except Exception as e:
            logger.error(f"Error in get_drug_indications: {e}")
            return f"Error getting indications for {chembl_id}: {str(e)}"

    # -------------------------------------------------------------------------
    # TOOL 9: check_drug_discovery_status
    # -------------------------------------------------------------------------
    @tool
    def check_drug_discovery_status() -> str:
        """
        List modalities containing drug discovery data and show workspace status.

        Scans the current workspace for modalities with drug-discovery-relevant
        names (target, compound, drug, chembl, bioactivity, etc.).
        """
        try:
            modalities = data_manager.list_modalities()

            drug_discovery_terms = [
                "target",
                "compound",
                "drug",
                "chembl",
                "bioactivity",
                "pharmacogenomics",
                "admet",
                "synergy",
                "binding",
                "docking",
                "fingerprint",
                "descriptor",
                "lipinski",
            ]

            dd_modalities = [
                m
                for m in modalities
                if any(term in m.lower() for term in drug_discovery_terms)
            ]

            if not dd_modalities and not modalities:
                return (
                    "No modalities found in workspace.\n"
                    "Start by searching for drug targets with search_drug_targets() "
                    "or compounds with search_compounds()."
                )

            if not dd_modalities:
                response = (
                    f"No drug discovery modalities found. "
                    f"Available modalities ({len(modalities)}): "
                    f"{', '.join(modalities[:20])}\n"
                )
                if len(modalities) > 20:
                    response += f"... and {len(modalities) - 20} more.\n"
                response += (
                    "\nUse search_drug_targets() or search_compounds() to begin."
                )
                return response

            response = f"**Drug Discovery Modalities** ({len(dd_modalities)}):\n\n"
            for mod_name in dd_modalities:
                try:
                    adata = data_manager.get_modality(mod_name)
                    if adata is not None:
                        response += (
                            f"- **{mod_name}**: "
                            f"{adata.n_obs} obs x {adata.n_vars} vars\n"
                        )
                    else:
                        response += f"- **{mod_name}**: (metadata only)\n"
                except Exception:
                    response += f"- **{mod_name}**: (could not load)\n"

            other_count = len(modalities) - len(dd_modalities)
            if other_count > 0:
                response += f"\n({other_count} other non-drug-discovery modalities in workspace)\n"

            return response

        except Exception as e:
            logger.error(f"Error in check_drug_discovery_status: {e}")
            return f"Error checking drug discovery status: {str(e)}"

    # -------------------------------------------------------------------------
    # TOOL 10: list_available_databases
    # -------------------------------------------------------------------------
    @tool
    def list_available_databases() -> str:
        """
        Check which drug discovery APIs are reachable from this environment.

        Performs lightweight HEAD requests to ChEMBL, Open Targets, and PubChem
        base URLs and reports connectivity status for each.
        """
        databases = {
            "ChEMBL": f"{CHEMBL_API_BASE}/status",
            "Open Targets": OPENTARGETS_GRAPHQL,
            "PubChem": f"{PUBCHEM_API_BASE}/compound/name/aspirin/property/MolecularWeight/JSON",
        }

        results: Dict[str, str] = {}

        for name, url in databases.items():
            try:
                with httpx.Client(timeout=10.0) as client:
                    response = client.head(url)
                    if response.status_code < 500:
                        results[name] = f"Reachable (HTTP {response.status_code})"
                    else:
                        results[name] = f"Server error (HTTP {response.status_code})"
            except httpx.TimeoutException:
                results[name] = "Timeout (unreachable)"
            except httpx.RequestError as exc:
                results[name] = f"Unreachable ({type(exc).__name__})"

        response = "**Drug Discovery Database Connectivity**\n\n"
        all_ok = True
        for name, status in results.items():
            is_ok = "Reachable" in status
            icon = "OK" if is_ok else "FAIL"
            if not is_ok:
                all_ok = False
            response += f"- [{icon}] **{name}**: {status}\n"

        if all_ok:
            response += "\nAll databases are reachable. Ready for analysis."
        else:
            response += (
                "\nSome databases are unreachable. "
                "Tools depending on those APIs may return errors."
            )

        return response

    # =========================================================================
    # Return all tools
    # =========================================================================

    return [
        search_drug_targets,
        score_drug_target,
        rank_targets,
        search_compounds,
        get_compound_bioactivity,
        get_target_compounds,
        get_compound_properties,
        get_drug_indications,
        check_drug_discovery_status,
        list_available_databases,
    ]
