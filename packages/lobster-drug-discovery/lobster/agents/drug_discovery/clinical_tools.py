"""
Tool factory for the clinical development expert child agent.

8 tools for target-disease evidence retrieval, drug synergy scoring (Bliss,
Loewe, HSA), combination matrix analysis, safety profiling, tractability
assessment, clinical trial lookup, indication mapping, and candidate comparison.

All tools follow the Lobster AI tool pattern:
- Accept string parameters (LLM-friendly)
- Call stateless services returning 3-tuples
- Log with ir=ir for provenance
- Return human-readable summary strings
"""

import json
from typing import Callable, List

from langchain_core.tools import tool

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


def create_clinical_tools(
    data_manager: DataManagerV2,
    opentargets_service,
    synergy_service,
    chembl_service,
    target_scoring_service,
) -> List[Callable]:
    """
    Create clinical development tools for the clinical_dev_expert child agent.

    Args:
        data_manager: DataManagerV2 instance for modality management.
        opentargets_service: OpenTargetsService for evidence, safety, tractability.
        synergy_service: SynergyScoringService for Bliss/Loewe/HSA scoring.
        chembl_service: ChEMBLService for bioactivity and clinical data.
        target_scoring_service: TargetScoringService for composite scoring.

    Returns:
        List of 8 tool functions.
    """

    @tool
    def get_target_disease_evidence(
        ensembl_id: str, disease_id: str = "", limit: int = 25
    ) -> str:
        """Get target-disease association evidence from Open Targets with datatype breakdown. Args: ensembl_id - Ensembl gene ID (e.g. ENSG00000157764 for BRAF), disease_id - optional EFO disease ID to filter (e.g. EFO_0000311), limit - max associations (default 25)."""
        try:
            disease_filter = disease_id if disease_id else None
            _, stats, ir = opentargets_service.get_target_disease_evidence(
                ensembl_id, disease_id=disease_filter, limit=limit
            )

            if "error" in stats:
                return f"Error: {stats['error']}"

            data_manager.log_tool_usage(
                "get_target_disease_evidence",
                {
                    "ensembl_id": ensembl_id,
                    "disease_id": disease_id,
                    "limit": limit,
                },
                stats,
                ir=ir,
            )

            symbol = stats.get("approved_symbol", "")
            name = stats.get("approved_name", "")
            n_assoc = stats.get("n_associations", 0)
            total = stats.get("total_associated_diseases", 0)
            associations = stats.get("associations", [])

            lines = [
                f"Target-disease evidence for {symbol} ({name}, {ensembl_id}):",
                f"  Total associated diseases: {total}",
                f"  Returned: {n_assoc}",
            ]

            for assoc in associations[:10]:
                disease_name = assoc.get("disease_name", "Unknown")
                score = assoc.get("overall_score", 0)
                dt_scores = assoc.get("datatype_scores", {})
                top_dt = sorted(
                    dt_scores.items(), key=lambda x: x[1], reverse=True
                )[:3]
                dt_str = ", ".join(f"{k}={v:.2f}" for k, v in top_dt)
                lines.append(
                    f"  - {disease_name}: score={score:.3f} ({dt_str})"
                )

            if n_assoc > 10:
                lines.append(f"  ... and {n_assoc - 10} more associations")

            return "\n".join(lines)
        except Exception as e:
            return f"Error getting target-disease evidence: {e}"

    @tool
    def score_drug_synergy(
        effect_a: float,
        effect_b: float,
        effect_ab: float,
        model: str = "bliss",
    ) -> str:
        """Score drug combination synergy using Bliss Independence, Loewe Additivity, or HSA model. Effects are fractional inhibition values (0.0=no effect, 1.0=full effect). Args: effect_a - effect of drug A alone (0-1), effect_b - effect of drug B alone (0-1), effect_ab - observed combination effect (0-1), model - synergy model: 'bliss', 'loewe', or 'hsa' (default 'bliss')."""
        try:
            if model == "bliss":
                _, stats, ir = synergy_service.bliss_independence(
                    effect_a, effect_b, effect_ab
                )
            elif model == "hsa":
                _, stats, ir = synergy_service.hsa_model(
                    effect_a, effect_b, effect_ab
                )
            elif model == "loewe":
                # Loewe requires dose and IC50 data; approximate using effects
                # For single-point scoring, use Bliss as fallback with Loewe note
                _, stats, ir = synergy_service.bliss_independence(
                    effect_a, effect_b, effect_ab
                )
                stats["note"] = (
                    "Loewe additivity requires dose and IC50 values. "
                    "Used Bliss Independence as approximation. For full Loewe "
                    "analysis, use combination_matrix with dose-response data."
                )
            else:
                return (
                    f"Error: Unknown synergy model '{model}'. "
                    "Supported: 'bliss', 'loewe', 'hsa'."
                )

            data_manager.log_tool_usage(
                "score_drug_synergy",
                {
                    "effect_a": effect_a,
                    "effect_b": effect_b,
                    "effect_ab": effect_ab,
                    "model": model,
                },
                stats,
                ir=ir,
            )

            classification = stats.get("classification", "unknown")
            model_name = stats.get("model", model)

            lines = [f"Synergy scoring ({model_name}):"]

            if "excess" in stats:
                lines.append(
                    f"  Excess effect: {stats['excess']:.4f}"
                )
            if "combination_index" in stats:
                lines.append(
                    f"  Combination Index: {stats['combination_index']:.4f}"
                )
            if "effect_ab_expected" in stats:
                lines.append(
                    f"  Expected: {stats['effect_ab_expected']:.4f}, "
                    f"Observed: {stats['effect_ab_observed']:.4f}"
                )
            if "hsa_reference" in stats:
                lines.append(
                    f"  HSA reference: {stats['hsa_reference']:.4f}, "
                    f"Observed: {stats['effect_ab_observed']:.4f}"
                )

            lines.append(f"  Classification: {classification.upper()}")

            if "note" in stats:
                lines.append(f"  Note: {stats['note']}")

            return "\n".join(lines)
        except Exception as e:
            return f"Error scoring drug synergy: {e}"

    @tool
    def combination_matrix(
        modality_name: str,
        drug_a_col: str,
        drug_b_col: str,
        response_col: str,
        model: str = "bliss",
    ) -> str:
        """Score a full dose-response combination matrix from AnnData. Identifies monotherapy controls (dose=0) and computes synergy scores for all combination points. Args: modality_name - dataset containing combination screen data, drug_a_col - column for drug A dose in obs, drug_b_col - column for drug B dose in obs, response_col - column for response (fractional inhibition 0-1), model - synergy model 'bliss' or 'hsa' (default 'bliss')."""
        try:
            adata = data_manager.get_modality(modality_name)
            if adata is None:
                return (
                    f"Error: Modality '{modality_name}' not found. "
                    "Use list_modalities or check_drug_discovery_status to see available datasets."
                )

            result_adata, stats, ir = synergy_service.score_combination_matrix(
                adata,
                drug_a_col=drug_a_col,
                drug_b_col=drug_b_col,
                response_col=response_col,
                model=model,
            )

            # Store scored modality
            new_name = f"{modality_name}_synergy_scored"
            data_manager.store_modality(new_name, result_adata)

            data_manager.log_tool_usage(
                "combination_matrix",
                {
                    "modality_name": modality_name,
                    "drug_a_col": drug_a_col,
                    "drug_b_col": drug_b_col,
                    "response_col": response_col,
                    "model": model,
                },
                stats,
                ir=ir,
            )

            n_syn = stats.get("n_synergistic", 0)
            n_add = stats.get("n_additive", 0)
            n_ant = stats.get("n_antagonistic", 0)
            n_combos = stats.get("n_combination_points", 0)
            mean_score = stats.get("mean_synergy_score")

            lines = [
                f"Combination matrix scored ({model} model):",
                f"  Observations: {stats.get('n_observations', 0)}",
                f"  Combination points: {n_combos}",
                f"  Monotherapy A: {stats.get('n_monotherapy_a', 0)}",
                f"  Monotherapy B: {stats.get('n_monotherapy_b', 0)}",
                f"  Synergistic: {n_syn} ({n_syn / max(n_combos, 1) * 100:.1f}%)",
                f"  Additive: {n_add} ({n_add / max(n_combos, 1) * 100:.1f}%)",
                f"  Antagonistic: {n_ant} ({n_ant / max(n_combos, 1) * 100:.1f}%)",
            ]

            if mean_score is not None:
                lines.append(f"  Mean synergy score: {mean_score:.4f}")

            lines.append(f"  Stored as '{new_name}'")
            lines.append(
                f"  Columns added: {', '.join(stats.get('columns_added', []))}"
            )

            return "\n".join(lines)
        except Exception as e:
            return f"Error scoring combination matrix: {e}"

    @tool
    def get_drug_safety_profile(target_id: str) -> str:
        """Get known adverse events and safety liabilities for a drug target from Open Targets. Args: target_id - Ensembl gene ID (e.g. ENSG00000157764)."""
        try:
            _, stats, ir = opentargets_service.get_safety_profile(target_id)

            if "error" in stats:
                return f"Error: {stats['error']}"

            data_manager.log_tool_usage(
                "get_drug_safety_profile",
                {"target_id": target_id},
                stats,
                ir=ir,
            )

            symbol = stats.get("approved_symbol", "")
            n_events = stats.get("n_safety_events", 0)
            risk = stats.get("risk_level", "unknown")
            events = stats.get("events", [])
            categories = stats.get("event_categories", {})
            tissues = stats.get("affected_tissues", {})

            lines = [
                f"Safety profile for {symbol} ({target_id}):",
                f"  Risk level: {risk.upper()}",
                f"  Safety events: {n_events}",
            ]

            if categories:
                lines.append("  Event categories:")
                for cat, count in sorted(
                    categories.items(), key=lambda x: x[1], reverse=True
                ):
                    lines.append(f"    - {cat}: {count}")

            if tissues:
                top_tissues = sorted(
                    tissues.items(), key=lambda x: x[1], reverse=True
                )[:5]
                lines.append("  Affected tissues:")
                for tissue, count in top_tissues:
                    lines.append(f"    - {tissue}: {count}")

            return "\n".join(lines)
        except Exception as e:
            return f"Error getting safety profile: {e}"

    @tool
    def assess_clinical_tractability(target_id: str) -> str:
        """Assess small molecule, antibody, and PROTAC tractability for a drug target from Open Targets. Args: target_id - Ensembl gene ID (e.g. ENSG00000157764)."""
        try:
            _, stats, ir = opentargets_service.assess_tractability(target_id)

            if "error" in stats:
                return f"Error: {stats['error']}"

            data_manager.log_tool_usage(
                "assess_clinical_tractability",
                {"target_id": target_id},
                stats,
                ir=ir,
            )

            symbol = stats.get("approved_symbol", "")
            is_tractable = stats.get("is_tractable", False)
            tractable_mods = stats.get("tractable_modalities", [])
            verdicts = stats.get("modality_verdicts", {})

            lines = [
                f"Tractability for {symbol} ({target_id}):",
                f"  Overall: {'TRACTABLE' if is_tractable else 'NOT TRACTABLE'}",
            ]

            if tractable_mods:
                lines.append(
                    f"  Tractable modalities: {', '.join(tractable_mods)}"
                )

            for modality, verdict in verdicts.items():
                n_pos = verdict.get("n_positive", 0)
                n_tot = verdict.get("n_total", 0)
                status = "YES" if verdict.get("tractable") else "NO"
                lines.append(
                    f"  {modality}: {status} ({n_pos}/{n_tot} positive assessments)"
                )
                for assessment in verdict.get("assessments", []):
                    label = assessment.get("label", "")
                    value = assessment.get("value", False)
                    lines.append(
                        f"    - {label}: {'Yes' if value else 'No'}"
                    )

            return "\n".join(lines)
        except Exception as e:
            return f"Error assessing tractability: {e}"

    @tool
    def search_clinical_trials(chembl_id: str) -> str:
        """Search for clinical trial phase data for a compound from ChEMBL bioactivity records. Args: chembl_id - ChEMBL molecule ID (e.g. CHEMBL25)."""
        try:
            _, stats, ir = chembl_service.get_bioactivity(chembl_id)

            if "error" in stats:
                return f"Error: {stats['error']}"

            data_manager.log_tool_usage(
                "search_clinical_trials",
                {"chembl_id": chembl_id},
                stats,
                ir=ir,
            )

            n_activities = stats.get("n_activities", 0)
            activities = stats.get("activities", [])
            type_counts = stats.get("activity_type_counts", {})

            # Filter for clinically relevant activity types
            clinical_types = {"IC50", "Ki", "EC50", "Kd", "ED50", "MIC"}
            clinical_activities = [
                a for a in activities
                if a.get("standard_type", "") in clinical_types
            ]

            lines = [
                f"Clinical trial data for {chembl_id}:",
                f"  Total bioactivity records: {n_activities}",
                f"  Activity types: {', '.join(f'{k}({v})' for k, v in type_counts.items())}",
                f"  Clinical-grade records: {len(clinical_activities)}",
            ]

            # Group by target
            target_map = {}
            for act in clinical_activities[:50]:
                target = act.get("target_pref_name", "Unknown")
                if target not in target_map:
                    target_map[target] = []
                target_map[target].append(act)

            for target, acts in list(target_map.items())[:5]:
                lines.append(f"  Target: {target}")
                for act in acts[:3]:
                    std_type = act.get("standard_type", "")
                    std_value = act.get("standard_value", "N/A")
                    std_units = act.get("standard_units", "")
                    pchembl = act.get("pchembl_value", "N/A")
                    lines.append(
                        f"    - {std_type}: {std_value} {std_units} "
                        f"(pChEMBL={pchembl})"
                    )

            return "\n".join(lines)
        except Exception as e:
            return f"Error searching clinical trials: {e}"

    @tool
    def indication_mapping(chembl_id: str) -> str:
        """Map a compound to its known therapeutic indications and clinical trial phases from Open Targets. Args: chembl_id - ChEMBL drug ID (e.g. CHEMBL25 for aspirin)."""
        try:
            _, stats, ir = opentargets_service.get_drug_indications(chembl_id)

            if "error" in stats:
                return f"Error: {stats['error']}"

            data_manager.log_tool_usage(
                "indication_mapping",
                {"chembl_id": chembl_id},
                stats,
                ir=ir,
            )

            drug_name = stats.get("drug_name", "")
            drug_type = stats.get("drug_type", "")
            max_phase = stats.get("max_clinical_trial_phase", 0)
            n_indications = stats.get("n_indications", 0)
            indications = stats.get("indications", [])
            phase_dist = stats.get("phase_distribution", {})
            mechanisms = stats.get("mechanisms_of_action", [])

            lines = [
                f"Indication mapping for {drug_name} ({chembl_id}):",
                f"  Drug type: {drug_type}",
                f"  Max clinical trial phase: {max_phase}",
                f"  Total indications: {n_indications}",
            ]

            if phase_dist:
                lines.append(
                    f"  Phase distribution: "
                    + ", ".join(
                        f"Phase {p}={c}" for p, c in sorted(phase_dist.items())
                    )
                )

            if mechanisms:
                lines.append("  Mechanisms of action:")
                for moa in mechanisms[:5]:
                    targets = [
                        t.get("symbol", t.get("id", ""))
                        for t in moa.get("targets", [])
                    ]
                    lines.append(
                        f"    - {moa.get('mechanism', 'Unknown')}"
                        + (f" (targets: {', '.join(targets)})" if targets else "")
                    )

            if indications:
                lines.append("  Indications:")
                for ind in indications[:10]:
                    lines.append(
                        f"    - {ind.get('disease_name', 'Unknown')} "
                        f"(phase {ind.get('max_phase', 0)})"
                    )
                if n_indications > 10:
                    lines.append(
                        f"    ... and {n_indications - 10} more indications"
                    )

            return "\n".join(lines)
        except Exception as e:
            return f"Error mapping indications: {e}"

    @tool
    def compare_drug_candidates(candidates: str) -> str:
        """Compare multiple drug candidates side-by-side using ChEMBL bioactivity data. Args: candidates - comma-separated ChEMBL molecule IDs (e.g. 'CHEMBL25, CHEMBL1201585')."""
        try:
            ids = [c.strip() for c in candidates.split(",") if c.strip()]
            if len(ids) < 2:
                return "Error: Provide at least 2 comma-separated ChEMBL IDs."

            comparison = []
            all_stats = []

            for chembl_id in ids:
                _, stats, ir = chembl_service.get_bioactivity(chembl_id)
                all_stats.append(stats)

                data_manager.log_tool_usage(
                    "compare_drug_candidates",
                    {"chembl_id": chembl_id},
                    stats,
                    ir=ir,
                )

                if "error" in stats:
                    comparison.append({
                        "chembl_id": chembl_id,
                        "error": stats["error"],
                    })
                    continue

                activities = stats.get("activities", [])
                type_counts = stats.get("activity_type_counts", {})

                # Find best IC50 and pChEMBL
                best_ic50 = None
                best_pchembl = None
                n_targets = set()
                for act in activities:
                    n_targets.add(act.get("target_chembl_id", ""))
                    pval = act.get("pchembl_value")
                    if pval is not None:
                        try:
                            pval_f = float(pval)
                            if best_pchembl is None or pval_f > best_pchembl:
                                best_pchembl = pval_f
                        except (ValueError, TypeError):
                            pass
                    if act.get("standard_type") == "IC50":
                        sval = act.get("standard_value")
                        if sval is not None:
                            try:
                                sval_f = float(sval)
                                if best_ic50 is None or sval_f < best_ic50:
                                    best_ic50 = sval_f
                            except (ValueError, TypeError):
                                pass

                comparison.append({
                    "chembl_id": chembl_id,
                    "n_activities": len(activities),
                    "n_targets": len(n_targets),
                    "best_ic50_nM": best_ic50,
                    "best_pchembl": best_pchembl,
                    "activity_types": type_counts,
                })

            lines = [
                f"Drug candidate comparison ({len(ids)} compounds):",
                f"  {'ChEMBL ID':<20} {'Activities':>10} {'Targets':>8} "
                f"{'Best IC50 (nM)':>15} {'Best pChEMBL':>13}",
                f"  {'-' * 66}",
            ]

            for comp in comparison:
                if "error" in comp:
                    lines.append(
                        f"  {comp['chembl_id']:<20} ERROR: {comp['error']}"
                    )
                    continue

                ic50_str = (
                    f"{comp['best_ic50_nM']:.1f}"
                    if comp["best_ic50_nM"] is not None
                    else "N/A"
                )
                pchembl_str = (
                    f"{comp['best_pchembl']:.2f}"
                    if comp["best_pchembl"] is not None
                    else "N/A"
                )

                lines.append(
                    f"  {comp['chembl_id']:<20} {comp['n_activities']:>10} "
                    f"{comp['n_targets']:>8} {ic50_str:>15} {pchembl_str:>13}"
                )

            return "\n".join(lines)
        except Exception as e:
            return f"Error comparing drug candidates: {e}"

    return [
        get_target_disease_evidence,
        score_drug_synergy,
        combination_matrix,
        get_drug_safety_profile,
        assess_clinical_tractability,
        search_clinical_trials,
        indication_mapping,
        compare_drug_candidates,
    ]
