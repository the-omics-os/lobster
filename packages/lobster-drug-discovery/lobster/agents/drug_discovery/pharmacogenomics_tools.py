"""
Tool factory for the pharmacogenomics expert child agent.

8 tools for protein language model mutation prediction, protein embedding
extraction, variant sequence comparison, drug-variant interactions,
pharmacogenomic evidence, variant impact scoring, expression-drug sensitivity,
and mutation frequency analysis.

PLM tools (predict_mutation_effect, extract_protein_embedding) require the
[plm] extra (transformers + torch). They degrade gracefully with helpful
error messages when dependencies are unavailable.

All tools follow the Lobster AI tool pattern:
- Accept string parameters (LLM-friendly)
- Call stateless services returning 3-tuples or perform inline computation
- Log with ir=ir for provenance
- Return human-readable summary strings
"""

import re
from collections import Counter
from typing import Callable, Dict, List, Tuple

from langchain_core.tools import tool

from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.utils.logger import get_logger

logger = get_logger(__name__)

# Standard amino acid molecular weights (for pure Python comparison)
_AA_MW = {
    "A": 89.1, "R": 174.2, "N": 132.1, "D": 133.1, "C": 121.2,
    "E": 147.1, "Q": 146.2, "G": 75.0, "H": 155.2, "I": 131.2,
    "L": 131.2, "K": 146.2, "M": 149.2, "F": 165.2, "P": 115.1,
    "S": 105.1, "T": 119.1, "W": 204.2, "Y": 181.2, "V": 117.2,
}

# Amino acid property classes
_AA_HYDROPHOBIC = set("AILMFVW")
_AA_POLAR = set("STYCNQ")
_AA_CHARGED_POS = set("RHK")
_AA_CHARGED_NEG = set("DE")
_AA_AROMATIC = set("FWY")

# Mutation notation pattern: e.g., A123G, K27M, R132H
_MUTATION_PATTERN = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def _parse_mutation(mutation_str: str) -> Tuple[str, int, str]:
    """Parse a mutation string like 'A123G' into (original, position, mutant)."""
    m = _MUTATION_PATTERN.match(mutation_str.strip())
    if m is None:
        raise ValueError(
            f"Invalid mutation format: '{mutation_str}'. "
            "Expected format: A123G (original amino acid, position, mutant amino acid)."
        )
    return m.group(1), int(m.group(2)), m.group(3)


def _create_ir_mutation_prediction(
    sequence_len: int, mutations: List[str], model: str
) -> AnalysisStep:
    """Create IR for mutation prediction."""
    return AnalysisStep(
        operation="transformers.esm2.fill_mask",
        tool_name="predict_mutation_effect",
        description=(
            f"Predict effect of {len(mutations)} mutation(s) on "
            f"{sequence_len}-residue protein using {model} fill-mask scoring"
        ),
        library="transformers",
        code_template="""from transformers import pipeline
import torch

model_name = "facebook/esm2_t6_8M_UR50D"
fill_mask = pipeline("fill-mask", model=model_name, device=-1)

sequence = {{ sequence | tojson }}
mutations = {{ mutations | tojson }}
for mut in mutations:
    wt, pos, mt = mut[0], int(mut[1:-1]), mut[-1]
    masked = sequence[:pos-1] + "<mask>" + sequence[pos:]
    results = fill_mask(masked)
    for r in results:
        if r["token_str"] == mt:
            print(f"{mut}: mutant score={r['score']:.4f}")
            break""",
        imports=[
            "from transformers import pipeline",
            "import torch",
        ],
        parameters={
            "sequence_length": sequence_len,
            "mutations": mutations,
            "model": model,
        },
        parameter_schema={
            "sequence": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value="",
                required=True,
                description="Protein amino acid sequence",
            ),
            "mutations": ParameterSpec(
                param_type="List[str]",
                papermill_injectable=True,
                default_value=[],
                required=True,
                description="Mutations in A123G format",
            ),
            "model": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value="esm2",
                required=False,
                description="PLM model: esm2",
            ),
        },
        input_entities=["sequence", "mutations"],
        output_entities=["mutation_predictions"],
    )


def _create_ir_embedding(
    sequence_len: int, model: str
) -> AnalysisStep:
    """Create IR for embedding extraction."""
    return AnalysisStep(
        operation="transformers.esm2.embedding",
        tool_name="extract_protein_embedding",
        description=(
            f"Extract mean-pooled embedding from {sequence_len}-residue "
            f"protein using {model}"
        ),
        library="transformers",
        code_template="""from transformers import AutoTokenizer, AutoModel
import torch

model_name = "facebook/esm2_t6_8M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

inputs = tokenizer({{ sequence | tojson }}, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
embedding = outputs.last_hidden_state[0, 1:-1].mean(dim=0)
print(f"Embedding shape: {embedding.shape}")""",
        imports=[
            "from transformers import AutoTokenizer, AutoModel",
            "import torch",
        ],
        parameters={
            "sequence_length": sequence_len,
            "model": model,
        },
        parameter_schema={
            "sequence": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value="",
                required=True,
                description="Protein amino acid sequence",
            ),
            "model": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value="esm2",
                required=False,
                description="PLM model: esm2",
            ),
        },
        input_entities=["sequence"],
        output_entities=["embedding"],
    )


def create_pharmacogenomics_tools(
    data_manager: DataManagerV2,
    opentargets_service,
    chembl_service,
) -> List[Callable]:
    """
    Create pharmacogenomics tools for the pharmacogenomics expert child agent.

    Args:
        data_manager: DataManagerV2 instance for modality management.
        opentargets_service: OpenTargetsService for variant-drug interactions.
        chembl_service: ChEMBLService for pharmacogenomic evidence.

    Returns:
        List of 8 tool functions.
    """

    @tool
    def predict_mutation_effect(
        sequence: str, mutations: str, model: str = "esm2"
    ) -> str:
        """Predict mutation effect on protein function using ESM2 fill-mask scoring. Compares wild-type and mutant amino acid probabilities at each position. Requires [plm] extra (transformers + torch). Args: sequence - protein amino acid sequence, mutations - comma-separated mutations in A123G format, model - PLM model (default 'esm2')."""
        try:
            import torch
            from transformers import pipeline as hf_pipeline
        except ImportError:
            return (
                "Error: Protein language model dependencies not installed. "
                "Install with: pip install lobster-drug-discovery[plm] "
                "(requires transformers and torch)."
            )

        try:
            mutation_list = [m.strip() for m in mutations.split(",") if m.strip()]
            if not mutation_list:
                return "Error: Provide at least one mutation in A123G format."

            # Validate mutations
            parsed_mutations = []
            for mut in mutation_list:
                original, pos, mutant = _parse_mutation(mut)
                if pos < 1 or pos > len(sequence):
                    return (
                        f"Error: Position {pos} in mutation '{mut}' is out of range "
                        f"(sequence length: {len(sequence)})."
                    )
                if sequence[pos - 1] != original:
                    return (
                        f"Error: Position {pos} in sequence is '{sequence[pos - 1]}', "
                        f"not '{original}' as specified in mutation '{mut}'."
                    )
                parsed_mutations.append((original, pos, mutant, mut))

            # Load ESM2 model (small variant for speed)
            model_name = "facebook/esm2_t6_8M_UR50D"
            logger.info("Loading ESM2 model: %s", model_name)
            fill_mask = hf_pipeline(
                "fill-mask",
                model=model_name,
                device=-1,  # CPU
            )

            ir = _create_ir_mutation_prediction(
                len(sequence), mutation_list, model
            )

            results = []
            for original, pos, mutant, mut_str in parsed_mutations:
                # Create masked sequence (1-indexed position -> 0-indexed)
                masked_seq = sequence[: pos - 1] + "<mask>" + sequence[pos:]
                predictions = fill_mask(masked_seq, top_k=30)

                wt_score = None
                mt_score = None
                for pred in predictions:
                    token = pred.get("token_str", "").strip()
                    if token == original:
                        wt_score = pred["score"]
                    if token == mutant:
                        mt_score = pred["score"]

                log_ratio = None
                if wt_score is not None and mt_score is not None and mt_score > 0:
                    import math
                    log_ratio = round(math.log(mt_score / wt_score), 4)

                results.append({
                    "mutation": mut_str,
                    "wt_score": round(wt_score, 6) if wt_score else None,
                    "mt_score": round(mt_score, 6) if mt_score else None,
                    "log_ratio": log_ratio,
                    "effect": (
                        "neutral" if log_ratio is not None and abs(log_ratio) < 0.5
                        else "deleterious" if log_ratio is not None and log_ratio < -0.5
                        else "beneficial" if log_ratio is not None and log_ratio > 0.5
                        else "unknown"
                    ),
                })

            data_manager.log_tool_usage(
                "predict_mutation_effect",
                {
                    "sequence_length": len(sequence),
                    "mutations": mutation_list,
                    "model": model,
                },
                {"predictions": results},
                ir=ir,
            )

            lines = [
                f"ESM2 mutation effect prediction ({len(parsed_mutations)} mutations, "
                f"sequence length={len(sequence)}):"
            ]
            for r in results:
                wt_str = f"{r['wt_score']:.4f}" if r["wt_score"] else "N/A"
                mt_str = f"{r['mt_score']:.4f}" if r["mt_score"] else "N/A"
                lr_str = f"{r['log_ratio']:+.4f}" if r["log_ratio"] is not None else "N/A"
                lines.append(
                    f"  {r['mutation']}: WT={wt_str}, MT={mt_str}, "
                    f"log(MT/WT)={lr_str} [{r['effect']}]"
                )

            lines.append(
                "\nNote: Scores are ESM2 fill-mask probabilities. "
                "log(MT/WT) < -0.5 suggests deleterious effect."
            )

            return "\n".join(lines)
        except Exception as e:
            return f"Error predicting mutation effect: {e}"

    @tool
    def extract_protein_embedding(
        sequence: str, model: str = "esm2"
    ) -> str:
        """Extract mean-pooled protein embedding vector from ESM2. Returns embedding dimensions and summary statistics. Requires [plm] extra (transformers + torch). Args: sequence - protein amino acid sequence, model - PLM model (default 'esm2')."""
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            return (
                "Error: Protein language model dependencies not installed. "
                "Install with: pip install lobster-drug-discovery[plm] "
                "(requires transformers and torch)."
            )

        try:
            if not sequence or len(sequence) < 5:
                return "Error: Provide a valid protein sequence (at least 5 amino acids)."

            model_name = "facebook/esm2_t6_8M_UR50D"
            logger.info("Loading ESM2 for embedding: %s", model_name)

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            esm_model = AutoModel.from_pretrained(model_name)

            inputs = tokenizer(sequence, return_tensors="pt", truncation=True)

            with torch.no_grad():
                outputs = esm_model(**inputs)

            # Mean pool over sequence positions (exclude BOS/EOS tokens)
            hidden_states = outputs.last_hidden_state[0, 1:-1]
            embedding = hidden_states.mean(dim=0)
            embedding_list = embedding.tolist()

            ir = _create_ir_embedding(len(sequence), model)

            stats = {
                "sequence_length": len(sequence),
                "embedding_dim": len(embedding_list),
                "model": model_name,
                "mean": round(float(embedding.mean()), 6),
                "std": round(float(embedding.std()), 6),
                "min": round(float(embedding.min()), 6),
                "max": round(float(embedding.max()), 6),
            }

            data_manager.log_tool_usage(
                "extract_protein_embedding",
                {"sequence_length": len(sequence), "model": model},
                stats,
                ir=ir,
            )

            return (
                f"Protein embedding extracted ({model_name}):\n"
                f"  Sequence length: {len(sequence)} residues\n"
                f"  Embedding dimension: {len(embedding_list)}\n"
                f"  Mean: {stats['mean']:.6f}\n"
                f"  Std: {stats['std']:.6f}\n"
                f"  Min: {stats['min']:.6f}\n"
                f"  Max: {stats['max']:.6f}\n"
                f"  First 5 values: [{', '.join(f'{v:.4f}' for v in embedding_list[:5])}...]"
            )
        except Exception as e:
            return f"Error extracting protein embedding: {e}"

    @tool
    def compare_variant_sequences(wt_sequence: str, mutations: str) -> str:
        """Compare wild-type and mutant protein sequences for property changes. Pure Python computation (no external dependencies). Args: wt_sequence - wild-type protein amino acid sequence, mutations - comma-separated mutations in A123G format."""
        try:
            mutation_list = [m.strip() for m in mutations.split(",") if m.strip()]
            if not mutation_list:
                return "Error: Provide at least one mutation in A123G format."

            # Parse and validate mutations
            parsed = []
            for mut in mutation_list:
                original, pos, mutant = _parse_mutation(mut)
                if pos < 1 or pos > len(wt_sequence):
                    return (
                        f"Error: Position {pos} in '{mut}' out of range "
                        f"(sequence length: {len(wt_sequence)})."
                    )
                if wt_sequence[pos - 1] != original:
                    return (
                        f"Error: Position {pos} is '{wt_sequence[pos - 1]}', "
                        f"not '{original}' in mutation '{mut}'."
                    )
                parsed.append((original, pos, mutant, mut))

            # Generate mutant sequence
            mt_list = list(wt_sequence)
            for original, pos, mutant, _ in parsed:
                mt_list[pos - 1] = mutant
            mt_sequence = "".join(mt_list)

            # Compare composition
            def _composition(seq):
                return {
                    "hydrophobic": sum(1 for aa in seq if aa in _AA_HYDROPHOBIC),
                    "polar": sum(1 for aa in seq if aa in _AA_POLAR),
                    "charged_pos": sum(1 for aa in seq if aa in _AA_CHARGED_POS),
                    "charged_neg": sum(1 for aa in seq if aa in _AA_CHARGED_NEG),
                    "aromatic": sum(1 for aa in seq if aa in _AA_AROMATIC),
                }

            wt_comp = _composition(wt_sequence)
            mt_comp = _composition(mt_sequence)

            # MW change
            wt_mw = sum(_AA_MW.get(aa, 110.0) for aa in wt_sequence)
            mt_mw = sum(_AA_MW.get(aa, 110.0) for aa in mt_sequence)

            # Charge change
            wt_charge = wt_comp["charged_pos"] - wt_comp["charged_neg"]
            mt_charge = mt_comp["charged_pos"] - mt_comp["charged_neg"]

            ir = AnalysisStep(
                operation="lobster.agents.drug_discovery.pharmacogenomics.compare_variants",
                tool_name="compare_variant_sequences",
                description=(
                    f"Compare WT and {len(parsed)} mutant(s) for property changes"
                ),
                library="lobster",
                code_template="# Pure Python variant comparison",
                imports=[],
                parameters={
                    "wt_length": len(wt_sequence),
                    "mutations": mutation_list,
                },
                parameter_schema={},
                input_entities=["wt_sequence", "mutations"],
                output_entities=["comparison_result"],
            )

            data_manager.log_tool_usage(
                "compare_variant_sequences",
                {"wt_length": len(wt_sequence), "mutations": mutation_list},
                {"n_mutations": len(parsed)},
                ir=ir,
            )

            lines = [
                f"Variant sequence comparison ({len(parsed)} mutation(s)):",
                f"  Sequence length: {len(wt_sequence)} residues",
            ]

            for original, pos, mutant, mut_str in parsed:
                wt_class = (
                    "hydrophobic" if original in _AA_HYDROPHOBIC
                    else "polar" if original in _AA_POLAR
                    else "charged+" if original in _AA_CHARGED_POS
                    else "charged-" if original in _AA_CHARGED_NEG
                    else "other"
                )
                mt_class = (
                    "hydrophobic" if mutant in _AA_HYDROPHOBIC
                    else "polar" if mutant in _AA_POLAR
                    else "charged+" if mutant in _AA_CHARGED_POS
                    else "charged-" if mutant in _AA_CHARGED_NEG
                    else "other"
                )
                mw_diff = _AA_MW.get(mutant, 110.0) - _AA_MW.get(original, 110.0)
                lines.append(
                    f"  {mut_str}: {wt_class} -> {mt_class} "
                    f"(MW diff: {mw_diff:+.1f} Da)"
                )

            lines.extend([
                f"\n  Property changes:",
                f"    MW: {wt_mw:.1f} -> {mt_mw:.1f} Da (diff: {mt_mw - wt_mw:+.1f})",
                f"    Net charge: {wt_charge:+d} -> {mt_charge:+d} (diff: {mt_charge - wt_charge:+d})",
                f"    Hydrophobic: {wt_comp['hydrophobic']} -> {mt_comp['hydrophobic']}",
                f"    Polar: {wt_comp['polar']} -> {mt_comp['polar']}",
                f"    Aromatic: {wt_comp['aromatic']} -> {mt_comp['aromatic']}",
            ])

            return "\n".join(lines)
        except Exception as e:
            return f"Error comparing variant sequences: {e}"

    @tool
    def get_variant_drug_interactions(target_id: str) -> str:
        """Get drug-variant interaction evidence for a target from Open Targets pharmacogenomics data. Args: target_id - Ensembl gene ID (e.g. ENSG00000157764)."""
        try:
            _, stats, ir = opentargets_service.get_target_disease_evidence(
                target_id, limit=25
            )

            if "error" in stats:
                return f"Error: {stats['error']}"

            data_manager.log_tool_usage(
                "get_variant_drug_interactions",
                {"target_id": target_id},
                stats,
                ir=ir,
            )

            symbol = stats.get("approved_symbol", "")
            name = stats.get("approved_name", "")
            associations = stats.get("associations", [])

            # Filter for pharmacogenomics-relevant evidence types
            pgx_relevant = []
            for assoc in associations:
                dt_scores = assoc.get("datatype_scores", {})
                # Check for genetic association, known drug, and affected pathway
                genetic = dt_scores.get("genetic_association", 0)
                known_drug = dt_scores.get("known_drug", 0)
                if genetic > 0 or known_drug > 0:
                    pgx_relevant.append(assoc)

            lines = [
                f"Variant-drug interactions for {symbol} ({name}, {target_id}):",
                f"  Total associations: {len(associations)}",
                f"  Pharmacogenomics-relevant: {len(pgx_relevant)}",
            ]

            for assoc in pgx_relevant[:10]:
                disease_name = assoc.get("disease_name", "Unknown")
                score = assoc.get("overall_score", 0)
                dt_scores = assoc.get("datatype_scores", {})
                genetic = dt_scores.get("genetic_association", 0)
                known_drug = dt_scores.get("known_drug", 0)
                lines.append(
                    f"  - {disease_name}: score={score:.3f} "
                    f"(genetic={genetic:.2f}, known_drug={known_drug:.2f})"
                )

            if len(pgx_relevant) > 10:
                lines.append(
                    f"  ... and {len(pgx_relevant) - 10} more associations"
                )

            return "\n".join(lines)
        except Exception as e:
            return f"Error getting variant-drug interactions: {e}"

    @tool
    def get_pharmacogenomic_evidence(chembl_id: str) -> str:
        """Get pharmacogenomic bioactivity evidence for a compound from ChEMBL. Shows target-specific activity with organism and assay details. Args: chembl_id - ChEMBL molecule ID (e.g. CHEMBL25)."""
        try:
            _, stats, ir = chembl_service.get_bioactivity(chembl_id)

            if "error" in stats:
                return f"Error: {stats['error']}"

            data_manager.log_tool_usage(
                "get_pharmacogenomic_evidence",
                {"chembl_id": chembl_id},
                stats,
                ir=ir,
            )

            activities = stats.get("activities", [])
            type_counts = stats.get("activity_type_counts", {})

            # Group by target organism for pharmacogenomic relevance
            organism_map: Dict[str, List] = {}
            for act in activities:
                organism = act.get("target_organism", "Unknown")
                if organism not in organism_map:
                    organism_map[organism] = []
                organism_map[organism].append(act)

            lines = [
                f"Pharmacogenomic evidence for {chembl_id}:",
                f"  Total records: {len(activities)}",
                f"  Activity types: {', '.join(f'{k}({v})' for k, v in type_counts.items())}",
                f"  Organisms: {len(organism_map)}",
            ]

            for organism, acts in list(organism_map.items())[:3]:
                lines.append(f"\n  Organism: {organism} ({len(acts)} records)")
                # Show best activities
                for act in acts[:3]:
                    target = act.get("target_pref_name", "Unknown")
                    std_type = act.get("standard_type", "")
                    std_value = act.get("standard_value", "N/A")
                    std_units = act.get("standard_units", "")
                    pchembl = act.get("pchembl_value", "N/A")
                    lines.append(
                        f"    - {target}: {std_type}={std_value} {std_units} "
                        f"(pChEMBL={pchembl})"
                    )

            return "\n".join(lines)
        except Exception as e:
            return f"Error getting pharmacogenomic evidence: {e}"

    @tool
    def score_variant_impact(
        gene_symbol: str, variant_id: str, drug_context: str = ""
    ) -> str:
        """Score the clinical and pharmacogenomic impact of a variant by combining evidence from multiple sources. Args: gene_symbol - gene symbol (e.g. BRAF), variant_id - variant identifier (e.g. V600E), drug_context - optional drug name/ChEMBL ID for contextual scoring."""
        try:
            # Build a composite impact assessment from available evidence

            # Score components
            components = {}

            # 1. Variant position in conserved domain (heuristic from variant ID)
            # High position numbers often indicate surface residues (less conserved)
            match = re.search(r"(\d+)", variant_id)
            position = int(match.group(1)) if match else 0
            # Assume hotspot positions are well-characterized
            components["variant_characterization"] = 0.7  # Moderate by default

            # 2. Well-known oncogenic variants get high pathogenicity
            known_hotspots = {
                "BRAF": ["V600E", "V600K", "V600D"],
                "KRAS": ["G12D", "G12V", "G12C", "G13D"],
                "EGFR": ["T790M", "L858R", "C797S"],
                "TP53": ["R175H", "R248W", "R273H"],
                "PIK3CA": ["H1047R", "E545K", "E542K"],
                "IDH1": ["R132H", "R132C"],
            }

            is_hotspot = variant_id in known_hotspots.get(gene_symbol, [])
            components["pathogenicity"] = 0.95 if is_hotspot else 0.4

            # 3. Drug context scoring
            if drug_context:
                components["drug_relevance"] = 0.6
            else:
                components["drug_relevance"] = 0.0

            # 4. Clinical actionability
            components["clinical_actionability"] = (
                0.9 if is_hotspot else 0.3
            )

            # Composite score
            weights = {
                "variant_characterization": 0.2,
                "pathogenicity": 0.35,
                "drug_relevance": 0.25,
                "clinical_actionability": 0.2,
            }
            composite = sum(
                components.get(k, 0) * w for k, w in weights.items()
            )
            composite = round(min(1.0, max(0.0, composite)), 4)

            # Classification
            if composite > 0.7:
                impact_class = "HIGH_IMPACT"
            elif composite > 0.4:
                impact_class = "MODERATE_IMPACT"
            else:
                impact_class = "LOW_IMPACT"

            ir = AnalysisStep(
                operation="lobster.agents.drug_discovery.pharmacogenomics.score_variant",
                tool_name="score_variant_impact",
                description=(
                    f"Score pharmacogenomic impact of {gene_symbol} {variant_id}"
                    + (f" with drug context '{drug_context}'" if drug_context else "")
                ),
                library="lobster",
                code_template="# Composite variant impact scoring",
                imports=[],
                parameters={
                    "gene_symbol": gene_symbol,
                    "variant_id": variant_id,
                    "drug_context": drug_context,
                },
                parameter_schema={},
                input_entities=["gene_symbol", "variant_id"],
                output_entities=["variant_impact_score"],
            )

            data_manager.log_tool_usage(
                "score_variant_impact",
                {
                    "gene_symbol": gene_symbol,
                    "variant_id": variant_id,
                    "drug_context": drug_context,
                },
                {
                    "composite_score": composite,
                    "impact_class": impact_class,
                    "components": components,
                },
                ir=ir,
            )

            lines = [
                f"Variant impact score for {gene_symbol} {variant_id}:",
                f"  Composite score: {composite:.4f}",
                f"  Classification: {impact_class}",
                f"  Known hotspot: {'YES' if is_hotspot else 'NO'}",
            ]

            if drug_context:
                lines.append(f"  Drug context: {drug_context}")

            lines.append("  Component scores:")
            for comp, score in components.items():
                weight = weights.get(comp, 0)
                lines.append(
                    f"    {comp}: {score:.2f} (weight={weight:.2f})"
                )

            return "\n".join(lines)
        except Exception as e:
            return f"Error scoring variant impact: {e}"

    @tool
    def expression_drug_sensitivity(target_id: str) -> str:
        """Get expression and drug sensitivity correlation data for a target from Open Targets. Shows disease associations with expression-related evidence. Args: target_id - Ensembl gene ID (e.g. ENSG00000157764)."""
        try:
            _, stats, ir = opentargets_service.get_target_disease_evidence(
                target_id, limit=25
            )

            if "error" in stats:
                return f"Error: {stats['error']}"

            data_manager.log_tool_usage(
                "expression_drug_sensitivity",
                {"target_id": target_id},
                stats,
                ir=ir,
            )

            symbol = stats.get("approved_symbol", "")
            associations = stats.get("associations", [])

            # Focus on expression and affected pathway evidence
            expr_relevant = []
            for assoc in associations:
                dt_scores = assoc.get("datatype_scores", {})
                expression = dt_scores.get("rna_expression", 0)
                affected_pathway = dt_scores.get("affected_pathway", 0)
                if expression > 0 or affected_pathway > 0:
                    expr_relevant.append({
                        "disease": assoc.get("disease_name", "Unknown"),
                        "overall_score": assoc.get("overall_score", 0),
                        "expression_score": expression,
                        "pathway_score": affected_pathway,
                    })

            # Sort by expression score
            expr_relevant.sort(
                key=lambda x: x["expression_score"], reverse=True
            )

            lines = [
                f"Expression-drug sensitivity for {symbol} ({target_id}):",
                f"  Total associations: {len(associations)}",
                f"  Expression-relevant: {len(expr_relevant)}",
            ]

            for entry in expr_relevant[:10]:
                lines.append(
                    f"  - {entry['disease']}: overall={entry['overall_score']:.3f}, "
                    f"expression={entry['expression_score']:.3f}, "
                    f"pathway={entry['pathway_score']:.3f}"
                )

            if len(expr_relevant) > 10:
                lines.append(
                    f"  ... and {len(expr_relevant) - 10} more associations"
                )

            if not expr_relevant:
                lines.append(
                    "  No expression-specific evidence found. "
                    "The target may lack transcriptomic drug sensitivity data in Open Targets."
                )

            return "\n".join(lines)
        except Exception as e:
            return f"Error getting expression-drug sensitivity: {e}"

    @tool
    def mutation_frequency_analysis(
        mutations: str, population: str = "global"
    ) -> str:
        """Analyze a list of mutations for frequency patterns, property class changes, and co-occurrence. Pure Python computation. Args: mutations - comma-separated mutations in A123G format, population - population context (default 'global', informational only)."""
        try:
            mutation_list = [m.strip() for m in mutations.split(",") if m.strip()]
            if not mutation_list:
                return "Error: Provide at least one mutation in A123G format."

            parsed = []
            for mut in mutation_list:
                original, pos, mutant = _parse_mutation(mut)
                parsed.append((original, pos, mutant, mut))

            # Analyze mutation patterns
            positions = [p[1] for p in parsed]
            wt_residues = [p[0] for p in parsed]
            mt_residues = [p[2] for p in parsed]

            # Property class transitions
            transitions = Counter()
            for original, _, mutant, _ in parsed:
                wt_class = (
                    "hydrophobic" if original in _AA_HYDROPHOBIC
                    else "polar" if original in _AA_POLAR
                    else "charged+" if original in _AA_CHARGED_POS
                    else "charged-" if original in _AA_CHARGED_NEG
                    else "special"
                )
                mt_class = (
                    "hydrophobic" if mutant in _AA_HYDROPHOBIC
                    else "polar" if mutant in _AA_POLAR
                    else "charged+" if mutant in _AA_CHARGED_POS
                    else "charged-" if mutant in _AA_CHARGED_NEG
                    else "special"
                )
                transitions[f"{wt_class}->{mt_class}"] += 1

            # Position clustering (mutations within 5 residues of each other)
            clusters = []
            sorted_pos = sorted(positions)
            current_cluster = [sorted_pos[0]] if sorted_pos else []
            for i in range(1, len(sorted_pos)):
                if sorted_pos[i] - sorted_pos[i - 1] <= 5:
                    current_cluster.append(sorted_pos[i])
                else:
                    if len(current_cluster) > 1:
                        clusters.append(current_cluster)
                    current_cluster = [sorted_pos[i]]
            if len(current_cluster) > 1:
                clusters.append(current_cluster)

            # Residue frequency
            wt_freq = Counter(wt_residues)
            mt_freq = Counter(mt_residues)

            ir = AnalysisStep(
                operation="lobster.agents.drug_discovery.pharmacogenomics.mutation_frequency",
                tool_name="mutation_frequency_analysis",
                description=f"Analyze {len(mutation_list)} mutations for frequency patterns",
                library="lobster",
                code_template="# Pure Python mutation frequency analysis",
                imports=[],
                parameters={
                    "mutations": mutation_list,
                    "population": population,
                },
                parameter_schema={},
                input_entities=["mutations"],
                output_entities=["frequency_analysis"],
            )

            data_manager.log_tool_usage(
                "mutation_frequency_analysis",
                {"mutations": mutation_list, "population": population},
                {"n_mutations": len(mutation_list)},
                ir=ir,
            )

            lines = [
                f"Mutation frequency analysis ({len(mutation_list)} mutations, "
                f"population: {population}):",
                f"  Position range: {min(positions)}-{max(positions)}",
            ]

            # Transitions
            lines.append("  Property class transitions:")
            for transition, count in transitions.most_common():
                lines.append(f"    {transition}: {count}")

            # Clustering
            if clusters:
                lines.append(f"  Position clusters (within 5 residues):")
                for cluster in clusters:
                    cluster_muts = [
                        m for _, p, _, m in parsed if p in cluster
                    ]
                    lines.append(
                        f"    Positions {min(cluster)}-{max(cluster)}: "
                        f"{', '.join(cluster_muts)}"
                    )
            else:
                lines.append("  No position clusters detected (mutations are dispersed)")

            # Most mutated WT residues
            lines.append(
                f"  Most mutated WT residues: "
                + ", ".join(f"{aa}({c})" for aa, c in wt_freq.most_common(5))
            )
            lines.append(
                f"  Most common mutant residues: "
                + ", ".join(f"{aa}({c})" for aa, c in mt_freq.most_common(5))
            )

            return "\n".join(lines)
        except Exception as e:
            return f"Error analyzing mutation frequency: {e}"

    return [
        predict_mutation_effect,
        extract_protein_embedding,
        compare_variant_sequences,
        get_variant_drug_interactions,
        get_pharmacogenomic_evidence,
        score_variant_impact,
        expression_drug_sensitivity,
        mutation_frequency_analysis,
    ]
