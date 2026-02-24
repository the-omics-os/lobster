"""
Variant Analysis Expert Sub-Agent for clinical variant interpretation.

This sub-agent handles post-GWAS clinical interpretation tools for genomic variants.
It is called by the parent genomics_expert via delegation tools.

Tools included:
1. normalize_variants - Left-align indels, split multiallelic
2. predict_consequences - VEP batch annotation with SIFT/PolyPhen/CADD
3. query_population_frequencies - gnomAD allele frequency lookup
4. query_clinical_databases - ClinVar pathogenicity and disease associations
5. prioritize_variants - Rank by consequence severity + frequency + pathogenicity
6. lookup_variant - Single-variant comprehensive lookup by rsID/coordinates
7. retrieve_sequence - Ensembl sequence fetch (relocated from parent)
8. summarize_modality - List/inspect loaded modalities
"""

# Agent configuration for entry point discovery (must be at top, before heavy imports)
from lobster.config.agent_registry import AgentRegistryConfig

AGENT_CONFIG = AgentRegistryConfig(
    name="variant_analysis_expert",
    display_name="Variant Analysis Expert",
    description="Clinical variant interpretation: normalization, VEP consequences, gnomAD frequencies, ClinVar pathogenicity, variant prioritization",
    factory_function="lobster.agents.genomics.variant_analysis_expert.variant_analysis_expert",
    handoff_tool_name=None,  # Not directly accessible from supervisor
    handoff_tool_description=None,
    supervisor_accessible=False,  # Only via genomics_expert parent
    tier_requirement="free",
)

# === Heavy imports below ===
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from lobster.config.llm_factory import create_llm
from lobster.config.settings import get_settings
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.analysis.variant_annotation_service import (
    VariantAnnotationService,
)
from lobster.tools.knowledgebase_tools import (
    create_sequence_retrieval_tool,
    create_summarize_modality_tool,
)
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class VariantAnalysisAgentError(Exception):
    """Base exception for variant analysis agent operations."""

    pass


class ModalityNotFoundError(VariantAnalysisAgentError):
    """Raised when requested modality doesn't exist."""

    pass


def variant_analysis_expert(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "variant_analysis_expert",
    delegation_tools: list = None,
    workspace_path: Optional[Path] = None,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
):
    """
    Factory function for variant analysis expert sub-agent.

    This sub-agent handles clinical variant interpretation: normalization,
    consequence prediction, population frequencies, clinical databases,
    and variant prioritization. It is delegated to by genomics_expert.

    Args:
        data_manager: DataManagerV2 instance for modality management
        callback_handler: Optional callback handler for LLM interactions
        agent_name: Name identifier for the agent instance
        delegation_tools: List of delegation tools for sub-agents
        workspace_path: Optional workspace path for LLM operations
        provider_override: Optional LLM provider override
        model_override: Optional model override

    Returns:
        Configured ReAct agent with variant analysis capabilities
    """
    settings = get_settings()
    model_params = settings.get_agent_llm_params("variant_analysis_expert")
    llm = create_llm(
        "variant_analysis_expert",
        model_params,
        provider_override=provider_override,
        model_override=model_override,
        workspace_path=workspace_path,
    )

    # Normalize callbacks to a flat list
    if callback_handler and hasattr(llm, "with_config"):
        callbacks = (
            callback_handler
            if isinstance(callback_handler, list)
            else [callback_handler]
        )
        llm = llm.with_config(callbacks=callbacks)

    # Initialize services
    annotation_service = VariantAnnotationService()

    # =========================================================================
    # VARIANT NORMALIZATION TOOL
    # =========================================================================

    @tool
    def normalize_variants(
        modality_name: str,
        ref_col: str = "REF",
        alt_col: str = "ALT",
    ) -> str:
        """
        Normalize variant representations: left-trim padding bases and split multiallelic variants.

        Run BEFORE annotation or frequency lookup to ensure consistent variant representation.
        Splits multiallelic variants (e.g., REF=A, ALT=C,T) into separate biallelic records.

        Args:
            modality_name: Name of modality with variant data
            ref_col: Column in adata.var containing reference alleles (default "REF")
            alt_col: Column in adata.var containing alternate alleles (default "ALT")

        Returns:
            Normalization summary with count of trimmed and split variants
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
                )

            adata = data_manager.get_modality(modality_name)
            logger.info(
                f"Normalizing variants in '{modality_name}': {adata.n_vars} variants"
            )

            # Run normalization
            adata_norm, stats, ir = annotation_service.normalize_variants(
                adata=adata,
                ref_col=ref_col,
                alt_col=alt_col,
            )

            # Store as new modality
            norm_modality_name = f"{modality_name}_normalized"
            data_manager.store_modality(
                name=norm_modality_name,
                adata=adata_norm,
                parent_name=modality_name,
                step_summary=f"Normalized: {stats['n_trimmed']} trimmed, {stats['n_multiallelic_split']} multiallelic split",
            )

            # Log operation with IR
            data_manager.log_tool_usage(
                tool_name="normalize_variants",
                parameters={
                    "modality_name": modality_name,
                    "ref_col": ref_col,
                    "alt_col": alt_col,
                },
                description=f"Normalized variants: {stats['n_variants_before']} -> {stats['n_variants_after']}",
                ir=ir,
            )

            response = f"""Variant normalization completed: '{norm_modality_name}'

**Normalization Summary:**
- Variants before: {stats['n_variants_before']:,}
- Variants after: {stats['n_variants_after']:,}
- Alleles trimmed: {stats['n_trimmed']:,}
- Multiallelic split: {stats['n_multiallelic_split']:,}

**New modality created**: '{norm_modality_name}'
**Next steps**: Run predict_consequences("{norm_modality_name}") to annotate variant consequences
"""
            return response

        except ModalityNotFoundError as e:
            logger.error(f"Error in variant normalization: {e}")
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in variant normalization: {e}")
            return f"Unexpected error: {str(e)}"

    # =========================================================================
    # CONSEQUENCE PREDICTION TOOL
    # =========================================================================

    @tool
    def predict_consequences(
        modality_name: str,
        annotation_source: str = "genebe",
        genome_build: str = "GRCh38",
    ) -> str:
        """
        Predict functional consequences of variants using VEP/genebe batch annotation.

        Annotates variants with gene names, consequence types (missense, synonymous, etc.),
        SIFT/PolyPhen impact predictions, and CADD pathogenicity scores. Operates on all
        variants in the modality (batch mode).

        For single-variant lookups by rsID or coordinates, use lookup_variant() instead.

        Args:
            modality_name: Name of modality with variant data
            annotation_source: "genebe" (default, faster) or "ensembl_vep" (REST API)
            genome_build: "GRCh38" (default) or "GRCh37"

        Returns:
            Annotation summary with gene coverage and consequence distribution
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
                )

            adata = data_manager.get_modality(modality_name)
            logger.info(
                f"Predicting consequences for '{modality_name}': {adata.n_vars} variants, source={annotation_source}"
            )

            # Run annotation
            adata_annotated, stats, ir = annotation_service.annotate_variants(
                adata=adata,
                annotation_source=annotation_source,
                genome_build=genome_build,
            )

            # Store as new modality
            cons_modality_name = f"{modality_name}_consequences"
            data_manager.store_modality(
                name=cons_modality_name,
                adata=adata_annotated,
                parent_name=modality_name,
                step_summary=f"Consequences: {stats['n_variants_annotated']}/{stats['n_variants']} annotated",
            )

            # Log operation with IR
            data_manager.log_tool_usage(
                tool_name="predict_consequences",
                parameters={
                    "modality_name": modality_name,
                    "annotation_source": annotation_source,
                    "genome_build": genome_build,
                },
                description=f"Consequence prediction: {stats['n_variants_annotated']}/{stats['n_variants']} variants annotated",
                ir=ir,
            )

            # Format response
            response = f"""Consequence prediction completed: '{cons_modality_name}'

**Annotation Summary:**
- Annotation source: {stats['annotation_source']}
- Genome build: {stats['genome_build']}
- Variants: {stats['n_variants']:,}
- Variants annotated: {stats['n_variants_annotated']:,} ({stats['annotation_rate_pct']:.1f}%)

**Annotations Added to adata.var:**
- gene_symbol, gene_id, consequence, biotype
- gnomad_af, clinvar_significance, cadd_score
- sift_score, polyphen_score"""

            # Add top consequences if available
            if stats.get("top_consequences"):
                response += "\n\n**Top Consequences:**"
                for cons, count in list(stats["top_consequences"].items())[:5]:
                    response += f"\n- {cons}: {count}"

            response += f"\n\n**New modality created**: '{cons_modality_name}'"
            response += f'\n**Next steps**: Run query_population_frequencies("{cons_modality_name}") or prioritize_variants("{cons_modality_name}")'

            return response

        except ModalityNotFoundError as e:
            logger.error(f"Error in consequence prediction: {e}")
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in consequence prediction: {e}")
            return f"Unexpected error: {str(e)}"

    # =========================================================================
    # POPULATION FREQUENCY TOOL
    # =========================================================================

    @tool
    def query_population_frequencies(
        modality_name: str,
        population: Optional[str] = None,
    ) -> str:
        """
        Look up gnomAD population allele frequencies for variants in a modality.

        Adds gnomAD allele frequency (AF) annotations to adata.var. Rare variants
        (AF < 0.01) are more likely to be pathogenic. Common variants (AF > 0.05)
        are usually benign.

        Args:
            modality_name: Name of modality with variant data
            population: Optional population filter (e.g., "EUR", "AFR", "EAS").
                       Default: global allele frequency across all populations.

        Returns:
            Frequency distribution summary with rare/common variant counts
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
                )

            adata = data_manager.get_modality(modality_name)
            logger.info(f"Querying population frequencies for '{modality_name}'")

            # Run frequency query
            adata_freq, stats, ir = annotation_service.query_population_frequencies(
                adata=adata,
                population=population,
            )

            # Store as new modality
            freq_modality_name = f"{modality_name}_frequencies"
            data_manager.store_modality(
                name=freq_modality_name,
                adata=adata_freq,
                parent_name=modality_name,
                step_summary=f"Frequencies: {stats['n_with_frequency']}/{stats['n_variants']} with gnomAD AF",
            )

            # Log operation with IR
            data_manager.log_tool_usage(
                tool_name="query_population_frequencies",
                parameters={
                    "modality_name": modality_name,
                    "population": population,
                },
                description=f"Population frequencies: {stats['n_with_frequency']}/{stats['n_variants']} with AF",
                ir=ir,
            )

            # Format response
            response = f"""Population frequency query completed: '{freq_modality_name}'

**Frequency Summary:**
- Variants queried: {stats['n_variants']:,}
- Variants with frequency: {stats['n_with_frequency']:,}
- Mean allele frequency: {stats['mean_af']:.6f}
- Median allele frequency: {stats['median_af']:.6f}

**Variant Classification by Frequency:**
- Rare (AF < 0.01): {stats['n_rare']:,}
- Common (AF > 0.05): {stats['n_common']:,}"""

            if stats.get("population_note"):
                response += f"\n\n**Note:** {stats['population_note']}"

            response += f"\n\n**New modality created**: '{freq_modality_name}'"
            response += f'\n**Next steps**: Run query_clinical_databases("{freq_modality_name}") or prioritize_variants("{freq_modality_name}")'

            return response

        except ModalityNotFoundError as e:
            logger.error(f"Error in population frequency query: {e}")
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in population frequency query: {e}")
            return f"Unexpected error: {str(e)}"

    # =========================================================================
    # CLINICAL DATABASE TOOL
    # =========================================================================

    @tool
    def query_clinical_databases(modality_name: str) -> str:
        """
        Query ClinVar for clinical significance of variants in a modality.

        Adds ClinVar pathogenicity classifications (pathogenic, likely_pathogenic,
        uncertain_significance, likely_benign, benign) to adata.var.

        Args:
            modality_name: Name of modality with variant data

        Returns:
            Clinical annotation summary with pathogenicity distribution
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
                )

            adata = data_manager.get_modality(modality_name)
            logger.info(f"Querying clinical databases for '{modality_name}'")

            # Run clinical query
            adata_clin, stats, ir = annotation_service.query_clinical_databases(
                adata=adata,
            )

            # Store as new modality
            clin_modality_name = f"{modality_name}_clinical"
            data_manager.store_modality(
                name=clin_modality_name,
                adata=adata_clin,
                parent_name=modality_name,
                step_summary=f"Clinical: {stats['n_with_clinvar']}/{stats['n_variants']} with ClinVar",
            )

            # Log operation with IR
            data_manager.log_tool_usage(
                tool_name="query_clinical_databases",
                parameters={"modality_name": modality_name},
                description=f"Clinical databases: {stats['n_with_clinvar']}/{stats['n_variants']} with ClinVar",
                ir=ir,
            )

            # Format response
            response = f"""Clinical database query completed: '{clin_modality_name}'

**Clinical Annotation Summary:**
- Variants queried: {stats['n_variants']:,}
- Variants with ClinVar: {stats['n_with_clinvar']:,}
- Pathogenic/Likely pathogenic: {stats['n_pathogenic']:,}
- Benign/Likely benign: {stats['n_benign']:,}
- Uncertain significance: {stats['n_uncertain']:,}"""

            # Add significance distribution
            if stats.get("significance_counts"):
                response += "\n\n**ClinVar Significance Distribution:**"
                for sig, count in list(stats["significance_counts"].items())[:8]:
                    response += f"\n- {sig}: {count}"

            response += f"\n\n**New modality created**: '{clin_modality_name}'"
            response += f'\n**Next steps**: Run prioritize_variants("{clin_modality_name}") to rank by composite score'

            return response

        except ModalityNotFoundError as e:
            logger.error(f"Error in clinical database query: {e}")
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in clinical database query: {e}")
            return f"Unexpected error: {str(e)}"

    # =========================================================================
    # VARIANT PRIORITIZATION TOOL
    # =========================================================================

    @tool
    def prioritize_variants(modality_name: str) -> str:
        """
        Rank variants by composite priority score combining consequence severity,
        population frequency, and pathogenicity evidence.

        Requires prior annotation (run predict_consequences and/or query_population_frequencies
        first). Produces a priority_score (0-1) and priority_rank (1=highest) for each variant.

        Score components:
        - Consequence severity (0-0.4): frameshift/stop_gained > missense > synonymous > intronic
        - Population rarity (0-0.3): absent > rare (AF<0.01) > low_freq > common
        - Pathogenicity (0-0.3): ClinVar pathogenic + CADD>20 + SIFT deleterious + PolyPhen damaging

        Args:
            modality_name: Name of annotated modality

        Returns:
            Prioritization summary with top variants and score distribution
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
                )

            adata = data_manager.get_modality(modality_name)
            logger.info(
                f"Prioritizing variants in '{modality_name}': {adata.n_vars} variants"
            )

            # Run prioritization
            adata_pri, stats, ir = annotation_service.prioritize_variants(
                adata=adata,
            )

            # Store as new modality
            pri_modality_name = f"{modality_name}_prioritized"
            data_manager.store_modality(
                name=pri_modality_name,
                adata=adata_pri,
                parent_name=modality_name,
                step_summary=f"Prioritized: {stats['n_high_priority']} high, {stats['n_medium_priority']} medium priority",
            )

            # Log operation with IR
            data_manager.log_tool_usage(
                tool_name="prioritize_variants",
                parameters={"modality_name": modality_name},
                description=f"Variant prioritization: {stats['n_high_priority']} high priority of {stats['n_variants_scored']}",
                ir=ir,
            )

            # Format response with top 10 variants
            response = f"""Variant prioritization completed: '{pri_modality_name}'

**Prioritization Summary:**
- Variants scored: {stats['n_variants_scored']:,}
- High priority (score > 0.6): {stats['n_high_priority']:,}
- Medium priority (0.3-0.6): {stats['n_medium_priority']:,}
- Low priority (< 0.3): {stats['n_low_priority']:,}

**Top Variants:**"""

            for i, var in enumerate(stats.get("top_variants", [])[:10], 1):
                line = f"\n{i}. {var['variant_id']}: score={var['priority_score']:.3f} (rank #{var['priority_rank']})"
                if var.get("consequence"):
                    line += f" [{var['consequence']}]"
                response += line

            response += f"\n\n**New modality created**: '{pri_modality_name}'"
            response += "\n**Results stored in**: adata.var['priority_score'] and adata.var['priority_rank']"

            return response

        except ModalityNotFoundError as e:
            logger.error(f"Error in variant prioritization: {e}")
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in variant prioritization: {e}")
            return f"Unexpected error: {str(e)}"

    # =========================================================================
    # SINGLE-VARIANT LOOKUP TOOL
    # =========================================================================

    # Lazy-loaded EnsemblService for single-variant lookups
    _ensembl_service = None

    def _get_ensembl():
        nonlocal _ensembl_service
        if _ensembl_service is None:
            from lobster.services.data_access.ensembl_service import EnsemblService

            _ensembl_service = EnsemblService()
        return _ensembl_service

    @tool
    def lookup_variant(
        notation: str,
        species: str = "human",
        notation_type: str = "id",
    ) -> str:
        """
        Comprehensive single-variant lookup by rsID or genomic coordinates.

        Returns all available information for one variant: VEP consequences, gnomAD
        frequencies, ClinVar significance, SIFT/PolyPhen scores, affected genes and
        transcripts. Does NOT require a loaded modality.

        Args:
            notation: Variant identifier in one of these formats:
                - rsID: "rs1042522" (use notation_type="id")
                - HGVS genomic: "9:g.22125503G>C" (use notation_type="hgvs")
                - Region: "9:22125503-22125503:1/C" (use notation_type="region")
            species: Species name (default "human")
            notation_type: "id" (default, for rsIDs), "hgvs", or "region"

        Returns:
            Comprehensive variant report with consequences, frequencies, clinical significance
        """
        try:
            service = _get_ensembl()
            results = service.get_variant_consequences(
                notation=notation,
                species=species,
                notation_type=notation_type,
            )

            if not results:
                return f"No results found for variant: {notation}"

            # Handle both list and single-result responses
            entries = results if isinstance(results, list) else [results]
            entry = entries[0]

            # === Build comprehensive report ===
            lines = [f"## Comprehensive Variant Report: {notation}\n"]

            # Basic Info
            most_severe = entry.get("most_severe_consequence", "unknown")
            allele_string = entry.get("allele_string", "")
            lines.append("### Basic Information")
            lines.append(f"- **Most severe consequence**: {most_severe}")
            if allele_string:
                lines.append(f"- **Alleles**: {allele_string}")

            # Consequences
            transcript_cons = entry.get("transcript_consequences", [])
            if transcript_cons:
                lines.append(
                    f"\n### Transcript Consequences ({len(transcript_cons)} transcripts)"
                )
                for tc in transcript_cons[:10]:
                    gene = tc.get("gene_symbol", tc.get("gene_id", "?"))
                    tx_id = tc.get("transcript_id", "?")
                    cons = ", ".join(tc.get("consequence_terms", []))
                    impact = tc.get("impact", "")
                    biotype = tc.get("biotype", "")

                    line = f"- **{gene}** ({tx_id}): {cons}"
                    if impact:
                        line += f" [{impact}]"
                    if biotype:
                        line += f" ({biotype})"

                    # Protein change
                    amino_acids = tc.get("amino_acids", "")
                    protein_pos = tc.get("protein_start", "")
                    if amino_acids and protein_pos:
                        line += f" -- p.{amino_acids} at pos {protein_pos}"

                    # SIFT/PolyPhen
                    sift = tc.get("sift_prediction", "")
                    sift_score = tc.get("sift_score", "")
                    polyphen = tc.get("polyphen_prediction", "")
                    polyphen_score = tc.get("polyphen_score", "")
                    if sift:
                        line += f" | SIFT: {sift}({sift_score})"
                    if polyphen:
                        line += f" | PolyPhen: {polyphen}({polyphen_score})"

                    lines.append(line)

                if len(transcript_cons) > 10:
                    lines.append(
                        f"\n... and {len(transcript_cons) - 10} more transcripts"
                    )

            # Population Frequencies (from colocated_variants)
            colocated = entry.get("colocated_variants", [])
            gnomad_info = []
            clinvar_info = []
            known_ids = []

            for cv in colocated:
                cv_id = cv.get("id", "")
                if cv_id:
                    known_ids.append(cv_id)

                # gnomAD frequencies
                if "frequencies" in cv:
                    freqs = cv["frequencies"]
                    for allele_key, freq_data in freqs.items():
                        if isinstance(freq_data, dict):
                            af = freq_data.get("af")
                            if af is not None:
                                gnomad_info.append((allele_key, af, freq_data))

                # ClinVar
                clin_sig = cv.get("clin_sig", [])
                if clin_sig:
                    clinvar_info.extend(clin_sig)

            if known_ids:
                lines.append(f"\n### Known Variant IDs")
                lines.append(f"- {', '.join(known_ids[:10])}")

            if gnomad_info:
                lines.append("\n### Population Frequencies (gnomAD)")
                for allele_key, af, freq_data in gnomad_info[:5]:
                    line = f"- **{allele_key}**: AF={af:.6f}"
                    # Extract population-specific if available
                    for pop_key in ["afr", "amr", "eas", "nfe", "sas"]:
                        pop_af = freq_data.get(pop_key)
                        if pop_af is not None:
                            line += f" | {pop_key.upper()}={pop_af:.4f}"
                    lines.append(line)

            if clinvar_info:
                lines.append("\n### Clinical Significance (ClinVar)")
                unique_sigs = list(set(clinvar_info))
                for sig in unique_sigs:
                    lines.append(f"- {sig}")

            # Log the lookup
            data_manager.log_tool_usage(
                tool_name="lookup_variant",
                parameters={
                    "notation": notation,
                    "species": species,
                    "notation_type": notation_type,
                },
                description=f"Variant lookup for {notation}: {most_severe}",
            )

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Error in variant lookup: {e}")
            return f"Error looking up variant: {str(e)}"

    # =========================================================================
    # RELOCATED TOOLS (from parent via factories)
    # =========================================================================

    retrieve_sequence = create_sequence_retrieval_tool(data_manager)
    summarize_modality = create_summarize_modality_tool(data_manager)

    # =========================================================================
    # COLLECT ALL TOOLS
    # =========================================================================

    variant_tools = [
        normalize_variants,
        predict_consequences,
        query_population_frequencies,
        query_clinical_databases,
        prioritize_variants,
        lookup_variant,
        retrieve_sequence,
        summarize_modality,
    ]

    tools = variant_tools
    if delegation_tools:
        tools = tools + delegation_tools

    # Create system prompt (lazy import to allow AGENT_CONFIG discovery before prompt exists)
    from lobster.agents.genomics.prompts import create_variant_analysis_expert_prompt

    system_prompt = create_variant_analysis_expert_prompt()

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        name=agent_name,
    )
