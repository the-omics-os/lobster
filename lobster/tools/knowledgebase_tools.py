"""
Knowledgebase tool factories for UniProt and Ensembl integration.

Three tool factory functions following the create_execute_custom_code_tool pattern:
- create_cross_database_id_mapping_tool → metadata_assistant
- create_variant_consequence_tool → genomics_expert
- create_sequence_retrieval_tool → genomics_expert

Each factory returns a @tool-decorated closure with lazy service loading
and provenance logging via data_manager.log_tool_usage().
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from langchain_core.tools import tool

from lobster.utils.logger import get_logger

if TYPE_CHECKING:
    from lobster.core.data_manager_v2 import DataManagerV2

logger = get_logger(__name__)


# =============================================================================
# Tool 1: Cross-Database ID Mapping (metadata_assistant)
# =============================================================================


def create_cross_database_id_mapping_tool(data_manager: DataManagerV2):
    """
    Factory for cross-database ID mapping tool.

    Consolidates UniProt ID mapping + Ensembl cross-references into one tool.
    Assigned to metadata_assistant (ID mapping is its core competency).

    Args:
        data_manager: DataManagerV2 instance for provenance logging

    Returns:
        LangChain tool for cross-database ID mapping
    """
    _uniprot_service = None
    _ensembl_service = None

    def _get_uniprot():
        nonlocal _uniprot_service
        if _uniprot_service is None:
            from lobster.services.data_access.uniprot_service import UniProtService

            _uniprot_service = UniProtService()
        return _uniprot_service

    def _get_ensembl():
        nonlocal _ensembl_service
        if _ensembl_service is None:
            from lobster.services.data_access.ensembl_service import EnsemblService

            _ensembl_service = EnsemblService()
        return _ensembl_service

    @tool
    def map_cross_database_ids(
        ids: str,
        from_db: str,
        to_db: str,
    ) -> str:
        """
        Map biological identifiers between databases using UniProt ID Mapping or Ensembl xrefs.

        Supports gene names → UniProt, Ensembl → UniProt, UniProt → PDB, and many more
        cross-database mappings. Automatically selects the right backend (UniProt or Ensembl).

        **Routing logic:**
        - If from_db starts with "Ensembl" and to_db is also an xref type → uses Ensembl xrefs API
        - Otherwise → uses UniProt ID Mapping service

        Args:
            ids: Comma-separated identifiers (e.g. "TP53,BRCA1" or "ENSG00000141510")
            from_db: Source database. Common values:
                - "Gene_Name" (gene symbols like TP53)
                - "UniProtKB_AC-ID" (UniProt accessions)
                - "Ensembl" (Ensembl gene IDs)
                - "Ensembl_Protein" (Ensembl protein IDs)
                - "RefSeq_Protein" (RefSeq protein IDs)
                - "PDB" (PDB structure IDs)
                - "GeneID" (NCBI Gene IDs)
            to_db: Target database (same values as from_db)

        Returns:
            Formatted mapping results showing source → target ID pairs

        Examples:
            # Gene names to UniProt
            map_cross_database_ids(ids="TP53,BRCA1", from_db="Gene_Name", to_db="UniProtKB_AC-ID")

            # Ensembl to UniProt
            map_cross_database_ids(ids="ENSG00000141510", from_db="Ensembl", to_db="UniProtKB_AC-ID")

            # UniProt to PDB structures
            map_cross_database_ids(ids="P04637", from_db="UniProtKB_AC-ID", to_db="PDB")
        """
        id_list = [i.strip() for i in ids.split(",") if i.strip()]
        if not id_list:
            return "Error: No identifiers provided. Pass comma-separated IDs."

        try:
            # Route: Ensembl xrefs for Ensembl-to-external lookups on single IDs
            if from_db.startswith("Ensembl") and len(id_list) == 1:
                ensembl_id = id_list[0]
                if ensembl_id.upper().startswith("ENS"):
                    service = _get_ensembl()
                    # Map to_db to Ensembl external_db filter
                    db_filter = _map_to_ensembl_external_db(to_db)
                    xrefs = service.get_xrefs(ensembl_id, external_db=db_filter)

                    if not xrefs:
                        return f"No cross-references found for {ensembl_id} → {to_db}"

                    lines = [
                        f"## Cross-Database Mapping: {ensembl_id} → {to_db}\n",
                        f"Found {len(xrefs)} reference(s):\n",
                    ]
                    for xref in xrefs:
                        primary = xref.get("primary_id", "N/A")
                        display = xref.get("display_id", "")
                        dbname = xref.get("dbname", "")
                        desc = xref.get("description", "")
                        line = f"- **{primary}**"
                        if display and display != primary:
                            line += f" ({display})"
                        if dbname:
                            line += f" [{dbname}]"
                        if desc:
                            line += f": {desc}"
                        lines.append(line)

                    data_manager.log_tool_usage(
                        tool_name="map_cross_database_ids",
                        parameters={
                            "ids": ids,
                            "from_db": from_db,
                            "to_db": to_db,
                            "backend": "ensembl_xrefs",
                        },
                        description=f"Mapped {ensembl_id} via Ensembl xrefs → {len(xrefs)} results",
                    )
                    return "\n".join(lines)

            # Default: UniProt ID Mapping
            service = _get_uniprot()
            result = service.map_ids(from_db=from_db, to_db=to_db, ids=id_list)

            results_list = result.get("results", [])
            if not results_list:
                failed = result.get("failedIds", [])
                msg = f"No mappings found for {from_db} → {to_db}."
                if failed:
                    msg += f" Failed IDs: {', '.join(failed)}"
                return msg

            lines = [
                f"## Cross-Database Mapping: {from_db} → {to_db}\n",
                f"Mapped {len(results_list)} identifier(s):\n",
            ]
            for mapping in results_list[:50]:  # Cap output
                source = mapping.get("from", "?")
                target = mapping.get("to", {})
                if isinstance(target, dict):
                    target_id = target.get(
                        "primaryAccession", target.get("uniProtkbId", str(target))
                    )
                else:
                    target_id = str(target)
                lines.append(f"- {source} → **{target_id}**")

            if len(results_list) > 50:
                lines.append(f"\n... and {len(results_list) - 50} more")

            failed = result.get("failedIds", [])
            if failed:
                lines.append(f"\nFailed IDs: {', '.join(failed)}")

            data_manager.log_tool_usage(
                tool_name="map_cross_database_ids",
                parameters={
                    "ids": ids,
                    "from_db": from_db,
                    "to_db": to_db,
                    "backend": "uniprot_idmapping",
                },
                description=f"Mapped {len(id_list)} IDs via UniProt → {len(results_list)} results",
            )
            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Cross-database ID mapping error: {e}")
            return f"Error mapping IDs: {str(e)}"

    return map_cross_database_ids


def _map_to_ensembl_external_db(to_db: str) -> Optional[str]:
    """Map tool's to_db parameter to Ensembl external_db filter name."""
    mapping = {
        "UniProtKB_AC-ID": "UniProt/SWISSPROT",
        "UniProtKB": "UniProt/SWISSPROT",
        "HGNC": "HGNC",
        "RefSeq_mRNA": "RefSeq_mRNA",
        "RefSeq_Protein": "RefSeq_peptide",
        "PDB": "PDB",
        "MIM_GENE": "MIM_GENE",
        "GeneID": "EntrezGene",
    }
    return mapping.get(to_db)


# =============================================================================
# Tool 2: Variant Consequence Prediction (genomics_expert)
# =============================================================================


def create_variant_consequence_tool(data_manager: DataManagerV2):
    """
    Factory for VEP variant consequence prediction tool.

    Assigned to genomics_expert (domain-specific variant analysis).

    Args:
        data_manager: DataManagerV2 instance for provenance logging

    Returns:
        LangChain tool for variant consequence prediction
    """
    _ensembl_service = None

    def _get_ensembl():
        nonlocal _ensembl_service
        if _ensembl_service is None:
            from lobster.services.data_access.ensembl_service import EnsemblService

            _ensembl_service = EnsemblService()
        return _ensembl_service

    @tool
    def predict_variant_consequences(
        notation: str,
        species: str = "human",
        notation_type: str = "hgvs",
    ) -> str:
        """
        Predict the functional consequences of a genetic variant using Ensembl VEP.

        Returns consequence types (missense, synonymous, splice, etc.), affected
        transcripts, protein changes, and SIFT/PolyPhen predictions when available.

        Args:
            notation: Variant notation in one of these formats:
                - HGVS genomic: "9:g.22125503G>C"
                - HGVS coding: "ENST00000269305.9:c.817C>T"
                - Region: "9:22125503-22125503:1/C"
                - rsID: "rs1042522"
            species: Species name (default "human"). Also accepts "mouse", "rat",
                    taxonomy IDs like "9606", or Ensembl names like "homo_sapiens"
            notation_type: Format of the notation:
                - "hgvs" (default): HGVS notation (genomic or coding)
                - "region": Ensembl region notation
                - "id": dbSNP rsID

        Returns:
            Formatted VEP results with consequence types, affected genes/transcripts,
            and impact predictions

        Examples:
            predict_variant_consequences("9:g.22125503G>C", species="human")
            predict_variant_consequences("rs1042522", notation_type="id")
        """
        try:
            service = _get_ensembl()
            results = service.get_variant_consequences(
                notation=notation,
                species=species,
                notation_type=notation_type,
            )

            if not results:
                return f"No consequences predicted for variant: {notation}"

            # Format results
            lines = [f"## Variant Effect Prediction: {notation}\n"]

            # Handle both list and single-result responses
            entries = results if isinstance(results, list) else [results]

            for entry in entries:
                most_severe = entry.get("most_severe_consequence", "unknown")
                lines.append(f"**Most severe consequence**: {most_severe}\n")

                # Input variant info
                allele_string = entry.get("allele_string", "")
                if allele_string:
                    lines.append(f"**Alleles**: {allele_string}")

                # Colocated variants (known variants at this position)
                colocated = entry.get("colocated_variants", [])
                if colocated:
                    known_ids = [
                        v.get("id", "?")
                        for v in colocated
                        if v.get("id")
                    ]
                    if known_ids:
                        lines.append(
                            f"**Known variants**: {', '.join(known_ids[:5])}"
                        )

                # Transcript consequences
                transcript_cons = entry.get("transcript_consequences", [])
                if transcript_cons:
                    lines.append(f"\n**Transcript consequences** ({len(transcript_cons)}):\n")
                    for tc in transcript_cons[:10]:  # Cap output
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
                            line += f" — p.{amino_acids} at pos {protein_pos}"

                        # SIFT/PolyPhen
                        sift = tc.get("sift_prediction", "")
                        polyphen = tc.get("polyphen_prediction", "")
                        if sift:
                            line += f" | SIFT: {sift}({tc.get('sift_score', '')})"
                        if polyphen:
                            line += f" | PolyPhen: {polyphen}({tc.get('polyphen_score', '')})"

                        lines.append(line)

                    if len(transcript_cons) > 10:
                        lines.append(
                            f"\n... and {len(transcript_cons) - 10} more transcripts"
                        )

            data_manager.log_tool_usage(
                tool_name="predict_variant_consequences",
                parameters={
                    "notation": notation,
                    "species": species,
                    "notation_type": notation_type,
                },
                description=f"VEP prediction for {notation}: {entries[0].get('most_severe_consequence', 'unknown')}",
            )
            return "\n".join(lines)

        except Exception as e:
            logger.error(f"VEP prediction error: {e}")
            return f"Error predicting variant consequences: {str(e)}"

    return predict_variant_consequences


# =============================================================================
# Tool 3: Sequence Retrieval (genomics_expert)
# =============================================================================


def create_sequence_retrieval_tool(data_manager: DataManagerV2):
    """
    Factory for Ensembl sequence retrieval tool.

    Assigned to genomics_expert (domain-specific sequence analysis).

    Args:
        data_manager: DataManagerV2 instance for provenance logging

    Returns:
        LangChain tool for sequence retrieval
    """
    _ensembl_service = None

    def _get_ensembl():
        nonlocal _ensembl_service
        if _ensembl_service is None:
            from lobster.services.data_access.ensembl_service import EnsemblService

            _ensembl_service = EnsemblService()
        return _ensembl_service

    @tool
    def get_ensembl_sequence(
        ensembl_id: str,
        seq_type: str = "cdna",
    ) -> str:
        """
        Retrieve a nucleotide or protein sequence from Ensembl by stable ID.

        Args:
            ensembl_id: Ensembl stable ID for a gene, transcript, or protein
                       (e.g. "ENSG00000141510", "ENST00000269305", "ENSP00000269305")
            seq_type: Type of sequence to retrieve:
                - "genomic": Full genomic sequence (can be very long for genes)
                - "cdna": Complementary DNA (spliced transcript, default)
                - "cds": Coding sequence only (start to stop codon)
                - "protein": Amino acid sequence

        Returns:
            Formatted sequence with metadata (ID, type, length, first 500 chars)

        Examples:
            get_ensembl_sequence("ENST00000269305", seq_type="cdna")
            get_ensembl_sequence("ENSP00000269305", seq_type="protein")
        """
        try:
            service = _get_ensembl()
            result = service.get_sequence(
                ensembl_id=ensembl_id,
                seq_type=seq_type,
            )

            seq = result.get("seq", "")
            seq_id = result.get("id", ensembl_id)
            desc = result.get("desc", "")
            molecule = result.get("molecule", seq_type)

            lines = [
                f"## Sequence: {seq_id}\n",
                f"**Type**: {seq_type} ({molecule})",
                f"**Length**: {len(seq):,} {'residues' if seq_type == 'protein' else 'bases'}",
            ]
            if desc:
                lines.append(f"**Description**: {desc}")

            # Show sequence preview (cap at 500 chars to avoid context overflow)
            if len(seq) <= 500:
                lines.append(f"\n**Full sequence**:\n```\n{seq}\n```")
            else:
                lines.append(
                    f"\n**Sequence preview** (first 500 of {len(seq):,}):\n"
                    f"```\n{seq[:500]}...\n```"
                )

            data_manager.log_tool_usage(
                tool_name="get_ensembl_sequence",
                parameters={
                    "ensembl_id": ensembl_id,
                    "seq_type": seq_type,
                },
                description=f"Retrieved {seq_type} sequence for {seq_id}: {len(seq):,} {'residues' if seq_type == 'protein' else 'bases'}",
            )
            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Sequence retrieval error: {e}")
            return f"Error retrieving sequence: {str(e)}"

    return get_ensembl_sequence


# =============================================================================
# Tool 4: Summarize Modality (shared across agents)
# =============================================================================


def create_summarize_modality_tool(data_manager: DataManagerV2):
    """
    Factory for summarize_modality tool — merges list_modalities + get_modality_info.

    This tool consolidates two previously separate tools into one, reducing
    LLM confusion about which to call. If modality_name is given, returns
    detailed info; if omitted, lists all loaded modalities.

    Assigned to: genomics_expert (pilot), will expand to other agents.

    Args:
        data_manager: DataManagerV2 instance for modality access and logging

    Returns:
        LangChain tool for modality inspection
    """

    @tool
    def summarize_modality(modality_name: Optional[str] = None) -> str:
        """
        List all loaded modalities or get detailed info about a specific one.

        If modality_name is provided, returns detailed information (dimensions,
        metadata, QC status, layers, columns). If omitted, lists all modalities
        with basic info.

        Args:
            modality_name: Optional name of specific modality to inspect.
                          If None, lists all loaded modalities.

        Returns:
            Modality summary or detailed information.
        """
        try:
            if modality_name is None:
                # List all modalities
                all_modalities = data_manager.list_modalities()
                if not all_modalities:
                    return "No modalities loaded yet. Use data loading tools to load data."

                lines = [f"**Loaded Modalities** ({len(all_modalities)} total):\n"]
                for mod_name in all_modalities:
                    try:
                        adata = data_manager.get_modality(mod_name)
                        data_type = adata.uns.get("data_type", "unknown")
                        layers = list(adata.layers.keys()) if adata.layers else []
                        lines.append(
                            f"  - **{mod_name}** ({data_type}): "
                            f"{adata.n_obs:,} obs x {adata.n_vars:,} vars"
                            f"{f', layers: {layers}' if layers else ''}"
                        )
                    except Exception:
                        lines.append(f"  - **{mod_name}**: (error reading)")

                data_manager.log_tool_usage(
                    tool_name="summarize_modality",
                    parameters={"modality_name": None},
                    description=f"Listed {len(all_modalities)} modalities",
                )
                return "\n".join(lines)

            else:
                # Detailed info for specific modality
                all_modalities = data_manager.list_modalities()
                if modality_name not in all_modalities:
                    return (
                        f"Modality '{modality_name}' not found. "
                        f"Available: {all_modalities}"
                    )

                adata = data_manager.get_modality(modality_name)

                data_type = adata.uns.get("data_type", "unknown")
                modality_type = adata.uns.get("modality", "unknown")
                source_file = adata.uns.get("source_file", "N/A")

                has_qc = "call_rate" in adata.obs.columns or "qc_pass" in adata.var.columns

                lines = [
                    f"**Modality: '{modality_name}'**\n",
                    f"**Dimensions:** {adata.n_obs:,} obs x {adata.n_vars:,} vars",
                    f"**Data type:** {data_type}",
                    f"**Modality:** {modality_type}",
                    f"**Source file:** {source_file}",
                    f"\n**obs columns ({len(adata.obs.columns)}):** {list(adata.obs.columns)}",
                    f"**var columns ({len(adata.var.columns)}):** {list(adata.var.columns)}",
                    f"**Layers:** {list(adata.layers.keys()) if adata.layers else 'None'}",
                    f"**obsm keys:** {list(adata.obsm.keys()) if adata.obsm else 'None'}",
                    f"**uns keys:** {list(adata.uns.keys()) if adata.uns else 'None'}",
                    f"**QC metrics present:** {'Yes' if has_qc else 'No'}",
                ]

                if has_qc:
                    if "call_rate" in adata.obs.columns:
                        lines.append(
                            f"  - Mean sample call rate: {adata.obs['call_rate'].mean():.4f}"
                        )
                    if "qc_pass" in adata.var.columns:
                        n_pass = int(adata.var["qc_pass"].sum())
                        lines.append(
                            f"  - Variants passing QC: {n_pass:,}/{adata.n_vars:,}"
                        )

                data_manager.log_tool_usage(
                    tool_name="summarize_modality",
                    parameters={"modality_name": modality_name},
                    description=f"Detailed info for modality '{modality_name}'",
                )
                return "\n".join(lines)

        except Exception as e:
            logger.error(f"Error in summarize_modality: {e}")
            return f"Error: {str(e)}"

    return summarize_modality


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "create_cross_database_id_mapping_tool",
    "create_variant_consequence_tool",
    "create_sequence_retrieval_tool",
    "create_summarize_modality_tool",
]
