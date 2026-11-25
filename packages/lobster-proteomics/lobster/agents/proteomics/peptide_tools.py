"""
Peptide expert tool factory.

8 domain tools + DELEGATE (handoff_to_supervisor) + CODE_EXEC (execute_custom_code).
All domain tools follow the 3-tuple service pattern with AQUADIF metadata.
"""

import logging
from pathlib import Path
from typing import Callable, List, Optional

from langchain_core.tools import tool

from lobster.core import session_inputs
from lobster.core.figures.export_guard import MANAGED_EXPORT_MESSAGE, export_refused
from lobster.core.runtime.data_manager import DataManagerV2

logger = logging.getLogger(__name__)


def create_peptide_tools(
    data_manager: DataManagerV2,
    workspace_path: Optional[Path] = None,
    agent_name: str = "peptide_expert",
) -> List[Callable]:
    """Create peptide analysis tools with service wiring.

    Returns list of 8 domain tools. DELEGATE and CODE_EXEC are added
    by the agent factory (not here).
    """
    from lobster.services.peptidomics.peptide_activity_service import (
        PeptideActivityService,
    )
    from lobster.services.peptidomics.peptide_digestion_service import (
        PeptideDigestionService,
    )
    from lobster.services.peptidomics.peptide_library_service import (
        PeptideLibraryService,
    )
    from lobster.services.peptidomics.peptide_property_service import (
        PeptidePropertyService,
    )

    property_service = PeptidePropertyService()
    activity_service = PeptideActivityService()
    digestion_service = PeptideDigestionService()
    library_service = PeptideLibraryService()

    # =========================================================================
    # TOOL 1: import_peptide_sequences
    # =========================================================================

    @tool
    def import_peptide_sequences(
        file_path: str,
        modality_name: str = "peptides",
        format: str = "auto",
    ) -> str:
        """Load peptide sequences from FASTA, CSV, or TSV. Auto-detects format.

        Args:
            file_path: Path to input file (FASTA/CSV/TSV)
            modality_name: Name for the peptide modality
            format: File format — "fasta", "csv", "tsv", or "auto"
        """
        from lobster.tools.filesystem_tools import _resolve_safe_path

        if workspace_path and not Path(file_path).is_absolute():
            try:
                resolved = _resolve_safe_path(workspace_path, file_path)
            except ValueError as e:
                return f"Security error: {e}"
        else:
            resolved = Path(file_path)
        if not resolved.exists():
            return f"File not found: {resolved}"

        try:
            adata, stats, ir = library_service.import_sequences(
                str(resolved), format=format
            )
        except ValueError as e:
            return str(e)
        # An export never overwrites the file this read.
        read_files = session_inputs.note_import(data_manager, resolved)

        data_manager.store_modality(
            name=modality_name,
            adata=adata,
            step_summary=f"Imported {stats['n_sequences']} peptide sequences from {resolved.name}",
        )

        data_manager.log_tool_usage(
            tool_name="import_peptide_sequences",
            parameters={
                "file_path": str(resolved),
                "format": stats["format"],
                "read_files": read_files,
            },
            description=f"Import {stats['n_sequences']} peptides from {resolved.name}",
            ir=ir,
            agent=agent_name,
        )

        result = f"STATUS: complete\npeptides_imported={stats['n_sequences']}"
        if stats["n_invalid"]:
            result += f"\ninvalid_skipped={stats['n_invalid']}"
        result += f"\nmodality='{modality_name}'"
        return result

    import_peptide_sequences.metadata = {"categories": ["IMPORT"], "provenance": True}
    import_peptide_sequences.tags = ["IMPORT"]

    # =========================================================================
    # TOOL 2: calculate_peptide_properties
    # =========================================================================

    @tool
    def calculate_peptide_properties(
        modality_name: str,
        properties: str = "all",
    ) -> str:
        """Calculate physicochemical properties (MW, pI, charge, GRAVY, Boman, etc.).

        Args:
            modality_name: Peptide modality to analyze
            properties: Comma-separated list or "all"
        """
        if modality_name not in data_manager.list_modalities():
            return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"

        adata = data_manager.get_modality(modality_name)

        prop_list = (
            None if properties == "all" else [p.strip() for p in properties.split(",")]
        )

        adata_out, stats, ir = property_service.calculate_properties(
            adata, properties=prop_list
        )

        result_name = f"{modality_name}_properties"
        data_manager.store_modality(
            name=result_name,
            adata=adata_out,
            parent_name=modality_name,
            step_summary=f"Properties: {', '.join(stats['properties_computed'])}",
        )
        data_manager.log_tool_usage(
            tool_name="calculate_peptide_properties",
            parameters={"modality_name": modality_name, "properties": properties},
            description=f"Computed {len(stats['properties_computed'])} properties for {stats['n_peptides']} peptides",
            ir=ir,
            agent=agent_name,
        )

        return (
            f"STATUS: complete\n"
            f"peptides_analyzed={stats['n_peptides']}\n"
            f"properties={','.join(stats['properties_computed'])}\n"
            f"modality='{result_name}'\n"
            f"method={stats['method']}"
        )

    calculate_peptide_properties.metadata = {
        "categories": ["ANALYZE"],
        "provenance": True,
    }
    calculate_peptide_properties.tags = ["ANALYZE"]

    # =========================================================================
    # TOOL 3: predict_peptide_activity
    # =========================================================================

    @tool
    def predict_peptide_activity(
        modality_name: str,
        activity_type: str = "antimicrobial",
    ) -> str:
        """Predict peptide activity via heuristic scoring.

        Args:
            modality_name: Peptide modality to analyze
            activity_type: "antimicrobial", "cell_penetrating", or "toxic"
        """
        if modality_name not in data_manager.list_modalities():
            return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"

        adata = data_manager.get_modality(modality_name)

        adata_out, stats, ir = activity_service.predict_activity(
            adata, activity_type=activity_type
        )

        result_name = f"{modality_name}_{activity_type}"
        data_manager.store_modality(
            name=result_name,
            adata=adata_out,
            parent_name=modality_name,
            step_summary=f"{activity_type}: {stats['n_positive']}/{stats['n_scored']} positive",
        )
        data_manager.log_tool_usage(
            tool_name="predict_peptide_activity",
            parameters={"modality_name": modality_name, "activity_type": activity_type},
            description=f"{activity_type} prediction: {stats['n_positive']} positive",
            ir=ir,
            agent=agent_name,
        )

        return (
            f"STATUS: complete\n"
            f"activity_type={activity_type}\n"
            f"peptides_scored={stats['n_scored']}\n"
            f"positive={stats['n_positive']}\n"
            f"negative={stats['n_negative']}\n"
            f"mean_score={stats['mean_score']:.3f}\n"
            f"method={stats['method']}\n"
            f"note={stats['confidence']}\n"
            f"modality='{result_name}'"
        )

    predict_peptide_activity.metadata = {"categories": ["ANALYZE"], "provenance": True}
    predict_peptide_activity.tags = ["ANALYZE"]

    # =========================================================================
    # TOOL 4: simulate_enzymatic_digestion
    # =========================================================================

    @tool
    def simulate_enzymatic_digestion(
        modality_name: str,
        enzyme: str = "trypsin",
        missed_cleavages: int = 0,
        min_length: int = 6,
        max_length: int = 50,
    ) -> str:
        """In silico enzymatic digestion of protein/peptide sequences.

        Args:
            modality_name: Modality containing sequences
            enzyme: Enzyme — "trypsin", "chymotrypsin", "pepsin", "lys_c", "asp_n", "glu_c"
            missed_cleavages: Allowed missed cleavages (0-3)
            min_length: Minimum fragment length
            max_length: Maximum fragment length
        """
        if modality_name not in data_manager.list_modalities():
            return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"

        adata = data_manager.get_modality(modality_name)

        adata_out, stats, ir = digestion_service.digest(
            adata,
            enzyme=enzyme,
            missed_cleavages=missed_cleavages,
            min_length=min_length,
            max_length=max_length,
        )

        result_name = f"{modality_name}_{enzyme}_digest"
        data_manager.store_modality(
            name=result_name,
            adata=adata_out,
            parent_name=modality_name,
            step_summary=f"{enzyme} digest: {stats['n_fragments']} fragments",
        )
        data_manager.log_tool_usage(
            tool_name="simulate_enzymatic_digestion",
            parameters={"modality_name": modality_name, "enzyme": enzyme},
            description=f"{enzyme} digest: {stats['n_fragments']} fragments from {stats['n_parents']} sequences",
            ir=ir,
            agent=agent_name,
        )

        return (
            f"STATUS: complete\n"
            f"enzyme={enzyme}\n"
            f"parent_sequences={stats['n_parents']}\n"
            f"fragments={stats['n_fragments']}\n"
            f"unique_fragments={stats['n_unique_fragments']}\n"
            f"mean_length={stats['mean_fragment_length']:.1f}\n"
            f"length_range={stats['length_range']}\n"
            f"modality='{result_name}'"
        )

    simulate_enzymatic_digestion.metadata = {
        "categories": ["PREPROCESS"],
        "provenance": True,
    }
    simulate_enzymatic_digestion.tags = ["PREPROCESS"]

    # =========================================================================
    # TOOL 5: filter_peptides
    # =========================================================================

    @tool
    def filter_peptides(
        modality_name: str,
        min_length: int = 0,
        max_length: int = 100,
        min_charge: float = -100,
        max_charge: float = 100,
        min_hydrophobicity: float = -10,
        max_hydrophobicity: float = 10,
        min_activity_score: float = 0.0,
        activity_column: str = "",
    ) -> str:
        """Filter peptide library by property thresholds.

        Args:
            modality_name: Peptide modality to filter
            min_length: Minimum peptide length
            max_length: Maximum peptide length
            min_charge: Minimum charge
            max_charge: Maximum charge
            min_hydrophobicity: Minimum GRAVY score
            max_hydrophobicity: Maximum GRAVY score
            min_activity_score: Minimum activity prediction score
            activity_column: Activity score column to filter on
        """
        if modality_name not in data_manager.list_modalities():
            return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"

        adata = data_manager.get_modality(modality_name)
        adata_filtered, stats, ir = library_service.filter_peptides(
            adata,
            min_length=min_length,
            max_length=max_length,
            min_charge=min_charge,
            max_charge=max_charge,
            min_hydrophobicity=min_hydrophobicity,
            max_hydrophobicity=max_hydrophobicity,
            min_activity_score=min_activity_score,
            activity_column=activity_column,
        )
        n_before, n_after = stats["n_before"], stats["n_after"]

        result_name = f"{modality_name}_filtered"
        data_manager.store_modality(
            name=result_name,
            adata=adata_filtered,
            parent_name=modality_name,
            step_summary=f"Filtered: {n_before} → {n_after} peptides",
        )

        data_manager.log_tool_usage(
            tool_name="filter_peptides",
            parameters={
                "modality_name": modality_name,
                "min_length": min_length,
                "max_length": max_length,
                "min_charge": min_charge,
                "max_charge": max_charge,
                "min_hydrophobicity": min_hydrophobicity,
                "max_hydrophobicity": max_hydrophobicity,
                "min_activity_score": min_activity_score,
                "activity_column": activity_column,
            },
            description=f"Filter: {n_before} → {n_after}",
            ir=ir,
            agent=agent_name,
        )

        return (
            f"STATUS: complete\n"
            f"before={n_before}\n"
            f"after={n_after}\n"
            f"removed={n_before - n_after}\n"
            f"modality='{result_name}'"
        )

    filter_peptides.metadata = {"categories": ["FILTER"], "provenance": True}
    filter_peptides.tags = ["FILTER"]

    # =========================================================================
    # TOOL 6: annotate_peptide_activity
    # =========================================================================

    @tool
    def annotate_peptide_activity(
        modality_name: str,
        database: str = "dbaasp",
    ) -> str:
        """Cross-reference sequences against peptide databases for known activities.

        Args:
            modality_name: Peptide modality to annotate
            database: Database to query — "dbaasp", "peptipedia", or "all"
        """
        if modality_name not in data_manager.list_modalities():
            return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"

        adata = data_manager.get_modality(modality_name)

        if "sequence" not in adata.obs.columns:
            return "No 'sequence' column found in modality"

        from lobster.tools.providers.dbaasp_provider import DBAASPProvider
        from lobster.tools.providers.peptipedia_provider import PeptipediaProvider

        providers_map = {"dbaasp": DBAASPProvider, "peptipedia": PeptipediaProvider}
        if database == "all":
            selected = list(providers_map.values())
        elif database in providers_map:
            selected = [providers_map[database]]
        else:
            return f"Unknown database '{database}'. Use: dbaasp, peptipedia, or all"

        adata_out, stats, ir = library_service.annotate_known_activity(
            adata, providers=selected, database=database
        )
        n_annotated, n_queried = stats["n_annotated"], stats["n_queried"]

        result_name = f"{modality_name}_annotated"
        data_manager.store_modality(
            name=result_name,
            adata=adata_out,
            parent_name=modality_name,
            step_summary=f"Annotated: {n_annotated}/{n_queried} with known activities",
        )

        data_manager.log_tool_usage(
            tool_name="annotate_peptide_activity",
            parameters={"modality_name": modality_name, "database": database},
            description=f"Annotated {n_annotated}/{n_queried} peptides",
            ir=ir,
            agent=agent_name,
        )

        return (
            f"STATUS: complete\n"
            f"peptides_queried={n_queried}\n"
            f"annotated={n_annotated}\n"
            f"database={database}\n"
            f"modality='{result_name}'"
        )

    annotate_peptide_activity.metadata = {
        "categories": ["ANNOTATE"],
        "provenance": True,
    }
    annotate_peptide_activity.tags = ["ANNOTATE"]

    # =========================================================================
    # TOOL 7: generate_peptide_variants
    # =========================================================================

    @tool
    def generate_peptide_variants(
        modality_name: str,
        variant_type: str = "alanine_scan",
        target_peptide: str = "",
    ) -> str:
        """Generate sequence variants for SAR (structure-activity relationships).

        Args:
            modality_name: Peptide modality
            variant_type: "alanine_scan", "point_mutations", or "scrambled"
            target_peptide: Unique peptide label to target (empty = first row by
                position, including when labels repeat). Missing or ambiguous
                explicit labels are rejected. Notebook replay fails closed if
                its current library differs from this recorded input.
        """
        if modality_name not in data_manager.list_modalities():
            return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"

        adata = data_manager.get_modality(modality_name)

        try:
            adata_out, stats, ir = library_service.generate_variants(
                adata, variant_type=variant_type, target_peptide=target_peptide
            )
        except ValueError as e:
            return str(e)
        parent_name, n_variants = stats["parent"], stats["n_variants"]

        result_name = f"{modality_name}_{variant_type}"
        data_manager.store_modality(
            name=result_name,
            adata=adata_out,
            parent_name=modality_name,
            step_summary=f"{variant_type}: {n_variants} variants of {parent_name}",
        )

        data_manager.log_tool_usage(
            tool_name="generate_peptide_variants",
            parameters={
                "modality_name": modality_name,
                "variant_type": variant_type,
                "target_peptide": target_peptide,
                "parent": parent_name,
                "parent_sequence": stats["parent_sequence"],
                "parent_position": stats["parent_position"],
            },
            description=f"{variant_type}: {n_variants} variants",
            ir=ir,
            agent=agent_name,
        )

        return (
            f"STATUS: complete\n"
            f"variant_type={variant_type}\n"
            f"parent={parent_name}\n"
            f"parent_sequence={stats['parent_sequence']}\n"
            f"variants_generated={n_variants}\n"
            f"modality='{result_name}'"
        )

    generate_peptide_variants.metadata = {
        "categories": ["PREPROCESS"],
        "provenance": True,
    }
    generate_peptide_variants.tags = ["PREPROCESS"]

    # =========================================================================
    # TOOL 8: export_peptide_report
    # =========================================================================

    @tool
    def export_peptide_report(
        modality_name: str,
        output_format: str = "markdown",
    ) -> str:
        """Export peptide analysis report with properties, predictions, and summary.

        Args:
            modality_name: Peptide modality to report on
            output_format: "markdown" or "csv"
        """
        if modality_name not in data_manager.list_modalities():
            return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"

        adata = data_manager.get_modality(modality_name)

        if output_format == "csv":
            import re as _re

            safe_name = _re.sub(r"[^\w\-]", "_", modality_name)
            csv_path = data_manager.exports_dir / f"{safe_name}_report.csv"
            if export_refused(data_manager.workspace_path, csv_path):
                return MANAGED_EXPORT_MESSAGE  # e.g. the file is a link into a figure
            adata.obs.to_csv(csv_path)

            data_manager.log_tool_usage(
                tool_name="export_peptide_report",
                parameters={"modality_name": modality_name, "format": "csv"},
                description=f"Exported CSV: {csv_path}",
                ir=None,
                agent=agent_name,
            )
            return f"STATUS: complete\nformat=csv\nfile={csv_path}\nrows={adata.n_obs}"

        # Markdown report
        lines = [
            f"# Peptide Analysis Report: {modality_name}",
            f"\n**Peptides**: {adata.n_obs}",
            f"**Columns**: {', '.join(adata.obs.columns[:20])}",
            "",
        ]

        # Summary statistics for property columns
        prop_cols = [c for c in adata.obs.columns if c.startswith("peptide_")]
        if prop_cols:
            lines.append("## Property Summary\n")
            lines.append("| Property | Mean | Std | Min | Max |")
            lines.append("|----------|------|-----|-----|-----|")
            for col in prop_cols:
                try:
                    vals = adata.obs[col].astype(float).dropna()
                    if len(vals) > 0:
                        lines.append(
                            f"| {col.replace('peptide_', '')} | "
                            f"{vals.mean():.3f} | {vals.std():.3f} | "
                            f"{vals.min():.3f} | {vals.max():.3f} |"
                        )
                except (ValueError, TypeError):
                    pass
            lines.append("")

        # Activity columns
        act_cols = [c for c in adata.obs.columns if "_score" in c or "_label" in c]
        if act_cols:
            lines.append("## Activity Predictions\n")
            for col in act_cols:
                if "_label" in col:
                    vc = adata.obs[col].value_counts()
                    lines.append(f"**{col}**: {vc.to_dict()}")
            lines.append("")

        # Top peptides table
        if "sequence" in adata.obs.columns:
            lines.append("## Top Peptides (first 20)\n")
            show_cols = (
                ["sequence", "length"]
                + [c for c in prop_cols[:3]]
                + [c for c in act_cols[:2]]
            )
            show_cols = [c for c in show_cols if c in adata.obs.columns]
            if show_cols:
                lines.append("| " + " | ".join(show_cols) + " |")
                lines.append("| " + " | ".join(["---"] * len(show_cols)) + " |")
                for idx in range(min(20, adata.n_obs)):
                    row_vals = []
                    for col in show_cols:
                        val = adata.obs[col].iloc[idx]
                        if isinstance(val, float):
                            row_vals.append(f"{val:.3f}")
                        else:
                            row_vals.append(str(val)[:30])
                    lines.append("| " + " | ".join(row_vals) + " |")

        report = "\n".join(lines)

        data_manager.log_tool_usage(
            tool_name="export_peptide_report",
            parameters={"modality_name": modality_name, "format": "markdown"},
            description=f"Generated report for {adata.n_obs} peptides",
            ir=None,
            agent=agent_name,
        )

        return report

    export_peptide_report.metadata = {"categories": ["UTILITY"], "provenance": False}
    export_peptide_report.tags = ["UTILITY"]

    # =========================================================================
    # Return all 8 domain tools
    # =========================================================================

    return [
        import_peptide_sequences,
        calculate_peptide_properties,
        predict_peptide_activity,
        simulate_enzymatic_digestion,
        filter_peptides,
        annotate_peptide_activity,
        generate_peptide_variants,
        export_peptide_report,
    ]
