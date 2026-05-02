"""
Peptide expert tool factory.

8 domain tools + DELEGATE (handoff_to_supervisor) + CODE_EXEC (execute_custom_code).
All domain tools follow the 3-tuple service pattern with AQUADIF metadata.
"""

import logging
from pathlib import Path
from typing import Callable, List, Optional

from langchain_core.tools import tool

from lobster.core.runtime.data_manager import DataManagerV2

logger = logging.getLogger(__name__)


def create_peptide_tools(
    data_manager: DataManagerV2,
    workspace_path: Optional[Path] = None,
) -> List[Callable]:
    """Create peptide analysis tools with service wiring.

    Returns list of 8 domain tools. DELEGATE and CODE_EXEC are added
    by the agent factory (not here).
    """
    from lobster.services.peptidomics.peptide_activity_service import PeptideActivityService
    from lobster.services.peptidomics.peptide_digestion_service import PeptideDigestionService
    from lobster.services.peptidomics.peptide_property_service import PeptidePropertyService

    property_service = PeptidePropertyService()
    activity_service = PeptideActivityService()
    digestion_service = PeptideDigestionService()

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
        import anndata as ad
        import numpy as np
        import pandas as pd

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

        content = resolved.read_text()
        ext = resolved.suffix.lower()

        # Auto-detect format
        if format == "auto":
            if ext in (".fa", ".fasta", ".faa"):
                format = "fasta"
            elif ext == ".tsv" or "\t" in content.split("\n")[0]:
                format = "tsv"
            else:
                format = "csv"

        sequences = []
        names = []

        if format == "fasta":
            current_name = ""
            current_seq = []
            for line in content.strip().split("\n"):
                if line.startswith(">"):
                    if current_seq:
                        sequences.append("".join(current_seq))
                        names.append(current_name)
                    current_name = line[1:].strip().split()[0]
                    current_seq = []
                else:
                    current_seq.append(line.strip())
            if current_seq:
                sequences.append("".join(current_seq))
                names.append(current_name)
        else:
            sep = "\t" if format == "tsv" else ","
            df = pd.read_csv(resolved, sep=sep)
            # Find sequence column
            seq_col = None
            for col in df.columns:
                if col.lower() in ("sequence", "seq", "peptide", "peptide_sequence"):
                    seq_col = col
                    break
            if seq_col is None:
                seq_col = df.columns[0]
            sequences = df[seq_col].astype(str).tolist()

            # Find name column
            name_col = None
            for col in df.columns:
                if col.lower() in ("name", "id", "accession", "peptide_id"):
                    name_col = col
                    break
            names = df[name_col].astype(str).tolist() if name_col else [f"pep_{i}" for i in range(len(sequences))]

        if not sequences:
            return "No sequences found in file"

        # Validate sequences
        from lobster.agents.proteomics.peptide_config import STANDARD_AA

        valid_seqs = []
        valid_names = []
        invalid_count = 0
        for name, seq in zip(names, sequences):
            clean = "".join(c for c in seq.upper() if c.isalpha())
            if clean and all(aa in STANDARD_AA for aa in clean) and 2 <= len(clean) <= 100:
                valid_seqs.append(clean)
                valid_names.append(name)
            else:
                invalid_count += 1

        obs = pd.DataFrame({
            "sequence": valid_seqs,
            "length": [len(s) for s in valid_seqs],
            "source_file": str(resolved.name),
        }, index=valid_names)

        adata = ad.AnnData(obs=obs)

        data_manager.store_modality(
            name=modality_name,
            adata=adata,
            step_summary=f"Imported {len(valid_seqs)} peptide sequences from {resolved.name}",
        )

        data_manager.log_tool_usage(
            tool_name="import_peptide_sequences",
            parameters={"file_path": str(resolved), "format": format},
            description=f"Import {len(valid_seqs)} peptides from {resolved.name}",
            ir=None,
        )

        result = f"STATUS: complete\npeptides_imported={len(valid_seqs)}"
        if invalid_count:
            result += f"\ninvalid_skipped={invalid_count}"
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

        prop_list = None if properties == "all" else [p.strip() for p in properties.split(",")]

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
        )

        return (
            f"STATUS: complete\n"
            f"peptides_analyzed={stats['n_peptides']}\n"
            f"properties={','.join(stats['properties_computed'])}\n"
            f"modality='{result_name}'\n"
            f"method={stats['method']}"
        )

    calculate_peptide_properties.metadata = {"categories": ["ANALYZE"], "provenance": True}
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

    simulate_enzymatic_digestion.metadata = {"categories": ["PREPROCESS"], "provenance": True}
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

        import pandas as pd

        adata = data_manager.get_modality(modality_name)
        n_before = adata.n_obs

        mask = pd.Series(True, index=adata.obs.index)

        # Length filter
        if "length" in adata.obs.columns or "peptide_length" in adata.obs.columns:
            len_col = "peptide_length" if "peptide_length" in adata.obs.columns else "length"
            lengths = adata.obs[len_col].astype(float)
            mask = mask & (lengths >= min_length) & (lengths <= max_length)
        elif "sequence" in adata.obs.columns:
            lengths = adata.obs["sequence"].str.len()
            mask = mask & (lengths >= min_length) & (lengths <= max_length)

        # Charge filter
        if "peptide_charge" in adata.obs.columns:
            charges = adata.obs["peptide_charge"].astype(float)
            mask = mask & (charges >= min_charge) & (charges <= max_charge)

        # Hydrophobicity filter
        if "peptide_gravy" in adata.obs.columns:
            gravy = adata.obs["peptide_gravy"].astype(float)
            mask = mask & (gravy >= min_hydrophobicity) & (gravy <= max_hydrophobicity)

        # Activity score filter
        if activity_column and activity_column in adata.obs.columns:
            scores = adata.obs[activity_column].astype(float)
            mask = mask & (scores >= min_activity_score)

        mask_array = mask.values
        adata_filtered = adata[mask_array].copy()
        n_after = adata_filtered.n_obs

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
            ir=None,
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

        annotations = []
        sequences = adata.obs["sequence"].astype(str).tolist()

        for seq in sequences:
            found_activities = []
            for provider_cls in selected:
                try:
                    provider = provider_cls()
                    results = provider.search_sequences(seq[:20], max_results=3)
                    for r in results:
                        if r.sequence == seq and r.activities:
                            found_activities.extend(r.activities)
                except Exception as e:
                    logger.warning(f"Annotation lookup failed: {e}")
            annotations.append(", ".join(set(found_activities)) if found_activities else "unknown")

        adata_out = adata.copy()
        adata_out.obs["known_activities"] = annotations

        n_annotated = sum(1 for a in annotations if a != "unknown")

        result_name = f"{modality_name}_annotated"
        data_manager.store_modality(
            name=result_name,
            adata=adata_out,
            parent_name=modality_name,
            step_summary=f"Annotated: {n_annotated}/{len(sequences)} with known activities",
        )

        data_manager.log_tool_usage(
            tool_name="annotate_peptide_activity",
            parameters={"modality_name": modality_name, "database": database},
            description=f"Annotated {n_annotated}/{len(sequences)} peptides",
            ir=None,
        )

        return (
            f"STATUS: complete\n"
            f"peptides_queried={len(sequences)}\n"
            f"annotated={n_annotated}\n"
            f"database={database}\n"
            f"modality='{result_name}'"
        )

    annotate_peptide_activity.metadata = {"categories": ["ANNOTATE"], "provenance": True}
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
            target_peptide: Specific peptide name/index to target (empty = first)
        """
        import anndata as ad
        import pandas as pd

        if modality_name not in data_manager.list_modalities():
            return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"

        adata = data_manager.get_modality(modality_name)

        if "sequence" not in adata.obs.columns:
            return "No 'sequence' column in modality"

        # Select target
        if target_peptide and target_peptide in adata.obs_names:
            seq = adata.obs.loc[target_peptide, "sequence"]
            parent_name = target_peptide
        else:
            seq = adata.obs["sequence"].iloc[0]
            parent_name = adata.obs_names[0]

        variants = []
        if variant_type == "alanine_scan":
            for i, aa in enumerate(seq):
                if aa != "A":
                    mutant = seq[:i] + "A" + seq[i + 1:]
                    variants.append({
                        "sequence": mutant,
                        "variant_type": "alanine_scan",
                        "mutation": f"{aa}{i + 1}A",
                        "position": i + 1,
                        "parent": parent_name,
                    })
        elif variant_type == "point_mutations":
            from lobster.agents.proteomics.peptide_config import STANDARD_AA
            for i, aa in enumerate(seq):
                for new_aa in sorted(STANDARD_AA):
                    if new_aa != aa:
                        mutant = seq[:i] + new_aa + seq[i + 1:]
                        variants.append({
                            "sequence": mutant,
                            "variant_type": "point_mutation",
                            "mutation": f"{aa}{i + 1}{new_aa}",
                            "position": i + 1,
                            "parent": parent_name,
                        })
        elif variant_type == "scrambled":
            import random
            for j in range(10):
                shuffled = list(seq)
                random.shuffle(shuffled)
                variants.append({
                    "sequence": "".join(shuffled),
                    "variant_type": "scrambled",
                    "mutation": f"scramble_{j + 1}",
                    "position": 0,
                    "parent": parent_name,
                })
        else:
            return f"Unknown variant_type '{variant_type}'. Use: alanine_scan, point_mutations, scrambled"

        if not variants:
            return "No variants generated"

        var_df = pd.DataFrame(variants)
        var_df["length"] = var_df["sequence"].str.len()
        var_df.index = [f"var_{i}" for i in range(len(var_df))]

        adata_out = ad.AnnData(obs=var_df)

        result_name = f"{modality_name}_{variant_type}"
        data_manager.store_modality(
            name=result_name,
            adata=adata_out,
            parent_name=modality_name,
            step_summary=f"{variant_type}: {len(variants)} variants of {parent_name}",
        )

        data_manager.log_tool_usage(
            tool_name="generate_peptide_variants",
            parameters={"modality_name": modality_name, "variant_type": variant_type},
            description=f"{variant_type}: {len(variants)} variants",
            ir=None,
        )

        return (
            f"STATUS: complete\n"
            f"variant_type={variant_type}\n"
            f"parent={parent_name}\n"
            f"parent_sequence={seq}\n"
            f"variants_generated={len(variants)}\n"
            f"modality='{result_name}'"
        )

    generate_peptide_variants.metadata = {"categories": ["PREPROCESS"], "provenance": True}
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
            safe_name = _re.sub(r'[^\w\-]', '_', modality_name)
            csv_path = (workspace_path or Path(".")) / f"{safe_name}_report.csv"
            adata.obs.to_csv(csv_path)

            data_manager.log_tool_usage(
                tool_name="export_peptide_report",
                parameters={"modality_name": modality_name, "format": "csv"},
                description=f"Exported CSV: {csv_path}",
                ir=None,
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
            show_cols = ["sequence", "length"] + [c for c in prop_cols[:3]] + [c for c in act_cols[:2]]
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
        )

        return report

    export_peptide_report.metadata = {"categories": ["SYNTHESIZE"], "provenance": True}
    export_peptide_report.tags = ["SYNTHESIZE"]

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
